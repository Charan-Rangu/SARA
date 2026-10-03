"""Data Download Service, Manifest Manager, and Archive Integrity Verification for SARA.

Manages data downloads from official CDSE endpoints with:
1. Strict user confirmation safeguards for multi-gigabyte downloads.
2. Storage budget & free disk space verification prior to acquisition.
3. Temporary (.part) file usage to prevent partial downloads being mistaken for complete products.
4. Resumable downloads via HTTP Range headers.
5. Product archive integrity checks (ZIP structure testzip & file size verification).
6. Non-destructive file handling (no silent overwriting of valid products).
7. Secure authentication credential handling (no hardcoded tokens/passwords logged).
"""

import os
import shutil
import zipfile
import hashlib
from datetime import datetime, timezone
from typing import Optional, Any
from pydantic import BaseModel, Field
import httpx

from backend.config import settings
from backend.services.sentinel1_catalog import Sentinel1Scene
from backend.services.download_planner import DownloadPlanner, DownloadPlan

CDSE_TOKEN_URL = "https://identity.dataspace.copernicus.eu/auth/realms/CDSE/protocol/openid-connect/token"


def get_cdse_auth_token(
    username: Optional[str] = None,
    password: Optional[str] = None,
    direct_token: Optional[str] = None,
    timeout_seconds: float = 15.0,
    force_refresh: bool = False
) -> tuple[Optional[str], str]:
    """Obtain a CDSE OAuth2 access token securely.

    Never prints, logs, or exposes passwords or tokens.

    Returns:
        tuple[Optional[str], str]: (access_token, status_message).
    """
    if not force_refresh:
        if direct_token:
            return direct_token, "CDSE access token provided directly."
        if settings.cdse_access_token:
            return settings.cdse_access_token, "CDSE access token provided via configuration/environment."

    user = username or settings.cdse_username
    pwd = password or settings.cdse_password
    if not user or not pwd:
        return None, "CDSE credentials not configured (CDSE_USERNAME / CDSE_PASSWORD not set in environment or .env)."

    try:
        data = {
            "client_id": "cdse-public",
            "grant_type": "password",
            "username": user,
            "password": pwd
        }
        with httpx.Client(timeout=timeout_seconds) as client:
            resp = client.post(CDSE_TOKEN_URL, data=data)
            if resp.status_code == 200:
                payload = resp.json()
                access_token = payload.get("access_token")
                if access_token:
                    settings.cdse_access_token = access_token
                    return access_token, "Authenticated successfully with CDSE Keycloak service."
                return None, "CDSE response did not contain an access_token."
            elif resp.status_code in (400, 401):
                return None, "CDSE authentication failed: invalid username or password."
            else:
                return None, f"CDSE token endpoint returned HTTP status {resp.status_code}."
    except Exception as e:
        return None, f"CDSE authentication error: {type(e).__name__}"


def check_cdse_auth_status() -> dict[str, Any]:
    """Check CDSE authentication readiness without exposing credentials.

    Returns:
        dict containing 'configured', 'method', and 'status_message'.
    """
    has_token = bool(settings.cdse_access_token)
    has_user = bool(settings.cdse_username)
    has_pwd = bool(settings.cdse_password)

    if has_token:
        return {
            "configured": True,
            "method": "DIRECT_TOKEN",
            "status_message": "CDSE direct access token configured in environment."
        }
    elif has_user and has_pwd:
        return {
            "configured": True,
            "method": "CREDENTIALS",
            "status_message": "CDSE username and password configured in environment."
        }
    elif has_user:
        return {
            "configured": False,
            "method": "INCOMPLETE",
            "status_message": "CDSE username set, but CDSE_PASSWORD is missing."
        }
    else:
        return {
            "configured": False,
            "method": "NONE",
            "status_message": "CDSE credentials not configured. Downloads require CDSE_USERNAME and CDSE_PASSWORD in .env."
        }


CDSE_USERINFO_URL = "https://identity.dataspace.copernicus.eu/auth/realms/CDSE/protocol/openid-connect/userinfo"


def verify_cdse_authentication(
    username: Optional[str] = None,
    password: Optional[str] = None,
    direct_token: Optional[str] = None,
    timeout_seconds: float = 15.0
) -> tuple[bool, str]:
    """Test and verify CDSE authentication without initiating any product downloads.

    Tests credentials against Keycloak token endpoint or validates an access token against
    the Keycloak userinfo endpoint.
    Never exposes passwords, tokens, or credential-bearing URLs.

    Returns:
        tuple[bool, str]: (is_authenticated, sanitized_status_message).
    """
    token_candidate = direct_token or settings.cdse_access_token
    token_expired = False
    if token_candidate:
        try:
            headers = {
                "Authorization": f"Bearer {token_candidate}",
                "User-Agent": "SARA-Rescue-Assessment/1.0"
            }
            with httpx.Client(timeout=timeout_seconds) as client:
                resp = client.get(CDSE_USERINFO_URL, headers=headers)
                if resp.status_code == 200:
                    return True, "Authentication successful: CDSE access token is valid and active."
                elif resp.status_code in (401, 403):
                    token_expired = True
                    if direct_token:
                        return False, "Authentication failed: CDSE access token is expired or invalid (HTTP 401 Unauthorized)."
                else:
                    if direct_token:
                        return False, f"Authentication test returned unexpected HTTP {resp.status_code}."
        except Exception as e:
            if direct_token:
                return False, f"Authentication verification network error: {type(e).__name__}"

    user = username or settings.cdse_username
    pwd = password or settings.cdse_password
    if not user or not pwd:
        if token_expired:
            return False, "Authentication failed: CDSE access token is expired or invalid (HTTP 401 Unauthorized)."
        return False, "Authentication not configured: CDSE_USERNAME and CDSE_PASSWORD are not set in .env."

    token, msg = get_cdse_auth_token(username=user, password=pwd, timeout_seconds=timeout_seconds, force_refresh=True)
    if token:
        return True, "Authentication successful: CDSE credentials verified with Keycloak token service."
    else:
        return False, f"Authentication failed: {msg}"




class ProductManifestEntry(BaseModel):
    """Manifest record tracking an individual product download status and integrity."""

    product_id: str
    title: str
    download_url: Optional[str] = None
    expected_size_bytes: Optional[int] = None
    downloaded_size_bytes: int = 0
    target_filepath: str
    part_filepath: Optional[str] = None
    status: str = "PENDING"  # "PENDING", "CONFIRMATION_REQUIRED", "IN_PROGRESS", "COMPLETED", "CORRUPTED", "AUTH_FAILED", "STORAGE_EXCEEDED"
    checksum_sha256: Optional[str] = None
    integrity_verified: bool = False
    error_message: Optional[str] = None
    timestamp_utc: str = Field(default_factory=lambda: datetime.now(timezone.utc).isoformat())


class DownloadManifest(BaseModel):
    """Manifest container tracking overall download batch progress and disk storage checks."""

    batch_id: str
    created_at_utc: str = Field(default_factory=lambda: datetime.now(timezone.utc).isoformat())
    total_products_count: int = 0
    total_estimated_size_mb: float = 0.0
    storage_budget_gb: float = 10.0
    free_disk_space_gb: float = 0.0
    user_confirmed: bool = False
    manifest_entries: list[ProductManifestEntry] = Field(default_factory=list)
    summary_notes: list[str] = Field(default_factory=list)


class DataDownloadService:
    """Service for managing secure, confirmed, and verified satellite data downloads."""

    def __init__(self, download_dir: str = os.path.join("data", "raw"), default_budget_gb: float = 10.0):
        self.download_dir = download_dir
        self.default_budget_gb = default_budget_gb
        os.makedirs(self.download_dir, exist_ok=True)

    def get_free_disk_space_gb(self, path: Optional[str] = None) -> float:
        """Check available disk space in GB for the target volume."""
        target_path = path or self.download_dir
        vol = os.path.abspath(target_path)
        while not os.path.exists(vol) and len(vol) > 3:
            vol = os.path.dirname(vol)
        usage = shutil.disk_usage(vol)
        return round(usage.free / (1024 ** 3), 2)

    def prepare_download_manifest(
        self,
        batch_id: str,
        scenes: list[Sentinel1Scene],
        storage_budget_gb: Optional[float] = None
    ) -> DownloadManifest:
        """Create a download manifest and assess storage budget & free space prior to download.

        Args:
            batch_id: Identifier for the download task or run.
            scenes: List of Sentinel1Scene objects to download.
            storage_budget_gb: Optional custom storage limit in GB.

        Returns:
            DownloadManifest containing pre-download analysis and entries requiring user confirmation.
        """
        budget = storage_budget_gb if storage_budget_gb is not None else self.default_budget_gb
        free_space = self.get_free_disk_space_gb()

        planner = DownloadPlanner(default_storage_budget_gb=budget)
        plan = planner.create_download_plan(scenes, storage_budget_gb=budget)

        entries: list[ProductManifestEntry] = []
        notes: list[str] = []

        if plan.total_estimated_size_gb > free_space:
            notes.append(
                f"STORAGE WARNING: Total download size ({plan.total_estimated_size_gb:.2f} GB) "
                f"exceeds available free disk space ({free_space:.2f} GB)."
            )

        for scene in scenes:
            target_path = os.path.join(self.download_dir, f"{scene.title}.zip")
            part_path = f"{target_path}.part"
            status = "CONFIRMATION_REQUIRED" if plan.total_estimated_size_gb > 0.1 else "PENDING"

            # Check if file already exists locally and passes integrity
            if os.path.exists(target_path):
                is_valid, msg = self.verify_archive_integrity(target_path, scene.size_bytes)
                if is_valid:
                    status = "COMPLETED"
                    notes.append(f"Product '{scene.title}' already exists locally and passed integrity check.")

            current_bytes = 0
            if os.path.exists(target_path):
                current_bytes = os.path.getsize(target_path)
            elif os.path.exists(part_path):
                current_bytes = os.path.getsize(part_path)

            entries.append(
                ProductManifestEntry(
                    product_id=scene.id,
                    title=scene.title,
                    download_url=scene.download_url,
                    expected_size_bytes=scene.size_bytes,
                    downloaded_size_bytes=current_bytes,
                    target_filepath=target_path,
                    part_filepath=part_path,
                    status=status
                )
            )

        return DownloadManifest(
            batch_id=batch_id,
            total_products_count=len(scenes),
            total_estimated_size_mb=plan.total_estimated_size_mb,
            storage_budget_gb=budget,
            free_disk_space_gb=free_space,
            user_confirmed=False,
            manifest_entries=entries,
            summary_notes=notes
        )

    def verify_archive_integrity(self, file_path: str, expected_size_bytes: Optional[int] = None) -> tuple[bool, str]:
        """Validate local downloaded product archive integrity.

        Checks:
        1. File existence.
        2. Non-zero size.
        3. Size consistency within 2% tolerance if expected bytes are provided.
        4. Valid ZIP archive headers and non-corrupted internal structure via testzip().

        Returns:
            Tuple of (is_valid: bool, status_message: str).
        """
        if not os.path.isfile(file_path):
            return False, f"File '{file_path}' does not exist."

        actual_size = os.path.getsize(file_path)
        if actual_size == 0:
            return False, f"File '{file_path}' is empty (0 bytes)."

        if expected_size_bytes is not None and expected_size_bytes > 0:
            diff_pct = abs(actual_size - expected_size_bytes) / expected_size_bytes * 100
            if diff_pct > 2.0:
                return False, f"Size mismatch: actual ({actual_size} bytes) vs expected ({expected_size_bytes} bytes)."

        # Test ZIP file header and internal structure
        if file_path.endswith(".zip") or file_path.endswith(".SAFE"):
            try:
                if zipfile.is_zipfile(file_path):
                    with zipfile.ZipFile(file_path, "r") as zf:
                        corrupted_file = zf.testzip()
                        if corrupted_file is not None:
                            return False, f"ZIP file corrupted at internal entry: '{corrupted_file}'."
                else:
                    return False, f"File '{file_path}' is not a valid ZIP archive."
            except Exception as e:
                return False, f"ZIP archive validation error: {e}"

        return True, f"Archive passed integrity verification ({actual_size / (1024**2):.1f} MB)."

    def download_product_with_safeguards(
        self,
        entry: ProductManifestEntry,
        user_confirmed: bool = False,
        auth_token: Optional[str] = None
    ) -> ProductManifestEntry:
        """Download a single satellite product with strict confirmation and error safeguards.

        Enforces:
        - Free disk space check prior to download.
        - User confirmation required for products > 100 MB.
        - Output written to temporary .part file so incomplete files are never mistaken for completed products.
        - Atomic rename from .part to final target only upon passing full integrity verification.
        - No credentials logged.
        """
        size_gb = (entry.expected_size_bytes or 0) / (1024 ** 3)
        free_space = self.get_free_disk_space_gb()

        # 1. Disk Space Check
        if size_gb > free_space:
            entry.status = "STORAGE_EXCEEDED"
            entry.error_message = (
                f"Insufficient disk space: required {size_gb:.2f} GB, available {free_space:.2f} GB."
            )
            return entry

        # 2. Existing Valid File Check (Never silently overwrite valid file)
        if os.path.exists(entry.target_filepath):
            is_valid, _ = self.verify_archive_integrity(entry.target_filepath, entry.expected_size_bytes)
            if is_valid:
                entry.status = "COMPLETED"
                entry.integrity_verified = True
                entry.downloaded_size_bytes = os.path.getsize(entry.target_filepath)
                entry.error_message = None
                if not entry.checksum_sha256:
                    h = hashlib.sha256()
                    with open(entry.target_filepath, "rb") as f_hash:
                        while c := f_hash.read(4 * 1024 * 1024):
                            h.update(c)
                    entry.checksum_sha256 = h.hexdigest()
                return entry

        # 3. Explicit User Confirmation Check for large downloads
        if size_gb > 0.1 and not user_confirmed:
            entry.status = "CONFIRMATION_REQUIRED"
            entry.error_message = (
                f"Multi-gigabyte download ({size_gb:.2f} GB) requires explicit user confirmation (`user_confirmed=True`)."
            )
            return entry

        if not entry.download_url:
            entry.status = "FAILED"
            entry.error_message = "No download URL available in scene metadata."
            return entry

        headers = {"User-Agent": "SARA-Rescue-Assessment/1.0 (contact@sara-rescue.org)"}
        active_token = auth_token
        if not active_token:
            active_token, _ = get_cdse_auth_token()
        if active_token:
            headers["Authorization"] = f"Bearer {active_token}"

        # Resume support using .part file
        part_path = entry.part_filepath or f"{entry.target_filepath}.part"
        resume_bytes = 0
        if os.path.exists(part_path):
            resume_bytes = os.path.getsize(part_path)
            if resume_bytes > 0:
                headers["Range"] = f"bytes={resume_bytes}-"

        entry.status = "IN_PROGRESS"
        try:
            with httpx.Client(timeout=60.0, follow_redirects=True, headers=headers) as client:
                with client.stream("GET", entry.download_url) as response:
                    if response.status_code in (401, 403):
                        # Attempt auto-refresh if username & password are available
                        new_token, _ = get_cdse_auth_token(
                            username=settings.cdse_username,
                            password=settings.cdse_password,
                            direct_token=None
                        )
                        if new_token and new_token != active_token:
                            active_token = new_token
                            headers["Authorization"] = f"Bearer {active_token}"
                            with httpx.Client(timeout=60.0, follow_redirects=True, headers=headers) as retry_client:
                                with retry_client.stream("GET", entry.download_url) as retry_resp:
                                    if retry_resp.status_code in (401, 403):
                                        entry.status = "AUTH_FAILED"
                                        entry.error_message = "CDSE authentication failed after token refresh attempt."
                                        return entry
                                    retry_resp.raise_for_status()
                                    mode = "ab" if (resume_bytes > 0 and retry_resp.status_code == 206) else "wb"
                                    with open(part_path, mode) as f:
                                        for chunk in retry_resp.iter_bytes(chunk_size=1024 * 1024):
                                            f.write(chunk)
                        else:
                            entry.status = "AUTH_FAILED"
                            entry.error_message = (
                                "CDSE authentication failed: Access token expired or invalid (HTTP 401/403)."
                            )
                            return entry
                    else:
                        response.raise_for_status()
                        mode = "ab" if (resume_bytes > 0 and response.status_code == 206) else "wb"
                        downloaded_now = resume_bytes if mode == "ab" else 0
                        total_expected = entry.expected_size_bytes or 0
                        last_logged_mb = downloaded_now // (100 * 1024 * 1024)
                        with open(part_path, mode) as f:
                            for chunk in response.iter_bytes(chunk_size=1024 * 1024):
                                f.write(chunk)
                                downloaded_now += len(chunk)
                                current_mb = downloaded_now // (100 * 1024 * 1024)
                                if current_mb > last_logged_mb:
                                    last_logged_mb = current_mb
                                    pct = (downloaded_now / total_expected * 100) if total_expected else 0
                                    print(f"    ... downloaded {downloaded_now / (1024*1024):.1f} MB / {total_expected / (1024*1024):.1f} MB ({pct:.1f}%)", flush=True)

            # Post-download integrity verification on .part file
            is_valid, msg = self.verify_archive_integrity(part_path, entry.expected_size_bytes)

            if is_valid:
                # Atomic rename from .part to final target
                if os.path.exists(entry.target_filepath):
                    os.remove(entry.target_filepath)
                shutil.move(part_path, entry.target_filepath)

                entry.downloaded_size_bytes = os.path.getsize(entry.target_filepath)
                entry.status = "COMPLETED"
                entry.integrity_verified = True
                entry.error_message = None

                # Compute SHA256 checksum on verified archive
                h = hashlib.sha256()
                with open(entry.target_filepath, "rb") as f_hash:
                    while c := f_hash.read(4 * 1024 * 1024):
                        h.update(c)
                entry.checksum_sha256 = h.hexdigest()
            else:
                entry.status = "CORRUPTED"
                entry.integrity_verified = False
                entry.error_message = msg

        except Exception as e:
            entry.status = "FAILED"
            entry.error_message = f"Download failed ({type(e).__name__}: {e})"

        return entry
