"""Global Configuration Settings & Environment Manager for SARA.

Centralizes path discovery, environment variables, storage limits, and ESA SNAP settings.
Loads .env configuration file if present.
"""

import os
import shutil
from typing import Optional
from pydantic import BaseModel, Field

REPO_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
ENV_PATH = os.path.join(REPO_ROOT, ".env")

try:
    from dotenv import load_dotenv
    if os.path.isfile(ENV_PATH):
        load_dotenv(dotenv_path=ENV_PATH, override=True)
    else:
        load_dotenv(override=True)
except ImportError:
    pass

# Standard candidate locations for ESA SNAP gpt.exe on Windows
STANDARD_SNAP_LOCATIONS = [
    r"C:\Program Files\esa-snap\bin\gpt.exe",
    r"C:\Program Files\snap\bin\gpt.exe",
    r"C:\Program Files\SNAP 14.0.0\bin\gpt.exe",
    r"C:\Program Files\SNAP\bin\gpt.exe",
    r"C:\Program Files (x86)\esa-snap\bin\gpt.exe",
    r"C:\Program Files (x86)\snap\bin\gpt.exe",
    os.path.expanduser(r"~\AppData\Local\Programs\esa-snap\bin\gpt.exe"),
    os.path.expanduser(r"~\AppData\Local\Programs\snap\bin\gpt.exe"),
    os.path.expanduser(r"~\AppData\Local\snap\bin\gpt.exe"),
    r"C:\snap\bin\gpt.exe"
]


def resolve_snap_gpt_path(custom_path: Optional[str] = None) -> Optional[str]:
    """Resolve the ESA SNAP gpt executable path in priority order:
    
    1. Custom explicit path passed to function.
    2. Environment variable 'SNAP_GPT_PATH'.
    3. Environment variable 'ESA_SNAP_HOME' (appending bin/gpt.exe).
    4. Environment variable 'SNAP_HOME' (appending bin/gpt.exe).
    5. Executable on system PATH ('gpt' or 'gpt.exe').
    6. Standard installation locations.
    """
    if custom_path and os.path.isfile(custom_path):
        return os.path.abspath(custom_path)

    # 1. Environment variable: SNAP_GPT_PATH
    env_gpt = os.getenv("SNAP_GPT_PATH")
    if env_gpt:
        if os.path.isfile(env_gpt):
            return os.path.abspath(env_gpt)
        candidate = os.path.join(env_gpt, "bin", "gpt.exe")
        if os.path.isfile(candidate):
            return os.path.abspath(candidate)
        candidate_direct = os.path.join(env_gpt, "gpt.exe")
        if os.path.isfile(candidate_direct):
            return os.path.abspath(candidate_direct)

    # 2. Environment variable: ESA_SNAP_HOME or SNAP_HOME
    for env_home in ["ESA_SNAP_HOME", "SNAP_HOME"]:
        home_val = os.getenv(env_home)
        if home_val:
            candidate = os.path.join(home_val, "bin", "gpt.exe")
            if os.path.isfile(candidate):
                return os.path.abspath(candidate)

    # 3. System PATH lookup
    path_which = shutil.which("gpt") or shutil.which("gpt.exe")
    if path_which and os.path.isfile(path_which):
        return os.path.abspath(path_which)

    # 4. Standard Windows filesystem locations
    for loc in STANDARD_SNAP_LOCATIONS:
        if os.path.isfile(loc):
            return os.path.abspath(loc)

    return None


class SARAConfig(BaseModel):
    """Configuration model for SARA runtime settings."""

    snap_gpt_path: Optional[str] = Field(default_factory=resolve_snap_gpt_path)
    storage_budget_gb: float = Field(default_factory=lambda: float(os.getenv("STORAGE_BUDGET_GB", "10.0")))
    default_dem_name: str = Field(default_factory=lambda: os.getenv("DEFAULT_DEM_NAME", "Copernicus 30m Global DEM"))
    target_crs: str = Field(default_factory=lambda: os.getenv("DEFAULT_CRS", "EPSG:4326"))
    cdse_username: Optional[str] = Field(default_factory=lambda: os.getenv("CDSE_USERNAME"))
    cdse_password: Optional[str] = Field(default_factory=lambda: os.getenv("CDSE_PASSWORD"))
    cdse_access_token: Optional[str] = Field(default_factory=lambda: os.getenv("CDSE_ACCESS_TOKEN"))
    data_raw_dir: str = os.path.join("data", "raw")
    data_processed_dir: str = os.path.join("data", "processed")
    data_output_dir: str = os.path.join("data", "output")


# Global singleton instance
settings = SARAConfig()
