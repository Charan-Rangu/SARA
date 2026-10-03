"""SARA — Safe CDSE Authentication Verification Tool.

Tests CDSE credentials against the Copernicus Data Space Ecosystem Keycloak authentication
service without downloading any satellite imagery or exposing sensitive values.

Usage:
  python scripts/check_cdse_auth.py
"""

import sys
import os

# Ensure repository root is on sys.path
REPO_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
if REPO_ROOT not in sys.path:
    sys.path.insert(0, REPO_ROOT)

from backend.config import settings
from backend.services.download_service import (
    check_cdse_auth_status,
    verify_cdse_authentication
)


def run_auth_check() -> int:
    print("=" * 72)
    print("      SARA — COPERNICUS DATA SPACE ECOSYSTEM (CDSE) AUTH CHECK")
    print("=" * 72)

    status_info = check_cdse_auth_status()

    # 1. Environment Variable Presence Audit (No values printed)
    has_user = bool(settings.cdse_username)
    has_pwd = bool(settings.cdse_password)
    has_token = bool(settings.cdse_access_token)

    print("\n[1/2] ENVIRONMENT CONFIGURATION AUDIT:")
    print(f"  CDSE_USERNAME:     {'[SET]' if has_user else '[NOT SET]'}")
    print(f"  CDSE_PASSWORD:     {'[SET]' if has_pwd else '[NOT SET]'}")
    print(f"  CDSE_ACCESS_TOKEN: {'[SET]' if has_token else '[NOT SET]'}")
    print(f"  Auth Mode:         {status_info['method']}")

    if not status_info["configured"]:
        print("\n[2/2] AUTHENTICATION TEST:")
        print(f"  Status:  [BLOCKED]")
        print(f"  Details: {status_info['status_message']}")
        print("\nTo configure CDSE authentication:")
        print("1. Open the .env file in the project root:")
        print(f"   {os.path.join(REPO_ROOT, '.env')}")
        print("2. Add your Copernicus Data Space Ecosystem credentials:")
        print("   CDSE_USERNAME=your_copernicus_email@example.com")
        print("   CDSE_PASSWORD=your_copernicus_password")
        print("3. Re-run this check: python scripts/check_cdse_auth.py")
        print("\nNote: Create a free account at https://dataspace.copernicus.eu/ if needed.")
        return 1

    # 2. Live Keycloak Service Authentication Test (No data downloaded)
    print("\n[2/2] AUTHENTICATION ENDPOINT TEST:")
    print("  Connecting to CDSE Keycloak authentication endpoint...")
    is_valid, msg = verify_cdse_authentication()

    if is_valid:
        print(f"  Status:  [PASS]")
        print(f"  Details: {msg}")
        print("\nAuthentication verified! SARA is ready for Sentinel-1 product downloads.")
        return 0
    else:
        print(f"  Status:  [FAIL]")
        print(f"  Details: {msg}")
        print("\nPlease check your credentials in .env and try again.")
        return 1


if __name__ == "__main__":
    sys.exit(run_auth_check())
