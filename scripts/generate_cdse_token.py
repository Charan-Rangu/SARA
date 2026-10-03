import getpass
import os
from pathlib import Path

import requests

TOKEN_URL = (
    "https://identity.dataspace.copernicus.eu/"
    "auth/realms/CDSE/protocol/openid-connect/token"
)

PROJECT_ROOT = Path(__file__).resolve().parent.parent
ENV_FILE = PROJECT_ROOT / ".env"


def save_token(token: str) -> None:
    """Save the access token to .env without printing it."""
    lines = []

    if ENV_FILE.exists():
        lines = ENV_FILE.read_text(encoding="utf-8").splitlines()

    # Remove any existing active or commented access-token setting.
    lines = [
        line
        for line in lines
        if not line.lstrip().startswith("CDSE_ACCESS_TOKEN=")
        and not line.lstrip().startswith("# CDSE_ACCESS_TOKEN=")
    ]

    lines.append(f"CDSE_ACCESS_TOKEN={token}")
    ENV_FILE.write_text("\n".join(lines) + "\n", encoding="utf-8")

    # Best effort: restrict access to the file on Windows.
    try:
        os.chmod(ENV_FILE, 0o600)
    except OSError:
        pass


def main():
    print("CDSE Access Token Generator")
    print("-" * 35)

    username = input("Enter your CDSE username/email: ").strip()
    password = getpass.getpass("Enter your CDSE password: ")

    if not username or not password:
        print("Username and password are required.")
        return

    payload = {
        "client_id": "cdse-public",
        "grant_type": "password",
        "username": username,
        "password": password,
    }

    # Ask for a one-time code only if your account uses TOTP.
    use_totp = input(
        "Does your account require a TOTP code? (y/N): "
    ).strip().lower()

    if use_totp == "y":
        payload["totp"] = input("Enter your current TOTP code: ").strip()

    try:
        response = requests.post(
            TOKEN_URL,
            data=payload,
            timeout=30,
        )

        if not response.ok:
            print(f"\nAuthentication failed (HTTP {response.status_code}).")

            try:
                error = response.json()
                print(
                    "Reason:",
                    error.get(
                        "error_description",
                        error.get("error", "Unknown authentication error"),
                    ),
                )
            except ValueError:
                print("The server returned an unreadable error response.")

            return

        result = response.json()
        token = result.get("access_token")

        if not token:
            print("The response did not contain an access token.")
            return

        save_token(token)

        print("\nSUCCESS: Access token saved to the project's .env file.")
        print("The token was not displayed.")
        print("Token lifetime (seconds):", result.get("expires_in", "Not provided"))
        print("\nNext, run:")
        print(r".\.venv\Scripts\python.exe scripts\check_cdse_auth.py")

    except requests.RequestException as exc:
        print("\nCould not contact the CDSE authentication service.")
        print("Error type:", type(exc).__name__)


if __name__ == "__main__":
    main()
