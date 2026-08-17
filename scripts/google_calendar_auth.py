"""Google Calendar OAuth2 helper.

Obtains a refresh token for the Google Calendar API using ``credentials.json``
downloaded from the Google Cloud Console (OAuth2 Client of type "Desktop app").

The script:
    1. Loads client credentials from ``credentials.json`` (never hardcoded).
    2. Runs the ``InstalledAppFlow`` local-server flow with the ``calendar`` scope.
    3. Opens the default browser for one-time Google authorization.
    4. Prints ``GOOGLE_CLIENT_ID``, ``GOOGLE_CLIENT_SECRET`` and
       ``GOOGLE_REFRESH_TOKEN`` for the AI Employee Calendar integration.
    5. Optionally writes the refresh token to ``.env`` (off by default).

No credentials or tokens are committed; ``credentials.json`` and ``.env`` are
git-ignored.

Requirements:
    google-auth-oauthlib>=1.0.0  (installed in the orchestrator service)

Usage (from repo root):
    python -m scripts.google_calendar_auth
    python -m scripts.google_calendar_auth --credentials path/to/credentials.json
    python -m scripts.google_calendar_auth --write-env        # persist token to .env
"""

import argparse
import sys
from pathlib import Path

SCOPE = "https://www.googleapis.com/auth/calendar"
CREDENTIALS_FILENAME = "credentials.json"


def _find_credentials(explicit: str | None) -> Path:
    if explicit:
        path = Path(explicit).resolve()
        if not path.exists():
            raise SystemExit(f"ERROR: credentials file not found: {explicit}")
        return path

    candidates = [
        Path.cwd() / CREDENTIALS_FILENAME,
        Path(__file__).resolve().parents[1] / CREDENTIALS_FILENAME,
    ]
    for candidate in candidates:
        if candidate.exists():
            return candidate
    raise SystemExit(
        "ERROR: credentials.json not found. Download it from the Google Cloud "
        "Console (APIs & Services > Credentials > Create OAuth client ID > "
        "Desktop app) and place it in the repo root, or pass --credentials."
    )


def _print_credentials(
    client_id: str,
    client_secret: str,
    refresh_token: str,
    write_env: bool,
) -> int:
    print("\n" + "=" * 72)
    print("Copy these values into the orchestrator .env file:")
    print("=" * 72)
    print(f"GOOGLE_CLIENT_ID={client_id}")
    print(f"GOOGLE_CLIENT_SECRET={client_secret}")
    print(f"GOOGLE_REFRESH_TOKEN={refresh_token}")
    print("=" * 72)

    if not write_env:
        print(
            "\nTip: pass --write-env to save the refresh token to .env automatically."
        )
        return 0

    env_file = Path(".env") if Path(".env").exists() else Path("services/orchestrator/.env")
    lines = env_file.read_text(encoding="utf-8").splitlines() if env_file.exists() else []
    updates = {
        "GOOGLE_CLIENT_ID": client_id,
        "GOOGLE_CLIENT_SECRET": client_secret,
        "GOOGLE_REFRESH_TOKEN": refresh_token,
        "GOOGLE_CALENDAR_SCOPES": SCOPE,
        "CALENDAR_ENABLED": "true",
        "CALENDAR_PROVIDER": "auto",
    }
    for key, value in updates.items():
        found = False
        for i, line in enumerate(lines):
            if line.strip().startswith(f"{key}="):
                lines[i] = f"{key}={value}"
                found = True
                break
        if not found:
            lines.append(f"{key}={value}")
    env_file.parent.mkdir(parents=True, exist_ok=True)
    env_file.write_text("\n".join(lines) + "\n", encoding="utf-8")
    print(f"\nToken saved to {env_file}. Restart the orchestrator to apply.")
    return 0


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(
        description="Obtain a Google Calendar OAuth2 refresh token."
    )
    parser.add_argument(
        "--credentials",
        default=None,
        help="Path to the OAuth2 client credentials.json file.",
    )
    parser.add_argument(
        "--write-env",
        action="store_true",
        help="Also write the refresh token to the project .env file.",
    )
    args = parser.parse_args(argv)

    try:
        from google_auth_oauthlib.flow import InstalledAppFlow
    except ImportError:
        print(
            "ERROR: google-auth-oauthlib is not installed.\n"
            "Run this script from the orchestrator service or install it with:\n"
            "    pip install 'google-auth-oauthlib>=1.0.0'",
            file=sys.stderr,
        )
        return 1

    credentials_path = _find_credentials(args.credentials)

    try:
        flow = InstalledAppFlow.from_client_secrets_file(
            str(credentials_path),
            scopes=[SCOPE],
        )
    except Exception as exc:  # noqa: BLE001 - surface the client config error
        print(f"ERROR: could not read OAuth client config: {exc}", file=sys.stderr)
        return 1

    client_id = flow.client_config.get("client_id", "")
    client_secret = flow.client_config.get("client_secret", "")
    if not client_id or not client_secret:
        print(
            "ERROR: credentials.json is missing client_id/client_secret. "
            "Download a valid Desktop app OAuth client.",
            file=sys.stderr,
        )
        return 1

    print(
        f"Opening browser for Google authorization (scope: {SCOPE})...\n"
        "If the browser does not open, copy the URL below into your browser:\n"
    )
    print(flow.authorization_url(access_type="offline", prompt="consent")[0])

    try:
        creds = flow.run_local_server(open_browser=True, access_type="offline", prompt="consent")
    except Exception as exc:  # noqa: BLE001 - local server may fail on some setups
        print(f"ERROR: authorization failed: {exc}", file=sys.stderr)
        print(
            "If a localhost redirect server is not available on this machine, "
            "use the old out-of-band flow or run this script on a machine with "
            "a local browser.",
            file=sys.stderr,
        )
        return 1

    refresh_token = creds.refresh_token
    if not refresh_token:
        print(
            "ERROR: no refresh_token returned. The flow requested offline access "
            "and consent, so a refresh token should have been issued.",
            file=sys.stderr,
        )
        return 1

    return _print_credentials(client_id, client_secret, refresh_token, args.write_env)


if __name__ == "__main__":
    raise SystemExit(main())
