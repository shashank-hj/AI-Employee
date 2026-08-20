#!/usr/bin/env python3
"""Live voice (CALL) session with the hosted Sarvam Samvaad agent.

Mirrors the SDK's canonical ``examples/async_audio_example.py`` but pulls the
credentials and agent identifiers from the platform's ``.env`` instead of
hardcoding them, and sets the required ``user_identifier_type``.

Requirements:
    - ``sarvam-conv-ai-sdk[audio]`` (installs the SDK + PyAudio)
    - ``.env`` populated with ``SAMVAAD_API_KEY``, ``SAMVAAD_AGENT_ID``,
      ``SAMVAAD_ORG_ID``, ``SAMVAAD_WORKSPACE_ID``.

Usage:
    python scripts/talk_to_agent.py
    python scripts/talk_to_agent.py --language Hindi
    python scripts/talk_to_agent.py --version 3   # pin a committed agent version

The agent streams your microphone to the hosted agent and plays the agent's
audio back through your speakers. Press Ctrl+C to end the session.
"""

import argparse
import asyncio
import logging
import os
import sys
from pathlib import Path
from urllib.request import Request, urlopen

from pydantic import SecretStr

logger = logging.getLogger("talk_to_agent")


def _load_env() -> None:
    """Best-effort load of .env so the script works standalone."""
    try:
        from dotenv import load_dotenv

        root = Path(__file__).resolve().parent.parent
        load_dotenv(root / ".env")
        load_dotenv(root / ".env-opencode")
    except Exception:
        pass


DEFAULT_TOOL_BASE_URL = "https://comic-paragraph-peroxide.ngrok-free.dev"


def _health_check(base_url: str, timeout: float = 10.0) -> tuple[bool, str]:
    """Probe the public tool-webhook base URL that the Sarvam agent calls.

    The hosted agent reaches the platform's tools (calendar, email, ...) via
    this URL. If it is down (e.g. the ngrok tunnel isn't running), every tool
    call the agent makes fails with "couldn't reach ... / couldn't send ...",
    so we check it before opening the session and warn loudly.
    """
    url = base_url.rstrip("/") + "/health"
    req = Request(url, headers={"User-Agent": "talk-to-agent", "ngrok-skip-browser-warning": "1"})
    try:
        with urlopen(req, timeout=timeout) as resp:
            body = resp.read().decode("utf-8", "ignore")
            if resp.status == 200 and '"status"' in body and "healthy" in body:
                return True, f"tool webhook base is healthy: {base_url}"
            snippet = body[:200]
            return (
                True,
                f"tool webhook base reachable ({resp.status}), unexpected body: {snippet}",
            )
    except Exception as exc:
        return False, f"tool webhook base unreachable: {base_url} ({exc})"


def _require_pyaudio() -> None:
    try:
        import pyaudio  # noqa: F401
    except ImportError:
        pyver = ".".join(map(str, sys.version_info[:2]))
        sys.exit(
            "PyAudio is required for a live voice (CALL) session.\n"
            f"It is not installed for Python {pyver} on this machine (no wheel "
            "for this Python version and the source build needs the PortAudio "
            "headers).\n"
            "Options:\n"
            "  - run this under Python <=3.13 where the PyAudio wheel exists, then\n"
            '    pip install "sarvam-conv-ai-sdk[audio]"\n'
            "  - or install a PortAudio dev build and: pip install pyaudio\n"
        )


async def _run(
    api_key: str,
    *,
    org_id: str,
    workspace_id: str,
    app_id: str,
    version: int | None,
    language: str | None,
) -> int:
    from sarvam_conv_ai_sdk import (
        AsyncDefaultAudioInterface,
        AsyncSamvaadAgent,
        InteractionConfig,
        InteractionType,
        Role,
        ServerTranscriptMsg,
    )
    from sarvam_conv_ai_sdk.messages.types import UserIdentifierType
    from sarvam_conv_ai_sdk.tool import SarvamToolLanguageName

    async def handle_transcript(msg: ServerTranscriptMsg) -> None:
        speaker = "User" if msg.role == Role.USER else "Bot"
        print(f"[{speaker}] {msg.content}")

    initial_language = None
    if language:
        for member in SarvamToolLanguageName:
            if member.value.lower() == language.strip().lower():
                initial_language = member
                break
        if initial_language is None:
            print(f"WARNING: unknown language {language!r}; using agent default.")

    config = InteractionConfig(
        user_identifier="talk-to-agent-demo",
        user_identifier_type=UserIdentifierType.CUSTOM,
        app_id=app_id,
        org_id=org_id,
        workspace_id=workspace_id,
        interaction_type=InteractionType.CALL,
        sample_rate=16000,
        version=version,
        initial_language_name=initial_language,
    )

    _require_pyaudio()
    agent = AsyncSamvaadAgent(
        api_key=SecretStr(api_key),
        config=config,
        audio_interface=AsyncDefaultAudioInterface(input_sample_rate=16000),
        transcript_callback=handle_transcript,
    )

    try:
        await agent.start()
        await agent.wait_for_connect(timeout=15.0)
        print(
            f"Connected! Interaction ID: {agent.get_interaction_id()}\n"
            "Voice conversation active - speak into your microphone.\n"
            "Press Ctrl+C to stop."
        )
        await agent.wait_for_disconnect()
        print("Connection closed.")
    except KeyboardInterrupt:
        print("\nStopping conversation...")
    except Exception as exc:
        msg = str(exc)
        if "402" in msg:
            print(
                "\nSarvam returned HTTP 402 (Payment Required) while opening the "
                "session.\n"
                "Confirmed cause: the Sarvam account is OUT OF CREDITS\n"
                "  (API: \"Insufficient credits. Top up to resume outbound calls.\")\n\n"
                "The agent is committed and working - it has run many successful "
                "sessions (see Analytics). The blocker is purely billing.\n\n"
                "To fix: top up credits on the Sarvam account at "
                "https://indus.sarvam.ai/samvaad (billing / credits), then re-run "
                "this script - no code change is needed."
            )
        else:
            logger.error("Error: %s", msg, exc_info=True)
        return 2
    finally:
        await agent.stop()
    return 0


def main() -> int:
    parser = argparse.ArgumentParser(description="Talk to the Samvaad voice agent")
    parser.add_argument("--language", default=None, help="Starting language, e.g. Hindi")
    parser.add_argument(
        "--version",
        type=int,
        default=None,
        help="Committed agent version to use (default: latest committed)",
    )
    parser.add_argument(
        "--skip-health-check",
        action="store_true",
        help="Skip the pre-flight tool-webhook (ngrok tunnel) health check",
    )
    parser.add_argument(
        "--tool-base-url",
        default=None,
        help="Public base URL the Sarvam agent uses for tool webhooks "
        "(default: SAMVAAD_TOOL_BASE_URL env or the known ngrok domain)",
    )
    args = parser.parse_args()

    _load_env()
    api_key = os.getenv("SAMVAAD_API_KEY", "")
    org_id = os.getenv("SAMVAAD_ORG_ID", "")
    workspace_id = os.getenv("SAMVAAD_WORKSPACE_ID", "")
    app_id = os.getenv("SAMVAAD_AGENT_ID", "")

    for name, value in (
        ("SAMVAAD_API_KEY", api_key),
        ("SAMVAAD_ORG_ID", org_id),
        ("SAMVAAD_WORKSPACE_ID", workspace_id),
        ("SAMVAAD_AGENT_ID", app_id),
    ):
        if not value:
            print(f"Missing {name} in .env - cannot start a session.")
            return 1

    tool_base_url = (
        args.tool_base_url
        or os.getenv("SAMVAAD_TOOL_BASE_URL", "")
        or DEFAULT_TOOL_BASE_URL
    )
    if not args.skip_health_check:
        ok, note = _health_check(tool_base_url)
        print(f"[pre-flight] {note}")
        if not ok:
            print(
                "\nWARNING: the Sarvam agent's tool webhooks are UNREACHABLE.\n"
                "The hosted agent needs this public URL to reach your "
                "orchestrator's tools (calendar, email, ...). If it is down, "
                "every tool call in the call will fail.\n\n"
                "To fix, start the ngrok tunnel:\n"
                f"  powershell -File scripts\\run-ngrok.ps1 -Domain <your-domain>\n"
                f"and confirm {tool_base_url}/health returns the orchestrator "
                "health JSON.\n"
                "You can bypass this check with --skip-health-check if you "
                "know the tools are reachable another way.\n"
            )

    return asyncio.run(
        _run(
            api_key,
            org_id=org_id,
            workspace_id=workspace_id,
            app_id=app_id,
            version=args.version,
            language=args.language,
        )
    )


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO, format="%(levelname)s %(name)s: %(message)s")
    raise SystemExit(main())
