#!/usr/bin/env python3
"""Verify the Samvaad (Sarvam Voice Agents) integration.

Checks the environment and config, then runs a short text (CHAT) session
against the hosted agent using the ``sarvam-conv-ai-sdk`` to prove the
credentials, org/workspace ids, and committed agent version all work.

Usage:
    python scripts/samvaad_verify.py            # default, from .env
    python scripts/samvaad_verify.py --text "Hello"  # send a custom message
"""

import argparse
import asyncio
import os
import sys
from typing import Any

try:
    from orchestrator.config import settings
except ImportError:
    # Fallback for running without the workspace installed (uv sync). Use
    # append, NOT insert(0): the repo's shared/queue would otherwise shadow the
    # stdlib queue module and break anyio/httpx inside the SDK.
    _ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    for _p in (os.path.join(_ROOT, "services"), os.path.join(_ROOT, "shared")):
        if _p not in sys.path:
            sys.path.append(_p)
    from orchestrator.config import settings  # noqa: E402


def _load_env() -> None:
    """Best-effort load of .env / .env-opencode so the script works standalone."""
    try:
        from dotenv import load_dotenv

        load_dotenv(os.path.join(_ROOT, ".env"))
        load_dotenv(os.path.join(_ROOT, ".env-opencode"))
    except Exception:
        pass


def _line(title: str = "") -> None:
    print("=" * 64)
    if title:
        print(title)


async def _open_agent(mode: str, on_text) -> Any:
    from pydantic import SecretStr
    from sarvam_conv_ai_sdk import (
        AsyncSamvaadAgent,
        InteractionConfig,
        InteractionType,
    )
    from sarvam_conv_ai_sdk.messages.types import UserIdentifierType

    config = InteractionConfig(
        user_identifier_type=UserIdentifierType.CUSTOM,
        user_identifier="samvaad-verify",
        org_id=settings.SAMVAAD_ORG_ID,
        workspace_id=settings.SAMVAAD_WORKSPACE_ID,
        app_id=settings.SAMVAAD_AGENT_ID,
        version=settings.SAMVAAD_AGENT_VERSION or None,
        interaction_type=(
            InteractionType.CALL if mode == "call" else InteractionType.CHAT
        ),
        sample_rate=settings.SAMVAAD_SAMPLE_RATE,
        agent_variables={"user_name": "Samvaad Verify"},
    )
    return AsyncSamvaadAgent(
        api_key=SecretStr(settings.SAMVAAD_API_KEY),
        config=config,
        text_callback=on_text,
        base_url=settings.SAMVAAD_APP_RUNTIME_URL,
    )


async def _run_check(text: str, mode: str) -> int:
    """Run a session. ``mode`` is 'chat', 'call', or 'auto' (fall back to call)."""
    for attempt_mode in ("chat", "call") if mode == "auto" else (mode,):
        replies: list[str] = []

        async def on_text(msg: Any, replies: list[str] = replies) -> None:
            t = getattr(msg, "text", "")
            status = getattr(getattr(msg, "status", None), "value", "completed")
            if status == "completed" and t:
                replies.append(t)
                print(f"[agent] {t}")

        agent = await _open_agent(attempt_mode, on_text)
        try:
            await agent.start()
        except Exception as exc:
            print(f"NOTE: {attempt_mode} mode unavailable: {type(exc).__name__}: {exc}")
            if attempt_mode == "chat" and mode == "auto":
                continue
            return 2

        try:
            connected = await agent.wait_for_connect(timeout=settings.SAMVAAD_CONNECT_TIMEOUT)
            if not connected:
                print("ERROR: agent did not connect. Is the agent committed/deployed?")
                return 2
            print(
                f"connected ({attempt_mode}), interaction_id={agent.get_interaction_id()}"
            )
            if attempt_mode == "call":
                print("NOTE: agent is a voice (CALL) agent; text round-trip is not available.")
                print("      Connectivity + committed version verified OK.")
                return 0
            await agent.send_text(text)
            await asyncio.sleep(8.0)
        finally:
            await agent.stop()

        if not replies:
            print("WARNING: no text reply received within the window.")
            return 1
        print("OK: agent replied.")
        return 0
    return 2


async def main() -> int:
    parser = argparse.ArgumentParser(description="Verify Samvaad integration")
    parser.add_argument("--text", default="Hello! Who are you and what can you help with?")
    parser.add_argument(
        "--mode",
        choices=["auto", "chat", "call"],
        default="auto",
        help="auto tries chat then falls back to call for voice agents",
    )
    args = parser.parse_args()

    _load_env()

    _line("Samvaad integration check")
    print(f"  SAMVAAD_ENABLED        : {settings.SAMVAAD_ENABLED}")
    print(f"  SAMVAAD_AGENT_ID       : {settings.SAMVAAD_AGENT_ID or '(empty)'}")
    print(f"  SAMVAAD_ORG_ID         : {settings.SAMVAAD_ORG_ID or '(empty)'}")
    print(f"  SAMVAAD_WORKSPACE_ID   : {settings.SAMVAAD_WORKSPACE_ID or '(empty)'}")
    print(f"  SAMVAAD_APP_RUNTIME_URL: {settings.SAMVAAD_APP_RUNTIME_URL}")
    print(f"  SAMVAAD_AGENT_VERSION  : {settings.SAMVAAD_AGENT_VERSION or '(latest)'}")
    print(f"  SAMVAAD_SAMPLE_RATE    : {settings.SAMVAAD_SAMPLE_RATE}")
    print(f"  API key set            : {bool(settings.SAMVAAD_API_KEY)}")

    if not settings.SAMVAAD_ENABLED:
        print("SKIPPED: SAMVAAD_ENABLED=false. Set it to true in .env first.")
        return 0
    if not settings.SAMVAAD_API_KEY:
        print("SKIPPED: SAMVAAD_API_KEY is empty. Generate one in the Sarvam")
        print("         dashboard (Settings -> API Key) and add it to .env.")
        return 0
    if not settings.SAMVAAD_AGENT_ID:
        print("SKIPPED: SAMVAAD_AGENT_ID is empty. Add your agent id (e.g.")
        print("         AI-Employee-33c6c05a-c14f) to .env.")
        return 0
    if not settings.SAMVAAD_ORG_ID or not settings.SAMVAAD_WORKSPACE_ID:
        print("SKIPPED: SAMVAAD_ORG_ID / SAMVAAD_WORKSPACE_ID are empty.")
        print("         Copy them from your dashboard URL.")
        return 0

    try:
        import sarvam_conv_ai_sdk  # noqa: F401
    except ImportError:
        print("ERROR: sarvam-conv-ai-sdk is not installed.")
        print("       Run: uv sync (or pip install sarvam-conv-ai-sdk)")
        return 2

    print(f"  probe text             : {args.text!r}")
    _line(f"Running {args.mode.upper()} session")
    return await _run_check(args.text, args.mode)


if __name__ == "__main__":
    raise SystemExit(asyncio.run(main()))
