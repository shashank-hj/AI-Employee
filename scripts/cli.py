#!/usr/bin/env python3
"""
AI Employee Platform - Interactive CLI
======================================
A terminal-based chat interface to the AI Employee orchestrator agent.
Includes built-in commands to test TTS, translation, and language detection.

Usage:
    python scripts/cli.py
"""

import base64
import io
import os
import sys
import tempfile
import time
import uuid

import httpx

sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding="utf-8", errors="replace")

ORCHESTRATOR_URL = "http://127.0.0.1:8001"
SPEECH_URL = "http://127.0.0.1:8006"
TEMP_DIR = os.path.join(tempfile.gettempdir(), "ai-employee-cli")

LINE = "-" * 60


def _ensure_temp_dir():
    os.makedirs(TEMP_DIR, exist_ok=True)


# ═══════════════════════════════════════════════════════════════════════════════
#  Commands
# ═══════════════════════════════════════════════════════════════════════════════

_COMMANDS: dict[str, dict] = {
    "/speech":    {"alias": "/s", "help": "<text> — speak the text (TTS)", "args": True},
    "/translate": {"alias": "/t", "help": "[lang] <text> — translate (default: en-IN)", "args": True},
    "/clear":   {"alias": "/c", "help": "— start a new conversation session"},
    "/verbose": {"alias": "/v", "help": "— toggle showing execution steps"},
    "/help":    {"alias": "/h", "help": "— show this help"},
    "/quit":    {"alias": "/q", "help": "— exit"},
    "/exit":    {"alias": None, "help": "— exit"},
}


def _print_help():
    print()
    print("  Commands:")
    for cmd, info in _COMMANDS.items():
        alias_text = f"  ({info['alias']})" if info.get("alias") else ""
        print(f"    {cmd:<14}{info['help']:<42}{alias_text}")
    print()


# ═══════════════════════════════════════════════════════════════════════════════
#  Speech / Translation helpers
# ═══════════════════════════════════════════════════════════════════════════════

def _handle_speech(client: httpx.Client, text: str) -> None:
    print("\n  Generating speech...", end="", flush=True)
    try:
        resp = client.post(
            f"{SPEECH_URL}/api/text-to-speech",
            json={"text": text, "language_code": "en-IN"},
            timeout=120.0,
        )
        if resp.status_code != 200:
            print(f"\n  TTS error [{resp.status_code}]: {resp.text[:200]}")
            return

        data = resp.json()
        if not data.get("audio_base64"):
            print(f"\n  TTS returned no audio. Error: {data.get('error', 'unknown')}")
            return

        audio_bytes = base64.b64decode(data["audio_base64"])
        wav_path = os.path.join(TEMP_DIR, f"tts_{uuid.uuid4().hex[:8]}.wav")
        with open(wav_path, "wb") as f:
            f.write(audio_bytes)

        duration = data.get("audio_bytes", 0) / 44100 * 8
        print(f"\r  Spoke: {text[:60]}{'...' if len(text) > 60 else ''}")
        print(f"  Saved: {wav_path}  ({data['audio_bytes']:,} bytes, ~{duration:.1f}s)")

        # Play on Windows
        try:
            os.startfile(wav_path)
        except Exception:
            print(f"  (audio saved to {wav_path})")
    except httpx.ConnectError:
        print("\n  Cannot reach speech service at http://127.0.0.1:8006")
    except Exception as e:
        print(f"\n  Speech error: {e}")


def _handle_translate(client: httpx.Client, text: str) -> None:
    # Support: /t <text>  or  /t <target_lang> <text>
    parts = text.split(maxsplit=1)
    lang_codes = {"en-IN", "hi-IN", "kn-IN", "ta-IN", "te-IN", "ml-IN", "mr-IN", "gu-IN", "pa-IN", "bn-IN", "or-IN", "as-IN", "ur-IN", "sd-IN", "sa-IN", "ne-IN", "brx-IN", "doi-IN", "kok-IN", "ks-IN", "mai-IN", "mni-IN", "sat-IN"}
    target_lang = "en-IN"
    text_to_translate = text

    if parts and parts[0] in lang_codes:
        target_lang = parts[0]
        text_to_translate = parts[1] if len(parts) > 1 else text_to_translate

    print("  Translating...", end="", flush=True)
    try:
        resp = client.post(
            f"{SPEECH_URL}/api/translate-text",
            json={
                "text": text_to_translate,
                "source_language_code": "auto",
                "target_language_code": target_lang,
            },
            timeout=30.0,
        )
        if resp.status_code != 200:
            print(f"\n  Translate error [{resp.status_code}]: {resp.text[:200]}")
            return
        data = resp.json()
        if data.get("error"):
            print(f"\n  Error: {data['error']}")
        else:
            print(f"\r  Translation ({target_lang}): {data['translated_text']}")
    except httpx.ConnectError:
        print("\n  Cannot reach speech service at http://127.0.0.1:8006")
    except Exception as e:
        print(f"\n  Translate error: {e}")


# ═══════════════════════════════════════════════════════════════════════════════
#  Agent communication
# ═══════════════════════════════════════════════════════════════════════════════

def _run_agent(client: httpx.Client, user_input: str, session_id: str, user_id: str) -> dict | None:
    try:
        start = time.time()
        resp = client.post(
            f"{ORCHESTRATOR_URL}/api/agent/run",
            json={
                "user_input": user_input,
                "user_id": user_id,
                "session_id": session_id,
            },
            timeout=120.0,
        )
        elapsed = (time.time() - start) * 1000

        if resp.status_code == 200:
            data = resp.json()
            data["_elapsed"] = elapsed
            return data
        elif resp.status_code == 422:
            detail = resp.json().get("detail", resp.text)
            print(f"\n  Validation error: {detail}")
        else:
            print(f"\n  Error [{resp.status_code}]: {resp.text[:300]}")
    except httpx.ConnectError:
        print("\n  Cannot connect to orchestrator at http://127.0.0.1:8001")
        print("  Is the orchestrator running?")
    except httpx.ReadTimeout:
        print("\n  Request timed out. The agent is taking too long.")
    except Exception as e:
        print(f"\n  Unexpected error: {e}")
    return None


def _display_response(data: dict, show_steps: bool):
    print()
    print(LINE)
    request_id = data.get("request_id", "?")
    elapsed = data.get("_elapsed", 0)
    print(f"  Agent  ({request_id[:8]}…)  |  {elapsed:.0f}ms")
    print(LINE)

    if show_steps:
        for step in data.get("steps", []):
            step_num = step.get("step_index", "?")
            tool_name = step.get("tool_name", "unknown")
            result = step.get("result") or {}
            print(f"\n  Step {step_num}: {tool_name}")
            if result.get("success"):
                d = result.get("data", {})
                if d:
                    for k, v in d.items():
                        line = str(v)[:120]
                        print(f"    {k}: {line}")
            else:
                print(f"    ERROR: {result.get('error', 'unknown')}")
        print()

    response_text = data.get("final_response", "")
    print(f"  {response_text}")
    print(LINE)
    print()


# ═══════════════════════════════════════════════════════════════════════════════
#  Main loop
# ═══════════════════════════════════════════════════════════════════════════════

def main():
    _ensure_temp_dir()
    session_id = str(uuid.uuid4())
    user_id = "cli_user"
    show_steps = False

    print()
    print("=" * 60)
    print("  AI Employee Platform — Interactive CLI")
    print("=" * 60)
    print()
    print("  Chat with the agent. Type /help for commands.")
    print(f"  Session: {session_id[:8]}…")
    print()

    client = httpx.Client(timeout=120.0)

    try:
        health = client.get(f"{ORCHESTRATOR_URL}/health", timeout=5.0)
        if health.status_code == 200:
            hd = health.json()
            print(f"  Connected to orchestrator v{hd.get('version', '?')}")
            print(f"  LLM: {hd.get('llm_provider', '?')} / {hd.get('llm_model', '?')}")
        else:
            print(f"  Orchestrator health: {health.status_code}")
    except Exception:
        print("  Warning: orchestrator not reachable. Make sure it's running.")
    print()

    while True:
        try:
            user_input = input("  You: ").strip()
        except (EOFError, KeyboardInterrupt):
            print("\n\n  Goodbye!")
            break

        if not user_input:
            continue

        # ── Built-in commands ──

        if user_input.startswith("/"):
            parts = user_input.split(maxsplit=1)
            cmd = parts[0].lower()
            arg = parts[1] if len(parts) > 1 else ""

            # Resolve aliases
            cmd_map = {c: c for c in _COMMANDS}
            alias_map = {}
            for c, info in _COMMANDS.items():
                if info.get("alias"):
                    alias_map[info["alias"]] = c
            cmd_map.update(alias_map)

            resolved = cmd_map.get(cmd)
            if resolved is None:
                print(f"\n  Unknown command: {cmd}. Type /help for commands.\n")
                continue

            needs_arg = _COMMANDS[resolved].get("args", False)

            if resolved == "/speech":
                if not arg:
                    print("\n  Usage: /speech <text>\n")
                    continue
                _handle_speech(client, arg)

            elif resolved == "/translate":
                if not arg:
                    print("\n  Usage: /translate <text>\n")
                    continue
                _handle_translate(client, arg)

            elif resolved == "/clear":
                session_id = str(uuid.uuid4())
                print(f"\n  New session: {session_id[:8]}…\n")

            elif resolved == "/verbose":
                show_steps = not show_steps
                print(f"\n  Verbose mode: {'ON' if show_steps else 'OFF'}\n")

            elif resolved == "/help":
                _print_help()

            elif resolved in ("/quit", "/exit"):
                print("\n  Goodbye!")
                break

            continue

        # ── Normal agent interaction ──
        print("\n  Agent: thinking…", end="\r")
        data = _run_agent(client, user_input, session_id, user_id)
        if data:
            _display_response(data, show_steps)

    client.close()


if __name__ == "__main__":
    main()
