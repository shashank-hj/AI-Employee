"""Configurable pricing table + cost estimators for the usage dashboard.

Prices are expressed in INR (Indian Rupees). LLM models are priced per 1M
tokens (input/output). Speech operations are priced per unit (characters or
audio seconds). All defaults are $0 until real rates are configured via the
``USAGE_PRICING`` env var, which is a JSON object that is deep-merged over the
defaults, e.g.:

    {"models": {"sarvam-105b": {"in": 4, "out": 16}},
     "operations": {"tts": {"unit": "characters", "rate": 0.003}}}
"""

import copy
import io
import json
import os
import wave
from functools import lru_cache

logger = __import__("structlog").get_logger(__name__)

PRICING_ENV_VAR = "USAGE_PRICING"

# USD per 1M tokens. Local models (Ollama) default to $0.
DEFAULT_MODEL_PRICING: dict[str, dict[str, float]] = {
    "sarvam-105b": {"in": 0.0, "out": 0.0},
    "qwen3:8b": {"in": 0.0, "out": 0.0},
    "llama-3.3-70b": {"in": 0.0, "out": 0.0},
    "gpt-4o": {"in": 2.5, "out": 10.0},
    "gpt-4o-mini": {"in": 0.15, "out": 0.60},
    "gpt-4.1": {"in": 2.0, "out": 8.0},
    "gpt-4.1-mini": {"in": 0.40, "out": 1.60},
    "claude-3-5-sonnet-20241022": {"in": 3.0, "out": 15.0},
    "claude-3-5-haiku-20241022": {"in": 0.80, "out": 4.0},
    "gemini-1.5-pro": {"in": 1.25, "out": 5.0},
    "gemini-1.5-flash": {"in": 0.075, "out": 0.30},
    # opencode (opencode serve / Zen gateway) — USD rates converted to INR at 84/$.
    "opencode": {"in": 0.0, "out": 0.0},
    "deepseek-v4-flash": {"in": 11.76, "out": 23.52},
    "deepseek-v4-pro": {"in": 146.16, "out": 292.32},
    "gpt-5.6-luna": {"in": 16.8, "out": 100.8},
    "big-pickle": {"in": 0.0, "out": 0.0},
}

# Per-unit rate per operation (speech + embeddings). Unit is billable quantity.
DEFAULT_OPERATION_PRICING: dict[str, dict] = {
    "stt": {"unit": "audio_seconds", "rate": 0.0},
    "speech_to_text_translate": {"unit": "audio_seconds", "rate": 0.0},
    "tts": {"unit": "characters", "rate": 0.0},
    "translate_text": {"unit": "characters", "rate": 0.0},
    "detect_language": {"unit": "characters", "rate": 0.0},
    "transliterate": {"unit": "characters", "rate": 0.0},
    "embed": {"unit": "tokens", "rate": 0.0},
    "embed_query": {"unit": "tokens", "rate": 0.0},
    "embed_docs": {"unit": "tokens", "rate": 0.0},
}


def _read_override_json() -> dict:
    raw = os.getenv(PRICING_ENV_VAR, "").strip()
    if not raw:
        return {}
    try:
        data = json.loads(raw)
        if isinstance(data, dict):
            return data
    except json.JSONDecodeError as exc:
        logger.warning("usage_pricing_override_invalid", error=str(exc))
    return {}


@lru_cache(maxsize=1)
def get_pricing() -> dict:
    """Return the merged pricing table: {"models": {...}, "operations": {...}}."""
    pricing = {
        "models": copy.deepcopy(DEFAULT_MODEL_PRICING),
        "operations": copy.deepcopy(DEFAULT_OPERATION_PRICING),
    }
    overrides = _read_override_json()
    models_override = overrides.get("models") or {}
    operations_override = overrides.get("operations") or {}
    if isinstance(models_override, dict):
        for model, rates in models_override.items():
            if isinstance(rates, dict):
                pricing["models"].setdefault(model, {"in": 0.0, "out": 0.0}).update(rates)
    if isinstance(operations_override, dict):
        for operation, conf in operations_override.items():
            if isinstance(conf, dict):
                pricing["operations"].setdefault(operation, {"unit": "characters", "rate": 0.0}).update(conf)
    return pricing


def get_model_pricing(model: str) -> dict | None:
    return get_pricing()["models"].get(model)


def get_operation_pricing(operation: str) -> dict | None:
    return get_pricing()["operations"].get(operation)


def estimate_tokens(text: str) -> int:
    """Rough token estimate (~4 chars/token) used when the provider omits usage."""
    if not text:
        return 0
    return max(1, len(text) // 4)


def estimate_audio_seconds(audio_bytes: bytes) -> float:
    """Estimate audio duration from WAV metadata, falling back to a byte-size guess."""
    if not audio_bytes:
        return 0.0
    try:
        with wave.open(io.BytesIO(audio_bytes), "rb") as wf:
            frames = wf.getnframes()
            rate = wf.getframerate()
            if rate and frames:
                return round(frames / rate, 2)
    except Exception:
        pass
    # Compressed audio (webm/mp3) ~ 12KB/s heuristic; PCM16 wav ~ 32KB/s.
    return round(max(0.1, len(audio_bytes) / 12000.0), 2)


def compute_cost(record) -> float:
    """Compute INR cost for a UsageRecord against the merged pricing table."""
    try:
        if record.category == "llm" or record.unit == "tokens" and record.category in ("llm", "embedding"):
            pricing = get_model_pricing(record.model)
            if not pricing:
                return 0.0
            return (record.input_units / 1_000_000.0) * float(pricing.get("in", 0.0)) + (
                record.output_units / 1_000_000.0
            ) * float(pricing.get("out", 0.0))
        op = get_operation_pricing(record.operation)
        if not op:
            return 0.0
        total = record.total_units if record.total_units is not None else (record.input_units + record.output_units)
        return total * float(op.get("rate", 0.0))
    except Exception:
        return 0.0
