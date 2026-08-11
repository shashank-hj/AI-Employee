"""Code-switching detection for voice turns.

Uses Unicode script ranges to detect when a single utterance mixes scripts
(e.g. Devanagari + Latin), and maps the dominant script to a primary language.
"""

import re

_LATIN_RE = re.compile(r"[A-Za-z]")

# Script → (unicode range start, end) for Indic scripts.
_SCRIPT_RANGES: dict[str, tuple[int, int]] = {
    "devanagari": (0x0900, 0x097F),
    "bengali": (0x0980, 0x09FF),
    "gurmukhi": (0x0A00, 0x0A7F),
    "gujarati": (0x0A80, 0x0AFF),
    "odia": (0x0B00, 0x0B7F),
    "tamil": (0x0B80, 0x0BFF),
    "telugu": (0x0C00, 0x0C7F),
    "kannada": (0x0C80, 0x0CFF),
    "malayalam": (0x0D00, 0x0D7F),
}

_SCRIPT_TO_LANGUAGE: dict[str, str] = {
    "devanagari": "hi-IN",
    "bengali": "bn-IN",
    "gurmukhi": "pa-IN",
    "gujarati": "gu-IN",
    "odia": "or-IN",
    "tamil": "ta-IN",
    "telugu": "te-IN",
    "kannada": "kn-IN",
    "malayalam": "ml-IN",
    "latin": "en-IN",
}

_MOSTLY_LATIN_RATIO = 0.5


def detect_code_switch(text: str, detected_language: str | None = None) -> tuple[bool, str | None]:
    """Detect code-switching (mixed scripts) in ``text``.

    Returns ``(code_switch, primary_language)``. When ``detected_language`` is
    supplied (e.g. from Sarvam's detector) it is used as the primary language.
    Otherwise the script with the most characters maps to a primary language.

    A mostly-English utterance containing a single Indic word is NOT flagged as
    code-switched to avoid false positives.
    """
    if not text:
        return False, None

    counts: dict[str, int] = {}
    for ch in text:
        if _LATIN_RE.match(ch):
            counts["latin"] = counts.get("latin", 0) + 1
            continue
        code = ord(ch)
        for name, (lo, hi) in _SCRIPT_RANGES.items():
            if lo <= code <= hi:
                counts[name] = counts.get(name, 0) + 1
                break

    if not counts:
        return False, None

    total = sum(counts.values())
    latin_share = counts.get("latin", 0) / max(1, total)

    if detected_language and detected_language != "unknown":
        return len(counts) > 1, detected_language

    code_switch = len(counts) > 1 and not (latin_share >= _MOSTLY_LATIN_RATIO)
    if not code_switch:
        dominant = max(counts, key=counts.get)
        return False, _SCRIPT_TO_LANGUAGE.get(dominant)

    dominant = max(counts, key=counts.get)
    return True, _SCRIPT_TO_LANGUAGE.get(dominant)
