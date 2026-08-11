"""Input sanitization utilities (O4)."""

import re
import unicodedata

_CONTROL_CHARS = re.compile(r"[\x00-\x08\x0b\x0c\x0e-\x1f\x7f]")
_MULTI_SPACE = re.compile(r"[ \t]{2,}")
_MULTI_NEWLINE = re.compile(r"\n{3,}")


class InputSanitizer:
    """Normalizes raw user input before it reaches the agent brain."""

    def __init__(self, max_length: int = 10000) -> None:
        self._max_length = max_length

    def sanitize(self, text: str) -> str:
        if not text:
            return text
        # Normalize unicode to NFC to defuse visually-confusable characters.
        text = unicodedata.normalize("NFC", text)
        # Strip dangerous control characters (keep \n and \t).
        text = _CONTROL_CHARS.sub("", text)
        # Collapse runs of spaces/tabs and stray newlines.
        text = _MULTI_SPACE.sub(" ", text)
        text = _MULTI_NEWLINE.sub("\n\n", text)
        text = text.strip()
        return text[: self._max_length]
