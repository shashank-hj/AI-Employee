"""PII redaction utilities (O4).

Redacts common personally-identifiable information from free text before it is
forwarded to downstream services. Each redaction is reported so it can be
logged/audited.
"""

import re
from dataclasses import dataclass, field
from re import Pattern


@dataclass
class Redaction:
    label: str
    start: int
    end: int
    original: str


@dataclass
class RedactionResult:
    text: str
    redactions: list[Redaction] = field(default_factory=list)

    @property
    def redacted(self) -> bool:
        return bool(self.redactions)


class PIIRedactor:
    """Regex-based redactor for common PII. Patterns are intentionally strict to
    avoid over-redacting ordinary prose."""

    def __init__(self, enable_all: bool = True) -> None:
        self._patterns: list[tuple[str, Pattern[str]]] = [
            ("EMAIL", re.compile(r"[A-Za-z0-9._%+-]+@[A-Za-z0-9.-]+\.[A-Za-z]{2,}")),
            ("CREDIT_CARD", re.compile(r"(?<!\d)(?:\d[ -]*?){13,19}(?!\d)")),
            ("SSN", re.compile(r"\b\d{3}-\d{2}-\d{4}\b")),
            ("AADHAAR", re.compile(r"\b[2-9]\d{3}\s?\d{4}\s?\d{4}\b")),
            ("PAN", re.compile(r"\b[A-Z]{5}\d{4}[A-Z]\b")),
            ("PHONE", re.compile(
                r"(?<!\w)(?:\+?\d{1,3}[\s.-]?)?\(?\d{3}\)?[\s.-]?\d{3}[\s.-]?\d{4}(?!\w)"
            )),
            ("IP_ADDRESS", re.compile(r"\b(?:\d{1,3}\.){3}\d{1,3}\b")),
        ]
        if not enable_all:
            self._patterns = []

    def redact(self, text: str) -> RedactionResult:
        if not text:
            return RedactionResult(text=text)
        redactions: list[Redaction] = []
        for label, pattern in self._patterns:
            for match in pattern.finditer(text):
                if not self._is_plausible(match.group(0), label):
                    continue
                redactions.append(Redaction(
                    label=label,
                    start=match.start(),
                    end=match.end(),
                    original=match.group(0),
                ))
        # Sort by start, then longest-first, so an overlapping match that spans
        # more text (e.g. a credit card vs. a substring Aadhaar/phone) wins.
        redactions.sort(key=lambda r: (r.start, -(r.end - r.start)))

        merged: list[Redaction] = []
        for redaction in redactions:
            if merged and redaction.start < merged[-1].end:
                continue
            merged.append(redaction)

        pieces: list[str] = []
        cursor = 0
        for redaction in merged:
            pieces.append(text[cursor:redaction.start])
            pieces.append(f"[{redaction.label}]")
            cursor = redaction.end
        pieces.append(text[cursor:])
        return RedactionResult(text="".join(pieces), redactions=merged)

    @staticmethod
    def _is_plausible(value: str, label: str) -> bool:
        if label == "CREDIT_CARD":
            digits = [d for d in value if d.isdigit()]
            if not 13 <= len(digits) <= 19:
                return False
            return not (len(set(digits)) == 1)
        return True
