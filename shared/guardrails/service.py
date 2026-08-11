"""Composed guardrail pipeline (O4): sanitize -> redact -> filter."""

from dataclasses import dataclass, field

from shared.guardrails.filter import ContentFilter, ContentViolation
from shared.guardrails.redactor import PIIRedactor, RedactionResult
from shared.guardrails.sanitizer import InputSanitizer


@dataclass
class GuardrailResult:
    text: str
    redactions: list = field(default_factory=list)
    violation: ContentViolation | None = None

    @property
    def allowed(self) -> bool:
        return self.violation is None


class GuardrailsService:
    """Runs the sanitizer, redactor, and content filter in order."""

    def __init__(
        self,
        sanitizer: InputSanitizer | None = None,
        redactor: PIIRedactor | None = None,
        content_filter: ContentFilter | None = None,
        redact_pii: bool = True,
        enabled: bool = True,
    ) -> None:
        self._sanitizer = sanitizer or InputSanitizer()
        self._redactor = redactor or PIIRedactor()
        self._filter = content_filter or ContentFilter()
        self._redact_pii = redact_pii
        self._enabled = enabled

    def apply(self, text: str) -> GuardrailResult:
        if not text or not self._enabled:
            return GuardrailResult(text=text)

        text = self._sanitizer.sanitize(text)

        redaction: RedactionResult = RedactionResult(text=text)
        if self._redact_pii:
            redaction = self._redactor.redact(text)
            text = redaction.text

        violation = self._filter.check(text)
        return GuardrailResult(text=text, redactions=redaction.redactions, violation=violation)
