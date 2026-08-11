"""Best-effort SSML → plain-text rendering for the TTS provider.

Sarvam's ``bulbul`` TTS endpoint accepts plain text only, so SSML documents are
parsed locally: recognized tags (``speak``/``p``/``s``/``break``/``prosody``/
``emphasis``/``say-as``/``sub``/``phoneme``) are honored for structure and text
substitution, and the result is stripped down to plain text for synthesis.
"""

import re
from html import unescape
from html.parser import HTMLParser

_KNOWN_TAGS = frozenset({
    "speak", "p", "s", "break", "prosody", "emphasis", "say-as", "sub",
    "phoneme", "mark", "voice", "lang", "amazon:breath", "audio",
})

_SSML_OPEN_RE = re.compile(
    r"<(/)?(speak|p|s|break|prosody|emphasis|say-as|sub|phoneme|mark|voice|lang)(\s[^>]*)?>",
    re.IGNORECASE,
)
_TAG_STRIP_RE = re.compile(r"<[^>]+>")


def is_ssml(text: str) -> bool:
    """Return True if ``text`` looks like an SSML document."""
    return bool(_SSML_OPEN_RE.search(text))


class _SSMLTextExtractor(HTMLParser):
    def __init__(self) -> None:
        super().__init__(convert_charrefs=True)
        self.parts: list[str] = []
        self._sub_alias: str | None = None

    def handle_starttag(self, tag: str, attrs: list[tuple[str, str | None]]) -> None:
        tag = tag.lower()
        if tag == "sub":
            self._sub_alias = dict(attrs).get("alias")
        if tag == "break":
            self.parts.append(" ")
        if tag in ("p", "s") and self.parts and not self.parts[-1].endswith("\n"):
            self.parts.append("\n")

    def handle_endtag(self, tag: str) -> None:
        tag = tag.lower()
        if tag in ("p", "s") and self.parts and not self.parts[-1].endswith("\n"):
            self.parts.append("\n")
        if tag == "sub":
            self._sub_alias = None

    def handle_data(self, data: str) -> None:
        if self._sub_alias:
            self.parts.append(self._sub_alias)
            self._sub_alias = None
            return
        self.parts.append(data)


def _fallback_strip(ssml: str) -> str:
    """Crude fallback: replace ``<sub alias="X">`` then strip remaining tags."""
    text = re.sub(
        r'<sub\s+alias="([^"]+)"[^>]*>.*?</sub>',
        r"\1",
        ssml,
        flags=re.IGNORECASE | re.DOTALL,
    )
    return _TAG_STRIP_RE.sub(" ", text)


def ssml_to_text(ssml: str) -> str:
    """Render an SSML document to plain text for downstream synthesis."""
    if not ssml:
        return ""
    if not is_ssml(ssml):
        return unescape(ssml).strip()

    parser = _SSMLTextExtractor()
    try:
        parser.feed(ssml)
        parser.close()
        text = "".join(parser.parts)
    except Exception:
        text = _fallback_strip(ssml)

    text = unescape(text)
    text = re.sub(r"[ \t]+", " ", text)
    text = re.sub(r"[ \t]*\n[ \t]*", "\n", text)
    text = re.sub(r"\n{3,}", "\n\n", text)
    return text.strip()
