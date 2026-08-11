import re

from speech.schemas.stt import WordTimestamp

_WORD_RE = re.compile(r"\s+")


def estimate_word_timestamps(transcript: str, audio_seconds: float) -> list[WordTimestamp]:
    """Estimate per-word start/end times from a transcript and total duration.

    Words are laid out sequentially with each word's duration proportional to its
    character count. This is a best-effort approximation used when the upstream
    STT provider does not return word-level timestamps.
    """
    words = _WORD_RE.split(transcript.strip())
    words = [w for w in words if w]
    if not words or audio_seconds <= 0:
        return []

    total_chars = sum(len(w) for w in words)
    cursor = 0.0
    timestamps: list[WordTimestamp] = []
    for word in words:
        duration = (len(word) / total_chars) * audio_seconds
        timestamps.append(
            WordTimestamp(
                word=word,
                start_sec=round(cursor, 3),
                end_sec=round(cursor + duration, 3),
            )
        )
        cursor += duration
    return timestamps
