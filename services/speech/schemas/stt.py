from pydantic import BaseModel


class WordTimestamp(BaseModel):
    """Word-level timestamp for a transcribed utterance.

    ``confidence`` is only populated when the upstream provider returns it;
    server-side estimates leave it as ``None``.
    """

    word: str
    start_sec: float
    end_sec: float
    confidence: float | None = None


class SpeechToTextResponse(BaseModel):
    transcript: str
    language_code: str
    word_timestamps: list[WordTimestamp] | None = None
