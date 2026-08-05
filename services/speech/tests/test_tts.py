import base64
import io
import struct
import wave
from unittest.mock import AsyncMock, MagicMock, patch

import pytest


def _mock_httpx_post(return_value):
    mock_response = MagicMock()
    mock_response.json.return_value = return_value
    mock_response.raise_for_status = MagicMock()

    mock_client = AsyncMock()
    mock_client.post = AsyncMock(return_value=mock_response)
    return mock_client


def _make_wav_bytes(data: bytes, nchannels: int = 1, sampwidth: int = 2, framerate: int = 24000) -> bytes:
    buf = io.BytesIO()
    with wave.open(buf, "wb") as wf:
        wf.setnchannels(nchannels)
        wf.setsampwidth(sampwidth)
        wf.setframerate(framerate)
        wf.writeframes(data)
    return buf.getvalue()


class TestTTSEndpoint:
    @pytest.mark.asyncio
    async def test_tts_invalid_request_missing_text(self, client):
        response = await client.post(
            "/api/text-to-speech",
            json={},
        )
        assert response.status_code == 422

    @pytest.mark.asyncio
    async def test_tts_empty_text_rejected(self, client):
        response = await client.post(
            "/api/text-to-speech",
            json={"text": ""},
        )
        assert response.status_code == 422

    @pytest.mark.asyncio
    async def test_tts_with_language_code(self):
        from speech.providers.tts import SarvamTTSProvider

        provider = SarvamTTSProvider(api_key="test-key", base_url="https://mock.local")
        test_audio = b"audio data"
        encoded = base64.b64encode(test_audio).decode("utf-8")

        mock_client = _mock_httpx_post({"audios": [encoded]})

        with patch("httpx.AsyncClient") as mock_client_class:
            mock_client_class.return_value.__aenter__.return_value = mock_client
            result = await provider.synthesize("नमस्ते", language_code="hi-IN")
            assert result == test_audio

    @pytest.mark.asyncio
    async def test_tts_no_audios_returns_empty(self):
        from speech.providers.tts import SarvamTTSProvider

        provider = SarvamTTSProvider(api_key="test-key", base_url="https://mock.local")

        mock_client = _mock_httpx_post({"audios": []})

        with patch("httpx.AsyncClient") as mock_client_class:
            mock_client_class.return_value.__aenter__.return_value = mock_client
            result = await provider.synthesize("Hello")
            assert result == b""


class TestTextChunking:
    def test_short_text_not_chunked(self):
        from speech.providers.tts import _split_text_into_chunks

        chunks = _split_text_into_chunks("Hello world.", max_chars=400)
        assert chunks == ["Hello world."]

    def test_long_text_split_at_sentences(self):
        from speech.providers.tts import _split_text_into_chunks

        text = "First. Second. Third. Fourth. Fifth. Sixth."
        chunks = _split_text_into_chunks(text, max_chars=15)
        assert len(chunks) > 1
        combined = " ".join(chunks)
        assert "First" in combined
        assert "Sixth" in combined

    def test_single_long_sentence(self):
        from speech.providers.tts import _split_text_into_chunks

        text = "A" * 500
        chunks = _split_text_into_chunks(text, max_chars=100)
        assert len(chunks) == 1
        assert chunks[0] == text

    def test_empty_text_returns_single_chunk(self):
        from speech.providers.tts import _split_text_into_chunks

        chunks = _split_text_into_chunks("", max_chars=100)
        assert chunks == [""]

    def test_removes_empty_sentences(self):
        from speech.providers.tts import _split_text_into_chunks

        text = "Hello.  . World."
        chunks = _split_text_into_chunks(text, max_chars=50)
        assert len(chunks) == 1
        assert "Hello" in chunks[0]
        assert "World" in chunks[0]


class TestWAVConcatenation:
    def test_single_segment_returns_unchanged(self):
        from speech.providers.tts import _concatenate_wav_bytes

        wav = _make_wav_bytes(b"\x00\x01" * 100)
        result = _concatenate_wav_bytes([wav])
        assert result == wav

    def test_empty_list_returns_empty(self):
        from speech.providers.tts import _concatenate_wav_bytes

        result = _concatenate_wav_bytes([])
        assert result == b""

    def test_multiple_segments_concatenated(self):
        from speech.providers.tts import _concatenate_wav_bytes

        wav1 = _make_wav_bytes(b"\x00\x01" * 100)
        wav2 = _make_wav_bytes(b"\x00\x02" * 50)
        combined = _concatenate_wav_bytes([wav1, wav2])

        with wave.open(io.BytesIO(combined), "rb") as wf:
            assert wf.getnchannels() == 1
            assert wf.getsampwidth() == 2
            assert wf.getframerate() == 24000
            frames = wf.readframes(wf.getnframes())
            assert len(frames) == 200 + 100

    def test_different_sample_rates_still_concatenates(self):
        from speech.providers.tts import _concatenate_wav_bytes

        wav1 = _make_wav_bytes(b"\x00\x01" * 50, framerate=22050)
        wav2 = _make_wav_bytes(b"\x00\x02" * 50, framerate=22050)
        combined = _concatenate_wav_bytes([wav1, wav2])

        with wave.open(io.BytesIO(combined), "rb") as wf:
            assert wf.getframerate() == 22050
            assert wf.getnframes() == 100


class TestLongTextTTS:
    @pytest.mark.asyncio
    async def test_long_text_triggers_chunking(self):
        from speech.providers.tts import SarvamTTSProvider

        provider = SarvamTTSProvider(api_key="test-key", base_url="https://mock.local")
        test_audio = _make_wav_bytes(b"\x00\x01" * 200)
        encoded = base64.b64encode(test_audio).decode("utf-8")

        mock_client = _mock_httpx_post({"audios": [encoded]})

        long_text = ". ".join(["Sentence number " + str(i) for i in range(1, 30)]) + "."
        assert len(long_text) > 400

        with patch("httpx.AsyncClient") as mock_client_class:
            mock_client_class.return_value.__aenter__.return_value = mock_client
            result = await provider.synthesize(long_text)
            assert len(result) > len(test_audio)
