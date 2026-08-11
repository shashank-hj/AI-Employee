import base64
import io
import wave
from unittest.mock import AsyncMock, MagicMock, patch

import pytest


def _mock_httpx_post(return_value):
    mock_response = MagicMock()
    mock_response.status_code = 200
    mock_response.json.return_value = return_value
    mock_response.raise_for_status = MagicMock()

    mock_client = AsyncMock()
    mock_client.post = AsyncMock(return_value=mock_response)
    return mock_client


def _make_wav_bytes(
    data: bytes,
    nchannels: int = 1,
    sampwidth: int = 2,
    framerate: int = 24000,
) -> bytes:
    buf = io.BytesIO()
    with wave.open(buf, "wb") as wf:
        wf.setnchannels(nchannels)
        wf.setsampwidth(sampwidth)
        wf.setframerate(framerate)
        wf.writeframes(data)
    return buf.getvalue()


class _FakeWebSocket:
    def __init__(self, config=None):
        self._config = config
        self.sent = []
        self.closed = False

    async def accept(self):
        pass

    async def receive_json(self):
        return self._config

    async def send_json(self, message):
        self.sent.append(message)

    async def close(self):
        self.closed = True


@pytest.fixture
def _noop_usage_recorder(monkeypatch):
    """Prevent SSE/WS handlers from touching the real usage DB."""
    monkeypatch.setattr("speech.routers.tts.get_usage_recorder", lambda: AsyncMock())
    monkeypatch.setattr("speech.routers.stt.get_usage_recorder", lambda: AsyncMock())


class TestSynthesizeStream:
    @pytest.mark.asyncio
    async def test_stream_yields_chunks(self):
        from speech.providers.tts import SarvamTTSProvider

        provider = SarvamTTSProvider(api_key="test-key", base_url="https://mock.local")
        wav = _make_wav_bytes(b"\x00\x01" * 100)
        mock_client = _mock_httpx_post({"audios": [base64.b64encode(wav).decode("utf-8")]})

        with patch("httpx.AsyncClient") as mock_client_class:
            mock_client_class.return_value.__aenter__.return_value = mock_client
            chunks = []
            async for chunk in provider.synthesize_stream("Hello world.", language_code="en-IN"):
                chunks.append(chunk)

        assert len(chunks) == 1
        assert chunks[0]["chunk_index"] == 0
        assert base64.b64decode(chunks[0]["audio_base64"]) == wav

    @pytest.mark.asyncio
    async def test_stream_multiple_chunks_ordered(self):
        from speech.providers.tts import SarvamTTSProvider

        provider = SarvamTTSProvider(api_key="test-key", base_url="https://mock.local")
        wav = _make_wav_bytes(b"\x00\x01" * 50)
        mock_client = _mock_httpx_post({"audios": [base64.b64encode(wav).decode("utf-8")]})

        text = ". ".join(["Sentence number " + str(i) for i in range(1, 30)]) + "."

        with patch("httpx.AsyncClient") as mock_client_class:
            mock_client_class.return_value.__aenter__.return_value = mock_client
            indices = []
            async for chunk in provider.synthesize_stream(text):
                indices.append(chunk["chunk_index"])

        assert len(indices) > 1
        assert indices == sorted(indices)

    @pytest.mark.asyncio
    async def test_stream_empty_text_yields_nothing(self):
        from speech.providers.tts import SarvamTTSProvider

        provider = SarvamTTSProvider(api_key="test-key", base_url="https://mock.local")
        chunks = []
        async for chunk in provider.synthesize_stream("   "):
            chunks.append(chunk)
        assert chunks == []

    @pytest.mark.asyncio
    async def test_stream_skips_failed_chunks(self):
        from speech.providers.tts import SarvamTTSProvider

        provider = SarvamTTSProvider(api_key="test-key", base_url="https://mock.local")
        failing = AsyncMock(side_effect=RuntimeError("boom"))
        with patch.object(provider, "_synthesize_single", failing):
            chunks = []
            async for chunk in provider.synthesize_stream("Hello world."):
                chunks.append(chunk)
        assert chunks == []


class TestTTSSseEndpoint:
    @pytest.mark.asyncio
    async def test_stream_sse_no_api_key_returns_event_error(self, client, monkeypatch):
        from speech.container import get_tts_provider

        monkeypatch.setattr("speech.config.settings.SARVAM_API_KEY", "")
        get_tts_provider.cache_clear()

        response = await client.post(
            "/api/text-to-speech/stream",
            json={"text": "Hello", "language_code": "en-IN"},
        )
        assert response.status_code == 200
        assert response.headers["content-type"].startswith("text/event-stream")
        body = response.text
        assert "not configured" in body.lower()

    @pytest.mark.asyncio
    async def test_stream_sse_emits_chunks_and_done(
        self,
        client,
        monkeypatch,
        _noop_usage_recorder,
    ):
        from speech.container import get_tts_provider

        monkeypatch.setattr("speech.config.settings.SARVAM_API_KEY", "test-key")
        get_tts_provider.cache_clear()

        wav = _make_wav_bytes(b"\x00\x01" * 100)
        encoded = base64.b64encode(wav).decode("utf-8")
        mock_client = _mock_httpx_post({"audios": [encoded]})

        with patch("httpx.AsyncClient") as mock_client_class:
            mock_client_class.return_value.__aenter__.return_value = mock_client
            response = await client.post(
                "/api/text-to-speech/stream",
                json={"text": "Hello world.", "language_code": "en-IN"},
            )

        assert response.status_code == 200
        body = response.text
        assert "[DONE]" in body
        assert encoded in body


class TestTTSWebSocket:
    @pytest.mark.asyncio
    async def test_tts_ws_no_api_key_rejects(self, monkeypatch):
        from speech.container import get_tts_provider
        from speech.routers.tts import text_to_speech_websocket

        monkeypatch.setattr("speech.config.settings.SARVAM_API_KEY", "")
        get_tts_provider.cache_clear()

        ws = _FakeWebSocket(config={"text": "Hello"})
        await text_to_speech_websocket(ws)
        assert ws.sent and "not configured" in ws.sent[0]["error"].lower()

    @pytest.mark.asyncio
    async def test_tts_ws_streams_audio(self, monkeypatch, _noop_usage_recorder):
        from speech.container import get_tts_provider
        from speech.routers.tts import text_to_speech_websocket

        monkeypatch.setattr("speech.config.settings.SARVAM_API_KEY", "test-key")
        get_tts_provider.cache_clear()

        wav = _make_wav_bytes(b"\x00\x01" * 100)
        encoded = base64.b64encode(wav).decode("utf-8")
        mock_client = _mock_httpx_post({"audios": [encoded]})

        with patch("httpx.AsyncClient") as mock_client_class:
            mock_client_class.return_value.__aenter__.return_value = mock_client
            ws = _FakeWebSocket(config={"text": "Hello world.", "language_code": "en-IN"})
            await text_to_speech_websocket(ws)

        assert ws.sent
        assert ws.sent[0]["chunk_index"] == 0
        assert ws.sent[-1].get("done") is True
