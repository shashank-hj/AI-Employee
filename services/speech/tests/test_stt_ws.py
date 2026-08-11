from unittest.mock import AsyncMock, MagicMock, patch

import pytest


class _FakeWebSocket:
    def __init__(self, config=None, binary_messages=()):
        self._config = config
        self._binary_messages = list(binary_messages)
        self.sent = []
        self.closed = False

    async def accept(self):
        pass

    async def receive_json(self):
        return self._config

    async def receive(self):
        if self._binary_messages:
            return {"type": "websocket.receive_bytes", "bytes": self._binary_messages.pop(0)}
        return {"type": "websocket.disconnect"}

    async def send_json(self, message):
        self.sent.append(message)

    async def close(self):
        self.closed = True


def _mock_httpx_post(return_value):
    mock_response = MagicMock()
    mock_response.status_code = 200
    mock_response.json.return_value = return_value
    mock_response.raise_for_status = MagicMock()

    mock_client = AsyncMock()
    mock_client.post = AsyncMock(return_value=mock_response)
    return mock_client


@pytest.fixture
def _noop_usage_recorder(monkeypatch):
    monkeypatch.setattr("speech.routers.stt.get_usage_recorder", lambda: AsyncMock())


class TestSTTWebSocket:
    @pytest.mark.asyncio
    async def test_stt_ws_no_api_key_rejects(self, monkeypatch):
        from speech.container import get_stt_provider
        from speech.routers.stt import speech_to_text_websocket

        monkeypatch.setattr("speech.config.settings.SARVAM_API_KEY", "")
        get_stt_provider.cache_clear()

        ws = _FakeWebSocket(config={"language_code": "en-IN"})
        await speech_to_text_websocket(ws)
        assert ws.sent and "not configured" in ws.sent[0]["error"].lower()

    @pytest.mark.asyncio
    async def test_stt_ws_transcribes_audio(self, monkeypatch, _noop_usage_recorder):
        from speech.container import get_stt_provider
        from speech.routers.stt import speech_to_text_websocket

        monkeypatch.setattr("speech.config.settings.SARVAM_API_KEY", "test-key")
        get_stt_provider.cache_clear()

        mock_client = _mock_httpx_post({"transcript": "Hello", "language_code": "en-IN"})

        with patch("httpx.AsyncClient") as mock_client_class:
            mock_client_class.return_value.__aenter__.return_value = mock_client
            ws = _FakeWebSocket(
                config={"language_code": "en-IN", "mode": "transcribe"},
                binary_messages=[b"audio"],
            )
            await speech_to_text_websocket(ws)

        assert ws.sent
        assert ws.sent[0]["transcript"] == "Hello"
        assert ws.sent[0]["language_code"] == "en-IN"
