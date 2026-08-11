import base64
from unittest.mock import AsyncMock, MagicMock, patch

import pytest


@pytest.mark.asyncio
async def test_stt_no_api_key_returns_informative_message(client, monkeypatch):
    from speech.container import get_stt_provider
    monkeypatch.setattr("speech.config.settings.SARVAM_API_KEY", "")
    get_stt_provider.cache_clear()
    files = {"file": ("test.webm", b"fake audio", "audio/webm")}
    response = await client.post("/api/speech-to-text", files=files)
    assert response.status_code == 200
    data = response.json()
    assert "not configured" in data["transcript"].lower()


@pytest.mark.asyncio
async def test_tts_no_api_key_returns_empty_audio(client, monkeypatch):
    from speech.container import get_tts_provider
    monkeypatch.setattr("speech.config.settings.SARVAM_API_KEY", "")
    get_tts_provider.cache_clear()
    response = await client.post(
        "/api/text-to-speech",
        json={"text": "Hello world"},
    )
    assert response.status_code == 200
    data = response.json()
    assert data["audio_base64"] == ""
    assert data["audio_bytes"] == 0


def _mock_httpx_post(return_value):
    mock_response = MagicMock()
    mock_response.status_code = 200
    mock_response.json.return_value = return_value
    mock_response.raise_for_status = MagicMock()

    mock_client = AsyncMock()
    mock_client.post = AsyncMock(return_value=mock_response)
    return mock_client, mock_response


class TestSTTProvider:
    def test_default_config(self):
        from speech.providers.stt import SarvamSTTProvider
        provider = SarvamSTTProvider(api_key="test-key")
        assert provider._model == "saaras:v3"
        assert provider._base_url == "https://api.sarvam.ai"

    @pytest.mark.asyncio
    async def test_transcribe_success(self):
        from speech.providers.stt import SarvamSTTProvider

        provider = SarvamSTTProvider(
            api_key="test-key",
            base_url="https://mock.local",
            model="saaras:v3",
            timeout=5.0,
        )

        mock_client, _ = _mock_httpx_post({
            "transcript": "Hello world",
            "language_code": "en-IN",
        })

        with patch("httpx.AsyncClient") as mock_client_class:
            mock_client_class.return_value.__aenter__.return_value = mock_client
            result = await provider.transcribe(b"fake audio")
            assert result["transcript"] == "Hello world"
            assert result["language_code"] == "en-IN"


class TestWordTimestamps:
    def test_estimate_word_timestamps_proportional(self):
        from speech.providers.word_timestamps import estimate_word_timestamps

        timestamps = estimate_word_timestamps("Hello world", 2.0)
        assert [ts.word for ts in timestamps] == ["Hello", "world"]
        assert timestamps[0].start_sec == 0.0
        assert timestamps[0].end_sec == pytest.approx(1.0, abs=0.001)
        assert timestamps[1].start_sec == pytest.approx(1.0, abs=0.001)
        assert timestamps[1].end_sec == pytest.approx(2.0, abs=0.001)
        assert all(ts.confidence is None for ts in timestamps)

    def test_estimate_word_timestamps_empty_or_zero_duration(self):
        from speech.providers.word_timestamps import estimate_word_timestamps

        assert estimate_word_timestamps("", 5.0) == []
        assert estimate_word_timestamps("Hello", 0.0) == []

    @pytest.mark.asyncio
    async def test_stt_endpoint_returns_word_timestamps(self, client, monkeypatch):
        from speech.container import get_stt_provider

        monkeypatch.setattr("speech.config.settings.SARVAM_API_KEY", "test-key")
        get_stt_provider.cache_clear()

        mock_client, _ = _mock_httpx_post({"transcript": "Hello world", "language_code": "en-IN"})

        monkeypatch.setattr("speech.routers.stt.get_usage_recorder", lambda: AsyncMock())

        with patch("httpx.AsyncClient") as mock_client_class:
            mock_client_class.return_value.__aenter__.return_value = mock_client
            response = await client.post(
                "/api/speech-to-text",
                files={"file": ("test.webm", b"fake audio", "audio/webm")},
                data={"with_word_timestamps": "true"},
            )

        assert response.status_code == 200
        data = response.json()
        assert [ts["word"] for ts in data["word_timestamps"]] == ["Hello", "world"]
        assert data["word_timestamps"][0]["start_sec"] == 0.0

    @pytest.mark.asyncio
    async def test_stt_endpoint_omits_timestamps_by_default(self, client, monkeypatch):
        from speech.container import get_stt_provider

        monkeypatch.setattr("speech.config.settings.SARVAM_API_KEY", "test-key")
        get_stt_provider.cache_clear()

        mock_client, _ = _mock_httpx_post({"transcript": "Hello world", "language_code": "en-IN"})

        with patch("httpx.AsyncClient") as mock_client_class:
            mock_client_class.return_value.__aenter__.return_value = mock_client
            response = await client.post(
                "/api/speech-to-text",
                files={"file": ("test.webm", b"fake audio", "audio/webm")},
            )

        assert response.status_code == 200
        assert response.json()["word_timestamps"] is None


class TestTTSProvider:
    def test_default_config(self):
        from speech.providers.tts import SarvamTTSProvider
        provider = SarvamTTSProvider(api_key="test-key")
        assert provider._speaker == "anushka"
        assert provider._lang == "en-IN"

    @pytest.mark.asyncio
    async def test_synthesize_success(self):
        from speech.providers.tts import SarvamTTSProvider

        provider = SarvamTTSProvider(
            api_key="test-key",
            base_url="https://mock.local",
            speaker="anushka",
            timeout=5.0,
        )

        test_audio = b"fake wav audio data"
        encoded = base64.b64encode(test_audio).decode("utf-8")

        mock_client, _ = _mock_httpx_post({"audios": [encoded]})

        with patch("httpx.AsyncClient") as mock_client_class:
            mock_client_class.return_value.__aenter__.return_value = mock_client
            result = await provider.synthesize("Hello")
            assert result == test_audio

    @pytest.mark.asyncio
    async def test_synthesize_text_truncation(self):
        from speech.providers.tts import SarvamTTSProvider

        provider = SarvamTTSProvider(api_key="test-key", base_url="https://mock.local")
        long_text = "x" * 3000
        test_audio = b"audio"
        encoded = base64.b64encode(test_audio).decode("utf-8")

        mock_client, _ = _mock_httpx_post({"audios": [encoded]})

        with patch("httpx.AsyncClient") as mock_client_class:
            mock_client_class.return_value.__aenter__.return_value = mock_client
            result = await provider.synthesize(long_text)
            assert result == test_audio
