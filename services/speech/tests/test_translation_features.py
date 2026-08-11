from unittest.mock import AsyncMock, MagicMock, patch

import pytest

from speech.providers.translation_utils import apply_glossary, compute_translation_confidence


def _mock_httpx_post(return_value):
    mock_response = MagicMock()
    mock_response.status_code = 200
    mock_response.json.return_value = return_value
    mock_response.raise_for_status = MagicMock()

    mock_client = AsyncMock()
    mock_client.post = AsyncMock(return_value=mock_response)
    return mock_client


class TestGlossary:
    def test_apply_glossary_substitutes_whole_words(self):
        text, matches = apply_glossary("I need a loan and a house", {"loan": "ऋण", "house": "घर"})
        assert "loan" not in text
        assert "ऋण" in text and "घर" in text
        assert matches == ["house", "loan"]

    def test_apply_glossary_respects_word_boundaries(self):
        text, matches = apply_glossary("loaner loans loan", {"loan": "ऋण"})
        assert "loaner loans " in text
        assert matches == ["loan"]

    def test_apply_glossary_longer_terms_win(self):
        text, matches = apply_glossary("New Delhi", {"New Delhi": "नई दिल्ली", "Delhi": "दिल्ली"})
        assert text == "नई दिल्ली"
        assert matches == ["New Delhi"]

    def test_apply_glossary_none(self):
        text, matches = apply_glossary("plain", None)
        assert text == "plain"
        assert matches == []


class TestConfidence:
    def test_empty_translation_is_zero(self):
        assert compute_translation_confidence("hola", "", "es-ES", "en-IN") == 0.0

    def test_same_language_noop_is_low(self):
        assert compute_translation_confidence("hello", "hello", "en-IN", "en-IN") == 0.3

    def test_real_translation_scores_high(self):
        confidence = compute_translation_confidence(
            "I need a loan", "मुझे ऋण चाहिए", "en-IN", "hi-IN", ["loan"]
        )
        assert 0.5 <= confidence <= 1.0

    def test_confidence_bounded(self):
        for _ in range(50):
            confidence = compute_translation_confidence(
                "a" * 100, "b" * 50, "en-IN", "hi-IN", ["a"]
            )
            assert 0.0 <= confidence <= 1.0


class TestTranslationFeatures:
    @pytest.mark.asyncio
    async def test_provider_applies_glossary_and_pipeline(self):
        from speech.providers.translation import SarvamTranslationProvider

        provider = SarvamTranslationProvider(api_key="test-key", base_url="https://mock.local")
        mock_client = _mock_httpx_post({"translated_text": "ऋण", "source_language_code": "en-IN"})

        with patch("httpx.AsyncClient") as mock_client_class:
            mock_client_class.return_value.__aenter__.return_value = mock_client
            result = await provider.translate_text(
                "loan",
                target_language_code="hi-IN",
                source_language_code="en-IN",
                pipeline_mode="pipeline",
                glossary={"loan": "ऋण"},
                with_confidence=True,
            )

        sent_payload = mock_client.post.call_args.kwargs["json"]
        assert sent_payload.get("pipeline") is True
        assert sent_payload["input"] == "ऋण"
        assert result["glossary_matches"] == ["loan"]
        assert result["confidence"] is not None

    @pytest.mark.asyncio
    async def test_endpoint_returns_glossary_and_confidence(self, client, monkeypatch):
        from speech.container import get_translation_provider

        monkeypatch.setattr("speech.config.settings.SARVAM_API_KEY", "test-key")
        get_translation_provider.cache_clear()
        monkeypatch.setattr(
            "speech.routers.translation.get_usage_recorder",
            lambda: AsyncMock(),
        )

        mock_client = _mock_httpx_post({"translated_text": "ऋण", "source_language_code": "en-IN"})

        with patch("httpx.AsyncClient") as mock_client_class:
            mock_client_class.return_value.__aenter__.return_value = mock_client
            response = await client.post(
                "/api/translate-text",
                json={
                    "text": "I need a loan",
                    "source_language_code": "en-IN",
                    "target_language_code": "hi-IN",
                    "pipeline_mode": "pipeline",
                    "glossary": {"loan": "ऋण"},
                    "with_confidence": True,
                },
            )

        assert response.status_code == 200
        data = response.json()
        assert data["pipeline_mode"] == "pipeline"
        assert data["glossary_matches"] == ["loan"]
        assert data["confidence"] is not None
