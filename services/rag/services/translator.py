"""Query translation for cross-lingual retrieval.

The embedding model is primarily English-capable, so non-English queries are
translated to the index language (default ``en-IN``) before embedding. A no-op
translator keeps the pipeline usable when translation is not configured.
"""

import httpx
import structlog

logger = structlog.get_logger(__name__)

INDEX_LANGUAGE = "en-IN"


class QueryTranslator:
    async def translate(self, text: str, target_language_code: str = INDEX_LANGUAGE) -> str:
        raise NotImplementedError


class NoopQueryTranslator(QueryTranslator):
    async def translate(self, text: str, target_language_code: str = INDEX_LANGUAGE) -> str:
        return text


class SpeechQueryTranslator(QueryTranslator):
    """Translates queries via the speech service's ``/api/translate-text``."""

    def __init__(
        self,
        base_url: str,
        api_key: str = "",
        timeout: float = 10.0,
    ) -> None:
        self._base_url = base_url.rstrip("/")
        self._api_key = api_key
        self._timeout = timeout

    async def translate(self, text: str, target_language_code: str = INDEX_LANGUAGE) -> str:
        if not text:
            return text
        headers = {"api-subscription-key": self._api_key} if self._api_key else {}
        try:
            async with httpx.AsyncClient(timeout=self._timeout) as client:
                response = await client.post(
                    f"{self._base_url}/api/translate-text",
                    json={
                        "text": text,
                        "source_language_code": "auto",
                        "target_language_code": target_language_code,
                    },
                    headers=headers,
                )
                response.raise_for_status()
                data = response.json()
            translated = (data.get("translated_text") or "").strip()
            if not translated:
                logger.warning("query_translation_empty", text=text[:80], fallback=True)
                return text
            logger.info(
                "query_translated",
                text=text[:60],
                target=target_language_code,
            )
            return translated
        except Exception as exc:
            logger.warning("query_translation_failed", error=str(exc)[:200], fallback=True)
            return text
