from shared.llm.openai_compatible_provider import (
    OpenAICompatibleProvider,
    DEFAULT_TIMEOUT,
    DEFAULT_TEMPERATURE,
    DEFAULT_MAX_TOKENS,
)

DEFAULT_BASE_URL = "https://api.sarvam.ai"
DEFAULT_MODEL = "sarvam-105b"


class SarvamProvider(OpenAICompatibleProvider):
    def __init__(
        self,
        api_key: str,
        base_url: str = DEFAULT_BASE_URL,
        model: str = DEFAULT_MODEL,
        timeout: float = DEFAULT_TIMEOUT,
        temperature: float = DEFAULT_TEMPERATURE,
        max_tokens: int = DEFAULT_MAX_TOKENS,
        usage_hook=None,
    ) -> None:
        super().__init__(
            base_url=base_url,
            model=model,
            api_key=api_key,
            timeout=timeout,
            temperature=temperature,
            max_tokens=max_tokens,
            extra_headers={"api-subscription-key": api_key},
            usage_hook=usage_hook,
        )
