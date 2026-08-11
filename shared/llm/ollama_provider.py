from shared.llm.openai_compatible_provider import (
    OpenAICompatibleProvider,
    DEFAULT_TIMEOUT,
    DEFAULT_TEMPERATURE,
    DEFAULT_MAX_TOKENS,
)

DEFAULT_BASE_URL = "http://localhost:11434"
DEFAULT_MODEL = "qwen3:8b"


class OllamaProvider(OpenAICompatibleProvider):
    def __init__(
        self,
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
            api_key="",  # Ollama runs locally — no auth required
            timeout=timeout,
            temperature=temperature,
            max_tokens=max_tokens,
            usage_hook=usage_hook,
        )
