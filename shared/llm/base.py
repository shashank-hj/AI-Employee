from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from typing import Any


@dataclass
class IntentClassification:
    intent: str
    confidence: float
    requires_human: bool = False
    reason: str = ""
    entities: list[dict[str, Any]] = field(default_factory=list)
    suggested_tools: list[str] = field(default_factory=list)


@dataclass
class LLMResponse:
    content: str
    model: str
    input_tokens: int = 0
    output_tokens: int = 0
    duration_ms: float = 0.0


class LLMProvider(ABC):
    @abstractmethod
    async def classify_intent(self, user_input: str) -> IntentClassification:
        ...

    @abstractmethod
    async def generate(self, system_prompt: str, user_message: str) -> LLMResponse:
        ...

    @abstractmethod
    async def health_check(self) -> bool:
        ...
