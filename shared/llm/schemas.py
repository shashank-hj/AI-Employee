from typing import Any

from pydantic import BaseModel, Field


class Entity(BaseModel):
    name: str
    type: str
    value: str


class IntentClassificationResult(BaseModel):
    intent: str = Field(
        description="Classified intent: sales, support, booking, general, complaint, or escalate",
    )
    confidence: float = Field(
        ge=0.0,
        le=1.0,
        description="Confidence score for the classification",
    )
    requires_human: bool = Field(
        default=False,
        description="Whether this request requires human intervention",
    )
    reason: str = Field(
        default="",
        description="Brief reasoning behind the classification",
    )
    entities: list[Entity] = Field(
        default_factory=list,
        description="Extracted entities from the user input",
    )
    suggested_tools: list[str] = Field(
        default_factory=list,
        description="Suggested tool names to fulfill this intent",
    )


class ChatMessage(BaseModel):
    role: str
    content: str


class ChatRequest(BaseModel):
    model: str
    messages: list[ChatMessage]
    temperature: float = 0.1
    max_tokens: int = 1024
    response_format: dict[str, Any] | None = None


class ChatChoice(BaseModel):
    index: int
    message: ChatMessage
    finish_reason: str | None = None


class ChatUsage(BaseModel):
    prompt_tokens: int = 0
    completion_tokens: int = 0
    total_tokens: int = 0


class ChatResponse(BaseModel):
    id: str
    object: str
    created: int
    model: str
    choices: list[ChatChoice]
    usage: ChatUsage | None = None



# Backward compatibility aliases
SarvamChatMessage = ChatMessage
SarvamChatRequest = ChatRequest
SarvamChatChoice = ChatChoice
SarvamChatUsage = ChatUsage
SarvamChatResponse = ChatResponse
