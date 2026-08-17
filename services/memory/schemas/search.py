from datetime import datetime
from typing import Any

from pydantic import BaseModel, Field


class MemorySearchRequest(BaseModel):
    query: str = Field(..., min_length=1, description="Natural language search query")
    user_id: str | None = Field(default=None, description="Filter to specific user")
    memory_type: str | None = Field(default=None, description="Filter by memory type")
    top_k: int = Field(default=10, ge=1, le=100)
    min_score: float = Field(
        default=0.0,
        ge=0.0,
        le=1.0,
        description="Minimum similarity score threshold",
    )
    importance_min: float | None = Field(
        default=None,
        ge=0.0,
        le=1.0,
        description="Minimum importance (inclusive)",
    )
    importance_max: float | None = Field(
        default=None,
        ge=0.0,
        le=1.0,
        description="Maximum importance (inclusive)",
    )
    sort: str = Field(
        default="score",
        pattern="^(score|importance|created_at)$",
        description="Result ordering",
    )


class MemorySearchResult(BaseModel):
    id: str
    user_id: str
    content: str
    memory_type: str
    importance: float
    score: float
    metadata: dict[str, Any] | None = None
    source: str | None = None
    created_at: datetime
