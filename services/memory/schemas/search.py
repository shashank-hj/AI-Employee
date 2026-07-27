from datetime import datetime
from typing import Any, Optional

from pydantic import BaseModel, Field


class MemorySearchRequest(BaseModel):
    query: str = Field(..., min_length=1, description="Natural language search query")
    user_id: Optional[str] = Field(default=None, description="Filter to specific user")
    memory_type: Optional[str] = Field(default=None, description="Filter by memory type")
    top_k: int = Field(default=10, ge=1, le=100)
    min_score: float = Field(default=0.0, ge=0.0, le=1.0, description="Minimum similarity score threshold")


class MemorySearchResult(BaseModel):
    id: str
    user_id: str
    content: str
    memory_type: str
    importance: float
    score: float
    metadata: Optional[dict[str, Any]] = None
    source: Optional[str] = None
    created_at: datetime
