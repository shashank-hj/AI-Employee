from datetime import datetime
from enum import Enum
from typing import Any, Optional

from pydantic import BaseModel, Field


class LongTermMemoryType(str, Enum):
    FACT = "fact"
    PREFERENCE = "preference"
    KNOWLEDGE = "knowledge"
    EVENT = "event"
    SUMMARY = "summary"


class LongTermMemoryCreate(BaseModel):
    user_id: str
    content: str = Field(..., min_length=1)
    memory_type: LongTermMemoryType = LongTermMemoryType.FACT
    importance: float = Field(default=0.5, ge=0.0, le=1.0)
    metadata: Optional[dict[str, Any]] = None
    source: Optional[str] = None


class LongTermMemoryResponse(BaseModel):
    id: str
    user_id: str
    content: str
    memory_type: LongTermMemoryType
    importance: float
    metadata: Optional[dict[str, Any]] = None
    source: Optional[str] = None
    created_at: datetime
    updated_at: datetime

    model_config = {"from_attributes": True}
