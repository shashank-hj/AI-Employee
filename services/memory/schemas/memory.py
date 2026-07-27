from enum import Enum
from typing import Optional, Any
from pydantic import BaseModel, Field
from datetime import datetime


class MemoryType(str, Enum):
    CONVERSATION = "conversation"
    FACT = "fact"
    PREFERENCE = "preference"
    TASK = "task"
    CONTEXT = "context"


class MemoryCreate(BaseModel):
    content: str
    memory_type: MemoryType = MemoryType.CONVERSATION
    metadata: Optional[dict[str, Any]] = None
    user_id: Optional[str] = None
    session_id: Optional[str] = None


class MemoryResponse(MemoryCreate):
    id: str
    created_at: datetime
    updated_at: Optional[datetime] = None
