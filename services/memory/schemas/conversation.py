from datetime import datetime
from typing import Optional

from pydantic import BaseModel, Field


class ConversationMessageCreate(BaseModel):
    session_id: str
    user_id: Optional[str] = None
    role: str = Field(..., pattern="^(user|assistant|system|tool)$")
    content: str = Field(..., min_length=1)
    sequence: Optional[int] = None


class ConversationMessageResponse(BaseModel):
    id: str
    session_id: str
    user_id: Optional[str] = None
    role: str
    content: str
    sequence: int
    created_at: datetime

    model_config = {"from_attributes": True}
