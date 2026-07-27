from datetime import datetime
from typing import Any, Optional

from pydantic import BaseModel, Field


class UserProfileCreate(BaseModel):
    user_id: str
    display_name: Optional[str] = None
    preferences: dict[str, Any] = Field(default_factory=dict)
    metadata: Optional[dict[str, Any]] = None


class UserProfileUpdate(BaseModel):
    display_name: Optional[str] = None
    preferences: Optional[dict[str, Any]] = None
    metadata: Optional[dict[str, Any]] = None


class UserProfileResponse(BaseModel):
    id: str
    user_id: str
    display_name: Optional[str] = None
    preferences: dict[str, Any]
    metadata: Optional[dict[str, Any]] = None
    created_at: datetime
    updated_at: datetime

    model_config = {"from_attributes": True}
