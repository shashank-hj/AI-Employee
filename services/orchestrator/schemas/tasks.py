from enum import Enum
from typing import Optional
from pydantic import BaseModel


class TaskStatus(str, Enum):
    PENDING = "pending"
    RUNNING = "running"
    COMPLETED = "completed"
    FAILED = "failed"


class TaskCreate(BaseModel):
    title: str
    description: Optional[str] = None
    priority: Optional[int] = 0


class TaskResponse(BaseModel):
    status: TaskStatus
    message: str
    tasks: list = []
