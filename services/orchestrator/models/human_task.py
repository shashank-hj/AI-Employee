import uuid
from datetime import datetime, timezone

from sqlalchemy import Column, String, DateTime, Integer, Text, JSON
from sqlalchemy.orm import Mapped, mapped_column

from shared.models.base import Base


class HumanTask(Base):
    __tablename__ = "human_tasks"

    id: Mapped[str] = mapped_column(String(36), primary_key=True, default=lambda: str(uuid.uuid4()))
    request_id: Mapped[str | None] = mapped_column(String(36), nullable=True, index=True)
    user_input: Mapped[str] = mapped_column(Text, nullable=False)
    intent: Mapped[str] = mapped_column(String(32), nullable=False)
    reason: Mapped[str | None] = mapped_column(Text, nullable=True)
    priority: Mapped[str] = mapped_column(String(16), default="NORMAL")
    status: Mapped[str] = mapped_column(String(16), default="pending", index=True)
    context: Mapped[dict] = mapped_column(JSON, default=dict)
    assigned_to: Mapped[str | None] = mapped_column(String(64), nullable=True)
    resolution_note: Mapped[str | None] = mapped_column(Text, nullable=True)
    created_at: Mapped[datetime] = mapped_column(DateTime(timezone=True), default=lambda: datetime.now(timezone.utc))
    resolved_at: Mapped[datetime | None] = mapped_column(DateTime(timezone=True), nullable=True)
