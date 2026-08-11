import uuid
from datetime import datetime, timezone

from sqlalchemy import DateTime, Float, Numeric, String, Text, JSON
from sqlalchemy.orm import Mapped, mapped_column

from shared.models.base import Base
from shared.usage.records import UsageRecord


class UsageEvent(Base):
    """One row per billable call to an external API (LLM / speech / embeddings)."""

    __tablename__ = "usage_events"

    id: Mapped[str] = mapped_column(String(36), primary_key=True, default=lambda: str(uuid.uuid4()))
    recorded_at: Mapped[datetime] = mapped_column(
        DateTime(timezone=True), default=lambda: datetime.now(timezone.utc), index=True
    )
    service: Mapped[str] = mapped_column(String(32), nullable=False, index=True)
    category: Mapped[str] = mapped_column(String(16), nullable=False)
    operation: Mapped[str] = mapped_column(String(40), nullable=False, index=True)
    model: Mapped[str] = mapped_column(String(64), nullable=False, default="")
    unit: Mapped[str] = mapped_column(String(16), nullable=False, default="tokens")
    input_units: Mapped[float] = mapped_column(Float, nullable=False, default=0.0)
    output_units: Mapped[float] = mapped_column(Float, nullable=False, default=0.0)
    total_units: Mapped[float] = mapped_column(Float, nullable=False, default=0.0)
    cost_inr: Mapped[float] = mapped_column(Numeric(14, 6), nullable=False, default=0.0)
    request_id: Mapped[str | None] = mapped_column(String(36), nullable=True, index=True)
    session_id: Mapped[str | None] = mapped_column(String(64), nullable=True, index=True)
    user_id: Mapped[str | None] = mapped_column(String(64), nullable=True, index=True)
    status: Mapped[str] = mapped_column(String(16), nullable=False, default="success")
    error: Mapped[str | None] = mapped_column(Text, nullable=True)
    duration_ms: Mapped[float] = mapped_column(Float, nullable=False, default=0.0)
    details: Mapped[dict | None] = mapped_column(JSON, nullable=True)

    @classmethod
    def from_record(cls, record: UsageRecord) -> "UsageEvent":
        total = record.total_units
        if total is None:
            total = record.input_units + record.output_units
        return cls(
            service=record.service,
            category=record.category,
            operation=record.operation,
            model=record.model,
            unit=record.unit,
            input_units=record.input_units,
            output_units=record.output_units,
            total_units=total,
            cost_inr=record.cost_inr if record.cost_inr is not None else 0.0,
            request_id=record.request_id,
            session_id=record.session_id,
            user_id=record.user_id,
            status=record.status,
            error=record.error,
            duration_ms=record.duration_ms,
            details=record.metadata or None,
        )
