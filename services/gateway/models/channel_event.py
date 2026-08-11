"""Channel event model: one row per inbound message outcome at the gateway.

Deliberately PII-free: we record the outcome (accepted / rate_limited / blocked),
the channel, a coarse scope key, and guardrail metadata — never the raw message
text. Stored user content is already redacted upstream, and persisting it here
would defeat that.
"""

import uuid
from datetime import UTC, datetime

from sqlalchemy import DateTime, Float, Integer, String
from sqlalchemy.orm import Mapped, mapped_column

from shared.models.base import Base


class ChannelEvent(Base):
    __tablename__ = "channel_events"

    id: Mapped[str] = mapped_column(String(36), primary_key=True, default=lambda: str(uuid.uuid4()))
    recorded_at: Mapped[datetime] = mapped_column(
        DateTime(timezone=True), default=lambda: datetime.now(UTC), index=True
    )
    channel: Mapped[str] = mapped_column(String(16), nullable=False, index=True)
    scope: Mapped[str] = mapped_column(String(128), nullable=False, default="", index=True)
    status: Mapped[str] = mapped_column(String(16), nullable=False, index=True)
    violation_category: Mapped[str | None] = mapped_column(String(40), nullable=True)
    reason: Mapped[str | None] = mapped_column(String(256), nullable=True)
    redactions_count: Mapped[int] = mapped_column(Integer, nullable=False, default=0)
    request_id: Mapped[str | None] = mapped_column(String(36), nullable=True, index=True)
    message_id: Mapped[str | None] = mapped_column(String(64), nullable=True)
    duration_ms: Mapped[float] = mapped_column(Float, nullable=False, default=0.0)
