import uuid

from sqlalchemy import Column, Integer, String, Text
from sqlalchemy.dialects.postgresql import JSONB, UUID

from shared.models.base import Base, TimestampMixin


class DocumentStatus:
    PENDING = "pending"
    PROCESSING = "processing"
    READY = "ready"
    FAILED = "failed"


class DocumentModel(Base, TimestampMixin):
    __tablename__ = "documents"

    id = Column(UUID(as_uuid=True), primary_key=True, default=uuid.uuid4)
    title = Column(String(500), nullable=False, index=True)
    content = Column(Text, nullable=False)
    source = Column(String(500), nullable=True)
    content_type = Column(String(100), default="text/plain")
    status = Column(String(50), default=DocumentStatus.PENDING, index=True)
    chunks_count = Column(Integer, default=0)
    metadata_ = Column("metadata", JSONB, nullable=True)
