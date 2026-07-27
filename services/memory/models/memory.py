import uuid
from sqlalchemy import Column, String, Text, DateTime
from sqlalchemy.dialects.postgresql import UUID, JSONB
from shared.models.base import Base, TimestampMixin


class MemoryModel(Base, TimestampMixin):
    __tablename__ = "memories"
    id = Column(UUID(as_uuid=True), primary_key=True, default=uuid.uuid4)
    content = Column(Text, nullable=False)
    memory_type = Column(String(50), nullable=False, index=True)
    metadata_ = Column("metadata", JSONB, nullable=True)
    user_id = Column(String(255), nullable=True, index=True)
    session_id = Column(String(255), nullable=True, index=True)
