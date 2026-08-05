import uuid

from pgvector.sqlalchemy import Vector
from sqlalchemy import Column, Float, String, Text
from sqlalchemy.dialects.postgresql import JSONB, UUID

from shared.models.base import Base, TimestampMixin


class LongTermMemoryModel(Base, TimestampMixin):
    __tablename__ = "long_term_memories"

    id = Column(UUID(as_uuid=True), primary_key=True, default=uuid.uuid4)
    user_id = Column(String(255), nullable=False, index=True)
    content = Column(Text, nullable=False)
    memory_type = Column(String(50), nullable=False, index=True)
    importance = Column(Float, default=0.5, nullable=False)
    embedding = Column(Vector(768), nullable=True)
    metadata_ = Column("metadata", JSONB, nullable=True)
    source = Column(String(255), nullable=True)
