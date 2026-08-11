import uuid

from sqlalchemy import Boolean, Column, Float, String, Text
from sqlalchemy.dialects.postgresql import ARRAY, JSONB, UUID

from shared.models.base import Base, TimestampMixin


class ToolModel(Base, TimestampMixin):
    __tablename__ = "tools"

    id = Column(UUID(as_uuid=True), primary_key=True, default=uuid.uuid4)
    name = Column(String(255), unique=True, nullable=False, index=True)
    description = Column(Text, nullable=True)
    version = Column(String(50), nullable=False, default="1.0.0")
    category = Column(String(100), nullable=False, index=True)
    permissions = Column(JSONB, nullable=False, default=list)
    input_schema = Column(JSONB, nullable=False, default=dict)
    output_schema = Column(JSONB, nullable=False, default=dict)
    timeout_seconds = Column(Float, nullable=False, default=30.0)
    retry_policy = Column(JSONB, nullable=False, default=dict)
    tags = Column(ARRAY(String), nullable=False, default=list)
    is_active = Column(Boolean, default=True, index=True)
    execution_type = Column(String(50), nullable=False, default="native")
    execution_config = Column(JSONB, nullable=False, default=dict)
