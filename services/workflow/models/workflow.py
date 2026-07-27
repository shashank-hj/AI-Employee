import uuid
from sqlalchemy import Column, String, Text, Integer
from sqlalchemy.dialects.postgresql import UUID, JSONB
from shared.models.base import Base, TimestampMixin


class WorkflowModel(Base, TimestampMixin):
    __tablename__ = "workflows"
    id = Column(UUID(as_uuid=True), primary_key=True, default=uuid.uuid4)
    name = Column(String(255), nullable=False, index=True)
    description = Column(Text, nullable=True)
    workflow_type = Column(String(100), default="default")
    status = Column(String(50), default="pending", index=True)
    current_step = Column(String(255), nullable=True)
    steps = Column(JSONB, nullable=True)
    input_data = Column(JSONB, nullable=True)
    output_data = Column(JSONB, nullable=True)
    error_message = Column(Text, nullable=True)
