import uuid

from sqlalchemy import Column, String
from sqlalchemy.dialects.postgresql import JSONB, UUID

from shared.models.base import Base, TimestampMixin


class UserProfileModel(Base, TimestampMixin):
    __tablename__ = "user_profiles"

    id = Column(UUID(as_uuid=True), primary_key=True, default=uuid.uuid4)
    user_id = Column(String(255), unique=True, nullable=False, index=True)
    display_name = Column(String(255), nullable=True)
    preferences = Column(JSONB, nullable=False, default=dict)
    metadata_ = Column("metadata", JSONB, nullable=True)
