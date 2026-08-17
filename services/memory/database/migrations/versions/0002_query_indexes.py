"""Add query-path indexes

Revision ID: 0002_query_indexes
Revises: 0001_memory
Create Date: 2026-08-11
"""
from collections.abc import Sequence

from alembic import op

revision: str = "0002_query_indexes"
down_revision: str | None = "0001_memory"
branch_labels: str | Sequence[str] | None = None
depends_on: str | Sequence[str] | None = None


def upgrade() -> None:
    op.create_index(
        "ix_conversation_messages_session_created",
        "conversation_messages",
        ["session_id", "created_at"],
    )
    op.create_index(
        "ix_long_term_memories_importance",
        "long_term_memories",
        ["importance"],
    )


def downgrade() -> None:
    op.drop_index("ix_long_term_memories_importance", table_name="long_term_memories")
    op.drop_index("ix_conversation_messages_session_created", table_name="conversation_messages")
