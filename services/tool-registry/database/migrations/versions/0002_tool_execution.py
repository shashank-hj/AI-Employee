"""Add tool execution fields (execution_type, execution_config)

Revision ID: 0002_tool_execution
Revises: 0001
Create Date: 2025-01-15 00:00:01.000000
"""
from collections.abc import Sequence

import sqlalchemy as sa
from alembic import op
from sqlalchemy.dialects import postgresql

revision: str = "0002"
down_revision: str | None = "0001"
branch_labels: str | Sequence[str] | None = None
depends_on: str | Sequence[str] | None = None


def upgrade() -> None:
    op.add_column(
        "tools",
        sa.Column("execution_type", sa.String(50), nullable=False, server_default="native"),
    )
    op.add_column(
        "tools",
        sa.Column(
            "execution_config",
            postgresql.JSONB(),
            nullable=False,
            server_default=sa.text("'{}'::jsonb"),
        ),
    )


def downgrade() -> None:
    op.drop_column("tools", "execution_config")
    op.drop_column("tools", "execution_type")
