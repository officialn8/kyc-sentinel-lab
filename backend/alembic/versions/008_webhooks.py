"""Add webhooks table.

Revision ID: 008_webhooks
Revises: 007_enum_constraints
Create Date: 2026-01-18
"""

from typing import Sequence, Union

from alembic import op
import sqlalchemy as sa

# revision identifiers, used by Alembic.
revision: str = "008_webhooks"
down_revision: Union[str, None] = "007_enum_constraints"
branch_labels: Union[str, Sequence[str], None] = None
depends_on: Union[str, Sequence[str], None] = None


def upgrade() -> None:
    """Create webhooks table."""
    op.create_table(
        "webhooks",
        sa.Column("id", sa.UUID(), nullable=False),
        sa.Column(
            "created_at",
            sa.DateTime(timezone=True),
            server_default=sa.text("now()"),
            nullable=False,
        ),
        sa.Column(
            "updated_at",
            sa.DateTime(timezone=True),
            server_default=sa.text("now()"),
            nullable=False,
        ),
        sa.Column("url", sa.String(2048), nullable=False),
        sa.Column("events", sa.JSON(), nullable=False),
        sa.Column("name", sa.String(255), nullable=True),
        sa.Column("enabled", sa.Boolean(), nullable=False, default=True),
        sa.Column("failure_count", sa.Integer(), nullable=False, default=0),
        sa.Column("last_failure", sa.DateTime(timezone=True), nullable=True),
        sa.Column("last_success", sa.DateTime(timezone=True), nullable=True),
        sa.PrimaryKeyConstraint("id"),
    )
    
    # Index for efficient event filtering
    op.create_index(
        "ix_webhooks_enabled",
        "webhooks",
        ["enabled"],
        postgresql_where=sa.text("enabled = true"),
    )


def downgrade() -> None:
    """Drop webhooks table."""
    op.drop_index("ix_webhooks_enabled", table_name="webhooks")
    op.drop_table("webhooks")
