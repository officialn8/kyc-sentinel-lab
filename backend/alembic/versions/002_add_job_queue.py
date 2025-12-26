"""Add durable job queue table for processing

Revision ID: 002_add_job_queue
Revises: 001_initial
Create Date: 2025-12-26
"""

from typing import Sequence, Union

from alembic import op
import sqlalchemy as sa


# revision identifiers, used by Alembic.
revision: str = "002_add_job_queue"
down_revision: Union[str, None] = "001_initial"
branch_labels: Union[str, Sequence[str], None] = None
depends_on: Union[str, Sequence[str], None] = None


def upgrade() -> None:
    op.create_table(
        "kyc_jobs",
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
        sa.Column("session_id", sa.UUID(), nullable=False),
        sa.Column("job_type", sa.String(length=50), nullable=False),
        sa.Column("status", sa.String(length=20), nullable=False),
        sa.Column("attempts", sa.Integer(), nullable=False, server_default="0"),
        sa.Column("last_error", sa.Text(), nullable=True),
        sa.Column("attack_family", sa.String(length=50), nullable=True),
        sa.Column("attack_severity", sa.String(length=20), nullable=True),
        sa.ForeignKeyConstraint(["session_id"], ["kyc_sessions.id"], ondelete="CASCADE"),
        sa.PrimaryKeyConstraint("id"),
    )

    op.create_index("ix_kyc_jobs_status_created", "kyc_jobs", ["status", "created_at"])
    op.create_index("ix_kyc_jobs_session_id", "kyc_jobs", ["session_id"])


def downgrade() -> None:
    op.drop_index("ix_kyc_jobs_session_id", table_name="kyc_jobs")
    op.drop_index("ix_kyc_jobs_status_created", table_name="kyc_jobs")
    op.drop_table("kyc_jobs")


