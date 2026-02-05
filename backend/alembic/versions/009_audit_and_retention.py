"""Add audit logging and retention fields.

Revision ID: 009_audit_and_retention
Revises: 008_webhooks
Create Date: 2026-02-04
"""

from typing import Sequence, Union

from alembic import op
import sqlalchemy as sa


# revision identifiers, used by Alembic.
revision: str = "009_audit_and_retention"
down_revision: Union[str, None] = "008_webhooks"
branch_labels: Union[str, Sequence[str], None] = None
depends_on: Union[str, Sequence[str], None] = None


def upgrade() -> None:
    op.create_table(
        "audit_chains",
        sa.Column("id", sa.UUID(), nullable=False),
        sa.Column("chain_date", sa.Date(), nullable=False),
        sa.Column("last_hash", sa.String(length=64), nullable=False),
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
        sa.PrimaryKeyConstraint("id"),
        sa.UniqueConstraint("chain_date", name="uq_audit_chains_chain_date"),
    )

    op.create_table(
        "audit_events",
        sa.Column("id", sa.UUID(), nullable=False),
        sa.Column(
            "created_at",
            sa.DateTime(timezone=True),
            server_default=sa.text("now()"),
            nullable=False,
        ),
        sa.Column("event_type", sa.String(length=100), nullable=False),
        sa.Column("status", sa.String(length=50), nullable=True),
        sa.Column("actor_type", sa.String(length=50), nullable=True),
        sa.Column("actor_id", sa.String(length=128), nullable=True),
        sa.Column("session_id", sa.UUID(), nullable=True),
        sa.Column("resource_type", sa.String(length=50), nullable=True),
        sa.Column("resource_id", sa.String(length=128), nullable=True),
        sa.Column("request_id", sa.String(length=128), nullable=True),
        sa.Column("ip_address", sa.String(length=45), nullable=True),
        sa.Column("user_agent", sa.String(length=512), nullable=True),
        sa.Column("prev_hash", sa.String(length=64), nullable=False),
        sa.Column("event_hash", sa.String(length=64), nullable=False),
        sa.Column("metadata", sa.JSON(), nullable=False),
        sa.Column("archived_at", sa.DateTime(timezone=True), nullable=True),
        sa.Column("archive_error", sa.String(length=512), nullable=True),
        sa.PrimaryKeyConstraint("id"),
    )

    op.create_index("ix_audit_events_event_type", "audit_events", ["event_type"])
    op.create_index("ix_audit_events_session_id", "audit_events", ["session_id"])
    op.create_index("ix_audit_events_created_at", "audit_events", ["created_at"])

    op.add_column(
        "kyc_sessions",
        sa.Column("deleted_at", sa.DateTime(timezone=True), nullable=True),
    )
    op.add_column(
        "kyc_sessions",
        sa.Column("deleted_reason", sa.String(length=200), nullable=True),
    )
    op.add_column(
        "kyc_sessions",
        sa.Column("media_purged_at", sa.DateTime(timezone=True), nullable=True),
    )
    op.create_index(
        "ix_kyc_sessions_deleted_at",
        "kyc_sessions",
        ["deleted_at"],
    )
    op.create_index(
        "ix_kyc_sessions_media_purged_at",
        "kyc_sessions",
        ["media_purged_at"],
    )


def downgrade() -> None:
    op.drop_index("ix_kyc_sessions_media_purged_at", table_name="kyc_sessions")
    op.drop_index("ix_kyc_sessions_deleted_at", table_name="kyc_sessions")
    op.drop_column("kyc_sessions", "media_purged_at")
    op.drop_column("kyc_sessions", "deleted_reason")
    op.drop_column("kyc_sessions", "deleted_at")

    op.drop_index("ix_audit_events_created_at", table_name="audit_events")
    op.drop_index("ix_audit_events_session_id", table_name="audit_events")
    op.drop_index("ix_audit_events_event_type", table_name="audit_events")
    op.drop_table("audit_events")
    op.drop_table("audit_chains")
