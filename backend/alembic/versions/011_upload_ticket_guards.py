"""Add upload ticket validation fields to sessions.

Revision ID: 011_upload_ticket_guards
Revises: 010_extend_result_versions
Create Date: 2026-02-09
"""

from typing import Sequence, Union

from alembic import op
import sqlalchemy as sa


# revision identifiers, used by Alembic.
revision: str = "011_upload_ticket_guards"
down_revision: Union[str, None] = "010_extend_result_versions"
branch_labels: Union[str, Sequence[str], None] = None
depends_on: Union[str, Sequence[str], None] = None


def upgrade() -> None:
    op.add_column(
        "kyc_sessions",
        sa.Column("selfie_expected_size_bytes", sa.BigInteger(), nullable=True),
    )
    op.add_column(
        "kyc_sessions",
        sa.Column("id_expected_size_bytes", sa.BigInteger(), nullable=True),
    )
    op.add_column(
        "kyc_sessions",
        sa.Column("selfie_expected_content_type", sa.String(length=100), nullable=True),
    )
    op.add_column(
        "kyc_sessions",
        sa.Column("id_expected_content_type", sa.String(length=100), nullable=True),
    )
    op.add_column(
        "kyc_sessions",
        sa.Column("selfie_upload_ticket_hash", sa.String(length=64), nullable=True),
    )
    op.add_column(
        "kyc_sessions",
        sa.Column("id_upload_ticket_hash", sa.String(length=64), nullable=True),
    )
    op.add_column(
        "kyc_sessions",
        sa.Column("upload_ticket_expires_at", sa.DateTime(timezone=True), nullable=True),
    )
    op.create_index(
        "ix_kyc_sessions_upload_ticket_expires_at",
        "kyc_sessions",
        ["upload_ticket_expires_at"],
    )


def downgrade() -> None:
    op.drop_index("ix_kyc_sessions_upload_ticket_expires_at", table_name="kyc_sessions")
    op.drop_column("kyc_sessions", "upload_ticket_expires_at")
    op.drop_column("kyc_sessions", "id_upload_ticket_hash")
    op.drop_column("kyc_sessions", "selfie_upload_ticket_hash")
    op.drop_column("kyc_sessions", "id_expected_content_type")
    op.drop_column("kyc_sessions", "selfie_expected_content_type")
    op.drop_column("kyc_sessions", "id_expected_size_bytes")
    op.drop_column("kyc_sessions", "selfie_expected_size_bytes")
