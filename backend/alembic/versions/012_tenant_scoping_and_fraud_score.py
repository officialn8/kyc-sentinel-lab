"""Add tenant scoping fields and fraud score persistence.

Revision ID: 012_tenant_scoping_fraud
Revises: 011_upload_ticket_guards
Create Date: 2026-02-10
"""

from typing import Sequence, Union

from alembic import op
import sqlalchemy as sa


# revision identifiers, used by Alembic.
revision: str = "012_tenant_scoping_fraud"
down_revision: Union[str, None] = "011_upload_ticket_guards"
branch_labels: Union[str, Sequence[str], None] = None
depends_on: Union[str, Sequence[str], None] = None


def upgrade() -> None:
    op.add_column(
        "kyc_sessions",
        sa.Column("tenant_id", sa.String(length=128), nullable=True),
    )
    op.add_column(
        "kyc_results",
        sa.Column("fraud_score", sa.Float(), nullable=True),
    )

    op.create_index(
        "ix_kyc_sessions_tenant_id",
        "kyc_sessions",
        ["tenant_id"],
    )
    op.create_index(
        "ix_kyc_sessions_tenant_created_at",
        "kyc_sessions",
        ["tenant_id", "created_at"],
    )
    op.create_index(
        "ix_kyc_sessions_tenant_device_created_at",
        "kyc_sessions",
        ["tenant_id", "device_fingerprint", "created_at"],
        postgresql_where=sa.text("device_fingerprint IS NOT NULL"),
    )
    op.create_index(
        "ix_kyc_sessions_tenant_ip_created_at",
        "kyc_sessions",
        ["tenant_id", "client_ip", "created_at"],
        postgresql_where=sa.text("client_ip IS NOT NULL"),
    )


def downgrade() -> None:
    op.drop_index("ix_kyc_sessions_tenant_ip_created_at", table_name="kyc_sessions")
    op.drop_index("ix_kyc_sessions_tenant_device_created_at", table_name="kyc_sessions")
    op.drop_index("ix_kyc_sessions_tenant_created_at", table_name="kyc_sessions")
    op.drop_index("ix_kyc_sessions_tenant_id", table_name="kyc_sessions")

    op.drop_column("kyc_results", "fraud_score")
    op.drop_column("kyc_sessions", "tenant_id")
