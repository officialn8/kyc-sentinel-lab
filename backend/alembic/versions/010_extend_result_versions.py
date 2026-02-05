"""Extend kyc_results version columns.

Revision ID: 010_extend_result_versions
Revises: 009_audit_and_retention
Create Date: 2026-02-05
"""

from typing import Sequence, Union

from alembic import op
import sqlalchemy as sa


# revision identifiers, used by Alembic.
revision: str = "010_extend_result_versions"
down_revision: Union[str, None] = "009_audit_and_retention"
branch_labels: Union[str, Sequence[str], None] = None
depends_on: Union[str, Sequence[str], None] = None


def upgrade() -> None:
    op.alter_column(
        "kyc_results",
        "model_version",
        existing_type=sa.String(length=20),
        type_=sa.String(length=128),
    )
    op.alter_column(
        "kyc_results",
        "rules_version",
        existing_type=sa.String(length=20),
        type_=sa.String(length=128),
    )


def downgrade() -> None:
    op.alter_column(
        "kyc_results",
        "rules_version",
        existing_type=sa.String(length=128),
        type_=sa.String(length=20),
    )
    op.alter_column(
        "kyc_results",
        "model_version",
        existing_type=sa.String(length=128),
        type_=sa.String(length=20),
    )
