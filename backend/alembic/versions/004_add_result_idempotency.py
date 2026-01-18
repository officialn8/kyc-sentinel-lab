"""Add unique constraint on kyc_results.session_id for idempotency

Revision ID: 004_add_result_idempotency
Revises: 003_fraud_detection
Create Date: 2024-01-18 00:00:00.000000

"""
from typing import Sequence, Union

from alembic import op


# revision identifiers, used by Alembic.
revision: str = '004_add_result_idempotency'
down_revision: Union[str, None] = '003_fraud_detection'
branch_labels: Union[str, Sequence[str], None] = None
depends_on: Union[str, Sequence[str], None] = None


def upgrade() -> None:
    # Add unique constraint on session_id to prevent duplicate results
    # This ensures idempotency - a session can only have one result
    op.create_index(
        'ix_kyc_results_session_id_unique',
        'kyc_results',
        ['session_id'],
        unique=True,
    )


def downgrade() -> None:
    op.drop_index('ix_kyc_results_session_id_unique', table_name='kyc_results')
