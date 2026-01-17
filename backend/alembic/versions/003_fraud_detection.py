"""Add fraud detection columns

Revision ID: 003_fraud_detection
Revises: 002_add_job_queue
Create Date: 2024-01-15 00:00:00.000000

"""
from typing import Sequence, Union

from alembic import op
import sqlalchemy as sa


# revision identifiers, used by Alembic.
revision: str = '003_fraud_detection'
down_revision: Union[str, None] = '002_add_job_queue'
branch_labels: Union[str, Sequence[str], None] = None
depends_on: Union[str, Sequence[str], None] = None


def upgrade() -> None:
    # Add device_fingerprint column for velocity tracking
    op.add_column(
        'kyc_sessions',
        sa.Column('device_fingerprint', sa.String(256), nullable=True)
    )
    
    # Add client_ip column for velocity tracking (IPv6 max = 45 chars)
    op.add_column(
        'kyc_sessions',
        sa.Column('client_ip', sa.String(45), nullable=True)
    )
    
    # Add device_timezone column for geo anomaly detection
    op.add_column(
        'kyc_sessions',
        sa.Column('device_timezone', sa.String(50), nullable=True)
    )
    
    # Create indexes for velocity queries
    op.create_index(
        'ix_kyc_sessions_device_fingerprint',
        'kyc_sessions',
        ['device_fingerprint']
    )
    op.create_index(
        'ix_kyc_sessions_client_ip',
        'kyc_sessions',
        ['client_ip']
    )
    
    # Add index on created_at for efficient time-range queries
    op.create_index(
        'ix_kyc_sessions_created_at',
        'kyc_sessions',
        ['created_at']
    )


def downgrade() -> None:
    # Drop indexes first
    op.drop_index('ix_kyc_sessions_created_at', table_name='kyc_sessions')
    op.drop_index('ix_kyc_sessions_client_ip', table_name='kyc_sessions')
    op.drop_index('ix_kyc_sessions_device_fingerprint', table_name='kyc_sessions')
    
    # Drop columns
    op.drop_column('kyc_sessions', 'device_timezone')
    op.drop_column('kyc_sessions', 'client_ip')
    op.drop_column('kyc_sessions', 'device_fingerprint')
