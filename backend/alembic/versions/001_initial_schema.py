"""Initial schema with KYC models

Revision ID: 001_initial
Revises: 
Create Date: 2024-01-01 00:00:00.000000

"""
from typing import Sequence, Union

from alembic import op
import sqlalchemy as sa
from pgvector.sqlalchemy import Vector
from sqlalchemy.dialects import postgresql

# revision identifiers, used by Alembic.
revision: str = '001_initial'
down_revision: Union[str, None] = None
branch_labels: Union[str, Sequence[str], None] = None
depends_on: Union[str, Sequence[str], None] = None


def upgrade() -> None:
    # Enable pgvector extension
    op.execute('CREATE EXTENSION IF NOT EXISTS vector')
    
    # Create kyc_sessions table
    op.create_table(
        'kyc_sessions',
        sa.Column('id', sa.UUID(), nullable=False),
        sa.Column('created_at', sa.DateTime(timezone=True), server_default=sa.text('now()'), nullable=False),
        sa.Column('updated_at', sa.DateTime(timezone=True), server_default=sa.text('now()'), nullable=False),
        sa.Column('status', sa.String(length=20), nullable=False),
        sa.Column('source', sa.String(length=20), nullable=False),
        sa.Column('attack_family', sa.String(length=50), nullable=True),
        sa.Column('attack_severity', sa.String(length=20), nullable=True),
        sa.Column('device_os', sa.String(length=50), nullable=True),
        sa.Column('device_model', sa.String(length=100), nullable=True),
        sa.Column('ip_country', sa.String(length=10), nullable=True),
        sa.Column('capture_fps', sa.Float(), nullable=True),
        sa.Column('resolution', sa.String(length=20), nullable=True),
        sa.Column('selfie_asset_key', sa.String(length=500), nullable=True),
        sa.Column('id_asset_key', sa.String(length=500), nullable=True),
        sa.Column('face_embedding', Vector(512), nullable=True),
        sa.PrimaryKeyConstraint('id')
    )
    
    # Create index for face embedding similarity search
    op.create_index(
        'ix_kyc_sessions_face_embedding',
        'kyc_sessions',
        ['face_embedding'],
        postgresql_using='ivfflat',
        postgresql_with={'lists': 100},
        postgresql_ops={'face_embedding': 'vector_cosine_ops'}
    )
    
    # Create kyc_results table
    op.create_table(
        'kyc_results',
        sa.Column('id', sa.UUID(), nullable=False),
        sa.Column('session_id', sa.UUID(), nullable=False),
        sa.Column('created_at', sa.DateTime(timezone=True), server_default=sa.text('now()'), nullable=False),
        sa.Column('risk_score', sa.Integer(), nullable=False),
        sa.Column('decision', sa.String(length=20), nullable=False),
        sa.Column('face_similarity', sa.Float(), nullable=True),
        sa.Column('pad_score', sa.Float(), nullable=True),
        sa.Column('doc_score', sa.Float(), nullable=True),
        sa.Column('model_version', sa.String(length=20), nullable=False),
        sa.Column('rules_version', sa.String(length=20), nullable=False),
        sa.ForeignKeyConstraint(['session_id'], ['kyc_sessions.id'], ondelete='CASCADE'),
        sa.PrimaryKeyConstraint('id')
    )
    
    # Create kyc_reasons table
    op.create_table(
        'kyc_reasons',
        sa.Column('id', sa.UUID(), nullable=False),
        sa.Column('session_id', sa.UUID(), nullable=False),
        sa.Column('code', sa.String(length=50), nullable=False),
        sa.Column('severity', sa.String(length=20), nullable=False),
        sa.Column('message', sa.Text(), nullable=False),
        sa.Column('evidence', postgresql.JSONB(astext_type=sa.Text()), nullable=False),
        sa.ForeignKeyConstraint(['session_id'], ['kyc_sessions.id'], ondelete='CASCADE'),
        sa.PrimaryKeyConstraint('id')
    )
    
    # Create kyc_frame_metrics table
    op.create_table(
        'kyc_frame_metrics',
        sa.Column('id', sa.UUID(), nullable=False),
        sa.Column('session_id', sa.UUID(), nullable=False),
        sa.Column('frame_idx', sa.Integer(), nullable=False),
        sa.Column('motion_entropy', sa.Float(), nullable=True),
        sa.Column('sharpness', sa.Float(), nullable=True),
        sa.Column('noise_score', sa.Float(), nullable=True),
        sa.Column('color_shift', sa.Float(), nullable=True),
        sa.Column('pad_flag', sa.Boolean(), nullable=False),
        sa.ForeignKeyConstraint(['session_id'], ['kyc_sessions.id'], ondelete='CASCADE'),
        sa.PrimaryKeyConstraint('id')
    )
    
    # Create indexes
    op.create_index('ix_kyc_sessions_status', 'kyc_sessions', ['status'])
    op.create_index('ix_kyc_sessions_source', 'kyc_sessions', ['source'])
    op.create_index('ix_kyc_sessions_attack_family', 'kyc_sessions', ['attack_family'])
    op.create_index('ix_kyc_results_session_id', 'kyc_results', ['session_id'])
    op.create_index('ix_kyc_results_decision', 'kyc_results', ['decision'])
    op.create_index('ix_kyc_reasons_session_id', 'kyc_reasons', ['session_id'])
    op.create_index('ix_kyc_reasons_code', 'kyc_reasons', ['code'])
    op.create_index('ix_kyc_frame_metrics_session_id', 'kyc_frame_metrics', ['session_id'])


def downgrade() -> None:
    op.drop_table('kyc_frame_metrics')
    op.drop_table('kyc_reasons')
    op.drop_table('kyc_results')
    op.drop_index('ix_kyc_sessions_face_embedding', table_name='kyc_sessions')
    op.drop_table('kyc_sessions')
    op.execute('DROP EXTENSION IF EXISTS vector')











