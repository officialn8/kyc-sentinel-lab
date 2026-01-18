"""Add GIN trigram indexes for faster text search

Revision ID: 005_search_performance
Revises: 004_add_result_idempotency
Create Date: 2024-01-18 00:00:00.000000

PERFORMANCE: The existing text search uses ILIKE on session ID, attack_family,
status, and source columns. Without proper indexes, this results in full table
scans as noted in code review.

This migration adds:
1. pg_trgm extension for trigram-based text search
2. GIN indexes on frequently searched columns
3. Partial index for common status queries

NOTE: pg_trgm must be enabled by a superuser. If running on a managed database
(Railway, Supabase), you may need to enable this extension manually or contact
support.

Alternative approaches if pg_trgm is unavailable:
- Use prefix matching (session_id LIKE 'abc%') with B-tree indexes
- Use PostgreSQL full-text search (to_tsvector/to_tsquery)
- Cache session metadata for faster searches
"""
from typing import Sequence, Union

from alembic import op
import sqlalchemy as sa


# revision identifiers, used by Alembic.
revision: str = '005_search_performance'
down_revision: Union[str, None] = '004_add_result_idempotency'
branch_labels: Union[str, Sequence[str], None] = None
depends_on: Union[str, Sequence[str], None] = None


def upgrade() -> None:
    # Enable pg_trgm extension for trigram-based text search
    # NOTE: This requires superuser privileges. On managed databases,
    # you may need to enable this manually via their dashboard.
    op.execute("CREATE EXTENSION IF NOT EXISTS pg_trgm")

    # Add computed column for session ID text search
    # (UUIDs can't be directly indexed with pg_trgm)
    op.execute("""
        CREATE INDEX IF NOT EXISTS ix_kyc_sessions_id_text_gin
        ON kyc_sessions USING gin (CAST(id AS TEXT) gin_trgm_ops)
    """)

    # GIN index on attack_family for pattern matching
    op.execute("""
        CREATE INDEX IF NOT EXISTS ix_kyc_sessions_attack_family_gin
        ON kyc_sessions USING gin (attack_family gin_trgm_ops)
        WHERE attack_family IS NOT NULL
    """)

    # GIN index on status for pattern matching
    op.execute("""
        CREATE INDEX IF NOT EXISTS ix_kyc_sessions_status_gin
        ON kyc_sessions USING gin (status gin_trgm_ops)
    """)

    # GIN index on source for pattern matching
    op.execute("""
        CREATE INDEX IF NOT EXISTS ix_kyc_sessions_source_gin
        ON kyc_sessions USING gin (source gin_trgm_ops)
    """)

    # Add partial index for common status queries (faster than full GIN)
    op.execute("""
        CREATE INDEX IF NOT EXISTS ix_kyc_sessions_status_pending
        ON kyc_sessions (status) WHERE status = 'pending'
    """)
    op.execute("""
        CREATE INDEX IF NOT EXISTS ix_kyc_sessions_status_processing
        ON kyc_sessions (status) WHERE status = 'processing'
    """)


def downgrade() -> None:
    op.execute("DROP INDEX IF EXISTS ix_kyc_sessions_status_processing")
    op.execute("DROP INDEX IF EXISTS ix_kyc_sessions_status_pending")
    op.execute("DROP INDEX IF EXISTS ix_kyc_sessions_source_gin")
    op.execute("DROP INDEX IF EXISTS ix_kyc_sessions_status_gin")
    op.execute("DROP INDEX IF EXISTS ix_kyc_sessions_attack_family_gin")
    op.execute("DROP INDEX IF EXISTS ix_kyc_sessions_id_text_gin")
    # Note: We don't drop pg_trgm as other things may depend on it
