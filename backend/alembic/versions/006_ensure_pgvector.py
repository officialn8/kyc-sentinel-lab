"""Ensure pgvector extension is available and properly configured.

Revision ID: 006_pgvector_check
Revises: 005_search_performance
Create Date: 2024-01-20 00:00:00.000000

This migration:
1. Verifies pgvector extension is available
2. Creates the extension if not exists
3. Validates vector column and index exist
4. Provides helpful error messages for common issues

Common Issues:
- Railway Postgres: Enable pgvector in the Railway dashboard
- Supabase: pgvector is enabled by default
- Self-hosted: apt install postgresql-16-pgvector (or appropriate version)
"""

from typing import Sequence, Union

from alembic import op
import sqlalchemy as sa
from sqlalchemy import text

# revision identifiers, used by Alembic.
revision: str = '006_pgvector_check'
down_revision: Union[str, None] = '005_search_performance'
branch_labels: Union[str, Sequence[str], None] = None
depends_on: Union[str, Sequence[str], None] = None


def upgrade() -> None:
    """Ensure pgvector extension is available and working."""
    connection = op.get_bind()

    # Step 1: Check if pgvector extension is available
    result = connection.execute(text("""
        SELECT EXISTS(
            SELECT 1 FROM pg_available_extensions WHERE name = 'vector'
        ) AS available
    """))
    row = result.fetchone()

    if not row or not row[0]:
        raise RuntimeError(
            "\n"
            "============================================================\n"
            "ERROR: pgvector extension is not available on this database.\n"
            "============================================================\n"
            "\n"
            "The KYC Sentinel Lab requires pgvector for face similarity search.\n"
            "\n"
            "How to fix:\n"
            "\n"
            "Railway Postgres:\n"
            "  1. Go to Railway Dashboard -> Your Postgres service\n"
            "  2. Click 'Data' tab -> 'Extensions'\n"
            "  3. Enable 'vector' extension\n"
            "\n"
            "Supabase:\n"
            "  pgvector should be enabled by default. If not:\n"
            "  1. Go to Database -> Extensions\n"
            "  2. Search for 'vector' and enable it\n"
            "\n"
            "Self-hosted PostgreSQL:\n"
            "  apt install postgresql-16-pgvector  # or appropriate version\n"
            "  # Then connect to database and run:\n"
            "  CREATE EXTENSION vector;\n"
            "\n"
            "After enabling, run migrations again.\n"
        )

    # Step 2: Create extension if not already created
    connection.execute(text("CREATE EXTENSION IF NOT EXISTS vector"))

    # Step 3: Verify extension is now installed
    result = connection.execute(text("""
        SELECT EXISTS(
            SELECT 1 FROM pg_extension WHERE extname = 'vector'
        ) AS installed
    """))
    row = result.fetchone()

    if not row or not row[0]:
        raise RuntimeError(
            "Failed to create pgvector extension. "
            "You may need superuser privileges or extension must be enabled in cloud dashboard."
        )

    # Step 4: Verify face_embedding column exists
    result = connection.execute(text("""
        SELECT EXISTS(
            SELECT 1 FROM information_schema.columns
            WHERE table_name = 'kyc_sessions'
            AND column_name = 'face_embedding'
        ) AS column_exists
    """))
    row = result.fetchone()

    if not row or not row[0]:
        # Column doesn't exist, add it
        connection.execute(text("""
            ALTER TABLE kyc_sessions
            ADD COLUMN IF NOT EXISTS face_embedding vector(512)
        """))
        print("Added face_embedding column to kyc_sessions")

    # Step 5: Ensure IVFFlat index exists for efficient similarity search
    result = connection.execute(text("""
        SELECT EXISTS(
            SELECT 1 FROM pg_indexes
            WHERE tablename = 'kyc_sessions'
            AND indexname = 'ix_kyc_sessions_face_embedding'
        ) AS index_exists
    """))
    row = result.fetchone()

    if not row or not row[0]:
        # Need to create index - requires at least some rows for IVFFlat
        # Check if we have any non-null embeddings
        result = connection.execute(text("""
            SELECT COUNT(*) FROM kyc_sessions WHERE face_embedding IS NOT NULL
        """))
        count = result.fetchone()[0]

        if count >= 100:
            # Enough rows for IVFFlat index
            connection.execute(text("""
                CREATE INDEX IF NOT EXISTS ix_kyc_sessions_face_embedding
                ON kyc_sessions USING ivfflat (face_embedding vector_cosine_ops)
                WITH (lists = 100)
            """))
            print("Created IVFFlat index on face_embedding")
        else:
            # Not enough rows, create basic HNSW index or skip
            # HNSW works better for small datasets
            try:
                connection.execute(text("""
                    CREATE INDEX IF NOT EXISTS ix_kyc_sessions_face_embedding
                    ON kyc_sessions USING hnsw (face_embedding vector_cosine_ops)
                """))
                print("Created HNSW index on face_embedding (small dataset)")
            except Exception:
                # HNSW may not be available in older pgvector versions
                # Fall back to no index for now (will still work, just slower)
                print(
                    "Note: Could not create vector index. "
                    "Similarity search will work but may be slower. "
                    "Index will be created once you have more data."
                )

    print("pgvector extension verified and configured successfully")


def downgrade() -> None:
    """No downgrade needed - this is a verification migration."""
    pass
