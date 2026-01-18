"""Add CHECK constraints for enum validation.

Revision ID: 007_enum_constraints
Revises: 006_ensure_pgvector
Create Date: 2026-01-18

Adds database-level CHECK constraints to enforce valid enum values:
- kyc_sessions.status: pending, processing, completed, failed
- kyc_sessions.attack_family: replay, injection, face_swap, doc_tamper, benign (nullable)
- kyc_sessions.attack_severity: low, medium, high (nullable)
- kyc_results.decision: pass, review, fail
"""

from alembic import op


# revision identifiers, used by Alembic.
revision = "007_enum_constraints"
down_revision = "006_ensure_pgvector"
branch_labels = None
depends_on = None


def upgrade() -> None:
    """Add CHECK constraints for enum fields."""
    
    # Session status constraint
    op.execute("""
        ALTER TABLE kyc_sessions 
        ADD CONSTRAINT check_session_status 
        CHECK (status IN ('pending', 'processing', 'completed', 'failed'))
    """)
    
    # Attack family constraint (NULL allowed)
    op.execute("""
        ALTER TABLE kyc_sessions 
        ADD CONSTRAINT check_attack_family 
        CHECK (attack_family IS NULL OR attack_family IN (
            'replay', 'injection', 'face_swap', 'doc_tamper', 'benign'
        ))
    """)
    
    # Attack severity constraint (NULL allowed)
    op.execute("""
        ALTER TABLE kyc_sessions 
        ADD CONSTRAINT check_attack_severity 
        CHECK (attack_severity IS NULL OR attack_severity IN ('low', 'medium', 'high'))
    """)
    
    # Decision constraint
    op.execute("""
        ALTER TABLE kyc_results 
        ADD CONSTRAINT check_decision 
        CHECK (decision IN ('pass', 'review', 'fail'))
    """)


def downgrade() -> None:
    """Remove CHECK constraints."""
    op.execute("ALTER TABLE kyc_results DROP CONSTRAINT IF EXISTS check_decision")
    op.execute("ALTER TABLE kyc_sessions DROP CONSTRAINT IF EXISTS check_attack_severity")
    op.execute("ALTER TABLE kyc_sessions DROP CONSTRAINT IF EXISTS check_attack_family")
    op.execute("ALTER TABLE kyc_sessions DROP CONSTRAINT IF EXISTS check_session_status")
