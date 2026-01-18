"""Enum definitions for KYC Sentinel models.

Centralizes all enum types to prevent magic strings and enable
type checking. Uses str mixin for JSON serialization compatibility.
"""

from enum import Enum


class SessionStatus(str, Enum):
    """Session processing status.
    
    State machine: pending -> processing -> completed|failed
    """
    PENDING = "pending"
    PROCESSING = "processing"
    COMPLETED = "completed"
    FAILED = "failed"


class AttackFamily(str, Enum):
    """Attack type categories for synthetic sessions.
    
    Used to classify the type of fraud attack being simulated.
    """
    REPLAY = "replay"           # Screen replay attack
    INJECTION = "injection"      # Camera bypass injection
    FACE_SWAP = "face_swap"     # Deepfake/face swap
    DOC_TAMPER = "doc_tamper"   # Document tampering
    BENIGN = "benign"           # No attack (genuine)


class AttackSeverity(str, Enum):
    """Attack intensity level.
    
    Controls artifact visibility in synthetic attack generation.
    """
    LOW = "low"
    MEDIUM = "medium"
    HIGH = "high"


class Decision(str, Enum):
    """Risk assessment decision.
    
    Final decision based on risk score and reason codes.
    """
    PASS = "pass"     # Low risk, approve
    REVIEW = "review" # Medium risk, manual review
    FAIL = "fail"     # High risk, reject


class ReasonSeverity(str, Enum):
    """Severity level for reason codes."""
    INFO = "info"
    WARN = "warn"
    HIGH = "high"
