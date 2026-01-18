"""Shared helper functions for processing backends.

Consolidates common utilities like reason code building and image decoding
that were previously duplicated across LocalBackend and ModalBackend.
"""

import logging
from typing import TYPE_CHECKING
from uuid import UUID

import cv2
import numpy as np

if TYPE_CHECKING:
    from app.models.reason import KYCReason

logger = logging.getLogger(__name__)


def decode_image(img_bytes: bytes, max_dim: int = 2048) -> np.ndarray | None:
    """Decode image bytes to numpy array (BGR format) and resize if too large.
    
    Args:
        img_bytes: Raw image bytes
        max_dim: Maximum dimension (width or height). Images larger than this 
                 will be resized proportionally to fit within this limit.
                 Default 2048 balances quality with memory constraints.
    
    Returns:
        BGR numpy array, or None if decode fails
    """
    nparr = np.frombuffer(img_bytes, np.uint8)
    img = cv2.imdecode(nparr, cv2.IMREAD_COLOR)
    
    if img is None:
        return img
        
    # Resize if image is too large (prevents memory issues and speeds up processing)
    h, w = img.shape[:2]
    if max(h, w) > max_dim:
        scale = max_dim / max(h, w)
        new_w = int(w * scale)
        new_h = int(h * scale)
        img = cv2.resize(img, (new_w, new_h), interpolation=cv2.INTER_AREA)
    
    return img


def is_video(asset_key: str) -> bool:
    """Check if asset is a video based on extension."""
    video_extensions = {'.mp4', '.mov', '.webm', '.avi', '.mkv'}
    return any(asset_key.lower().endswith(ext) for ext in video_extensions)


def build_reasons(
    session_id: UUID,
    codes: list[str],
    evidence: dict,
    message_overrides: dict[str, str] | None = None,
) -> list["KYCReason"]:
    """Build KYCReason objects from reason codes.
    
    Consolidates the duplicated reason code assembly logic that was
    previously copy-pasted 6 times across LocalBackend and ModalBackend.
    
    Args:
        session_id: UUID of the session
        codes: List of reason code strings (e.g., "FACE_MISMATCH")
        evidence: Dictionary of evidence data
        message_overrides: Optional dict of code -> formatted message
    
    Returns:
        List of KYCReason model instances
    """
    from app.models.reason import KYCReason
    from app.services.reason_codes import ReasonCode, REASON_MESSAGES, get_reason_severity
    
    reasons = []
    message_overrides = message_overrides or {}
    
    for code in codes:
        reason_code = getattr(ReasonCode, code, None)
        if reason_code:
            # Use override if provided, otherwise default message
            message = message_overrides.get(code) or REASON_MESSAGES.get(reason_code, code)
            
            reasons.append(
                KYCReason(
                    session_id=session_id,
                    code=code,
                    severity=get_reason_severity(reason_code),
                    message=message,
                    evidence=evidence,
                )
            )
    
    return reasons


def format_face_message(template: str, evidence: dict) -> str:
    """Format a face analysis message template with evidence."""
    if "{similarity" in template and evidence.get("face_similarity"):
        return template.format(similarity=evidence["face_similarity"])
    return template


def format_doc_message(template: str, evidence: dict) -> str:
    """Format a document analysis message template with evidence."""
    if "{confidence" in template and evidence.get("avg_ocr_confidence"):
        return template.format(confidence=evidence["avg_ocr_confidence"])
    return template
