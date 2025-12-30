"""Reason code taxonomy and messages."""

from enum import Enum


class ReasonCode(str, Enum):
    """Enumeration of all reason codes."""

    # Face matching
    FACE_MISMATCH = "FACE_MISMATCH"
    FACE_NOT_DETECTED_SELFIE = "FACE_NOT_DETECTED_SELFIE"
    FACE_NOT_DETECTED_ID = "FACE_NOT_DETECTED_ID"
    MULTIPLE_FACES_SELFIE = "MULTIPLE_FACES_SELFIE"

    # Presentation Attack Detection (PAD)
    PAD_SUSPECT_REPLAY = "PAD_SUSPECT_REPLAY"
    PAD_SCREEN_ARTIFACTS = "PAD_SCREEN_ARTIFACTS"
    PAD_LOW_MOTION_ENTROPY = "PAD_LOW_MOTION_ENTROPY"
    PAD_FRAME_STUTTER = "PAD_FRAME_STUTTER"
    PAD_INJECTION_ARTIFACTS = "PAD_INJECTION_ARTIFACTS"
    PAD_FACE_BOUNDARY_MISMATCH = "PAD_FACE_BOUNDARY_MISMATCH"

    # Document analysis
    DOC_TEMPLATE_MISMATCH = "DOC_TEMPLATE_MISMATCH"
    DOC_OCR_LOW_CONFIDENCE = "DOC_OCR_LOW_CONFIDENCE"
    DOC_FONT_INCONSISTENT = "DOC_FONT_INCONSISTENT"
    DOC_TEXT_MISALIGNED = "DOC_TEXT_MISALIGNED"
    DOC_EDGE_ARTIFACTS = "DOC_EDGE_ARTIFACTS"
    DOC_METADATA_SUSPICIOUS = "DOC_METADATA_SUSPICIOUS"

    # Metadata / device
    META_HIGH_RISK_DEVICE = "META_HIGH_RISK_DEVICE"
    META_SUSPICIOUS_TIMING = "META_SUSPICIOUS_TIMING"
    META_MULTIPLE_RETRIES = "META_MULTIPLE_RETRIES"


REASON_MESSAGES: dict[ReasonCode, str] = {
    # Face matching
    ReasonCode.FACE_MISMATCH: "Selfie face does not match ID photo (similarity: {similarity:.2f})",
    ReasonCode.FACE_NOT_DETECTED_SELFIE: "No face detected in selfie image",
    ReasonCode.FACE_NOT_DETECTED_ID: "No face detected in ID document",
    ReasonCode.MULTIPLE_FACES_SELFIE: "Multiple faces detected in selfie image",

    # PAD
    ReasonCode.PAD_SUSPECT_REPLAY: "Detected screen capture artifacts suggesting replay attack",
    ReasonCode.PAD_SCREEN_ARTIFACTS: "Detected moiré patterns or screen glare in image",
    ReasonCode.PAD_LOW_MOTION_ENTROPY: "Insufficient natural movement detected in video",
    ReasonCode.PAD_FRAME_STUTTER: "Inconsistent frame timing detected (possible video replay)",
    ReasonCode.PAD_INJECTION_ARTIFACTS: "Detected artifacts consistent with virtual camera injection",
    ReasonCode.PAD_FACE_BOUNDARY_MISMATCH: "Face boundary inconsistent with background (possible face swap)",

    # Document
    ReasonCode.DOC_TEMPLATE_MISMATCH: "Document layout does not match expected template",
    ReasonCode.DOC_OCR_LOW_CONFIDENCE: "Low confidence in document text extraction (confidence: {confidence:.2f})",
    ReasonCode.DOC_FONT_INCONSISTENT: "Inconsistent fonts detected in document text",
    ReasonCode.DOC_TEXT_MISALIGNED: "Text alignment does not match expected document format",
    ReasonCode.DOC_EDGE_ARTIFACTS: "Detected editing artifacts around document edges",
    ReasonCode.DOC_METADATA_SUSPICIOUS: "Document image metadata is suspicious or missing",

    # Metadata
    ReasonCode.META_HIGH_RISK_DEVICE: "Session originated from high-risk device profile",
    ReasonCode.META_SUSPICIOUS_TIMING: "Session timing patterns suggest automation",
    ReasonCode.META_MULTIPLE_RETRIES: "Multiple retry attempts detected in short period",
}


SEVERITY_LEVELS = {
    "info": 0,
    "warn": 1,
    "high": 2,
}


def get_reason_severity(code: ReasonCode) -> str:
    """Get the default severity level for a reason code."""
    high_severity = {
        ReasonCode.FACE_MISMATCH,
        ReasonCode.FACE_NOT_DETECTED_SELFIE,
        ReasonCode.FACE_NOT_DETECTED_ID,
        ReasonCode.PAD_SUSPECT_REPLAY,
        ReasonCode.PAD_INJECTION_ARTIFACTS,
        ReasonCode.PAD_FACE_BOUNDARY_MISMATCH,
    }

    warn_severity = {
        ReasonCode.PAD_SCREEN_ARTIFACTS,
        ReasonCode.PAD_LOW_MOTION_ENTROPY,
        ReasonCode.PAD_FRAME_STUTTER,
        ReasonCode.DOC_TEMPLATE_MISMATCH,
        ReasonCode.DOC_OCR_LOW_CONFIDENCE,
        ReasonCode.DOC_FONT_INCONSISTENT,
        ReasonCode.DOC_TEXT_MISALIGNED,
        ReasonCode.DOC_EDGE_ARTIFACTS,
    }

    if code in high_severity:
        return "high"
    elif code in warn_severity:
        return "warn"
    else:
        return "info"











