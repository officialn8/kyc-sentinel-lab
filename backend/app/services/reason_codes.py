"""Reason code taxonomy and messages."""

from enum import Enum


class ReasonCode(str, Enum):
    """Enumeration of all reason codes."""

    # Face matching
    FACE_MISMATCH = "FACE_MISMATCH"
    FACE_NOT_DETECTED_SELFIE = "FACE_NOT_DETECTED_SELFIE"
    FACE_NOT_DETECTED_ID = "FACE_NOT_DETECTED_ID"
    MULTIPLE_FACES_SELFIE = "MULTIPLE_FACES_SELFIE"
    # Phase 1: Face quality
    FACE_LOW_QUALITY_SELFIE = "FACE_LOW_QUALITY_SELFIE"
    FACE_LOW_QUALITY_ID = "FACE_LOW_QUALITY_ID"
    FACE_EXTREME_POSE = "FACE_EXTREME_POSE"
    # Phase 2: Age verification
    FACE_AGE_DISCREPANCY = "FACE_AGE_DISCREPANCY"

    # Presentation Attack Detection (PAD)
    PAD_SUSPECT_REPLAY = "PAD_SUSPECT_REPLAY"
    PAD_SCREEN_ARTIFACTS = "PAD_SCREEN_ARTIFACTS"
    PAD_LOW_MOTION_ENTROPY = "PAD_LOW_MOTION_ENTROPY"
    PAD_FRAME_STUTTER = "PAD_FRAME_STUTTER"
    PAD_INJECTION_ARTIFACTS = "PAD_INJECTION_ARTIFACTS"
    PAD_FACE_BOUNDARY_MISMATCH = "PAD_FACE_BOUNDARY_MISMATCH"
    # Phase 1: Liveness detection
    PAD_NO_BLINK_DETECTED = "PAD_NO_BLINK_DETECTED"
    PAD_PRINTED_TEXTURE = "PAD_PRINTED_TEXTURE"
    PAD_FLAT_DEPTH = "PAD_FLAT_DEPTH"
    # Phase 2: Virtual camera and audio liveness
    PAD_VIRTUAL_CAMERA = "PAD_VIRTUAL_CAMERA"
    PAD_LIP_SYNC_MISMATCH = "PAD_LIP_SYNC_MISMATCH"

    # Document analysis
    DOC_TEMPLATE_MISMATCH = "DOC_TEMPLATE_MISMATCH"
    DOC_OCR_LOW_CONFIDENCE = "DOC_OCR_LOW_CONFIDENCE"
    DOC_FONT_INCONSISTENT = "DOC_FONT_INCONSISTENT"
    DOC_TEXT_MISALIGNED = "DOC_TEXT_MISALIGNED"
    DOC_EDGE_ARTIFACTS = "DOC_EDGE_ARTIFACTS"
    DOC_METADATA_SUSPICIOUS = "DOC_METADATA_SUSPICIOUS"
    # Phase 1: MRZ validation
    DOC_MRZ_INVALID = "DOC_MRZ_INVALID"
    DOC_MRZ_MISMATCH = "DOC_MRZ_MISMATCH"
    DOC_BARCODE_MISMATCH = "DOC_BARCODE_MISMATCH"
    # Phase 2: ELA and copy-move detection
    DOC_ELA_MANIPULATION = "DOC_ELA_MANIPULATION"
    DOC_COPY_MOVE_DETECTED = "DOC_COPY_MOVE_DETECTED"

    # Metadata / device
    META_HIGH_RISK_DEVICE = "META_HIGH_RISK_DEVICE"
    META_SUSPICIOUS_TIMING = "META_SUSPICIOUS_TIMING"
    META_MULTIPLE_RETRIES = "META_MULTIPLE_RETRIES"

    # Phase 2: Fraud pattern detection
    FRAUD_HIGH_VELOCITY = "FRAUD_HIGH_VELOCITY"
    FRAUD_GEO_MISMATCH = "FRAUD_GEO_MISMATCH"
    FRAUD_SIMILAR_FACE = "FRAUD_SIMILAR_FACE"
    FRAUD_RING_LINKED = "FRAUD_RING_LINKED"


REASON_MESSAGES: dict[ReasonCode, str] = {
    # Face matching
    ReasonCode.FACE_MISMATCH: "Selfie face does not match ID photo (similarity: {similarity:.2f})",
    ReasonCode.FACE_NOT_DETECTED_SELFIE: "No face detected in selfie image",
    ReasonCode.FACE_NOT_DETECTED_ID: "No face detected in ID document",
    ReasonCode.MULTIPLE_FACES_SELFIE: "Multiple faces detected in selfie image",
    # Phase 1: Face quality
    ReasonCode.FACE_LOW_QUALITY_SELFIE: "Selfie image quality too low for reliable matching",
    ReasonCode.FACE_LOW_QUALITY_ID: "ID photo quality too low for reliable matching",
    ReasonCode.FACE_EXTREME_POSE: "Face pose angle too extreme for reliable matching (deviation: {pose_deviation:.1f}°)",
    # Phase 2: Age verification
    ReasonCode.FACE_AGE_DISCREPANCY: "Apparent age differs from ID date of birth by {difference_years} years (tolerance: {tolerance_years})",

    # PAD
    ReasonCode.PAD_SUSPECT_REPLAY: "Detected screen capture artifacts suggesting replay attack",
    ReasonCode.PAD_SCREEN_ARTIFACTS: "Detected moiré patterns or screen glare in image",
    ReasonCode.PAD_LOW_MOTION_ENTROPY: "Insufficient natural movement detected in video",
    ReasonCode.PAD_FRAME_STUTTER: "Inconsistent frame timing detected (possible video replay)",
    ReasonCode.PAD_INJECTION_ARTIFACTS: "Detected artifacts consistent with virtual camera injection",
    ReasonCode.PAD_FACE_BOUNDARY_MISMATCH: "Face boundary inconsistent with background (possible face swap)",
    # Phase 1: Liveness detection
    ReasonCode.PAD_NO_BLINK_DETECTED: "No natural eye blinks detected in video (expected at least {expected_blinks})",
    ReasonCode.PAD_PRINTED_TEXTURE: "Skin texture analysis suggests printed photo rather than live face",
    ReasonCode.PAD_FLAT_DEPTH: "Depth analysis indicates flat surface (possible photo/screen attack)",
    # Phase 2: Virtual camera and audio liveness
    ReasonCode.PAD_VIRTUAL_CAMERA: "Virtual camera software detected ({software_name})",
    ReasonCode.PAD_LIP_SYNC_MISMATCH: "Lip movement does not correlate with audio (correlation: {correlation:.2f})",

    # Document
    ReasonCode.DOC_TEMPLATE_MISMATCH: "Document layout does not match expected template",
    ReasonCode.DOC_OCR_LOW_CONFIDENCE: "Low confidence in document text extraction (confidence: {confidence:.2f})",
    ReasonCode.DOC_FONT_INCONSISTENT: "Inconsistent fonts detected in document text",
    ReasonCode.DOC_TEXT_MISALIGNED: "Text alignment does not match expected document format",
    ReasonCode.DOC_EDGE_ARTIFACTS: "Detected editing artifacts around document edges",
    ReasonCode.DOC_METADATA_SUSPICIOUS: "Document image metadata is suspicious or missing",
    # Phase 1: MRZ validation
    ReasonCode.DOC_MRZ_INVALID: "Machine Readable Zone (MRZ) check digits are invalid",
    ReasonCode.DOC_MRZ_MISMATCH: "MRZ data does not match OCR-extracted text fields",
    ReasonCode.DOC_BARCODE_MISMATCH: "Barcode data does not match document text",
    # Phase 2: ELA and copy-move detection
    ReasonCode.DOC_ELA_MANIPULATION: "Error Level Analysis suggests image manipulation (score: {score:.2f})",
    ReasonCode.DOC_COPY_MOVE_DETECTED: "Copy-move forgery detected in document image",

    # Metadata
    ReasonCode.META_HIGH_RISK_DEVICE: "Session originated from high-risk device profile",
    ReasonCode.META_SUSPICIOUS_TIMING: "Session timing patterns suggest automation",
    ReasonCode.META_MULTIPLE_RETRIES: "Multiple retry attempts detected in short period",

    # Phase 2: Fraud pattern detection
    ReasonCode.FRAUD_HIGH_VELOCITY: "High submission velocity detected ({submissions} in {period})",
    ReasonCode.FRAUD_GEO_MISMATCH: "Document country ({doc_country}) differs from device location ({device_country})",
    ReasonCode.FRAUD_SIMILAR_FACE: "Similar face found in {similar_count} other sessions",
    ReasonCode.FRAUD_RING_LINKED: "Session linked to potential fraud ring (matches: {match_count}, max similarity: {max_similarity})",
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
        # Phase 1: High severity liveness indicators
        ReasonCode.PAD_PRINTED_TEXTURE,
        ReasonCode.PAD_FLAT_DEPTH,
        # Phase 1: MRZ validation failures
        ReasonCode.DOC_MRZ_INVALID,
        # Phase 2: High severity new codes
        ReasonCode.PAD_VIRTUAL_CAMERA,  # Virtual camera is strong injection indicator
        ReasonCode.PAD_LIP_SYNC_MISMATCH,  # Audio mismatch is strong fraud indicator
        ReasonCode.DOC_COPY_MOVE_DETECTED,  # Copy-move forgery is definite tampering
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
        # Phase 1: Warning-level new codes
        ReasonCode.MULTIPLE_FACES_SELFIE,
        ReasonCode.FACE_LOW_QUALITY_SELFIE,
        ReasonCode.FACE_LOW_QUALITY_ID,
        ReasonCode.FACE_EXTREME_POSE,
        ReasonCode.PAD_NO_BLINK_DETECTED,
        ReasonCode.DOC_METADATA_SUSPICIOUS,
        ReasonCode.DOC_MRZ_MISMATCH,
        ReasonCode.DOC_BARCODE_MISMATCH,
        # Phase 2: Warning-level new codes
        ReasonCode.FACE_AGE_DISCREPANCY,  # Age mismatch needs manual review
        ReasonCode.DOC_ELA_MANIPULATION,  # ELA is heuristic, may have false positives
        ReasonCode.FRAUD_HIGH_VELOCITY,  # May be legitimate multiple attempts
        ReasonCode.FRAUD_GEO_MISMATCH,  # Could be VPN or travel
        ReasonCode.FRAUD_SIMILAR_FACE,  # Could be same person legitimate retry
        ReasonCode.FRAUD_RING_LINKED,  # Advisory signal for ring linkage
    }

    if code in high_severity:
        return "high"
    elif code in warn_severity:
        return "warn"
    else:
        return "info"











