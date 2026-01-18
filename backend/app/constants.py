"""Documented constants for the KYC Sentinel application.

Centralizes magic numbers with rationale explanations.
All constants should have a comment explaining why that value was chosen.
"""

# =============================================================================
# Image Processing
# =============================================================================

# Maximum image dimension for processing.
# InsightFace models trained on 640px faces, but we allow larger to preserve
# document text quality. 2048 balances quality with memory constraints.
MAX_IMAGE_DIMENSION = 2048

# JPEG encoding quality for face crops and frame storage.
# 95 provides excellent quality with reasonable file size.
JPEG_ENCODE_QUALITY = 95


# =============================================================================
# Face Analysis
# =============================================================================

# Face similarity threshold for match vs. mismatch.
# Conservative threshold based on InsightFace FMR/FNMR tradeoff at 1e-6 FMR.
# Higher values = stricter matching = more false rejections.
DEFAULT_FACE_SIMILARITY_THRESHOLD = 0.45


# =============================================================================
# Presigned URLs
# =============================================================================

# Presigned URL expiration for Modal backend workers (seconds).
# 5 minutes is enough for processing, short enough to limit exposure.
MODAL_PRESIGNED_URL_EXPIRATION = 300


# =============================================================================
# Concurrency Limits
# =============================================================================

# Maximum concurrent frame downloads to prevent memory exhaustion.
# 10 provides good parallelism without overwhelming storage or memory.
MAX_CONCURRENT_FRAME_DOWNLOADS = 10

# Maximum frames to extract from video for PAD analysis.
# 30 frames is enough for motion analysis, limits processing time.
MAX_FRAMES_FOR_PAD = 30


# =============================================================================
# Rate Limiting
# =============================================================================

# Maximum entries in in-memory rate limiter before emergency eviction.
# Prevents memory exhaustion in single-instance deployments.
RATE_LIMITER_MAX_BUCKETS = 100_000


# =============================================================================
# Risk Scoring
# =============================================================================

# Risk score thresholds - see scoring.py for profile-specific values.
# These are the DEFAULT profile values for reference.
DEFAULT_FAIL_THRESHOLD = 70
DEFAULT_REVIEW_THRESHOLD = 40

# Quality score threshold below which a penalty applies.
FACE_QUALITY_PENALTY_THRESHOLD = 0.5
