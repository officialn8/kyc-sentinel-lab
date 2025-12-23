"""Detection module for KYC verification.

This module provides ML/CV-based detection capabilities:
- Face detection and matching using InsightFace
- Document OCR and anomaly detection using PaddleOCR
- Presentation Attack Detection (PAD) heuristics using OpenCV
- Video frame extraction
"""

from app.detection.face_analyzer import (
    FaceAnalyzer,
    FaceAnalysisResult,
    FaceDetection,
    get_face_analyzer,
)
from app.detection.document_analyzer import (
    DocumentAnalyzer,
    DocumentAnalysisResult,
    TextBox,
    OCRResult,
    get_document_analyzer,
)
from app.detection.pad_heuristics import (
    PADAnalyzer,
    PADResult,
    FrameMetrics,
    get_pad_analyzer,
)
from app.detection.frame_extractor import (
    FrameExtractor,
    ExtractionResult,
    VideoMetadata,
    get_frame_extractor,
)

__all__ = [
    # Face Analysis
    "FaceAnalyzer",
    "FaceAnalysisResult",
    "FaceDetection",
    "get_face_analyzer",
    # Document Analysis
    "DocumentAnalyzer",
    "DocumentAnalysisResult",
    "TextBox",
    "OCRResult",
    "get_document_analyzer",
    # PAD Heuristics
    "PADAnalyzer",
    "PADResult",
    "FrameMetrics",
    "get_pad_analyzer",
    # Frame Extraction
    "FrameExtractor",
    "ExtractionResult",
    "VideoMetadata",
    "get_frame_extractor",
]
