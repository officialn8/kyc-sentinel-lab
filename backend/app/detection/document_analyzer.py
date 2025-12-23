"""
Document OCR and anomaly detection using PaddleOCR.

Responsibilities:
- Extract text from ID document
- Detect layout/structure anomalies
- Flag OCR confidence issues
- Identify font inconsistencies
- Check for text alignment problems
"""

import numpy as np
from dataclasses import dataclass, field
from typing import Optional
from statistics import mean, stdev

import cv2


@dataclass
class TextBox:
    """OCR text detection result."""

    text: str
    confidence: float
    bbox: list[list[int]]  # 4 corner points [[x1,y1], [x2,y2], [x3,y3], [x4,y4]]
    center: tuple[int, int]


@dataclass
class OCRResult:
    """Result from OCR text extraction (backward compatible)."""

    text: str
    confidence: float
    bbox: tuple[int, int, int, int]  # x1, y1, x2, y2


@dataclass
class DocumentAnalysisResult:
    """Result of document analysis."""

    text_boxes: list[TextBox]
    full_text: str
    avg_confidence: float
    anomalies: list[dict]
    reason_codes: list[str]
    evidence: dict
    doc_score: float  # 0-1, higher = more suspicious
    
    # Backward compatible fields
    detected: bool = True
    ocr_results: list[OCRResult] = field(default_factory=list)
    overall_confidence: float = 0.0
    template_match_score: float = 0.0
    font_consistency_score: float = 0.0
    alignment_score: float = 0.0


class DocumentAnalyzer:
    """Document analysis using PaddleOCR.
    
    Performs OCR text extraction and anomaly detection to identify
    potentially tampered or fraudulent documents.
    """

    def __init__(
        self,
        lang: str = "en",
        use_gpu: bool = True,
        min_confidence_threshold: float = 0.7,
        low_confidence_threshold: float = 0.5,
    ) -> None:
        """
        Initialize PaddleOCR.
        
        Args:
            lang: Language code for OCR
            use_gpu: Whether to use GPU acceleration
            min_confidence_threshold: Minimum confidence to accept text
            low_confidence_threshold: Threshold to flag low confidence
        """
        self.lang = lang
        self.use_gpu = use_gpu
        self.min_confidence_threshold = min_confidence_threshold
        self.low_confidence_threshold = low_confidence_threshold
        self._ocr = None

    def _ensure_model(self) -> None:
        """Ensure PaddleOCR is loaded (lazy initialization)."""
        if self._ocr is None:
            from paddleocr import PaddleOCR
            
            # PaddleOCR v2.8+ uses new API parameters
            self._ocr = PaddleOCR(
                use_textline_orientation=True,
                lang=self.lang,
                device="gpu" if self.use_gpu else "cpu",
            )

    def analyze(self, image: np.ndarray) -> DocumentAnalysisResult:
        """
        Analyze document image for text extraction and anomalies.
        
        Args:
            image: BGR image array
        
        Returns:
            DocumentAnalysisResult with extracted text and anomaly flags
        """
        reason_codes: list[str] = []
        evidence: dict = {}
        anomalies: list[dict] = []

        self._ensure_model()
        
        # Run OCR using the new predict() API (PaddleOCR v2.8+)
        result = self._ocr.predict(image)

        # Parse OCR results - PaddleOCR v2.8+ returns [OCRResult] with rec_texts, rec_scores, rec_polys
        text_boxes: list[TextBox] = []
        ocr_results: list[OCRResult] = []
        confidences: list[float] = []

        if not result or len(result) == 0:
            reason_codes.append("DOC_OCR_LOW_CONFIDENCE")
            evidence["ocr_result"] = "no_text_detected"
            return DocumentAnalysisResult(
                text_boxes=[],
                full_text="",
                avg_confidence=0.0,
                anomalies=[],
                reason_codes=reason_codes,
                evidence=evidence,
                doc_score=1.0,
                detected=False,
                ocr_results=[],
                overall_confidence=0.0,
            )

        # Extract from new PaddleOCR v2.8+ format
        ocr_data = result[0]
        rec_texts = ocr_data.get('rec_texts', []) if hasattr(ocr_data, 'get') else getattr(ocr_data, 'rec_texts', [])
        rec_scores = ocr_data.get('rec_scores', []) if hasattr(ocr_data, 'get') else getattr(ocr_data, 'rec_scores', [])
        rec_polys = ocr_data.get('rec_polys', []) if hasattr(ocr_data, 'get') else getattr(ocr_data, 'rec_polys', [])
        
        # Fallback to rec_boxes if rec_polys is empty
        if not rec_polys:
            rec_polys = ocr_data.get('rec_boxes', []) if hasattr(ocr_data, 'get') else getattr(ocr_data, 'rec_boxes', [])

        # Check if we have any text detected
        if not rec_texts:
            reason_codes.append("DOC_OCR_LOW_CONFIDENCE")
            evidence["ocr_result"] = "no_text_detected"
            return DocumentAnalysisResult(
                text_boxes=[],
                full_text="",
                avg_confidence=0.0,
                anomalies=[],
                reason_codes=reason_codes,
                evidence=evidence,
                doc_score=1.0,
                detected=False,
                ocr_results=[],
                overall_confidence=0.0,
            )

        for i, text in enumerate(rec_texts):
            confidence = float(rec_scores[i]) if i < len(rec_scores) else 0.0
            bbox = rec_polys[i] if i < len(rec_polys) else [[0, 0], [0, 0], [0, 0], [0, 0]]
            
            # Convert bbox to list of lists if it's a numpy array or other format
            if hasattr(bbox, 'tolist'):
                bbox = bbox.tolist()
            
            # Ensure bbox is in correct format [[x1,y1], [x2,y2], [x3,y3], [x4,y4]]
            if len(bbox) == 4 and not isinstance(bbox[0], (list, tuple)):
                # Format is [x1, y1, x2, y2] - convert to corner points
                x1, y1, x2, y2 = bbox
                bbox = [[x1, y1], [x2, y1], [x2, y2], [x1, y2]]
            
            center = self._get_center(bbox) if bbox else (0, 0)

            text_boxes.append(
                TextBox(
                    text=text,
                    confidence=confidence,
                    bbox=bbox,
                    center=center,
                )
            )
            
            # Also create backward-compatible OCRResult
            if bbox:
                xs = [p[0] for p in bbox]
                ys = [p[1] for p in bbox]
                ocr_results.append(
                    OCRResult(
                        text=text,
                        confidence=confidence,
                        bbox=(int(min(xs)), int(min(ys)), int(max(xs)), int(max(ys))),
                    )
                )
            
            confidences.append(confidence)

        avg_confidence = mean(confidences) if confidences else 0.0
        evidence["avg_ocr_confidence"] = avg_confidence
        evidence["text_box_count"] = len(text_boxes)

        # Check for low confidence regions
        low_conf_boxes = [
            tb for tb in text_boxes if tb.confidence < self.low_confidence_threshold
        ]
        if low_conf_boxes:
            reason_codes.append("DOC_OCR_LOW_CONFIDENCE")
            evidence["low_confidence_regions"] = len(low_conf_boxes)
            anomalies.append(
                {
                    "type": "low_confidence",
                    "boxes": [
                        {"text": tb.text, "confidence": tb.confidence}
                        for tb in low_conf_boxes
                    ],
                }
            )

        # Check for font/spacing inconsistencies
        font_anomaly = self._check_font_consistency(text_boxes, image)
        font_consistency_score = 1.0
        if font_anomaly:
            reason_codes.append("DOC_FONT_INCONSISTENT")
            anomalies.append(font_anomaly)
            evidence["font_inconsistency"] = True
            font_consistency_score = 1.0 - font_anomaly.get("height_cv", 0.5)

        # Check text alignment
        alignment_anomaly = self._check_alignment(text_boxes)
        alignment_score = 1.0
        if alignment_anomaly:
            reason_codes.append("DOC_TEXT_MISALIGNED")
            anomalies.append(alignment_anomaly)
            evidence["alignment_issue"] = True
            alignment_score = 0.5

        # Check for edge artifacts (signs of tampering)
        edge_anomaly = self._check_edge_artifacts(image)
        if edge_anomaly:
            reason_codes.append("DOC_EDGE_ARTIFACTS")
            anomalies.append(edge_anomaly)
            evidence["edge_artifacts"] = True

        # Compute overall document suspicion score
        doc_score = self._compute_doc_score(
            avg_confidence=avg_confidence,
            anomaly_count=len(anomalies),
            low_conf_ratio=len(low_conf_boxes) / max(len(text_boxes), 1),
        )

        return DocumentAnalysisResult(
            text_boxes=text_boxes,
            full_text=" ".join(tb.text for tb in text_boxes),
            avg_confidence=avg_confidence,
            anomalies=anomalies,
            reason_codes=reason_codes,
            evidence=evidence,
            doc_score=doc_score,
            detected=True,
            ocr_results=ocr_results,
            overall_confidence=avg_confidence,
            template_match_score=0.85,  # Placeholder - would need template matching
            font_consistency_score=font_consistency_score,
            alignment_score=alignment_score,
        )

    def analyze_document(self, image: np.ndarray) -> DocumentAnalysisResult:
        """Alias for analyze() for backward compatibility."""
        return self.analyze(image)

    def extract_text(self, image: np.ndarray) -> list[OCRResult]:
        """
        Extract text from document image.
        
        Args:
            image: Input image as numpy array
            
        Returns:
            List of OCR results with text and confidence
        """
        result = self.analyze(image)
        return result.ocr_results

    def _get_center(self, bbox: list) -> tuple[int, int]:
        """Get center point of bounding box."""
        xs = [p[0] for p in bbox]
        ys = [p[1] for p in bbox]
        return (int(mean(xs)), int(mean(ys)))

    def _check_font_consistency(
        self, text_boxes: list[TextBox], image: np.ndarray
    ) -> Optional[dict]:
        """
        Check for font size/style inconsistencies that might indicate tampering.
        Uses bounding box height variance as proxy for font size consistency.
        """
        if len(text_boxes) < 3:
            return None

        heights = []
        for tb in text_boxes:
            ys = [p[1] for p in tb.bbox]
            height = max(ys) - min(ys)
            heights.append(height)

        if len(heights) < 2:
            return None

        height_std = stdev(heights)
        height_mean = mean(heights)
        cv = height_std / height_mean if height_mean > 0 else 0

        # High coefficient of variation suggests inconsistent fonts
        if cv > 0.5:
            return {
                "type": "font_inconsistency",
                "height_cv": cv,
                "description": "Significant variation in text heights detected",
            }

        return None

    def _check_alignment(self, text_boxes: list[TextBox]) -> Optional[dict]:
        """
        Check for suspicious text alignment issues.
        Real IDs have consistent alignment; tampered ones often don't.
        """
        if len(text_boxes) < 3:
            return None

        # Check horizontal alignment (left edges should align for same-column text)
        left_edges = [min(p[0] for p in tb.bbox) for tb in text_boxes]

        # Cluster left edges and check for outliers
        sorted_edges = sorted(left_edges)
        edge_diffs = [
            sorted_edges[i + 1] - sorted_edges[i]
            for i in range(len(sorted_edges) - 1)
        ]

        if edge_diffs:
            max_gap = max(edge_diffs)
            median_gap = sorted(edge_diffs)[len(edge_diffs) // 2]

            # Large gap relative to median suggests misalignment
            if median_gap > 0 and max_gap > median_gap * 5:
                return {
                    "type": "alignment_anomaly",
                    "max_gap": max_gap,
                    "median_gap": median_gap,
                    "description": "Unusual text alignment pattern detected",
                }

        return None

    def _check_edge_artifacts(self, image: np.ndarray) -> Optional[dict]:
        """
        Check for edge artifacts that might indicate cut-paste tampering.
        Uses edge detection to find unnatural boundaries.
        """
        gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

        # Detect edges
        edges = cv2.Canny(gray, 50, 150)

        # Look for rectangular edge patterns (signs of pasted regions)
        contours, _ = cv2.findContours(
            edges, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE
        )

        suspicious_contours = 0
        for contour in contours:
            # Approximate contour to polygon
            epsilon = 0.02 * cv2.arcLength(contour, True)
            approx = cv2.approxPolyDP(contour, epsilon, True)

            # Rectangular shapes with specific size ranges are suspicious
            if len(approx) == 4:
                area = cv2.contourArea(contour)
                img_area = image.shape[0] * image.shape[1]

                # Small-to-medium rectangles (5-40% of image) might be pasted
                if 0.05 < area / img_area < 0.4:
                    suspicious_contours += 1

        if suspicious_contours >= 2:
            return {
                "type": "edge_artifacts",
                "suspicious_regions": suspicious_contours,
                "description": "Multiple rectangular edge patterns detected",
            }

        return None

    def _compute_doc_score(
        self,
        avg_confidence: float,
        anomaly_count: int,
        low_conf_ratio: float,
    ) -> float:
        """
        Compute overall document suspicion score (0-1, higher = more suspicious).
        """
        # Start with confidence-based score
        confidence_score = 1.0 - avg_confidence

        # Add anomaly penalty
        anomaly_penalty = min(anomaly_count * 0.15, 0.45)

        # Add low confidence ratio penalty
        low_conf_penalty = low_conf_ratio * 0.3

        score = confidence_score * 0.4 + anomaly_penalty + low_conf_penalty
        return min(max(score, 0.0), 1.0)

    def check_font_consistency(self, image: np.ndarray) -> float:
        """Check for font consistency in document.
        
        Returns:
            Score from 0-1 where 1 = perfectly consistent
        """
        result = self.analyze(image)
        return result.font_consistency_score

    def check_alignment(self, image: np.ndarray) -> float:
        """Check text alignment against expected template.
        
        Returns:
            Score from 0-1 where 1 = perfectly aligned
        """
        result = self.analyze(image)
        return result.alignment_score


# Singleton
_doc_analyzer: Optional[DocumentAnalyzer] = None


def get_document_analyzer() -> DocumentAnalyzer:
    """Get cached document analyzer instance."""
    global _doc_analyzer
    if _doc_analyzer is None:
        _doc_analyzer = DocumentAnalyzer()
    return _doc_analyzer
