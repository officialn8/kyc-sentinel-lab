"""
Document OCR and anomaly detection using PaddleOCR.

Responsibilities:
- Extract text from ID document
- Detect layout/structure anomalies
- Flag OCR confidence issues
- Identify font inconsistencies
- Check for text alignment problems
- Validate MRZ (Machine Readable Zone) checksums
- Analyze EXIF metadata for tampering signs
"""

import numpy as np
import re
from dataclasses import dataclass, field
from typing import Optional
from statistics import mean, stdev
from io import BytesIO

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
class MRZValidation:
    """Result of MRZ (Machine Readable Zone) validation."""
    
    detected: bool
    mrz_type: Optional[str]  # TD1, TD2, TD3, or None
    mrz_lines: list[str]
    parsed_data: dict  # Extracted fields (name, dob, doc_number, etc.)
    checksums_valid: bool
    checksum_details: dict  # Per-field checksum results
    confidence: float


@dataclass
class EXIFAnalysis:
    """Result of EXIF metadata analysis."""
    
    has_exif: bool
    software_edited: bool  # True if editing software detected
    software_name: Optional[str]
    creation_date: Optional[str]
    modification_date: Optional[str]
    camera_make: Optional[str]
    camera_model: Optional[str]
    gps_present: bool
    suspicious_flags: list[str]
    is_suspicious: bool


@dataclass
class ELAResult:
    """Result of Error Level Analysis for forgery detection."""
    
    manipulation_detected: bool
    manipulation_score: float  # 0-1, higher = more likely manipulated
    high_error_regions: list[dict]  # Regions with abnormally high error levels
    avg_error_level: float
    max_error_level: float
    error_std: float  # Standard deviation of error levels
    suspicious_region_count: int
    confidence: float


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
    
    # MRZ and EXIF analysis
    mrz_validation: Optional[MRZValidation] = None
    exif_analysis: Optional[EXIFAnalysis] = None
    
    # Phase 2: Error Level Analysis
    ela_result: Optional[ELAResult] = None
    
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
    
    Supports multiple languages for international ID documents.
    """

    # Supported language codes for PaddleOCR
    # See: https://github.com/PaddlePaddle/PaddleOCR/blob/main/doc/doc_en/multi_languages_en.md
    SUPPORTED_LANGUAGES = {
        # Latin-based
        "en": "English",
        "fr": "French",
        "de": "German (German)",
        "es": "Spanish",
        "pt": "Portuguese",
        "it": "Italian",
        "nl": "Dutch",
        "pl": "Polish",
        "ro": "Romanian",
        "cs": "Czech",
        "hu": "Hungarian",
        "sv": "Swedish",
        "da": "Danish",
        "no": "Norwegian",
        "fi": "Finnish",
        "tr": "Turkish",
        "vi": "Vietnamese",
        "id": "Indonesian",
        "ms": "Malay",
        # Asian
        "ch": "Chinese (Simplified)",
        "cht": "Chinese (Traditional)",
        "ja": "Japanese",
        "ko": "Korean",
        "th": "Thai",
        "ta": "Tamil",
        "te": "Telugu",
        "hi": "Hindi",
        "mr": "Marathi",
        "ne": "Nepali",
        # Cyrillic
        "ru": "Russian",
        "uk": "Ukrainian",
        "be": "Belarusian",
        "bg": "Bulgarian",
        "sr": "Serbian (Cyrillic)",
        # Arabic script
        "ar": "Arabic",
        "fa": "Persian/Farsi",
        "ur": "Urdu",
        # Other
        "he": "Hebrew",
        "el": "Greek",
    }

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
            lang: Language code for OCR. Supported codes include:
                - Latin: en, fr, de, es, pt, it, nl, pl, ro, cs, hu, sv, da, no, fi, tr, vi, id, ms
                - Asian: ch (Chinese Simplified), cht (Traditional), ja, ko, th, ta, te, hi, mr, ne
                - Cyrillic: ru, uk, be, bg, sr
                - Arabic script: ar, fa, ur
                - Other: he, el
            use_gpu: Whether to use GPU acceleration
            min_confidence_threshold: Minimum confidence to accept text
            low_confidence_threshold: Threshold to flag low confidence
        """
        self.lang = lang
        self.use_gpu = use_gpu
        self.min_confidence_threshold = min_confidence_threshold
        self.low_confidence_threshold = low_confidence_threshold
        self._ocr = None
        self._ocr_instances: dict[str, any] = {}  # Cache OCR instances per language

    @property
    def version(self) -> str:
        """Return model version string for reproducibility tracking."""
        return f"paddleocr-{self.lang}"

    def _ensure_model(self, lang: Optional[str] = None) -> None:
        """Ensure PaddleOCR is loaded for the specified language.
        
        Args:
            lang: Language code (defaults to self.lang)
        """
        target_lang = lang or self.lang
        
        if target_lang not in self._ocr_instances:
            from paddleocr import PaddleOCR
            
            # PaddleOCR v2.8+ uses new API parameters
            self._ocr_instances[target_lang] = PaddleOCR(
                use_textline_orientation=True,
                lang=target_lang,
                device="gpu" if self.use_gpu else "cpu",
            )
        
        self._ocr = self._ocr_instances[target_lang]

    @classmethod
    def get_supported_languages(cls) -> dict[str, str]:
        """Get dictionary of supported language codes and names."""
        return cls.SUPPORTED_LANGUAGES.copy()

    def analyze(
        self,
        image: np.ndarray,
        lang: Optional[str] = None,
    ) -> DocumentAnalysisResult:
        """
        Analyze document image for text extraction and anomalies.
        
        Args:
            image: BGR image array
            lang: Optional language code to use for this analysis.
                If not provided, uses the instance's default language.
                See SUPPORTED_LANGUAGES for available codes.
        
        Returns:
            DocumentAnalysisResult with extracted text and anomaly flags
        """
        reason_codes: list[str] = []
        evidence: dict = {}
        anomalies: list[dict] = []

        # Use specified language or default
        target_lang = lang or self.lang
        self._ensure_model(target_lang)
        evidence["ocr_language"] = target_lang
        
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

        # Compute template match score
        template_match_score = self._compute_template_match_score(image, text_boxes)
        evidence["template_match_score"] = template_match_score
        
        # Add reason code if template doesn't match well
        if template_match_score < 0.5:
            reason_codes.append("DOC_TEMPLATE_MISMATCH")
            evidence["template_mismatch"] = True

        # === MRZ Validation ===
        full_text = " ".join(tb.text for tb in text_boxes)
        mrz_validation = self._validate_mrz(text_boxes, full_text)
        if mrz_validation and mrz_validation.detected:
            evidence["mrz_detected"] = True
            evidence["mrz_type"] = mrz_validation.mrz_type
            evidence["mrz_checksums_valid"] = mrz_validation.checksums_valid
            
            if not mrz_validation.checksums_valid:
                reason_codes.append("DOC_MRZ_INVALID")
                evidence["mrz_checksum_details"] = mrz_validation.checksum_details

        # === EXIF Metadata Analysis ===
        exif_analysis = self._analyze_exif(image)
        if exif_analysis:
            evidence["has_exif"] = exif_analysis.has_exif
            evidence["software_edited"] = exif_analysis.software_edited
            
            if exif_analysis.is_suspicious:
                reason_codes.append("DOC_METADATA_SUSPICIOUS")
                evidence["exif_suspicious_flags"] = exif_analysis.suspicious_flags

        # Compute overall document suspicion score
        doc_score = self._compute_doc_score(
            avg_confidence=avg_confidence,
            anomaly_count=len(anomalies),
            low_conf_ratio=len(low_conf_boxes) / max(len(text_boxes), 1),
        )

        return DocumentAnalysisResult(
            text_boxes=text_boxes,
            full_text=full_text,
            avg_confidence=avg_confidence,
            anomalies=anomalies,
            reason_codes=reason_codes,
            evidence=evidence,
            doc_score=doc_score,
            mrz_validation=mrz_validation,
            exif_analysis=exif_analysis,
            detected=True,
            ocr_results=ocr_results,
            overall_confidence=avg_confidence,
            template_match_score=template_match_score,
            font_consistency_score=font_consistency_score,
            alignment_score=alignment_score,
        )

    def analyze_document(
        self,
        image: np.ndarray,
        lang: Optional[str] = None,
    ) -> DocumentAnalysisResult:
        """Alias for analyze() for backward compatibility.
        
        Args:
            image: Input image as numpy array (BGR format)
            lang: Optional language code for OCR
        """
        return self.analyze(image, lang=lang)

    def extract_text(
        self,
        image: np.ndarray,
        lang: Optional[str] = None,
    ) -> list[OCRResult]:
        """
        Extract text from document image.
        
        Args:
            image: Input image as numpy array
            lang: Optional language code for OCR
            
        Returns:
            List of OCR results with text and confidence
        """
        result = self.analyze(image, lang=lang)
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

    def _compute_template_match_score(
        self,
        image: np.ndarray,
        text_boxes: list[TextBox],
    ) -> float:
        """Compute how well the document matches expected ID card structure.
        
        Uses structural analysis rather than pixel-level template matching:
        - Aspect ratio check (ID cards are typically ~1.58:1)
        - Card edge detection (rectangular boundary)
        - Text region distribution (text in expected areas)
        - Photo region detection (face photo area)
        
        Args:
            image: BGR image array
            text_boxes: Detected text regions
            
        Returns:
            Score from 0-1 where 1 = perfect match to ID card structure
        """
        h, w = image.shape[:2]
        score = 0.0
        
        # 1. Aspect ratio check (ID cards: CR-80 standard is 85.6mm x 53.98mm = 1.586:1)
        aspect = w / h if h > 0 else 0
        # Allow for portrait orientation too
        if aspect < 1:
            aspect = 1 / aspect if aspect > 0 else 0
        
        # ID cards typically have aspect ratio between 1.4 and 1.8
        if 1.4 <= aspect <= 1.8:
            # Closer to 1.586 is better
            aspect_diff = abs(aspect - 1.586)
            aspect_score = max(0, 1.0 - aspect_diff * 2)
            score += aspect_score * 0.25
        elif 1.2 <= aspect <= 2.0:
            # Still somewhat card-like
            score += 0.1
        
        # 2. Card edge detection (looking for rectangular boundary)
        card_score = self._detect_card_edges(image)
        score += card_score * 0.25
        
        # 3. Text region distribution (IDs have text in specific patterns)
        if text_boxes:
            text_score = self._analyze_text_distribution(text_boxes, w, h)
            score += text_score * 0.25
        
        # 4. Photo region detection (IDs typically have a face photo area)
        photo_score = self._detect_photo_region(image)
        score += photo_score * 0.25
        
        return min(max(score, 0.0), 1.0)

    def _detect_card_edges(self, image: np.ndarray) -> float:
        """Detect if image has clear rectangular card boundaries.
        
        Returns:
            Score from 0-1 where 1 = clear card boundary detected
        """
        h, w = image.shape[:2]
        gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        
        # Apply edge detection
        edges = cv2.Canny(gray, 50, 150)
        
        # Find contours
        contours, _ = cv2.findContours(edges, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        
        if not contours:
            return 0.0
        
        # Look for large rectangular contour (the card itself)
        img_area = h * w
        best_score = 0.0
        
        for contour in contours:
            area = cv2.contourArea(contour)
            
            # Card should be a significant portion of the image
            if area < img_area * 0.3:
                continue
            
            # Approximate to polygon
            epsilon = 0.02 * cv2.arcLength(contour, True)
            approx = cv2.approxPolyDP(contour, epsilon, True)
            
            # Rectangles have 4 vertices
            if len(approx) == 4:
                # Check if it's roughly rectangular (angles close to 90 degrees)
                rect = cv2.minAreaRect(contour)
                box = cv2.boxPoints(rect)
                box_area = cv2.contourArea(box.astype(np.int32))
                
                # Ratio of contour area to bounding rect area (1.0 = perfect rectangle)
                if box_area > 0:
                    rectangularity = area / box_area
                    if rectangularity > 0.85:
                        # Good rectangular shape
                        coverage = area / img_area
                        score = min(rectangularity, 1.0) * min(coverage / 0.7, 1.0)
                        best_score = max(best_score, score)
        
        return best_score

    def _analyze_text_distribution(
        self,
        text_boxes: list[TextBox],
        img_width: int,
        img_height: int,
    ) -> float:
        """Analyze if text is distributed like an ID card.
        
        ID cards typically have:
        - Multiple text fields arranged in rows
        - Text concentrated in certain regions (not random)
        - Mix of shorter labels and longer values
        
        Returns:
            Score from 0-1 where 1 = text distribution matches ID pattern
        """
        if len(text_boxes) < 3:
            return 0.3  # Too few text boxes to be a good ID
        
        score = 0.0
        
        # Check for multiple rows of text
        y_positions = []
        for tb in text_boxes:
            ys = [p[1] for p in tb.bbox]
            center_y = mean(ys)
            y_positions.append(center_y)
        
        # Cluster y positions to find rows
        y_positions.sort()
        rows = 1
        prev_y = y_positions[0]
        row_threshold = img_height * 0.05  # 5% of image height
        
        for y in y_positions[1:]:
            if y - prev_y > row_threshold:
                rows += 1
            prev_y = y
        
        # ID cards typically have 4-10 rows of text
        if 4 <= rows <= 10:
            score += 0.4
        elif 2 <= rows <= 12:
            score += 0.2
        
        # Check for left-aligned text (common in IDs)
        x_positions = []
        for tb in text_boxes:
            xs = [p[0] for p in tb.bbox]
            left_x = min(xs)
            x_positions.append(left_x)
        
        # Look for clustering at left margin
        margin_threshold = img_width * 0.15
        left_aligned = sum(1 for x in x_positions if x < margin_threshold)
        left_ratio = left_aligned / len(x_positions)
        
        if left_ratio > 0.3:
            score += 0.3
        
        # Check for varying text lengths (labels vs values)
        widths = []
        for tb in text_boxes:
            xs = [p[0] for p in tb.bbox]
            widths.append(max(xs) - min(xs))
        
        if len(widths) >= 2:
            width_variance = stdev(widths) / mean(widths) if mean(widths) > 0 else 0
            if width_variance > 0.3:  # Good variety in text lengths
                score += 0.3
        
        return min(score, 1.0)

    def _detect_photo_region(self, image: np.ndarray) -> float:
        """Detect if image contains a photo region typical of ID cards.
        
        Looks for:
        - A rectangular region with skin-tone colors
        - Region in expected position (usually left side or top)
        - Appropriate size relative to card
        
        Returns:
            Score from 0-1 where 1 = clear photo region detected
        """
        h, w = image.shape[:2]
        
        # Convert to HSV for skin tone detection
        hsv = cv2.cvtColor(image, cv2.COLOR_BGR2HSV)
        
        # Skin tone ranges in HSV (broad range to catch various skin tones)
        lower_skin = np.array([0, 20, 70], dtype=np.uint8)
        upper_skin = np.array([25, 255, 255], dtype=np.uint8)
        
        # Also check for darker skin tones
        lower_skin2 = np.array([0, 10, 40], dtype=np.uint8)
        upper_skin2 = np.array([20, 150, 200], dtype=np.uint8)
        
        mask1 = cv2.inRange(hsv, lower_skin, upper_skin)
        mask2 = cv2.inRange(hsv, lower_skin2, upper_skin2)
        skin_mask = cv2.bitwise_or(mask1, mask2)
        
        # Clean up mask
        kernel = np.ones((5, 5), np.uint8)
        skin_mask = cv2.morphologyEx(skin_mask, cv2.MORPH_CLOSE, kernel)
        skin_mask = cv2.morphologyEx(skin_mask, cv2.MORPH_OPEN, kernel)
        
        # Find contours of skin regions
        contours, _ = cv2.findContours(skin_mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        
        if not contours:
            return 0.0
        
        # Look for a face-sized region in expected positions
        img_area = h * w
        expected_photo_ratio = 0.08  # Photo is typically 8-20% of card
        
        best_score = 0.0
        
        for contour in contours:
            area = cv2.contourArea(contour)
            area_ratio = area / img_area
            
            # Photo should be 5-25% of image
            if not (0.05 <= area_ratio <= 0.25):
                continue
            
            # Get bounding box
            x, y, bw, bh = cv2.boundingRect(contour)
            
            # Check if in expected position (left third or top-left quadrant)
            center_x = x + bw / 2
            center_y = y + bh / 2
            
            in_left_third = center_x < w / 3
            in_upper_half = center_y < h * 0.6
            
            # Photo aspect ratio should be roughly portrait (taller than wide)
            aspect = bh / bw if bw > 0 else 0
            is_portrait = 1.0 <= aspect <= 1.8
            
            if in_left_third and in_upper_half and is_portrait:
                # Good position and shape for ID photo
                position_score = 0.6
                shape_score = 0.4 * (1.0 - abs(area_ratio - expected_photo_ratio) / 0.15)
                score = position_score + max(0, shape_score)
                best_score = max(best_score, score)
            elif is_portrait:
                # Right shape but wrong position
                best_score = max(best_score, 0.3)
        
        return min(best_score, 1.0)

    # === MRZ VALIDATION METHODS ===

    def _validate_mrz(
        self,
        text_boxes: list[TextBox],
        full_text: str,
    ) -> Optional[MRZValidation]:
        """Validate Machine Readable Zone (MRZ) if present.
        
        Supports:
        - TD1: ID cards (3 lines, 30 chars each)
        - TD2: Some ID cards (2 lines, 36 chars each)
        - TD3: Passports (2 lines, 44 chars each)
        
        Returns:
            MRZValidation result or None if no MRZ detected
        """
        # MRZ character set: A-Z, 0-9, <
        mrz_pattern = r'[A-Z0-9<]{20,44}'
        
        # Collect potential MRZ lines from OCR
        mrz_candidates: list[str] = []
        
        for tb in text_boxes:
            text = tb.text.upper().replace(' ', '').replace('O', '0')
            # Clean common OCR errors
            text = re.sub(r'[^A-Z0-9<]', '<', text)
            
            if len(text) >= 20 and re.match(mrz_pattern, text):
                mrz_candidates.append(text)
        
        if not mrz_candidates:
            return None
        
        # Try to identify MRZ type and validate
        mrz_type = None
        mrz_lines: list[str] = []
        
        # TD3 (Passport): 2 lines of 44 characters
        td3_lines = [l for l in mrz_candidates if len(l) >= 42 and len(l) <= 46]
        if len(td3_lines) >= 2:
            mrz_type = "TD3"
            mrz_lines = [l[:44].ljust(44, '<') for l in td3_lines[:2]]
        
        # TD1 (ID Card): 3 lines of 30 characters
        elif len([l for l in mrz_candidates if 28 <= len(l) <= 32]) >= 3:
            mrz_type = "TD1"
            td1_candidates = [l for l in mrz_candidates if 28 <= len(l) <= 32]
            mrz_lines = [l[:30].ljust(30, '<') for l in td1_candidates[:3]]
        
        # TD2: 2 lines of 36 characters
        elif len([l for l in mrz_candidates if 34 <= len(l) <= 38]) >= 2:
            mrz_type = "TD2"
            td2_candidates = [l for l in mrz_candidates if 34 <= len(l) <= 38]
            mrz_lines = [l[:36].ljust(36, '<') for l in td2_candidates[:2]]
        
        if not mrz_type:
            return MRZValidation(
                detected=False,
                mrz_type=None,
                mrz_lines=mrz_candidates,
                parsed_data={},
                checksums_valid=False,
                checksum_details={},
                confidence=0.0,
            )
        
        # Validate checksums
        checksums_valid, checksum_details = self._validate_mrz_checksums(
            mrz_type, mrz_lines
        )
        
        # Parse MRZ data
        parsed_data = self._parse_mrz_data(mrz_type, mrz_lines)
        
        # Compute confidence based on checksum validation and line lengths
        confidence = 0.7 if checksums_valid else 0.3
        if all(len(l) == (44 if mrz_type == "TD3" else 36 if mrz_type == "TD2" else 30) 
               for l in mrz_lines):
            confidence += 0.2
        
        return MRZValidation(
            detected=True,
            mrz_type=mrz_type,
            mrz_lines=mrz_lines,
            parsed_data=parsed_data,
            checksums_valid=checksums_valid,
            checksum_details=checksum_details,
            confidence=min(confidence, 1.0),
        )

    def _validate_mrz_checksums(
        self,
        mrz_type: str,
        mrz_lines: list[str],
    ) -> tuple[bool, dict]:
        """Validate MRZ check digits per ICAO 9303.
        
        Returns:
            Tuple of (all_valid, details_dict)
        """
        def compute_check_digit(data: str) -> int:
            """Compute check digit for MRZ field."""
            weights = [7, 3, 1]
            values = {'<': 0}
            # A-Z = 10-35
            for i, c in enumerate('ABCDEFGHIJKLMNOPQRSTUVWXYZ'):
                values[c] = 10 + i
            # 0-9 = 0-9
            for i in range(10):
                values[str(i)] = i
            
            total = 0
            for i, char in enumerate(data):
                total += values.get(char, 0) * weights[i % 3]
            
            return total % 10
        
        details = {}
        all_valid = True
        
        if mrz_type == "TD3" and len(mrz_lines) >= 2:
            line2 = mrz_lines[1]
            
            # Document number (positions 0-8, check digit at 9)
            doc_num = line2[0:9]
            doc_check = line2[9] if len(line2) > 9 else ''
            doc_computed = compute_check_digit(doc_num)
            doc_valid = doc_check.isdigit() and int(doc_check) == doc_computed
            details['document_number'] = {
                'value': doc_num.replace('<', ''),
                'check_digit': doc_check,
                'computed': doc_computed,
                'valid': doc_valid
            }
            if not doc_valid:
                all_valid = False
            
            # Date of birth (positions 13-18, check digit at 19)
            dob = line2[13:19]
            dob_check = line2[19] if len(line2) > 19 else ''
            dob_computed = compute_check_digit(dob)
            dob_valid = dob_check.isdigit() and int(dob_check) == dob_computed
            details['date_of_birth'] = {
                'value': dob,
                'check_digit': dob_check,
                'computed': dob_computed,
                'valid': dob_valid
            }
            if not dob_valid:
                all_valid = False
            
            # Expiry date (positions 21-26, check digit at 27)
            expiry = line2[21:27]
            exp_check = line2[27] if len(line2) > 27 else ''
            exp_computed = compute_check_digit(expiry)
            exp_valid = exp_check.isdigit() and int(exp_check) == exp_computed
            details['expiry_date'] = {
                'value': expiry,
                'check_digit': exp_check,
                'computed': exp_computed,
                'valid': exp_valid
            }
            if not exp_valid:
                all_valid = False
        
        elif mrz_type == "TD1" and len(mrz_lines) >= 3:
            # TD1 has different layout
            line1 = mrz_lines[0]
            line2 = mrz_lines[1]
            
            # Document number (line 1, positions 5-13, check at 14)
            if len(line1) >= 15:
                doc_num = line1[5:14]
                doc_check = line1[14]
                doc_computed = compute_check_digit(doc_num)
                doc_valid = doc_check.isdigit() and int(doc_check) == doc_computed
                details['document_number'] = {
                    'value': doc_num.replace('<', ''),
                    'check_digit': doc_check,
                    'computed': doc_computed,
                    'valid': doc_valid
                }
                if not doc_valid:
                    all_valid = False
            
            # DOB (line 2, positions 0-5, check at 6)
            if len(line2) >= 7:
                dob = line2[0:6]
                dob_check = line2[6]
                dob_computed = compute_check_digit(dob)
                dob_valid = dob_check.isdigit() and int(dob_check) == dob_computed
                details['date_of_birth'] = {
                    'value': dob,
                    'check_digit': dob_check,
                    'computed': dob_computed,
                    'valid': dob_valid
                }
                if not dob_valid:
                    all_valid = False
        
        return all_valid, details

    def _parse_mrz_data(self, mrz_type: str, mrz_lines: list[str]) -> dict:
        """Parse MRZ data into structured fields."""
        data = {}
        
        if mrz_type == "TD3" and len(mrz_lines) >= 2:
            line1 = mrz_lines[0]
            line2 = mrz_lines[1]
            
            # Document type (P = passport)
            data['document_type'] = line1[0:2].replace('<', '')
            
            # Issuing country (3-letter code)
            data['issuing_country'] = line1[2:5].replace('<', '')
            
            # Names (surname<<given_names)
            names = line1[5:].split('<<')
            data['surname'] = names[0].replace('<', ' ').strip() if names else ''
            data['given_names'] = names[1].replace('<', ' ').strip() if len(names) > 1 else ''
            
            # Line 2
            data['document_number'] = line2[0:9].replace('<', '')
            data['nationality'] = line2[10:13].replace('<', '')
            data['date_of_birth'] = line2[13:19]
            data['sex'] = line2[20] if len(line2) > 20 else ''
            data['expiry_date'] = line2[21:27]
        
        elif mrz_type == "TD1" and len(mrz_lines) >= 3:
            line1 = mrz_lines[0]
            line2 = mrz_lines[1]
            line3 = mrz_lines[2]
            
            data['document_type'] = line1[0:2].replace('<', '')
            data['issuing_country'] = line1[2:5].replace('<', '')
            data['document_number'] = line1[5:14].replace('<', '')
            
            data['date_of_birth'] = line2[0:6]
            data['sex'] = line2[7] if len(line2) > 7 else ''
            data['expiry_date'] = line2[8:14]
            data['nationality'] = line2[15:18].replace('<', '')
            
            # Name in line 3
            names = line3.split('<<')
            data['surname'] = names[0].replace('<', ' ').strip() if names else ''
            data['given_names'] = names[1].replace('<', ' ').strip() if len(names) > 1 else ''
        
        return data

    # === EXIF METADATA ANALYSIS ===

    def _analyze_exif(self, image: np.ndarray) -> Optional[EXIFAnalysis]:
        """Analyze EXIF metadata for suspicious editing indicators.
        
        Note: This requires the original image bytes, not the decoded array.
        When only a numpy array is available, we can only do limited analysis.
        
        For full EXIF analysis, the image should be passed as bytes.
        """
        suspicious_flags: list[str] = []
        
        # Known editing software signatures
        editing_software = [
            'photoshop', 'gimp', 'paint', 'lightroom', 'affinity',
            'pixlr', 'canva', 'snapseed', 'vsco', 'facetune',
            'adobe', 'corel', 'acdsee', 'capture one'
        ]
        
        # Try to extract EXIF from numpy array
        # This is limited - for full analysis, pass original image bytes
        try:
            from PIL import Image
            from PIL.ExifTags import TAGS
            
            # Convert numpy array to PIL Image
            pil_image = Image.fromarray(cv2.cvtColor(image, cv2.COLOR_BGR2RGB))
            
            # Try to get EXIF data
            exif_data = pil_image._getexif()
            
            if exif_data is None:
                # No EXIF data - could be stripped (suspicious for camera photos)
                suspicious_flags.append("no_exif_data")
                return EXIFAnalysis(
                    has_exif=False,
                    software_edited=False,
                    software_name=None,
                    creation_date=None,
                    modification_date=None,
                    camera_make=None,
                    camera_model=None,
                    gps_present=False,
                    suspicious_flags=suspicious_flags,
                    is_suspicious=len(suspicious_flags) > 0,
                )
            
            # Parse EXIF tags
            exif_dict = {}
            for tag_id, value in exif_data.items():
                tag = TAGS.get(tag_id, tag_id)
                exif_dict[tag] = value
            
            # Check for editing software
            software = exif_dict.get('Software', '')
            software_edited = False
            software_name = None
            
            if software:
                software_lower = software.lower()
                for editor in editing_software:
                    if editor in software_lower:
                        software_edited = True
                        software_name = software
                        suspicious_flags.append(f"editing_software:{software}")
                        break
            
            # Check for camera info
            camera_make = exif_dict.get('Make', None)
            camera_model = exif_dict.get('Model', None)
            
            # Check dates
            creation_date = exif_dict.get('DateTimeOriginal', None)
            modification_date = exif_dict.get('DateTime', None)
            
            # Date mismatch is suspicious
            if creation_date and modification_date:
                if creation_date != modification_date:
                    suspicious_flags.append("date_mismatch")
            
            # Check for GPS data
            gps_present = 'GPSInfo' in exif_dict
            
            # Missing camera info for a "photo" is suspicious
            if not camera_make and not camera_model:
                suspicious_flags.append("no_camera_info")
            
            is_suspicious = len(suspicious_flags) > 0 and software_edited
            
            return EXIFAnalysis(
                has_exif=True,
                software_edited=software_edited,
                software_name=software_name,
                creation_date=str(creation_date) if creation_date else None,
                modification_date=str(modification_date) if modification_date else None,
                camera_make=str(camera_make) if camera_make else None,
                camera_model=str(camera_model) if camera_model else None,
                gps_present=gps_present,
                suspicious_flags=suspicious_flags,
                is_suspicious=is_suspicious,
            )
            
        except Exception:
            # EXIF extraction failed - likely no EXIF or unsupported format
            return EXIFAnalysis(
                has_exif=False,
                software_edited=False,
                software_name=None,
                creation_date=None,
                modification_date=None,
                camera_make=None,
                camera_model=None,
                gps_present=False,
                suspicious_flags=["exif_extraction_failed"],
                is_suspicious=False,
            )

    def analyze_with_bytes(
        self,
        image_bytes: bytes,
        image: Optional[np.ndarray] = None,
    ) -> DocumentAnalysisResult:
        """Analyze document with full EXIF support from original bytes.
        
        Args:
            image_bytes: Original image file bytes
            image: Optional decoded numpy array (will decode if not provided)
            
        Returns:
            DocumentAnalysisResult with enhanced EXIF analysis
        """
        # Decode image if not provided
        if image is None:
            nparr = np.frombuffer(image_bytes, np.uint8)
            image = cv2.imdecode(nparr, cv2.IMREAD_COLOR)
        
        # Run standard analysis
        result = self.analyze(image)
        
        # Enhanced EXIF analysis from bytes
        try:
            from PIL import Image
            from PIL.ExifTags import TAGS
            
            pil_image = Image.open(BytesIO(image_bytes))
            exif_data = pil_image._getexif()
            
            if exif_data:
                # Re-run EXIF analysis with better data from bytes
                result.exif_analysis = self._analyze_exif_from_pil(pil_image, exif_data)
                
                if result.exif_analysis.is_suspicious:
                    if "DOC_METADATA_SUSPICIOUS" not in result.reason_codes:
                        result.reason_codes.append("DOC_METADATA_SUSPICIOUS")
                    result.evidence["exif_suspicious_flags"] = result.exif_analysis.suspicious_flags
        except Exception:
            pass
        
        return result

    def _analyze_exif_from_pil(self, pil_image, exif_data) -> EXIFAnalysis:
        """Analyze EXIF data from PIL Image object."""
        from PIL.ExifTags import TAGS
        
        suspicious_flags: list[str] = []
        editing_software = [
            'photoshop', 'gimp', 'paint', 'lightroom', 'affinity',
            'pixlr', 'canva', 'snapseed', 'vsco', 'facetune',
            'adobe', 'corel', 'acdsee', 'capture one'
        ]
        
        exif_dict = {}
        for tag_id, value in exif_data.items():
            tag = TAGS.get(tag_id, tag_id)
            exif_dict[tag] = value
        
        software = exif_dict.get('Software', '')
        software_edited = False
        software_name = None
        
        if software:
            software_lower = software.lower()
            for editor in editing_software:
                if editor in software_lower:
                    software_edited = True
                    software_name = software
                    suspicious_flags.append(f"editing_software:{software}")
                    break
        
        camera_make = exif_dict.get('Make', None)
        camera_model = exif_dict.get('Model', None)
        creation_date = exif_dict.get('DateTimeOriginal', None)
        modification_date = exif_dict.get('DateTime', None)
        
        if creation_date and modification_date:
            if creation_date != modification_date:
                suspicious_flags.append("date_mismatch")
        
        gps_present = 'GPSInfo' in exif_dict
        
        if not camera_make and not camera_model:
            suspicious_flags.append("no_camera_info")
        
        is_suspicious = len(suspicious_flags) > 0 and software_edited
        
        return EXIFAnalysis(
            has_exif=True,
            software_edited=software_edited,
            software_name=software_name,
            creation_date=str(creation_date) if creation_date else None,
            modification_date=str(modification_date) if modification_date else None,
            camera_make=str(camera_make) if camera_make else None,
            camera_model=str(camera_model) if camera_model else None,
            gps_present=gps_present,
            suspicious_flags=suspicious_flags,
            is_suspicious=is_suspicious,
        )

    # === PHASE 2: ERROR LEVEL ANALYSIS (ELA) ===

    def compute_ela(
        self,
        image: np.ndarray,
        quality: int = 90,
        scale: float = 15.0,
    ) -> np.ndarray:
        """Compute Error Level Analysis for an image.
        
        ELA works by re-compressing an image at a known quality level and
        comparing it with the original. Areas that have been edited or
        pasted will show different error levels because they've been
        compressed at different quality levels.
        
        Args:
            image: BGR image array
            quality: JPEG quality level for re-compression (default 90)
            scale: Scaling factor to amplify error differences (default 15)
            
        Returns:
            ELA image (grayscale) showing error levels
        """
        # Encode to JPEG at specified quality
        encode_params = [cv2.IMWRITE_JPEG_QUALITY, quality]
        _, encoded = cv2.imencode('.jpg', image, encode_params)
        
        # Decode back
        recompressed = cv2.imdecode(encoded, cv2.IMREAD_COLOR)
        
        # Compute absolute difference
        ela = cv2.absdiff(image, recompressed)
        
        # Convert to grayscale
        ela_gray = cv2.cvtColor(ela, cv2.COLOR_BGR2GRAY)
        
        # Scale to amplify differences
        ela_scaled = np.clip(ela_gray.astype(float) * scale, 0, 255).astype(np.uint8)
        
        return ela_scaled

    def detect_ela_manipulation(
        self,
        image: np.ndarray,
        quality: int = 90,
        threshold_percentile: float = 95,
        min_region_size: int = 100,
    ) -> ELAResult:
        """Detect potential image manipulation using Error Level Analysis.
        
        Analyzes the ELA image to find regions with abnormally high
        error levels, which may indicate tampering.
        
        Args:
            image: BGR image array
            quality: JPEG quality for ELA (default 90)
            threshold_percentile: Percentile threshold for "high" error (default 95)
            min_region_size: Minimum pixel count for a suspicious region
            
        Returns:
            ELAResult with manipulation detection details
        """
        # Compute ELA
        ela = self.compute_ela(image, quality=quality)
        
        # Calculate statistics
        avg_error = float(np.mean(ela))
        max_error = float(np.max(ela))
        error_std = float(np.std(ela))
        
        # Determine threshold for "high" error
        threshold = np.percentile(ela, threshold_percentile)
        
        # Find high-error regions
        high_error_mask = ela > threshold
        
        # Find contours of high-error regions
        contours, _ = cv2.findContours(
            high_error_mask.astype(np.uint8),
            cv2.RETR_EXTERNAL,
            cv2.CHAIN_APPROX_SIMPLE,
        )
        
        high_error_regions = []
        h, w = image.shape[:2]
        
        for contour in contours:
            area = cv2.contourArea(contour)
            if area < min_region_size:
                continue
            
            x, y, rw, rh = cv2.boundingRect(contour)
            
            # Calculate region statistics
            region_mask = np.zeros_like(ela)
            cv2.drawContours(region_mask, [contour], -1, 255, -1)
            region_pixels = ela[region_mask > 0]
            
            region_info = {
                "bbox": (x, y, rw, rh),
                "area": int(area),
                "mean_error": float(np.mean(region_pixels)),
                "max_error": float(np.max(region_pixels)),
                "relative_position": (
                    round(x / w, 3),
                    round(y / h, 3),
                    round((x + rw) / w, 3),
                    round((y + rh) / h, 3),
                ),
            }
            high_error_regions.append(region_info)
        
        # Sort by mean error (most suspicious first)
        high_error_regions.sort(key=lambda r: r["mean_error"], reverse=True)
        
        # Calculate manipulation score based on error statistics
        # Higher variance and more high-error regions = more suspicious
        
        # Normalize average error to 0-1 range (assuming 255 max)
        norm_avg_error = avg_error / 50  # ELA values rarely exceed 50 even with scaling
        norm_avg_error = min(norm_avg_error, 1.0)
        
        # Region count score
        region_score = min(len(high_error_regions) / 5, 1.0)
        
        # Variance score (high variance suggests mixed content)
        norm_std = min(error_std / 30, 1.0)
        
        # Combined manipulation score
        manipulation_score = (
            norm_avg_error * 0.3 +
            region_score * 0.4 +
            norm_std * 0.3
        )
        
        # Determine if manipulation is likely
        # High score + significant regions = likely manipulated
        manipulation_detected = (
            manipulation_score > 0.5 and
            len(high_error_regions) >= 2
        ) or (
            manipulation_score > 0.7
        )
        
        # Confidence based on clarity of signals
        confidence = min(manipulation_score * 1.2, 1.0) if manipulation_detected else 0.3
        
        return ELAResult(
            manipulation_detected=manipulation_detected,
            manipulation_score=round(manipulation_score, 3),
            high_error_regions=high_error_regions[:10],  # Limit to top 10
            avg_error_level=round(avg_error, 2),
            max_error_level=round(max_error, 2),
            error_std=round(error_std, 2),
            suspicious_region_count=len(high_error_regions),
            confidence=round(confidence, 3),
        )

    def analyze_with_ela(
        self,
        image: np.ndarray,
        lang: Optional[str] = None,
        run_ela: bool = True,
    ) -> DocumentAnalysisResult:
        """Analyze document with optional Error Level Analysis.
        
        Extended version of analyze() that also runs ELA to detect
        potential Photoshopping or tampering.
        
        Args:
            image: BGR image array
            lang: Optional language code for OCR
            run_ela: Whether to run ELA analysis (default True)
            
        Returns:
            DocumentAnalysisResult with ELA results
        """
        # Run standard analysis
        result = self.analyze(image, lang=lang)
        
        # Run ELA if requested
        if run_ela:
            ela_result = self.detect_ela_manipulation(image)
            result.ela_result = ela_result
            
            result.evidence["ela_manipulation_score"] = ela_result.manipulation_score
            result.evidence["ela_suspicious_regions"] = ela_result.suspicious_region_count
            
            if ela_result.manipulation_detected:
                result.reason_codes.append("DOC_ELA_MANIPULATION")
                result.evidence["ela_high_error_regions"] = ela_result.high_error_regions[:5]
        
        return result


# Singleton
_doc_analyzer: Optional[DocumentAnalyzer] = None


def get_document_analyzer() -> DocumentAnalyzer:
    """Get cached document analyzer instance."""
    global _doc_analyzer
    if _doc_analyzer is None:
        _doc_analyzer = DocumentAnalyzer()
    return _doc_analyzer
