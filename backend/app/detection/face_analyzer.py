"""
Face detection, embedding extraction, and similarity scoring using InsightFace.

Responsibilities:
- Detect faces in selfie and ID images
- Extract 512-dimensional embeddings
- Compute cosine similarity between selfie and ID faces
- Return face crops for evidence storage
- Flag multiple faces or no face detected
- Compute face quality scores for adaptive thresholds
- Estimate pose angle from facial landmarks
- Estimate age and detect discrepancies with ID DOB
"""

import numpy as np
from dataclasses import dataclass, field
from datetime import date
from typing import Optional

import cv2

# InsightFace imports - lazy loaded to avoid slow startup
_insightface_app = None


@dataclass
class AgeEstimation:
    """Result of age estimation from face image."""
    
    estimated_age: int
    # Confidence interval for age (min, max)
    age_range: tuple[int, int]
    # Confidence score (0-1)
    confidence: float
    # Whether age estimation is reliable
    is_reliable: bool = True


@dataclass
class FaceQuality:
    """Face quality metrics for adaptive threshold computation."""
    
    # Overall quality score (0-1, higher = better)
    overall_score: float
    # Detection confidence from InsightFace
    detection_score: float
    # Sharpness/blur score (0-1)
    sharpness_score: float
    # Lighting quality (0-1)
    lighting_score: float
    # Pose deviation from frontal (degrees)
    pose_deviation: float
    # Face size relative to image
    face_size_ratio: float
    # Whether face is suitable for matching
    is_suitable: bool
    # Quality issues found
    issues: list[str] = field(default_factory=list)


@dataclass
class FaceDetection:
    """Result of face detection."""

    bbox: tuple[int, int, int, int]  # x1, y1, x2, y2
    embedding: np.ndarray  # 512d normalized vector
    confidence: float
    landmarks: np.ndarray  # 5-point landmarks
    crop: np.ndarray  # Face crop image
    quality: Optional[FaceQuality] = None  # Quality metrics
    age_estimation: Optional[AgeEstimation] = None  # Phase 2: Age estimation


@dataclass
class AgeDiscrepancyResult:
    """Result of age discrepancy check between estimated age and DOB."""
    
    estimated_age: int
    age_from_dob: int
    difference_years: int
    is_discrepant: bool
    tolerance_years: int = 10


@dataclass
class FaceAnalysisResult:
    """Result of face analysis comparing selfie to ID."""

    selfie_faces: list[FaceDetection]
    id_faces: list[FaceDetection]
    similarity: Optional[float]  # Cosine similarity if both have faces
    match: bool
    reason_codes: list[str]
    evidence: dict
    # Adaptive threshold used for matching
    threshold_used: float = 0.45
    # Quality scores
    selfie_quality: Optional[FaceQuality] = None
    id_quality: Optional[FaceQuality] = None
    # Phase 2: Age estimation
    selfie_age: Optional[AgeEstimation] = None
    id_age: Optional[AgeEstimation] = None
    age_discrepancy: Optional[AgeDiscrepancyResult] = None


class FaceAnalyzer:
    """Face detection and embedding extraction using InsightFace.
    
    Uses InsightFace's buffalo_l model pack for high-accuracy face detection
    and recognition. The model extracts 512-dimensional embeddings that can
    be compared using cosine similarity.
    
    Features adaptive thresholds based on image quality scores to handle
    degraded ID photos and suboptimal selfie conditions.
    """

    def __init__(
        self,
        model_pack: str = "buffalo_l",
        det_size: tuple[int, int] = (640, 640),
        similarity_threshold: float = 0.45,
        use_adaptive_threshold: bool = True,
        min_threshold: float = 0.35,
        max_threshold: float = 0.50,
    ) -> None:
        """
        Initialize InsightFace analyzer.
        
        Args:
            model_pack: Model pack name. Options:
                - "buffalo_l": Best accuracy, larger (default)
                - "buffalo_s": Faster, smaller
            det_size: Detection size (width, height)
            similarity_threshold: Base threshold for face matching (0-1)
            use_adaptive_threshold: Whether to adjust threshold based on quality
            min_threshold: Minimum threshold (used for low quality images)
            max_threshold: Maximum threshold (used for high quality images)
        """
        self.model_pack = model_pack
        self.det_size = det_size
        self.similarity_threshold = similarity_threshold
        self.use_adaptive_threshold = use_adaptive_threshold
        self.min_threshold = min_threshold
        self.max_threshold = max_threshold
        self._app = None

    @property
    def version(self) -> str:
        """Return model version string for reproducibility tracking."""
        return f"insightface-{self.model_pack}"

    def _ensure_model(self) -> None:
        """Ensure the InsightFace model is loaded (lazy initialization)."""
        if self._app is None:
            from insightface.app import FaceAnalysis
            
            self._app = FaceAnalysis(
                name=self.model_pack,
                providers=["CUDAExecutionProvider", "CPUExecutionProvider"],
            )
            self._app.prepare(ctx_id=0, det_size=self.det_size)

    def analyze(
        self, selfie_image: np.ndarray, id_image: np.ndarray
    ) -> FaceAnalysisResult:
        """
        Analyze selfie and ID images for face matching.
        
        Uses adaptive thresholds based on image quality when enabled.
        Lower quality images get more lenient thresholds.
        
        Args:
            selfie_image: BGR image array (OpenCV format)
            id_image: BGR image array (OpenCV format)
        
        Returns:
            FaceAnalysisResult with detections, similarity, and reason codes
        """
        import logging
        logger = logging.getLogger(__name__)
        
        reason_codes: list[str] = []
        evidence: dict = {}
        selfie_quality: Optional[FaceQuality] = None
        id_quality: Optional[FaceQuality] = None
        threshold_used = self.similarity_threshold

        # Detect faces with quality scoring
        selfie_detections = self._detect_faces(selfie_image, "selfie")
        id_detections = self._detect_faces(id_image, "id")
        
        # Debug logging
        logger.info(f"Face detection: selfie has {len(selfie_detections)} face(s), ID has {len(id_detections)} face(s)")
        for i, det in enumerate(selfie_detections):
            logger.info(f"  Selfie face {i}: confidence={det.confidence:.3f}, bbox={det.bbox}")
            if det.quality:
                logger.info(f"    Quality: overall={det.quality.overall_score:.3f}, pose={det.quality.pose_deviation:.1f}deg")
        for i, det in enumerate(id_detections):
            logger.info(f"  ID face {i}: confidence={det.confidence:.3f}, bbox={det.bbox}")
            if det.quality:
                logger.info(f"    Quality: overall={det.quality.overall_score:.3f}, pose={det.quality.pose_deviation:.1f}deg")

        # Check for missing faces
        if not selfie_detections:
            reason_codes.append("FACE_NOT_DETECTED_SELFIE")
            evidence["selfie_face_count"] = 0

        if not id_detections:
            reason_codes.append("FACE_NOT_DETECTED_ID")
            evidence["id_face_count"] = 0

        # Check for multiple faces in selfie
        if len(selfie_detections) > 1:
            reason_codes.append("MULTIPLE_FACES_SELFIE")
            evidence["selfie_face_count"] = len(selfie_detections)

        # Compute similarity if both have faces
        similarity: Optional[float] = None
        match = False

        if selfie_detections and id_detections:
            # Use highest confidence face from each
            selfie_face = max(selfie_detections, key=lambda f: f.confidence)
            id_face = max(id_detections, key=lambda f: f.confidence)
            
            selfie_quality = selfie_face.quality
            id_quality = id_face.quality
            
            # Check for quality issues and add reason codes
            if selfie_quality:
                evidence["selfie_quality_score"] = selfie_quality.overall_score
                evidence["selfie_pose_deviation"] = selfie_quality.pose_deviation
                
                if not selfie_quality.is_suitable:
                    reason_codes.append("FACE_LOW_QUALITY_SELFIE")
                    evidence["selfie_quality_issues"] = selfie_quality.issues
                
                if selfie_quality.pose_deviation > 30:
                    reason_codes.append("FACE_EXTREME_POSE")
                    evidence["selfie_pose_deviation"] = selfie_quality.pose_deviation
            
            if id_quality:
                evidence["id_quality_score"] = id_quality.overall_score
                evidence["id_pose_deviation"] = id_quality.pose_deviation
                
                if not id_quality.is_suitable:
                    reason_codes.append("FACE_LOW_QUALITY_ID")
                    evidence["id_quality_issues"] = id_quality.issues
            
            # Compute adaptive threshold based on quality
            if self.use_adaptive_threshold and selfie_quality and id_quality:
                threshold_used = self._compute_adaptive_threshold(
                    selfie_quality, id_quality
                )
                evidence["adaptive_threshold"] = threshold_used
                evidence["base_threshold"] = self.similarity_threshold
                logger.info(f"Adaptive threshold: {threshold_used:.4f} (base: {self.similarity_threshold})")
            else:
                threshold_used = self.similarity_threshold

            similarity = self._compute_similarity(
                selfie_face.embedding, id_face.embedding
            )
            evidence["face_similarity"] = float(similarity)
            evidence["threshold_used"] = threshold_used
            
            logger.info(f"Face comparison: selfie_conf={selfie_face.confidence:.3f}, id_conf={id_face.confidence:.3f}, similarity={similarity:.4f}")

            match = similarity >= threshold_used

            if not match:
                reason_codes.append("FACE_MISMATCH")
                evidence["similarity_threshold"] = threshold_used
                logger.info(f"FACE_MISMATCH: similarity {similarity:.4f} < threshold {threshold_used}")
            else:
                logger.info(f"Face match: similarity {similarity:.4f} >= threshold {threshold_used}")

        return FaceAnalysisResult(
            selfie_faces=selfie_detections,
            id_faces=id_detections,
            similarity=similarity,
            match=match,
            reason_codes=reason_codes,
            evidence=evidence,
            threshold_used=threshold_used,
            selfie_quality=selfie_quality,
            id_quality=id_quality,
        )

    def _detect_faces(self, image: np.ndarray, source: str) -> list[FaceDetection]:
        """Detect faces and extract embeddings from image."""
        self._ensure_model()
        
        faces = self._app.get(image)
        h, w = image.shape[:2]

        detections = []
        for face in faces:
            bbox = tuple(map(int, face.bbox))
            x1, y1, x2, y2 = bbox

            # Extract face crop with padding
            pad = 20
            crop = image[
                max(0, y1 - pad) : min(h, y2 + pad),
                max(0, x1 - pad) : min(w, x2 + pad),
            ]
            
            # Compute quality metrics
            quality = self._compute_face_quality(
                image=image,
                bbox=bbox,
                landmarks=face.kps,
                det_score=float(face.det_score),
                source=source,
            )
            
            # Phase 2: Extract age estimation if available
            age_estimation = self._extract_age_estimation(face)

            detections.append(
                FaceDetection(
                    bbox=bbox,
                    embedding=face.embedding,  # Already normalized by InsightFace
                    confidence=float(face.det_score),
                    landmarks=face.kps,
                    crop=crop,
                    quality=quality,
                    age_estimation=age_estimation,
                )
            )

        return detections
    
    def _compute_face_quality(
        self,
        image: np.ndarray,
        bbox: tuple[int, int, int, int],
        landmarks: np.ndarray,
        det_score: float,
        source: str,
    ) -> FaceQuality:
        """Compute face quality metrics for adaptive threshold computation.
        
        Args:
            image: Full BGR image
            bbox: Face bounding box (x1, y1, x2, y2)
            landmarks: 5-point facial landmarks
            det_score: Detection confidence
            source: "selfie" or "id"
            
        Returns:
            FaceQuality with comprehensive quality metrics
        """
        h, w = image.shape[:2]
        x1, y1, x2, y2 = bbox
        face_w, face_h = x2 - x1, y2 - y1
        
        issues: list[str] = []
        
        # 1. Face size ratio (larger = better)
        face_area = face_w * face_h
        img_area = w * h
        face_size_ratio = face_area / img_area if img_area > 0 else 0
        
        # For selfies, face should be prominent (10-50% of image)
        # For ID photos, face is typically smaller (5-30%)
        if source == "selfie":
            if face_size_ratio < 0.05:
                issues.append("face_too_small")
            elif face_size_ratio > 0.6:
                issues.append("face_too_large")
        else:  # ID
            if face_size_ratio < 0.02:
                issues.append("face_too_small")
        
        # 2. Sharpness score (Laplacian variance on face region)
        face_crop = image[max(0, y1):min(h, y2), max(0, x1):min(w, x2)]
        sharpness_score = self._compute_sharpness(face_crop)
        
        if sharpness_score < 0.3:
            issues.append("blurry_face")
        
        # 3. Lighting quality (check for under/over exposure)
        lighting_score = self._compute_lighting_quality(face_crop)
        
        if lighting_score < 0.3:
            issues.append("poor_lighting")
        
        # 4. Pose deviation from frontal (using landmarks)
        pose_deviation = self._estimate_pose_deviation(landmarks, (face_w, face_h))
        
        if pose_deviation > 25:
            issues.append("non_frontal_pose")
        if pose_deviation > 40:
            issues.append("extreme_pose")
        
        # 5. Overall quality score (weighted combination)
        # Normalize each component to 0-1
        size_score = min(face_size_ratio / 0.15, 1.0)  # Optimal around 15%
        pose_score = max(0, 1.0 - pose_deviation / 45)  # 45 degrees = 0 score
        det_normalized = min(det_score / 0.9, 1.0)  # Normalize detection score
        
        overall_score = (
            det_normalized * 0.25 +
            sharpness_score * 0.25 +
            lighting_score * 0.20 +
            pose_score * 0.20 +
            size_score * 0.10
        )
        
        # Face is suitable if quality is acceptable and no critical issues
        critical_issues = {"extreme_pose", "face_too_small", "blurry_face"}
        has_critical = bool(critical_issues & set(issues))
        is_suitable = overall_score >= 0.4 and not has_critical
        
        return FaceQuality(
            overall_score=overall_score,
            detection_score=det_score,
            sharpness_score=sharpness_score,
            lighting_score=lighting_score,
            pose_deviation=pose_deviation,
            face_size_ratio=face_size_ratio,
            is_suitable=is_suitable,
            issues=issues,
        )
    
    def _compute_sharpness(self, face_crop: np.ndarray) -> float:
        """Compute sharpness score for face region.
        
        Uses Laplacian variance normalized to 0-1 range.
        """
        if face_crop.size == 0:
            return 0.0
        
        gray = cv2.cvtColor(face_crop, cv2.COLOR_BGR2GRAY)
        laplacian = cv2.Laplacian(gray, cv2.CV_64F)
        variance = laplacian.var()
        
        # Normalize: typical good sharpness > 100, excellent > 500
        # Map to 0-1 with sigmoid-like curve
        score = min(variance / 200, 1.0)
        return float(score)
    
    def _compute_lighting_quality(self, face_crop: np.ndarray) -> float:
        """Compute lighting quality score.
        
        Checks for:
        - Underexposure (too dark)
        - Overexposure (too bright)
        - Uneven lighting
        """
        if face_crop.size == 0:
            return 0.0
        
        gray = cv2.cvtColor(face_crop, cv2.COLOR_BGR2GRAY)
        
        # Mean brightness (optimal around 100-150)
        mean_brightness = np.mean(gray)
        
        # Standard deviation (optimal range indicates good contrast)
        std_brightness = np.std(gray)
        
        # Score based on brightness (penalize extremes)
        if mean_brightness < 50:
            brightness_score = mean_brightness / 50
        elif mean_brightness > 200:
            brightness_score = (255 - mean_brightness) / 55
        else:
            # Optimal range: linear interpolation with peak at 125
            brightness_score = 1.0 - abs(mean_brightness - 125) / 125
        
        # Score based on contrast (too low = flat, too high = harsh)
        if std_brightness < 20:
            contrast_score = std_brightness / 20
        elif std_brightness > 80:
            contrast_score = max(0, 1.0 - (std_brightness - 80) / 80)
        else:
            contrast_score = 1.0
        
        # Check for uneven lighting using histogram
        hist = cv2.calcHist([gray], [0], None, [8], [0, 256])
        hist = hist.flatten() / hist.sum()
        # High entropy = more even distribution
        hist = hist[hist > 0]
        entropy = -np.sum(hist * np.log2(hist)) / 3  # Normalize by max entropy
        
        # Combine scores
        score = brightness_score * 0.4 + contrast_score * 0.3 + entropy * 0.3
        return float(max(0, min(1, score)))
    
    def _estimate_pose_deviation(
        self,
        landmarks: np.ndarray,
        face_size: tuple[int, int],
    ) -> float:
        """Estimate face pose deviation from frontal in degrees.
        
        Uses 5-point landmarks (left eye, right eye, nose, left mouth, right mouth)
        to estimate yaw and pitch angles.
        
        Args:
            landmarks: 5x2 array of landmark coordinates
            face_size: (width, height) of face bounding box
            
        Returns:
            Estimated pose deviation in degrees (0 = frontal)
        """
        if landmarks is None or len(landmarks) < 5:
            return 0.0
        
        face_w, face_h = face_size
        if face_w <= 0 or face_h <= 0:
            return 0.0
        
        # 5-point landmarks: [left_eye, right_eye, nose, left_mouth, right_mouth]
        left_eye = landmarks[0]
        right_eye = landmarks[1]
        nose = landmarks[2]
        left_mouth = landmarks[3]
        right_mouth = landmarks[4]
        
        # Eye center
        eye_center = (left_eye + right_eye) / 2
        mouth_center = (left_mouth + right_mouth) / 2
        
        # Yaw estimation: asymmetry in eye-to-nose distances
        left_eye_to_nose = np.linalg.norm(left_eye - nose)
        right_eye_to_nose = np.linalg.norm(right_eye - nose)
        
        if left_eye_to_nose + right_eye_to_nose > 0:
            yaw_ratio = abs(left_eye_to_nose - right_eye_to_nose) / (left_eye_to_nose + right_eye_to_nose)
        else:
            yaw_ratio = 0
        
        # Convert ratio to approximate degrees (empirical mapping)
        yaw_degrees = yaw_ratio * 90  # 0.5 ratio ≈ 45 degrees
        
        # Pitch estimation: nose position relative to eye-mouth line
        eye_to_mouth = np.linalg.norm(eye_center - mouth_center)
        eye_to_nose = np.linalg.norm(eye_center - nose)
        
        if eye_to_mouth > 0:
            # Frontal face: nose is about 40-50% of eye-mouth distance from eyes
            expected_ratio = 0.45
            actual_ratio = eye_to_nose / eye_to_mouth
            pitch_deviation = abs(actual_ratio - expected_ratio) / expected_ratio
        else:
            pitch_deviation = 0
        
        pitch_degrees = pitch_deviation * 45  # Deviation of 1.0 ≈ 45 degrees
        
        # Combine yaw and pitch into overall deviation
        total_deviation = np.sqrt(yaw_degrees ** 2 + pitch_degrees ** 2)
        
        return float(min(total_deviation, 90))
    
    def _compute_adaptive_threshold(
        self,
        selfie_quality: FaceQuality,
        id_quality: FaceQuality,
    ) -> float:
        """Compute adaptive similarity threshold based on image quality.
        
        Lower quality images get more lenient thresholds to account for
        degradation in embedding accuracy.
        
        Args:
            selfie_quality: Quality metrics for selfie
            id_quality: Quality metrics for ID photo
            
        Returns:
            Adjusted similarity threshold
        """
        # Average quality score
        avg_quality = (selfie_quality.overall_score + id_quality.overall_score) / 2
        
        # Pose penalty: high pose deviation reduces threshold
        max_pose = max(selfie_quality.pose_deviation, id_quality.pose_deviation)
        pose_factor = max(0, 1 - max_pose / 45)  # 45 degrees = full penalty
        
        # ID photos are often lower quality (older, scanned, printed)
        # Apply asymmetric weighting: ID quality matters less
        weighted_quality = (
            selfie_quality.overall_score * 0.6 +
            id_quality.overall_score * 0.4
        )
        
        # Map quality to threshold range
        # High quality (1.0) -> max_threshold
        # Low quality (0.0) -> min_threshold
        quality_range = self.max_threshold - self.min_threshold
        threshold = self.min_threshold + weighted_quality * quality_range
        
        # Apply pose penalty
        threshold = threshold * (0.7 + 0.3 * pose_factor)
        
        # Clamp to valid range
        return max(self.min_threshold, min(self.max_threshold, threshold))

    def _compute_similarity(self, emb1: np.ndarray, emb2: np.ndarray) -> float:
        """Compute cosine similarity between two embeddings."""
        import logging
        logger = logging.getLogger(__name__)
        
        # Log embedding norms before normalization
        norm1 = np.linalg.norm(emb1)
        norm2 = np.linalg.norm(emb2)
        logger.info(f"Embedding norms: selfie={norm1:.4f}, id={norm2:.4f}")
        
        # Normalize embeddings to unit vectors
        emb1_norm = emb1 / norm1
        emb2_norm = emb2 / norm2
        # Cosine similarity is dot product of normalized vectors
        similarity = float(np.dot(emb1_norm, emb2_norm))
        # Clamp to valid range to handle floating point precision
        return max(-1.0, min(1.0, similarity))

    def get_embedding(self, image: np.ndarray) -> Optional[np.ndarray]:
        """
        Get face embedding for pgvector storage.
        Returns None if no face detected.
        """
        self._ensure_model()
        
        faces = self._app.get(image)
        if not faces:
            return None
        # Return highest confidence face embedding
        best_face = max(faces, key=lambda f: f.det_score)
        return best_face.embedding

    def detect_face(self, image: np.ndarray) -> list[FaceDetection]:
        """
        Detect faces in an image.
        
        Args:
            image: Input image as numpy array (BGR format)
            
        Returns:
            List of detected faces
        """
        return self._detect_faces(image, "")

    def compare_faces(
        self, selfie_image: np.ndarray, id_image: np.ndarray
    ) -> FaceAnalysisResult:
        """
        Compare faces between selfie and ID images.
        
        Alias for analyze() for backward compatibility.
        """
        return self.analyze(selfie_image, id_image)

    # === PHASE 2: AGE ESTIMATION ===

    def _extract_age_estimation(self, face) -> Optional[AgeEstimation]:
        """Extract age estimation from InsightFace face object.
        
        InsightFace's analysis models can provide age estimation.
        The 'age' attribute is available when using models that include
        age prediction (e.g., buffalo_l includes this).
        
        Args:
            face: InsightFace Face object
            
        Returns:
            AgeEstimation or None if not available
        """
        try:
            # Check if face has age attribute
            if hasattr(face, 'age') and face.age is not None:
                age = int(face.age)
                
                # InsightFace age estimation typically has ~5-7 year error margin
                # Provide a reasonable confidence interval
                age_error_margin = 6
                age_range = (max(0, age - age_error_margin), age + age_error_margin)
                
                # Confidence based on detection score (proxy for image quality)
                confidence = min(float(face.det_score), 1.0) * 0.8
                
                return AgeEstimation(
                    estimated_age=age,
                    age_range=age_range,
                    confidence=confidence,
                    is_reliable=confidence >= 0.5,
                )
            
            return None
            
        except Exception:
            return None

    def estimate_age(self, image: np.ndarray) -> Optional[AgeEstimation]:
        """Estimate age from a face image.
        
        Args:
            image: BGR image containing a face
            
        Returns:
            AgeEstimation or None if no face detected
        """
        detections = self._detect_faces(image, "age_estimate")
        
        if not detections:
            return None
        
        # Use highest confidence face
        best_face = max(detections, key=lambda f: f.confidence)
        return best_face.age_estimation

    def check_age_discrepancy(
        self,
        estimated_age: int,
        dob: date,
        tolerance_years: int = 10,
        reference_date: Optional[date] = None,
    ) -> AgeDiscrepancyResult:
        """Check if estimated age differs significantly from DOB.
        
        Useful for detecting fraudulent ID documents where the
        photo doesn't match the stated date of birth.
        
        Args:
            estimated_age: Age estimated from face image
            dob: Date of birth from ID document
            tolerance_years: Maximum acceptable difference (default 10 years)
            reference_date: Date to calculate age from (defaults to today)
            
        Returns:
            AgeDiscrepancyResult with comparison details
        """
        if reference_date is None:
            reference_date = date.today()
        
        # Calculate actual age from DOB
        age_from_dob = reference_date.year - dob.year
        
        # Adjust if birthday hasn't occurred this year
        if (reference_date.month, reference_date.day) < (dob.month, dob.day):
            age_from_dob -= 1
        
        difference = abs(estimated_age - age_from_dob)
        is_discrepant = difference > tolerance_years
        
        return AgeDiscrepancyResult(
            estimated_age=estimated_age,
            age_from_dob=age_from_dob,
            difference_years=difference,
            is_discrepant=is_discrepant,
            tolerance_years=tolerance_years,
        )

    def analyze_with_age_check(
        self,
        selfie_image: np.ndarray,
        id_image: np.ndarray,
        dob: Optional[date] = None,
        age_tolerance_years: int = 10,
    ) -> FaceAnalysisResult:
        """Analyze faces with optional age discrepancy check.
        
        Extended version of analyze() that also checks if the
        estimated age matches the DOB from the ID document.
        
        Args:
            selfie_image: Selfie image (BGR)
            id_image: ID document image (BGR)
            dob: Optional date of birth from ID for age verification
            age_tolerance_years: Maximum acceptable age difference
            
        Returns:
            FaceAnalysisResult with age estimation and discrepancy check
        """
        # Run standard analysis
        result = self.analyze(selfie_image, id_image)
        
        # Extract age estimates
        selfie_age = None
        id_age = None
        
        if result.selfie_faces:
            best_selfie = max(result.selfie_faces, key=lambda f: f.confidence)
            selfie_age = best_selfie.age_estimation
            result.selfie_age = selfie_age
            
            if selfie_age:
                result.evidence["selfie_estimated_age"] = selfie_age.estimated_age
                result.evidence["selfie_age_range"] = selfie_age.age_range
        
        if result.id_faces:
            best_id = max(result.id_faces, key=lambda f: f.confidence)
            id_age = best_id.age_estimation
            result.id_age = id_age
            
            if id_age:
                result.evidence["id_estimated_age"] = id_age.estimated_age
                result.evidence["id_age_range"] = id_age.age_range
        
        # Check age discrepancy if DOB provided
        if dob and selfie_age and selfie_age.is_reliable:
            discrepancy = self.check_age_discrepancy(
                estimated_age=selfie_age.estimated_age,
                dob=dob,
                tolerance_years=age_tolerance_years,
            )
            result.age_discrepancy = discrepancy
            
            result.evidence["age_from_dob"] = discrepancy.age_from_dob
            result.evidence["age_difference_years"] = discrepancy.difference_years
            result.evidence["age_tolerance_years"] = age_tolerance_years
            
            if discrepancy.is_discrepant:
                result.reason_codes.append("FACE_AGE_DISCREPANCY")
        
        return result


# Singleton instance for reuse (model loading is expensive)
_face_analyzer: Optional[FaceAnalyzer] = None


def get_face_analyzer() -> FaceAnalyzer:
    """Get cached face analyzer instance."""
    global _face_analyzer
    if _face_analyzer is None:
        _face_analyzer = FaceAnalyzer()
    return _face_analyzer
