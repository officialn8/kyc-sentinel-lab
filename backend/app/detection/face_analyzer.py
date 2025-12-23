"""
Face detection, embedding extraction, and similarity scoring using InsightFace.

Responsibilities:
- Detect faces in selfie and ID images
- Extract 512-dimensional embeddings
- Compute cosine similarity between selfie and ID faces
- Return face crops for evidence storage
- Flag multiple faces or no face detected
"""

import numpy as np
from dataclasses import dataclass
from typing import Optional

import cv2

# InsightFace imports - lazy loaded to avoid slow startup
_insightface_app = None


@dataclass
class FaceDetection:
    """Result of face detection."""

    bbox: tuple[int, int, int, int]  # x1, y1, x2, y2
    embedding: np.ndarray  # 512d normalized vector
    confidence: float
    landmarks: np.ndarray  # 5-point landmarks
    crop: np.ndarray  # Face crop image


@dataclass
class FaceAnalysisResult:
    """Result of face analysis comparing selfie to ID."""

    selfie_faces: list[FaceDetection]
    id_faces: list[FaceDetection]
    similarity: Optional[float]  # Cosine similarity if both have faces
    match: bool
    reason_codes: list[str]
    evidence: dict


class FaceAnalyzer:
    """Face detection and embedding extraction using InsightFace.
    
    Uses InsightFace's buffalo_l model pack for high-accuracy face detection
    and recognition. The model extracts 512-dimensional embeddings that can
    be compared using cosine similarity.
    """

    def __init__(
        self,
        model_pack: str = "buffalo_l",
        det_size: tuple[int, int] = (640, 640),
        similarity_threshold: float = 0.45,
    ) -> None:
        """
        Initialize InsightFace analyzer.
        
        Args:
            model_pack: Model pack name. Options:
                - "buffalo_l": Best accuracy, larger (default)
                - "buffalo_s": Faster, smaller
            det_size: Detection size (width, height)
            similarity_threshold: Threshold for face matching (0-1)
        """
        self.model_pack = model_pack
        self.det_size = det_size
        self.similarity_threshold = similarity_threshold
        self._app = None

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
        
        Args:
            selfie_image: BGR image array (OpenCV format)
            id_image: BGR image array (OpenCV format)
        
        Returns:
            FaceAnalysisResult with detections, similarity, and reason codes
        """
        reason_codes: list[str] = []
        evidence: dict = {}

        # Detect faces
        selfie_detections = self._detect_faces(selfie_image, "selfie")
        id_detections = self._detect_faces(id_image, "id")

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

            similarity = self._compute_similarity(
                selfie_face.embedding, id_face.embedding
            )
            evidence["face_similarity"] = float(similarity)

            match = similarity >= self.similarity_threshold

            if not match:
                reason_codes.append("FACE_MISMATCH")
                evidence["similarity_threshold"] = self.similarity_threshold

        return FaceAnalysisResult(
            selfie_faces=selfie_detections,
            id_faces=id_detections,
            similarity=similarity,
            match=match,
            reason_codes=reason_codes,
            evidence=evidence,
        )

    def _detect_faces(self, image: np.ndarray, source: str) -> list[FaceDetection]:
        """Detect faces and extract embeddings from image."""
        self._ensure_model()
        
        faces = self._app.get(image)

        detections = []
        for face in faces:
            bbox = tuple(map(int, face.bbox))
            x1, y1, x2, y2 = bbox

            # Extract face crop with padding
            pad = 20
            h, w = image.shape[:2]
            crop = image[
                max(0, y1 - pad) : min(h, y2 + pad),
                max(0, x1 - pad) : min(w, x2 + pad),
            ]

            detections.append(
                FaceDetection(
                    bbox=bbox,
                    embedding=face.embedding,  # Already normalized by InsightFace
                    confidence=float(face.det_score),
                    landmarks=face.kps,
                    crop=crop,
                )
            )

        return detections

    def _compute_similarity(self, emb1: np.ndarray, emb2: np.ndarray) -> float:
        """Compute cosine similarity between two embeddings."""
        # InsightFace embeddings are already normalized
        return float(np.dot(emb1, emb2))

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


# Singleton instance for reuse (model loading is expensive)
_face_analyzer: Optional[FaceAnalyzer] = None


def get_face_analyzer() -> FaceAnalyzer:
    """Get cached face analyzer instance."""
    global _face_analyzer
    if _face_analyzer is None:
        _face_analyzer = FaceAnalyzer()
    return _face_analyzer
