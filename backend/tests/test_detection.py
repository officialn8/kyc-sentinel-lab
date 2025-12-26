"""Tests for detection modules."""

import numpy as np
import pytest

from app.detection.face_analyzer import FaceAnalyzer
from app.detection.document_analyzer import DocumentAnalyzer
from app.detection.pad_heuristics import PADAnalyzer
from app.detection.frame_extractor import FrameExtractor


class TestFaceAnalyzer:
    """Tests for FaceAnalyzer."""

    def test_detect_face(self) -> None:
        """Test face detection returns valid result."""
        analyzer = FaceAnalyzer()
        image = np.random.randint(0, 255, (480, 640, 3), dtype=np.uint8)
        
        faces = analyzer.detect_face(image)
        
        assert len(faces) > 0
        assert faces[0].detected is True
        assert faces[0].confidence > 0

    def test_get_embedding(self) -> None:
        """Test embedding extraction."""
        analyzer = FaceAnalyzer()
        image = np.random.randint(0, 255, (480, 640, 3), dtype=np.uint8)
        
        embedding = analyzer.get_embedding(image)
        
        assert embedding is not None
        assert len(embedding) == 512

    def test_compare_faces(self) -> None:
        """Test face comparison."""
        analyzer = FaceAnalyzer()
        selfie = np.random.randint(0, 255, (480, 640, 3), dtype=np.uint8)
        id_image = np.random.randint(0, 255, (480, 640, 3), dtype=np.uint8)
        
        result = analyzer.compare_faces(selfie, id_image)
        
        assert result.selfie_face.detected
        assert result.id_face.detected
        assert 0 <= result.similarity <= 1


class TestDocumentAnalyzer:
    """Tests for DocumentAnalyzer."""

    def test_extract_text(self) -> None:
        """Test OCR text extraction."""
        analyzer = DocumentAnalyzer()
        image = np.random.randint(0, 255, (800, 600, 3), dtype=np.uint8)
        
        results = analyzer.extract_text(image)
        
        assert len(results) > 0
        assert results[0].text
        assert results[0].confidence > 0

    def test_analyze_document(self) -> None:
        """Test document analysis."""
        analyzer = DocumentAnalyzer()
        image = np.random.randint(0, 255, (800, 600, 3), dtype=np.uint8)
        
        result = analyzer.analyze_document(image)
        
        assert result.detected
        assert 0 <= result.overall_confidence <= 1
        assert 0 <= result.doc_score <= 1


class TestPADAnalyzer:
    """Tests for PADAnalyzer."""

    def test_analyze_frames(self) -> None:
        """Test PAD analysis on frame sequence."""
        analyzer = PADAnalyzer()
        frames = [
            np.random.randint(0, 255, (480, 640, 3), dtype=np.uint8)
            for _ in range(10)
        ]
        
        result = analyzer.analyze_frames(frames)
        
        assert 0 <= result.pad_score <= 1
        assert len(result.frame_metrics) == len(frames)

    def test_analyze_empty_frames(self) -> None:
        """Test PAD analysis with no frames."""
        analyzer = PADAnalyzer()
        
        result = analyzer.analyze_frames([])
        
        assert result.pad_score == 0.5
        assert result.low_motion is True


class TestFrameExtractor:
    """Tests for FrameExtractor."""

    def test_initialization(self) -> None:
        """Test extractor initialization."""
        extractor = FrameExtractor(target_fps=10.0)
        assert extractor.target_fps == 10.0

    def test_extract_from_bytes(self) -> None:
        """Test frame extraction from bytes."""
        extractor = FrameExtractor()
        
        # Placeholder test - actual implementation would use real video
        result = extractor.extract_from_bytes(b"fake_video_data")
        
        assert len(result.frames) > 0
        assert result.metadata.fps > 0






