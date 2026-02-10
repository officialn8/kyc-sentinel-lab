"""Deterministic unit tests for detection module interfaces."""

import numpy as np
import pytest

from app.detection.document_analyzer import DocumentAnalysisResult, DocumentAnalyzer, OCRResult
from app.detection.face_analyzer import FaceAnalysisResult, FaceAnalyzer, FaceQuality
from app.detection.frame_extractor import FrameExtractor
from app.detection.pad_heuristics import PADAnalyzer


class TestFaceAnalyzer:
    """Tests for FaceAnalyzer without depending on model outputs."""

    def test_compute_similarity_bounds(self) -> None:
        analyzer = FaceAnalyzer()

        same = analyzer._compute_similarity(np.array([1.0, 0.0]), np.array([1.0, 0.0]))
        opposite = analyzer._compute_similarity(
            np.array([1.0, 0.0]), np.array([-1.0, 0.0])
        )

        assert same == pytest.approx(1.0)
        assert opposite == pytest.approx(-1.0)

    def test_compute_adaptive_threshold_monotonic(self) -> None:
        analyzer = FaceAnalyzer()

        high_quality = FaceQuality(
            overall_score=0.95,
            detection_score=0.99,
            sharpness_score=0.95,
            lighting_score=0.9,
            pose_deviation=2.0,
            face_size_ratio=0.2,
            is_suitable=True,
        )
        low_quality = FaceQuality(
            overall_score=0.2,
            detection_score=0.4,
            sharpness_score=0.2,
            lighting_score=0.2,
            pose_deviation=35.0,
            face_size_ratio=0.05,
            is_suitable=False,
        )

        high_threshold = analyzer._compute_adaptive_threshold(high_quality, high_quality)
        low_threshold = analyzer._compute_adaptive_threshold(low_quality, low_quality)

        assert analyzer.min_threshold <= low_threshold <= analyzer.max_threshold
        assert analyzer.min_threshold <= high_threshold <= analyzer.max_threshold
        assert low_threshold <= high_threshold

    def test_get_embedding_returns_none_when_no_face(self, monkeypatch: pytest.MonkeyPatch) -> None:
        analyzer = FaceAnalyzer()

        class DummyApp:
            @staticmethod
            def get(_: np.ndarray) -> list[object]:
                return []

        analyzer._app = DummyApp()
        monkeypatch.setattr(analyzer, "_ensure_model", lambda: None)

        image = np.zeros((64, 64, 3), dtype=np.uint8)
        embedding = analyzer.get_embedding(image)
        assert embedding is None

    def test_compare_faces_aliases_analyze(self, monkeypatch: pytest.MonkeyPatch) -> None:
        analyzer = FaceAnalyzer()
        expected = FaceAnalysisResult(
            selfie_faces=[],
            id_faces=[],
            similarity=None,
            match=False,
            reason_codes=[],
            evidence={},
        )

        monkeypatch.setattr(analyzer, "analyze", lambda _selfie, _id: expected)

        result = analyzer.compare_faces(
            np.zeros((32, 32, 3), dtype=np.uint8),
            np.zeros((32, 32, 3), dtype=np.uint8),
        )
        assert result is expected


class TestDocumentAnalyzer:
    """Tests for DocumentAnalyzer alias behavior."""

    def _mock_analysis_result(self) -> DocumentAnalysisResult:
        return DocumentAnalysisResult(
            text_boxes=[],
            full_text="ABC123",
            avg_confidence=0.95,
            anomalies=[],
            reason_codes=[],
            evidence={},
            doc_score=0.1,
            detected=True,
            ocr_results=[OCRResult(text="ABC123", confidence=0.95, bbox=(0, 0, 10, 10))],
            overall_confidence=0.95,
        )

    def test_extract_text_returns_ocr_results(
        self, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        analyzer = DocumentAnalyzer()
        expected = self._mock_analysis_result()
        monkeypatch.setattr(analyzer, "analyze", lambda _image, lang=None: expected)

        result = analyzer.extract_text(np.zeros((64, 64, 3), dtype=np.uint8))
        assert len(result) == 1
        assert result[0].text == "ABC123"
        assert result[0].confidence == pytest.approx(0.95)

    def test_analyze_document_aliases_analyze(self, monkeypatch: pytest.MonkeyPatch) -> None:
        analyzer = DocumentAnalyzer()
        expected = self._mock_analysis_result()
        monkeypatch.setattr(analyzer, "analyze", lambda _image, lang=None: expected)

        result = analyzer.analyze_document(np.zeros((64, 64, 3), dtype=np.uint8))
        assert result is expected


class TestPADAnalyzer:
    """Tests for PADAnalyzer edge-case behavior."""

    def test_analyze_empty_frames(self) -> None:
        analyzer = PADAnalyzer()
        result = analyzer.analyze_frames([])

        assert result.pad_score == pytest.approx(0.5)
        assert result.reason_codes == ["PAD_NO_FRAMES"]
        assert result.frame_metrics == []

    def test_analyze_frames_returns_frame_metrics(self) -> None:
        analyzer = PADAnalyzer()
        frames = [np.zeros((64, 64, 3), dtype=np.uint8) for _ in range(3)]

        result = analyzer.analyze_frames(frames)

        assert len(result.frame_metrics) == len(frames)
        assert 0.0 <= result.pad_score <= 1.0


class TestFrameExtractor:
    """Tests for FrameExtractor control flow."""

    def test_initialization(self) -> None:
        extractor = FrameExtractor(target_fps=10.0)
        assert extractor.target_fps == 10.0

    def test_uniform_sample(self) -> None:
        extractor = FrameExtractor()
        assert extractor._uniform_sample(100, 10) == [0, 10, 20, 30, 40, 50, 60, 70, 80, 90]

    def test_extract_from_bytes_raises_for_invalid_video(self) -> None:
        extractor = FrameExtractor()
        with pytest.raises(ValueError, match="Could not open video"):
            extractor.extract_from_bytes(b"fake_video_data")
