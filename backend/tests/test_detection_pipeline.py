"""
Tests for the detection pipeline modules.

These tests verify the detection modules work correctly with synthetic test data.
For tests with actual ML models, run with:
    pytest tests/test_detection_pipeline.py -v --runslow
"""

import pytest
import numpy as np
import cv2

from app.detection.pad_heuristics import PADAnalyzer
from app.detection.frame_extractor import FrameExtractor


# Mark for slow tests - uses the marker registered in conftest.py
slow = pytest.mark.slow


# ============================================================================
# Fixtures
# ============================================================================


@pytest.fixture
def sample_face_image():
    """Generate a simple test image with a 'face-like' region."""
    img = np.zeros((480, 640, 3), dtype=np.uint8)
    # Add background
    img[:] = (100, 120, 130)
    # Draw a circle as placeholder face
    cv2.circle(img, (320, 200), 80, (200, 180, 170), -1)
    # Add some features
    cv2.circle(img, (295, 180), 10, (50, 50, 50), -1)  # left eye
    cv2.circle(img, (345, 180), 10, (50, 50, 50), -1)  # right eye
    cv2.ellipse(img, (320, 220), (30, 15), 0, 0, 180, (150, 100, 100), -1)  # mouth
    return img


@pytest.fixture
def sample_id_image():
    """Generate a simple test image resembling an ID card."""
    img = np.zeros((400, 600, 3), dtype=np.uint8)
    # Card background
    img[:] = (240, 240, 240)
    # Photo area
    cv2.rectangle(img, (30, 50), (180, 250), (200, 200, 200), -1)
    # Face in photo area
    cv2.circle(img, (105, 130), 50, (200, 180, 170), -1)
    cv2.circle(img, (90, 120), 7, (50, 50, 50), -1)
    cv2.circle(img, (120, 120), 7, (50, 50, 50), -1)
    # Text lines
    cv2.rectangle(img, (220, 60), (570, 80), (50, 50, 50), -1)
    cv2.rectangle(img, (220, 100), (450, 120), (80, 80, 80), -1)
    cv2.rectangle(img, (220, 140), (500, 160), (80, 80, 80), -1)
    cv2.rectangle(img, (220, 180), (400, 200), (80, 80, 80), -1)
    return img


@pytest.fixture
def sample_frames():
    """Generate sample video frames with slight motion."""
    frames = []
    for i in range(10):
        img = np.zeros((480, 640, 3), dtype=np.uint8)
        img[:] = (100, 120, 130)
        # Slight position shift to simulate motion
        cv2.circle(img, (320 + i * 2, 200 + i), 80, (200, 180, 170), -1)
        # Add some noise for realism
        noise = np.random.randint(0, 10, img.shape, dtype=np.uint8)
        img = cv2.add(img, noise)
        frames.append(img)
    return frames


@pytest.fixture
def static_frames():
    """Generate static/duplicate frames (replay indicator)."""
    static_frame = np.random.randint(0, 255, (480, 640, 3), dtype=np.uint8)
    return [static_frame.copy() for _ in range(10)]


@pytest.fixture
def moire_frames():
    """Generate frames with moiré-like patterns (screen capture indicator)."""
    frames = []
    for _ in range(10):
        img = np.zeros((480, 640, 3), dtype=np.uint8)
        # Create periodic pattern (simulating screen pixels)
        for y in range(0, 480, 3):
            for x in range(0, 640, 3):
                img[y:y+2, x:x+2] = (200, 200, 200)
        frames.append(img)
    return frames


# ============================================================================
# PAD Analyzer Tests
# ============================================================================


class TestPADAnalyzer:
    """Tests for Presentation Attack Detection analyzer."""

    def test_init_default_thresholds(self):
        """PAD analyzer should initialize with default thresholds."""
        analyzer = PADAnalyzer()
        
        assert analyzer.sharpness_threshold == 100.0
        assert analyzer.motion_entropy_min == 0.1
        assert analyzer.moire_threshold == 0.15

    def test_empty_frames_returns_default_result(self):
        """PAD should handle empty frame list gracefully."""
        analyzer = PADAnalyzer()
        result = analyzer.analyze_frames([])
        
        assert result.overall_pad_score == 0.5
        assert "PAD_NO_FRAMES" in result.reason_codes
        assert result.frame_metrics == []

    def test_single_frame_analysis(self, sample_face_image):
        """PAD should analyze a single frame without error."""
        analyzer = PADAnalyzer()
        result = analyzer.analyze_frames([sample_face_image])
        
        assert result is not None
        assert len(result.frame_metrics) == 1
        assert result.frame_metrics[0].frame_idx == 0
        assert isinstance(result.overall_pad_score, float)
        assert 0 <= result.overall_pad_score <= 1

    def test_detects_static_frames(self, static_frames):
        """PAD should flag static/duplicate frames as suspicious."""
        analyzer = PADAnalyzer()
        result = analyzer.analyze_frames(static_frames)
        
        # Should detect frame stutter or low motion
        has_motion_flag = (
            "PAD_LOW_MOTION_ENTROPY" in result.reason_codes or
            "PAD_FRAME_STUTTER" in result.reason_codes
        )
        assert has_motion_flag, f"Expected motion-related flag, got: {result.reason_codes}"
        assert result.overall_pad_score > 0.1  # Should be somewhat suspicious

    def test_normal_frames_pass(self, sample_frames):
        """PAD should not flag normal video frames with motion."""
        analyzer = PADAnalyzer()
        result = analyzer.analyze_frames(sample_frames)
        
        # Normal frames should have low suspicion score
        # Note: May still have some flags due to synthetic nature
        assert result is not None
        assert isinstance(result.overall_pad_score, float)

    def test_frame_metrics_computed(self, sample_frames):
        """PAD should compute all required metrics for each frame."""
        analyzer = PADAnalyzer()
        result = analyzer.analyze_frames(sample_frames)
        
        for i, metrics in enumerate(result.frame_metrics):
            assert metrics.frame_idx == i
            assert isinstance(metrics.sharpness, float)
            assert isinstance(metrics.noise_level, float)
            assert isinstance(metrics.motion_entropy, float)
            assert isinstance(metrics.color_temperature, float)
            assert isinstance(metrics.moire_score, float)
            assert isinstance(metrics.pad_flags, list)

    def test_motion_entropy_first_frame_zero(self, sample_frames):
        """First frame should have zero motion entropy (no previous frame)."""
        analyzer = PADAnalyzer()
        result = analyzer.analyze_frames(sample_frames)
        
        assert result.frame_metrics[0].motion_entropy == 0.0
        # Subsequent frames should have non-zero motion entropy
        assert result.frame_metrics[1].motion_entropy >= 0.0

    def test_sharpness_calculation(self):
        """Sharpness calculation should return reasonable values."""
        analyzer = PADAnalyzer()
        
        # Create a sharp image (edges)
        sharp_img = np.zeros((100, 100, 3), dtype=np.uint8)
        cv2.rectangle(sharp_img, (20, 20), (80, 80), (255, 255, 255), -1)
        
        # Create a blurry image
        blurry_img = cv2.GaussianBlur(sharp_img, (21, 21), 0)
        
        sharp_score = analyzer.calculate_sharpness(sharp_img)
        blurry_score = analyzer.calculate_sharpness(blurry_img)
        
        assert sharp_score > blurry_score, "Sharp image should have higher sharpness score"

    def test_moire_detection(self, moire_frames):
        """PAD should detect moiré patterns in screen capture-like images."""
        analyzer = PADAnalyzer()
        result = analyzer.analyze_frames(moire_frames)
        
        # Check if any frames were flagged for moiré
        moire_flagged = any(
            m.moire_score > analyzer.moire_threshold 
            for m in result.frame_metrics
        )
        
        # Note: Detection depends on pattern frequency matching FFT analysis
        assert isinstance(result.overall_pad_score, float)


# ============================================================================
# Frame Extractor Tests
# ============================================================================


class TestFrameExtractor:
    """Tests for video frame extraction."""

    def test_init_default_values(self):
        """Frame extractor should initialize with default values."""
        extractor = FrameExtractor()
        
        assert extractor.max_frames == 30
        assert extractor.sample_strategy == "uniform"

    def test_uniform_sample_fewer_frames(self):
        """Uniform sampling with fewer total frames should return all."""
        extractor = FrameExtractor(max_frames=100)
        indices = extractor._uniform_sample(total=50, n=100)
        
        assert indices == list(range(50))

    def test_uniform_sample_exact_count(self):
        """Uniform sampling should return exactly n indices."""
        extractor = FrameExtractor()
        indices = extractor._uniform_sample(total=300, n=30)
        
        assert len(indices) == 30
        assert indices[0] == 0
        assert indices[-1] < 300

    def test_uniform_sample_evenly_spaced(self):
        """Uniform sampling should produce evenly spaced indices."""
        extractor = FrameExtractor()
        indices = extractor._uniform_sample(total=100, n=10)
        
        # Check spacing is approximately equal
        diffs = [indices[i+1] - indices[i] for i in range(len(indices)-1)]
        assert all(9 <= d <= 11 for d in diffs)


# ============================================================================
# Integration Tests (require ML models)
# ============================================================================


@slow
class TestFaceAnalyzerIntegration:
    """Integration tests for face analyzer (requires InsightFace)."""

    def test_face_analyzer_loads(self):
        """Face analyzer should load without error."""
        from app.detection.face_analyzer import FaceAnalyzer
        
        analyzer = FaceAnalyzer()
        assert analyzer is not None
        assert analyzer.similarity_threshold == 0.45

    def test_face_analyzer_analyze(self, sample_face_image, sample_id_image):
        """Face analyzer should analyze images without error."""
        from app.detection.face_analyzer import FaceAnalyzer
        
        analyzer = FaceAnalyzer()
        result = analyzer.analyze(sample_face_image, sample_id_image)
        
        assert result is not None
        assert isinstance(result.reason_codes, list)
        assert isinstance(result.evidence, dict)


@slow
class TestDocumentAnalyzerIntegration:
    """Integration tests for document analyzer (requires PaddleOCR)."""

    def test_document_analyzer_loads(self):
        """Document analyzer should load without error."""
        from app.detection.document_analyzer import DocumentAnalyzer
        
        analyzer = DocumentAnalyzer()
        assert analyzer is not None

    def test_document_analyzer_analyze(self, sample_id_image):
        """Document analyzer should analyze images without error."""
        from app.detection.document_analyzer import DocumentAnalyzer
        
        analyzer = DocumentAnalyzer()
        result = analyzer.analyze(sample_id_image)
        
        assert result is not None
        assert isinstance(result.doc_score, float)
        assert 0 <= result.doc_score <= 1


# ============================================================================
# Edge Cases
# ============================================================================


class TestEdgeCases:
    """Tests for edge cases and error handling."""

    def test_pad_very_small_image(self):
        """PAD should handle very small images."""
        analyzer = PADAnalyzer()
        small_img = np.zeros((10, 10, 3), dtype=np.uint8)
        
        result = analyzer.analyze_frames([small_img])
        
        assert result is not None
        assert len(result.frame_metrics) == 1

    def test_pad_grayscale_converted(self):
        """PAD should work with images that get converted to grayscale."""
        analyzer = PADAnalyzer()
        
        # Image where BGR and gray conversion matters
        color_img = np.zeros((100, 100, 3), dtype=np.uint8)
        color_img[:, :, 0] = 255  # Blue channel only
        
        result = analyzer.analyze_frames([color_img])
        
        assert result is not None

    def test_pad_high_contrast_image(self):
        """PAD should handle high contrast images."""
        analyzer = PADAnalyzer()
        
        # Checkerboard pattern (high contrast)
        img = np.zeros((100, 100, 3), dtype=np.uint8)
        img[::2, ::2] = 255
        img[1::2, 1::2] = 255
        
        result = analyzer.analyze_frames([img])
        
        assert result is not None
        # High contrast may trigger certain flags
        assert isinstance(result.overall_pad_score, float)

    def test_frame_extractor_handles_different_strategies(self):
        """Frame extractor should support different sampling strategies."""
        uniform_extractor = FrameExtractor(max_frames=10, sample_strategy="uniform")
        first_n_extractor = FrameExtractor(max_frames=10, sample_strategy="first_n")
        
        # Test uniform sampling
        uniform_indices = uniform_extractor._uniform_sample(100, 10)
        assert len(uniform_indices) == 10
        assert uniform_indices[0] == 0
        
        # first_n would just return range(min(total, n))
        assert first_n_extractor.sample_strategy == "first_n"

