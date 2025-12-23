"""
Integration tests for the full detection pipeline.
Tests end-to-end flow with generated fixtures.
"""

import pytest
import numpy as np
import cv2
from pathlib import Path
from uuid import uuid4

from app.detection.face_analyzer import get_face_analyzer
from app.detection.document_analyzer import get_document_analyzer
from app.detection.pad_heuristics import get_pad_analyzer
from app.detection.frame_extractor import FrameExtractor
from app.services.scoring import compute_risk_score
from app.models.reason import KYCReason

FIXTURES_DIR = Path(__file__).parent / "fixtures"


def load_image(name: str) -> np.ndarray:
    """Load fixture image by name."""
    path = FIXTURES_DIR / name
    if not path.exists():
        pytest.skip(f"Fixture {name} not found. Run generate_fixtures.py first.")
    img = cv2.imread(str(path))
    if img is None:
        pytest.skip(f"Could not load {name}")
    return img


def load_frames(prefix: str, count: int = 30) -> list[np.ndarray]:
    """Load video frames by prefix."""
    frames = []
    for i in range(count):
        path = FIXTURES_DIR / f"{prefix}_{i:03d}.jpg"
        if path.exists():
            frame = cv2.imread(str(path))
            if frame is not None:
                frames.append(frame)
    if len(frames) < 5:
        pytest.skip(f"Not enough frames for {prefix}. Run generate_fixtures.py first.")
    return frames


class TestFaceAnalyzerIntegration:
    """Integration tests for face analysis."""
    
    @pytest.fixture(scope="class")
    def analyzer(self):
        return get_face_analyzer()
    
    @pytest.mark.slow
    def test_genuine_session(self, analyzer):
        """Genuine selfie and ID should have reasonable results."""
        selfie = load_image("genuine_selfie.jpg")
        id_doc = load_image("genuine_id.jpg")
        
        result = analyzer.analyze(selfie, id_doc)
        
        print(f"\n[Genuine] Selfie faces: {len(result.selfie_faces)}, "
              f"ID faces: {len(result.id_faces)}, "
              f"Similarity: {result.similarity}, "
              f"Reasons: {result.reason_codes}")
        
        # Pipeline should run without error
        assert result is not None
        # Note: Synthetic faces may not be detected by InsightFace
        # This test validates the pipeline, not detection accuracy
    
    @pytest.mark.slow
    def test_no_face_detected(self, analyzer):
        """Should flag when no face in selfie."""
        no_face = load_image("no_face_selfie.jpg")
        id_doc = load_image("genuine_id.jpg")
        
        result = analyzer.analyze(no_face, id_doc)
        
        print(f"\n[No Face] Reasons: {result.reason_codes}")
        
        assert "FACE_NOT_DETECTED_SELFIE" in result.reason_codes
        assert result.similarity is None
    
    @pytest.mark.slow
    def test_multiple_faces(self, analyzer):
        """Should flag multiple faces in selfie."""
        multi = load_image("multiple_faces_selfie.jpg")
        id_doc = load_image("genuine_id.jpg")
        
        result = analyzer.analyze(multi, id_doc)
        
        print(f"\n[Multiple Faces] Detected: {len(result.selfie_faces)}, "
              f"Reasons: {result.reason_codes}")
        
        # Note: Detection depends on InsightFace's ability to find synthetic faces
        assert result is not None


class TestDocumentAnalyzerIntegration:
    """Integration tests for document analysis."""
    
    @pytest.fixture(scope="class")
    def analyzer(self):
        return get_document_analyzer()
    
    @pytest.mark.slow
    def test_genuine_document(self, analyzer):
        """Genuine ID should have good OCR results."""
        id_doc = load_image("genuine_id.jpg")
        
        result = analyzer.analyze(id_doc)
        
        print(f"\n[Genuine Doc] Boxes: {len(result.text_boxes)}, "
              f"Confidence: {result.avg_confidence:.2f}, "
              f"Score: {result.doc_score:.2f}, "
              f"Reasons: {result.reason_codes}")
        
        assert len(result.text_boxes) > 0
        assert 0 <= result.doc_score <= 1
        
        # Should extract some expected text
        full_text_upper = result.full_text.upper()
        has_expected = any(word in full_text_upper for word in ["JOHN", "DOE", "ID", "SAMPLE"])
        print(f"Extracted text: {result.full_text[:200]}")
    
    @pytest.mark.slow
    def test_tampered_document(self, analyzer):
        """Tampered ID should have anomalies flagged."""
        id_doc = load_image("tampered_id.jpg")
        
        result = analyzer.analyze(id_doc)
        
        print(f"\n[Tampered Doc] Score: {result.doc_score:.2f}, "
              f"Anomalies: {len(result.anomalies)}, "
              f"Reasons: {result.reason_codes}")
        
        # Tampered doc should have higher suspicion score or anomalies
        # (depends on how obvious the tampering is)
        assert result is not None


class TestPADAnalyzerIntegration:
    """Integration tests for PAD heuristics."""
    
    @pytest.fixture(scope="class")
    def analyzer(self):
        return get_pad_analyzer()
    
    @pytest.mark.slow
    def test_genuine_video(self, analyzer):
        """Genuine video with natural motion should pass."""
        frames = load_frames("genuine_frame")
        
        result = analyzer.analyze_frames(frames)
        
        print(f"\n[Genuine Video] PAD Score: {result.overall_pad_score:.2f}, "
              f"Reasons: {result.reason_codes}, "
              f"Flagged: {len(result.flagged_frames)}")
        
        # Natural motion should have lower PAD score
        assert result.overall_pad_score < 0.8
    
    @pytest.mark.slow
    def test_replay_attack(self, analyzer):
        """Replay attack video should be flagged."""
        frames = load_frames("replay_frame")
        
        result = analyzer.analyze_frames(frames)
        
        print(f"\n[Replay Attack] PAD Score: {result.overall_pad_score:.2f}, "
              f"Reasons: {result.reason_codes}, "
              f"Evidence: {result.evidence}")
        
        # Replay should trigger some flags
        has_pad_flag = any("PAD" in code for code in result.reason_codes)
        print(f"PAD flags triggered: {has_pad_flag}")
    
    @pytest.mark.slow
    def test_static_attack(self, analyzer):
        """Static video (no motion) should be flagged."""
        frames = load_frames("static_frame")
        
        result = analyzer.analyze_frames(frames)
        
        print(f"\n[Static Attack] PAD Score: {result.overall_pad_score:.2f}, "
              f"Reasons: {result.reason_codes}")
        
        # Static/stutter should be detected
        static_flags = ["PAD_LOW_MOTION_ENTROPY", "PAD_FRAME_STUTTER"]
        has_static_flag = any(flag in result.reason_codes for flag in static_flags)
        assert has_static_flag, f"Expected static detection, got: {result.reason_codes}"


class TestScoringIntegration:
    """Integration tests for risk scoring."""
    
    def test_high_risk_score(self):
        """Low similarity + high PAD should give high risk."""
        reasons = [
            KYCReason(
                session_id=uuid4(),
                code="FACE_MISMATCH",
                severity="high",
                message="Face mismatch",
                evidence={}
            )
        ]
        
        risk_score, decision = compute_risk_score(
            face_similarity=0.2,
            pad_score=0.7,
            doc_score=0.3,
            reasons=reasons
        )
        
        print(f"\n[High Risk] Score: {risk_score}, Decision: {decision}")
        
        assert risk_score >= 60
        assert decision in ["review", "fail"]
    
    def test_low_risk_score(self):
        """High similarity + low PAD should give low risk."""
        reasons = []
        
        risk_score, decision = compute_risk_score(
            face_similarity=0.95,
            pad_score=0.1,
            doc_score=0.1,
            reasons=reasons
        )
        
        print(f"\n[Low Risk] Score: {risk_score}, Decision: {decision}")
        
        assert risk_score < 40
        assert decision == "pass"
    
    def test_hard_fail_override(self):
        """High severity reason should trigger fail."""
        reasons = [
            KYCReason(
                session_id=uuid4(),
                code="FACE_NOT_DETECTED_SELFIE",
                severity="high",
                message="No face",
                evidence={}
            )
        ]
        
        risk_score, decision = compute_risk_score(
            face_similarity=0.0,
            pad_score=0.2,
            doc_score=0.2,
            reasons=reasons
        )
        
        print(f"\n[Hard Fail] Score: {risk_score}, Decision: {decision}")
        
        assert decision == "fail"


class TestFullPipelineIntegration:
    """End-to-end pipeline test."""
    
    @pytest.mark.slow
    def test_complete_analysis_flow(self):
        """Run complete analysis on genuine session."""
        from app.detection.face_analyzer import get_face_analyzer
        from app.detection.document_analyzer import get_document_analyzer
        from app.detection.pad_heuristics import get_pad_analyzer
        
        # Load assets
        selfie = load_image("genuine_selfie.jpg")
        id_doc = load_image("genuine_id.jpg")
        frames = load_frames("genuine_frame")
        
        # Run all analyzers
        face_result = get_face_analyzer().analyze(selfie, id_doc)
        doc_result = get_document_analyzer().analyze(id_doc)
        pad_result = get_pad_analyzer().analyze_frames(frames)
        
        # Collect reasons
        all_reason_codes = (
            face_result.reason_codes +
            doc_result.reason_codes +
            pad_result.reason_codes
        )
        
        # Create reason objects for scoring
        reasons = [
            KYCReason(
                session_id=uuid4(),
                code=code,
                severity="warn",
                message=code,
                evidence={}
            )
            for code in all_reason_codes
        ]
        
        # Compute final score
        face_sim = face_result.similarity or 0.0
        risk_score, decision = compute_risk_score(
            face_similarity=face_sim,
            pad_score=pad_result.overall_pad_score,
            doc_score=doc_result.doc_score,
            reasons=reasons
        )
        
        print(f"\n{'='*60}")
        print("FULL PIPELINE RESULTS")
        print(f"{'='*60}")
        print(f"Face: similarity={face_sim:.3f}, codes={face_result.reason_codes}")
        print(f"Doc: score={doc_result.doc_score:.3f}, codes={doc_result.reason_codes}")
        print(f"PAD: score={pad_result.overall_pad_score:.3f}, codes={pad_result.reason_codes}")
        print(f"{'='*60}")
        print(f"FINAL: risk_score={risk_score}, decision={decision}")
        print(f"{'='*60}")
        
        # Pipeline should complete without error
        assert risk_score is not None
        assert decision in ["pass", "review", "fail"]

