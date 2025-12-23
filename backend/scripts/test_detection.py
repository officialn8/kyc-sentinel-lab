#!/usr/bin/env python3
"""
Interactive detection testing script.

Usage:
    # Test with images
    python scripts/test_detection.py --selfie path/to/selfie.jpg --id path/to/id.jpg
    
    # Test with video
    python scripts/test_detection.py --video path/to/selfie.mp4
    
    # Test with fixtures
    python scripts/test_detection.py --fixtures
    
    # Save output
    python scripts/test_detection.py --selfie img.jpg --id id.jpg --output /tmp/results
"""

import argparse
import json
import sys
from pathlib import Path

import cv2
import numpy as np

# Add backend to path
sys.path.insert(0, str(Path(__file__).parent.parent))

from app.detection.face_analyzer import get_face_analyzer
from app.detection.document_analyzer import get_document_analyzer
from app.detection.pad_heuristics import get_pad_analyzer
from app.detection.frame_extractor import get_frame_extractor


def print_section(title: str):
    print(f"\n{'='*60}")
    print(f" {title}")
    print(f"{'='*60}")


def test_images(selfie_path: str, id_path: str, output_dir: str = None):
    """Run detection pipeline on image pair."""
    
    print_section("LOADING IMAGES")
    
    selfie = cv2.imread(selfie_path)
    id_doc = cv2.imread(id_path)
    
    if selfie is None:
        print(f"ERROR: Could not load selfie from {selfie_path}")
        return
    if id_doc is None:
        print(f"ERROR: Could not load ID from {id_path}")
        return
    
    print(f"Selfie: {selfie.shape} from {selfie_path}")
    print(f"ID:     {id_doc.shape} from {id_path}")
    
    # Face Analysis
    print_section("FACE ANALYSIS (InsightFace)")
    
    print("Loading model... ", end="", flush=True)
    face_analyzer = get_face_analyzer()
    print("done")
    
    print("Analyzing... ", end="", flush=True)
    face_result = face_analyzer.analyze(selfie, id_doc)
    print("done")
    
    print(f"\nSelfie faces detected: {len(face_result.selfie_faces)}")
    for i, face in enumerate(face_result.selfie_faces):
        print(f"  Face {i}: confidence={face.confidence:.3f}, bbox={face.bbox}")
    
    print(f"\nID faces detected: {len(face_result.id_faces)}")
    for i, face in enumerate(face_result.id_faces):
        print(f"  Face {i}: confidence={face.confidence:.3f}, bbox={face.bbox}")
    
    print(f"\nSimilarity: {face_result.similarity}")
    print(f"Match: {face_result.match}")
    print(f"Reason codes: {face_result.reason_codes}")
    print(f"Evidence: {json.dumps(face_result.evidence, indent=2, default=str)}")
    
    # Document Analysis
    print_section("DOCUMENT ANALYSIS (PaddleOCR)")
    
    print("Loading model... ", end="", flush=True)
    doc_analyzer = get_document_analyzer()
    print("done")
    
    print("Analyzing... ", end="", flush=True)
    doc_result = doc_analyzer.analyze(id_doc)
    print("done")
    
    print(f"\nText boxes found: {len(doc_result.text_boxes)}")
    print(f"Extracted text:")
    for tb in doc_result.text_boxes[:10]:
        print(f"  [{tb.confidence:.2f}] {tb.text}")
    if len(doc_result.text_boxes) > 10:
        print(f"  ... and {len(doc_result.text_boxes) - 10} more")
    
    print(f"\nAvg OCR confidence: {doc_result.avg_confidence:.2f}")
    print(f"Doc suspicion score: {doc_result.doc_score:.2f}")
    print(f"Reason codes: {doc_result.reason_codes}")
    print(f"Anomalies: {doc_result.anomalies}")
    
    # Summary
    print_section("SUMMARY")
    
    all_reasons = face_result.reason_codes + doc_result.reason_codes
    print(f"Total reason codes: {len(all_reasons)}")
    for code in all_reasons:
        print(f"  - {code}")
    
    # Save output if requested
    if output_dir:
        save_results(output_dir, face_result, doc_result, None)


def test_video(video_path: str, output_dir: str = None):
    """Run PAD analysis on video."""
    
    print_section("EXTRACTING FRAMES")
    
    print(f"Video: {video_path}")
    extractor = get_frame_extractor(max_frames=30)
    extraction = extractor.extract_from_path(video_path)
    
    print(f"Resolution: {extraction.resolution}")
    print(f"FPS: {extraction.fps:.1f}")
    print(f"Duration: {extraction.duration_seconds:.1f}s")
    print(f"Extracted frames: {len(extraction.frames)}")
    
    # PAD Analysis
    print_section("PAD ANALYSIS (Heuristics)")
    
    pad_analyzer = get_pad_analyzer()
    pad_result = pad_analyzer.analyze_frames(extraction.frames)
    
    print(f"\nOverall PAD score: {pad_result.overall_pad_score:.2f}")
    print(f"Reason codes: {pad_result.reason_codes}")
    print(f"Flagged frames: {pad_result.flagged_frames}")
    print(f"\nEvidence:")
    print(json.dumps(pad_result.evidence, indent=2, default=str))
    
    print(f"\nPer-frame metrics (first 10):")
    print(f"{'Frame':>6} {'Sharp':>8} {'Noise':>8} {'Motion':>8} {'Moire':>8} {'Flags'}")
    print("-" * 60)
    for fm in pad_result.frame_metrics[:10]:
        flags = ",".join(fm.pad_flags) if fm.pad_flags else "-"
        print(f"{fm.frame_idx:>6} {fm.sharpness:>8.1f} {fm.noise_level:>8.1f} "
              f"{fm.motion_entropy:>8.3f} {fm.moire_score:>8.3f} {flags}")
    
    if len(pad_result.frame_metrics) > 10:
        print(f"... and {len(pad_result.frame_metrics) - 10} more frames")


def test_fixtures():
    """Run tests on all fixture files."""
    fixtures_dir = Path(__file__).parent.parent / "tests" / "fixtures"
    
    if not fixtures_dir.exists():
        print(f"Fixtures directory not found: {fixtures_dir}")
        print("Run: python tests/fixtures/generate_fixtures.py")
        return
    
    print_section("TESTING FIXTURES")
    
    test_cases = [
        ("Genuine", "genuine_selfie.jpg", "genuine_id.jpg", "genuine_frame"),
        ("Mismatch", "mismatch_selfie.jpg", "mismatch_id.jpg", None),
        ("No Face", "no_face_selfie.jpg", "genuine_id.jpg", None),
        ("Tampered", "genuine_selfie.jpg", "tampered_id.jpg", None),
    ]
    
    results = []
    
    for name, selfie_name, id_name, frames_prefix in test_cases:
        print(f"\n--- {name} ---")
        
        selfie_path = fixtures_dir / selfie_name
        id_path = fixtures_dir / id_name
        
        if not selfie_path.exists() or not id_path.exists():
            print(f"  Skipped: missing files")
            continue
        
        selfie = cv2.imread(str(selfie_path))
        id_doc = cv2.imread(str(id_path))
        
        # Face analysis
        face_result = get_face_analyzer().analyze(selfie, id_doc)
        
        # Doc analysis
        doc_result = get_document_analyzer().analyze(id_doc)
        
        # PAD analysis (if frames exist)
        pad_score = 0.0
        pad_reasons = []
        if frames_prefix:
            frames = []
            for i in range(30):
                frame_path = fixtures_dir / f"{frames_prefix}_{i:03d}.jpg"
                if frame_path.exists():
                    frames.append(cv2.imread(str(frame_path)))
            
            if frames:
                pad_result = get_pad_analyzer().analyze_frames(frames)
                pad_score = pad_result.overall_pad_score
                pad_reasons = pad_result.reason_codes
        
        all_reasons = face_result.reason_codes + doc_result.reason_codes + pad_reasons
        
        print(f"  Face: {len(face_result.selfie_faces)} detected, "
              f"sim={face_result.similarity or 'N/A'}")
        print(f"  Doc: score={doc_result.doc_score:.2f}")
        print(f"  PAD: score={pad_score:.2f}")
        print(f"  Reasons: {all_reasons}")
        
        results.append({
            "name": name,
            "face_detected": len(face_result.selfie_faces) > 0,
            "similarity": face_result.similarity,
            "doc_score": doc_result.doc_score,
            "pad_score": pad_score,
            "reasons": all_reasons
        })
    
    print_section("RESULTS SUMMARY")
    print(f"{'Test Case':<15} {'Face':<6} {'Sim':<8} {'Doc':<8} {'PAD':<8} {'Reasons'}")
    print("-" * 70)
    for r in results:
        sim_str = f"{r['similarity']:.2f}" if r['similarity'] else "N/A"
        print(f"{r['name']:<15} {'Yes' if r['face_detected'] else 'No':<6} "
              f"{sim_str:<8} {r['doc_score']:<8.2f} {r['pad_score']:<8.2f} "
              f"{len(r['reasons'])}")


def save_results(output_dir: str, face_result, doc_result, pad_result):
    """Save analysis results to output directory."""
    output_path = Path(output_dir)
    output_path.mkdir(parents=True, exist_ok=True)
    
    # Save face crops
    for i, face in enumerate(face_result.selfie_faces):
        crop_path = output_path / f"selfie_face_{i}.jpg"
        cv2.imwrite(str(crop_path), face.crop)
        print(f"Saved: {crop_path}")
    
    for i, face in enumerate(face_result.id_faces):
        crop_path = output_path / f"id_face_{i}.jpg"
        cv2.imwrite(str(crop_path), face.crop)
        print(f"Saved: {crop_path}")
    
    # Save results JSON
    results = {
        "face": {
            "selfie_faces": len(face_result.selfie_faces),
            "id_faces": len(face_result.id_faces),
            "similarity": face_result.similarity,
            "match": face_result.match,
            "reason_codes": face_result.reason_codes,
        },
        "document": {
            "text_boxes": len(doc_result.text_boxes),
            "full_text": doc_result.full_text,
            "avg_confidence": doc_result.avg_confidence,
            "doc_score": doc_result.doc_score,
            "reason_codes": doc_result.reason_codes,
        }
    }
    
    if pad_result:
        results["pad"] = {
            "overall_score": pad_result.overall_pad_score,
            "reason_codes": pad_result.reason_codes,
            "flagged_frames": pad_result.flagged_frames,
        }
    
    results_path = output_path / "results.json"
    with open(results_path, "w") as f:
        json.dump(results, f, indent=2, default=str)
    print(f"Saved: {results_path}")


def main():
    parser = argparse.ArgumentParser(
        description="Test KYC detection pipeline",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
    python scripts/test_detection.py --selfie photo.jpg --id id.jpg
    python scripts/test_detection.py --video selfie.mp4
    python scripts/test_detection.py --fixtures
    python scripts/test_detection.py --selfie photo.jpg --id id.jpg --output /tmp/results
        """
    )
    
    parser.add_argument("--selfie", help="Path to selfie image")
    parser.add_argument("--id", help="Path to ID document image")
    parser.add_argument("--video", help="Path to selfie video")
    parser.add_argument("--fixtures", action="store_true", help="Test all fixtures")
    parser.add_argument("--output", help="Output directory for results")
    
    args = parser.parse_args()
    
    if args.fixtures:
        test_fixtures()
    elif args.selfie and args.id:
        test_images(args.selfie, args.id, args.output)
    elif args.video:
        test_video(args.video, args.output)
    else:
        parser.print_help()


if __name__ == "__main__":
    main()

