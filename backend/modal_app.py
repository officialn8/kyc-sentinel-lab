"""
Modal serverless GPU functions for KYC Sentinel.

This module defines three GPU-accelerated functions:
1. extract_frames - Direct R2 access for video processing
2. analyze_face - Face detection and matching using InsightFace
3. analyze_document - OCR and document analysis using PaddleOCR

The hybrid storage pattern:
- Video extraction uses direct R2 credentials (for byte-range requests)
- Face/document analysis use presigned URLs (stateless, secure)
"""

import modal
from typing import Optional

# Define the GPU container image with all ML dependencies
# Note: InsightFace uses onnxruntime-gpu (works with T4 GPU)
# PaddleOCR uses CPU version (simpler, no CUDA driver issues)
gpu_image = (
    modal.Image.debian_slim(python_version="3.11")
    .apt_install("libgl1-mesa-glx", "libglib2.0-0")  # OpenCV dependencies
    .pip_install(
        # Core ML libraries
        "numpy>=1.24.0,<2.0.0",
        "opencv-python-headless>=4.8.0",
        "scipy>=1.11.0",
        # Face analysis (GPU-accelerated via onnxruntime)
        "insightface>=0.7.3",
        "onnxruntime-gpu>=1.16.0",
        # Document OCR - use explicit CPU-only version
        "paddlepaddle==3.0.0",  # CPU-only version
        "paddleocr>=2.8.0",
        # HTTP client for presigned URLs
        "httpx>=0.26.0",
        # S3 client for direct R2 access
        "boto3>=1.34.0",
    )
)

# Create Modal app
app = modal.App("kyc-sentinel", image=gpu_image)

# R2 credentials secret for video extraction worker
r2_secret = modal.Secret.from_name("r2-credentials")


@app.function(gpu="T4", timeout=300, secrets=[r2_secret])
def extract_frames(
    session_id: str,
    video_key: str,
    max_frames: int = 30,
) -> dict:
    """
    Extract frames from video stored in R2 using direct access.
    
    This function has direct R2 credentials to perform efficient
    byte-range requests for large video files.
    
    Args:
        session_id: Session identifier for logging
        video_key: R2 object key for the video file
        max_frames: Maximum number of frames to extract
    
    Returns:
        dict with:
            - frames: List of base64-encoded JPEG frames
            - metadata: Video metadata (fps, duration, resolution)
            - timestamps: Frame timestamps
    """
    import os
    import cv2
    import boto3
    import base64
    import tempfile
    import numpy as np
    
    # Get R2 credentials from environment (injected by Modal secret)
    r2_endpoint = os.environ["R2_ENDPOINT"]
    r2_access_key = os.environ["R2_ACCESS_KEY"]
    r2_secret_key = os.environ["R2_SECRET_KEY"]
    r2_bucket = os.environ["R2_BUCKET"]
    
    # Create S3 client for R2
    s3_client = boto3.client(
        "s3",
        endpoint_url=r2_endpoint,
        aws_access_key_id=r2_access_key,
        aws_secret_access_key=r2_secret_key,
    )
    
    # Download video to temp file
    with tempfile.NamedTemporaryFile(suffix=".mp4", delete=False) as f:
        s3_client.download_fileobj(r2_bucket, video_key, f)
        video_path = f.name
    
    try:
        # Open video
        cap = cv2.VideoCapture(video_path)
        if not cap.isOpened():
            raise ValueError(f"Could not open video: {video_key}")
        
        # Get video metadata
        fps = cap.get(cv2.CAP_PROP_FPS)
        total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
        width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
        duration = total_frames / fps if fps > 0 else 0
        
        # Uniform sampling
        if total_frames <= max_frames:
            indices = list(range(total_frames))
        else:
            step = total_frames / max_frames
            indices = [int(i * step) for i in range(max_frames)]
        
        # Extract frames
        frames_b64 = []
        timestamps = []
        
        for idx in indices:
            cap.set(cv2.CAP_PROP_POS_FRAMES, idx)
            ret, frame = cap.read()
            if ret:
                # Encode frame as JPEG
                _, buffer = cv2.imencode('.jpg', frame, [cv2.IMWRITE_JPEG_QUALITY, 85])
                frames_b64.append(base64.b64encode(buffer).decode('utf-8'))
                timestamps.append(idx / fps if fps > 0 else 0.0)
        
        cap.release()
        
        return {
            "session_id": session_id,
            "frames": frames_b64,
            "metadata": {
                "fps": fps,
                "total_frames": total_frames,
                "width": width,
                "height": height,
                "duration": duration,
            },
            "timestamps": timestamps,
        }
    
    finally:
        # Cleanup temp file
        import os as os_module
        os_module.unlink(video_path)


@app.function(gpu="T4", timeout=120)
def analyze_face(
    selfie_url: str,
    id_url: str,
    similarity_threshold: float = 0.45,
) -> dict:
    """
    Analyze faces in selfie and ID images using InsightFace.
    
    Uses presigned URLs to download images (stateless, no credentials needed).
    
    Args:
        selfie_url: Presigned URL to download selfie image
        id_url: Presigned URL to download ID image
        similarity_threshold: Threshold for face matching (0-1)
    
    Returns:
        dict with:
            - similarity: Cosine similarity between faces (or None)
            - match: Whether faces match
            - reason_codes: List of detected issues
            - evidence: Additional analysis data
            - selfie_crop_b64: Base64-encoded face crop from selfie
            - id_crop_b64: Base64-encoded face crop from ID
            - selfie_embedding: 512d face embedding from selfie
    """
    import cv2
    import httpx
    import base64
    import numpy as np
    from insightface.app import FaceAnalysis
    
    # Initialize InsightFace
    face_app = FaceAnalysis(
        name="buffalo_l",
        providers=["CUDAExecutionProvider", "CPUExecutionProvider"],
    )
    face_app.prepare(ctx_id=0, det_size=(640, 640))
    
    def download_image(url: str) -> np.ndarray:
        """Download image from presigned URL."""
        response = httpx.get(url, timeout=30.0)
        response.raise_for_status()
        nparr = np.frombuffer(response.content, np.uint8)
        img = cv2.imdecode(nparr, cv2.IMREAD_COLOR)
        
        # Resize if too large (>4000px on any dimension)
        MAX_DIM = 4000
        h, w = img.shape[:2]
        if h > MAX_DIM or w > MAX_DIM:
            scale = MAX_DIM / max(h, w)
            img = cv2.resize(img, (int(w * scale), int(h * scale)), interpolation=cv2.INTER_AREA)
        
        return img
    
    def detect_faces(image: np.ndarray) -> list:
        """Detect faces and extract embeddings."""
        faces = face_app.get(image)
        results = []
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
            
            results.append({
                "bbox": bbox,
                "embedding": face.embedding,
                "confidence": float(face.det_score),
                "crop": crop,
            })
        return results
    
    # Download images
    selfie_img = download_image(selfie_url)
    id_img = download_image(id_url)
    
    # Detect faces
    selfie_faces = detect_faces(selfie_img)
    id_faces = detect_faces(id_img)
    
    # Build results
    reason_codes = []
    evidence = {}
    similarity = None
    match = False
    selfie_crop_b64 = None
    id_crop_b64 = None
    selfie_embedding = None
    
    # Check for missing/multiple faces
    if not selfie_faces:
        reason_codes.append("FACE_NOT_DETECTED_SELFIE")
        evidence["selfie_face_count"] = 0
    else:
        evidence["selfie_face_count"] = len(selfie_faces)
        # Encode selfie crop
        best_selfie = max(selfie_faces, key=lambda f: f["confidence"])
        _, buffer = cv2.imencode('.jpg', best_selfie["crop"])
        selfie_crop_b64 = base64.b64encode(buffer).decode('utf-8')
        selfie_embedding = best_selfie["embedding"].tolist()
        
        if len(selfie_faces) > 1:
            reason_codes.append("MULTIPLE_FACES_SELFIE")
    
    if not id_faces:
        reason_codes.append("FACE_NOT_DETECTED_ID")
        evidence["id_face_count"] = 0
    else:
        evidence["id_face_count"] = len(id_faces)
        # Encode ID crop
        best_id = max(id_faces, key=lambda f: f["confidence"])
        _, buffer = cv2.imencode('.jpg', best_id["crop"])
        id_crop_b64 = base64.b64encode(buffer).decode('utf-8')
    
    # Compute similarity if both have faces
    if selfie_faces and id_faces:
        best_selfie = max(selfie_faces, key=lambda f: f["confidence"])
        best_id = max(id_faces, key=lambda f: f["confidence"])
        
        # Cosine similarity (embeddings are normalized)
        similarity = float(np.dot(best_selfie["embedding"], best_id["embedding"]))
        evidence["face_similarity"] = similarity
        
        match = similarity >= similarity_threshold
        if not match:
            reason_codes.append("FACE_MISMATCH")
            evidence["similarity_threshold"] = similarity_threshold
    
    return {
        "similarity": similarity,
        "match": match,
        "reason_codes": reason_codes,
        "evidence": evidence,
        "selfie_crop_b64": selfie_crop_b64,
        "id_crop_b64": id_crop_b64,
        "selfie_embedding": selfie_embedding,
    }


@app.function(gpu="T4", timeout=120)
def analyze_document(
    id_url: str,
    min_confidence_threshold: float = 0.7,
    low_confidence_threshold: float = 0.5,
) -> dict:
    """
    Analyze ID document using PaddleOCR.
    
    Uses presigned URL to download image (stateless, no credentials needed).
    
    Args:
        id_url: Presigned URL to download ID image
        min_confidence_threshold: Minimum confidence to accept text
        low_confidence_threshold: Threshold to flag low confidence
    
    Returns:
        dict with:
            - text_boxes: List of detected text regions
            - full_text: Concatenated text content
            - avg_confidence: Average OCR confidence
            - reason_codes: List of detected issues
            - evidence: Additional analysis data
            - doc_score: Suspicion score (0-1, higher = more suspicious)
    """
    import cv2
    import httpx
    import numpy as np
    from statistics import mean, stdev
    from paddleocr import PaddleOCR
    
    def download_image(url: str) -> np.ndarray:
        """Download image from presigned URL."""
        response = httpx.get(url, timeout=30.0)
        response.raise_for_status()
        nparr = np.frombuffer(response.content, np.uint8)
        img = cv2.imdecode(nparr, cv2.IMREAD_COLOR)
        
        # Resize if too large (>4000px on any dimension)
        MAX_DIM = 4000
        h, w = img.shape[:2]
        if h > MAX_DIM or w > MAX_DIM:
            scale = MAX_DIM / max(h, w)
            img = cv2.resize(img, (int(w * scale), int(h * scale)), interpolation=cv2.INTER_AREA)
        
        return img
    
    # Download image
    id_img = download_image(id_url)
    
    reason_codes = []
    evidence = {}
    anomalies = []
    text_boxes = []
    confidences = []
    
    # Initialize PaddleOCR (v2.8+ API with CPU)
    import os
    os.environ["CUDA_VISIBLE_DEVICES"] = ""  # Force CPU mode
    
    ocr = PaddleOCR(
        use_textline_orientation=True,
        lang="en",
        device="cpu",
    )
    result = ocr.predict(id_img)
    
    if not result or len(result) == 0:
        reason_codes.append("DOC_OCR_LOW_CONFIDENCE")
        evidence["ocr_result"] = "no_text_detected"
        return {
            "text_boxes": [],
            "full_text": "",
            "avg_confidence": 0.0,
            "reason_codes": reason_codes,
            "evidence": evidence,
            "doc_score": 1.0,
        }
    
    # Parse v2.8+ format
    ocr_data = result[0]
    rec_texts = getattr(ocr_data, 'rec_texts', []) if hasattr(ocr_data, 'rec_texts') else ocr_data.get('rec_texts', [])
    rec_scores = getattr(ocr_data, 'rec_scores', []) if hasattr(ocr_data, 'rec_scores') else ocr_data.get('rec_scores', [])
    rec_polys = getattr(ocr_data, 'rec_polys', []) if hasattr(ocr_data, 'rec_polys') else ocr_data.get('rec_polys', [])
    
    if not rec_polys:
        rec_polys = getattr(ocr_data, 'rec_boxes', []) if hasattr(ocr_data, 'rec_boxes') else ocr_data.get('rec_boxes', [])
    
    if not rec_texts:
        reason_codes.append("DOC_OCR_LOW_CONFIDENCE")
        evidence["ocr_result"] = "no_text_detected"
        return {
            "text_boxes": [],
            "full_text": "",
            "avg_confidence": 0.0,
            "reason_codes": reason_codes,
            "evidence": evidence,
            "doc_score": 1.0,
        }
    
    for i, text in enumerate(rec_texts):
        confidence = float(rec_scores[i]) if i < len(rec_scores) else 0.0
        bbox = rec_polys[i] if i < len(rec_polys) else [[0, 0], [0, 0], [0, 0], [0, 0]]
        
        if hasattr(bbox, 'tolist'):
            bbox = bbox.tolist()
        
        if len(bbox) == 4 and not isinstance(bbox[0], (list, tuple)):
            x1, y1, x2, y2 = bbox
            bbox = [[x1, y1], [x2, y1], [x2, y2], [x1, y2]]
        
        xs = [p[0] for p in bbox]
        ys = [p[1] for p in bbox]
        center = (int(mean(xs)), int(mean(ys)))
        
        text_boxes.append({
            "text": text,
            "confidence": confidence,
            "bbox": bbox,
            "center": center,
        })
        confidences.append(confidence)
    
    if not text_boxes:
        reason_codes.append("DOC_OCR_LOW_CONFIDENCE")
        evidence["ocr_result"] = "no_text_detected"
        return {
            "text_boxes": [],
            "full_text": "",
            "avg_confidence": 0.0,
            "reason_codes": reason_codes,
            "evidence": evidence,
            "doc_score": 1.0,
        }
    
    avg_confidence = mean(confidences) if confidences else 0.0
    evidence["avg_ocr_confidence"] = avg_confidence
    evidence["text_box_count"] = len(text_boxes)
    
    # Check for low confidence regions
    low_conf_boxes = [tb for tb in text_boxes if tb["confidence"] < low_confidence_threshold]
    if low_conf_boxes:
        reason_codes.append("DOC_OCR_LOW_CONFIDENCE")
        evidence["low_confidence_regions"] = len(low_conf_boxes)
        anomalies.append({
            "type": "low_confidence",
            "boxes": [{"text": tb["text"], "confidence": tb["confidence"]} for tb in low_conf_boxes],
        })
    
    # Check font consistency (using height variance)
    if len(text_boxes) >= 3:
        heights = []
        for tb in text_boxes:
            ys = [p[1] for p in tb["bbox"]]
            heights.append(max(ys) - min(ys))
        
        if len(heights) >= 2:
            height_std = stdev(heights)
            height_mean = mean(heights)
            cv_val = height_std / height_mean if height_mean > 0 else 0
            
            if cv_val > 0.5:
                reason_codes.append("DOC_FONT_INCONSISTENT")
                anomalies.append({
                    "type": "font_inconsistency",
                    "height_cv": cv_val,
                })
                evidence["font_inconsistency"] = True
    
    # Check text alignment
    if len(text_boxes) >= 3:
        left_edges = [min(p[0] for p in tb["bbox"]) for tb in text_boxes]
        sorted_edges = sorted(left_edges)
        edge_diffs = [sorted_edges[i + 1] - sorted_edges[i] for i in range(len(sorted_edges) - 1)]
        
        if edge_diffs:
            max_gap = max(edge_diffs)
            median_gap = sorted(edge_diffs)[len(edge_diffs) // 2]
            
            if median_gap > 0 and max_gap > median_gap * 5:
                reason_codes.append("DOC_TEXT_MISALIGNED")
                anomalies.append({
                    "type": "alignment_anomaly",
                    "max_gap": max_gap,
                    "median_gap": median_gap,
                })
                evidence["alignment_issue"] = True
    
    # Check for edge artifacts
    gray = cv2.cvtColor(id_img, cv2.COLOR_BGR2GRAY)
    edges = cv2.Canny(gray, 50, 150)
    contours, _ = cv2.findContours(edges, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    
    suspicious_contours = 0
    for contour in contours:
        epsilon = 0.02 * cv2.arcLength(contour, True)
        approx = cv2.approxPolyDP(contour, epsilon, True)
        
        if len(approx) == 4:
            area = cv2.contourArea(contour)
            img_area = id_img.shape[0] * id_img.shape[1]
            if 0.05 < area / img_area < 0.4:
                suspicious_contours += 1
    
    if suspicious_contours >= 2:
        reason_codes.append("DOC_EDGE_ARTIFACTS")
        anomalies.append({
            "type": "edge_artifacts",
            "suspicious_regions": suspicious_contours,
        })
        evidence["edge_artifacts"] = True
    
    # Compute document suspicion score
    confidence_score = 1.0 - avg_confidence
    anomaly_penalty = min(len(anomalies) * 0.15, 0.45)
    low_conf_ratio = len(low_conf_boxes) / max(len(text_boxes), 1)
    low_conf_penalty = low_conf_ratio * 0.3
    doc_score = min(max(confidence_score * 0.4 + anomaly_penalty + low_conf_penalty, 0.0), 1.0)
    
    return {
        "text_boxes": text_boxes,
        "full_text": " ".join(tb["text"] for tb in text_boxes),
        "avg_confidence": avg_confidence,
        "reason_codes": reason_codes,
        "evidence": evidence,
        "doc_score": doc_score,
        "anomalies": anomalies,
    }


# Local entrypoint for testing
@app.local_entrypoint()
def main():
    """Test the Modal functions locally."""
    print("KYC Sentinel Modal App initialized.")
    print("Functions available:")
    print("  - extract_frames (GPU, direct R2 access)")
    print("  - analyze_face (GPU, presigned URLs)")
    print("  - analyze_document (GPU, presigned URLs)")

