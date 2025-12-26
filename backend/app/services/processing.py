"""
Processing backend abstraction for easy migration to Modal.

This module provides the local processing backend that orchestrates the
detection pipeline using InsightFace, PaddleOCR, and OpenCV.
"""

from abc import abstractmethod
from dataclasses import dataclass
from typing import Protocol, Optional
from uuid import UUID

import cv2
import numpy as np

from app.config import settings


@dataclass
class FrameResult:
    """Result from frame extraction."""

    frame_count: int
    frame_keys: list[str]
    fps: float
    resolution: str


@dataclass
class FaceResult:
    """Result from face analysis."""

    selfie_detected: bool
    id_detected: bool
    similarity: float
    embedding: list[float] | None
    pad_score: float


@dataclass
class DocResult:
    """Result from document analysis."""

    detected: bool
    ocr_confidence: float
    template_match: float
    doc_score: float


class ProcessingBackend(Protocol):
    """Protocol for processing backend implementations."""

    @abstractmethod
    async def extract_frames(self, session_id: str, video_key: str) -> FrameResult:
        """Extract frames from a video."""
        ...

    @abstractmethod
    async def analyze_face(
        self, session_id: str, selfie_key: str, id_key: str
    ) -> FaceResult:
        """Analyze faces in selfie and ID images."""
        ...

    @abstractmethod
    async def analyze_document(self, session_id: str, id_key: str) -> DocResult:
        """Analyze document for OCR and tampering."""
        ...

    @abstractmethod
    async def process_session(self, session_id: str) -> None:
        """Orchestrate full session processing."""
        ...

    @abstractmethod
    async def generate_synthetic_session(
        self,
        session_id: str,
        attack_family: str,
        attack_severity: str,
    ) -> None:
        """Generate a synthetic session with attack artifacts."""
        ...


class LocalBackend:
    """FastAPI BackgroundTasks implementation using real ML/CV models.
    
    Orchestrates the detection pipeline:
    1. Download assets from storage
    2. Run face analysis with InsightFace
    3. Run document analysis with PaddleOCR
    4. Run PAD heuristics with OpenCV (if video)
    5. Compute risk scores and save results
    """

    def __init__(self) -> None:
        """Initialize the local backend."""
        # Lazy-load detection modules to avoid slow startup
        self._face_analyzer = None
        self._doc_analyzer = None
        self._pad_analyzer = None
        self._frame_extractor = None
        self._storage = None

    @property
    def storage(self):
        """Get storage service (lazy load)."""
        if self._storage is None:
            from app.services.storage import get_storage_service
            self._storage = get_storage_service()
        return self._storage

    @property
    def face_analyzer(self):
        """Get face analyzer (lazy load)."""
        if self._face_analyzer is None:
            from app.detection.face_analyzer import get_face_analyzer
            self._face_analyzer = get_face_analyzer()
        return self._face_analyzer

    @property
    def doc_analyzer(self):
        """Get document analyzer (lazy load)."""
        if self._doc_analyzer is None:
            from app.detection.document_analyzer import get_document_analyzer
            self._doc_analyzer = get_document_analyzer()
        return self._doc_analyzer

    @property
    def pad_analyzer(self):
        """Get PAD analyzer (lazy load)."""
        if self._pad_analyzer is None:
            from app.detection.pad_heuristics import get_pad_analyzer
            self._pad_analyzer = get_pad_analyzer()
        return self._pad_analyzer

    @property
    def frame_extractor(self):
        """Get frame extractor (lazy load)."""
        if self._frame_extractor is None:
            from app.detection.frame_extractor import get_frame_extractor
            self._frame_extractor = get_frame_extractor()
        return self._frame_extractor

    def _decode_image(self, img_bytes: bytes, max_dim: int = 2048) -> np.ndarray:
        """Decode image bytes to numpy array (BGR format) and resize if too large.
        
        Args:
            img_bytes: Raw image bytes
            max_dim: Maximum dimension (width or height). Images larger than this 
                     will be resized proportionally to fit within this limit.
        """
        nparr = np.frombuffer(img_bytes, np.uint8)
        img = cv2.imdecode(nparr, cv2.IMREAD_COLOR)
        
        if img is None:
            return img
            
        # Resize if image is too large (prevents memory issues and speeds up processing)
        h, w = img.shape[:2]
        if max(h, w) > max_dim:
            scale = max_dim / max(h, w)
            new_w = int(w * scale)
            new_h = int(h * scale)
            img = cv2.resize(img, (new_w, new_h), interpolation=cv2.INTER_AREA)
        
        return img

    def _is_video(self, asset_key: str) -> bool:
        """Check if asset is a video based on extension."""
        video_extensions = {'.mp4', '.mov', '.webm', '.avi', '.mkv'}
        return any(asset_key.lower().endswith(ext) for ext in video_extensions)

    async def extract_frames(self, session_id: str, video_key: str) -> FrameResult:
        """Extract frames from video."""
        video_bytes = await self.storage.download_file(video_key)
        extraction = self.frame_extractor.extract_from_bytes(video_bytes)

        # Store frames to storage
        frame_keys = []
        for idx, frame in enumerate(extraction.frames):
            key = f"sessions/{session_id}/frames/{idx:04d}.jpg"
            _, encoded = cv2.imencode('.jpg', frame)
            await self.storage.upload_file(key, encoded.tobytes(), "image/jpeg")
            frame_keys.append(key)

        return FrameResult(
            frame_count=len(frame_keys),
            frame_keys=frame_keys,
            fps=extraction.fps,
            resolution=f"{extraction.resolution[0]}x{extraction.resolution[1]}",
        )

    async def analyze_face(
        self, session_id: str, selfie_key: str, id_key: str
    ) -> FaceResult:
        """Analyze faces in selfie and ID images."""
        # Download images
        selfie_bytes = await self.storage.download_file(selfie_key)
        id_bytes = await self.storage.download_file(id_key)

        # Decode
        selfie_img = self._decode_image(selfie_bytes)
        id_img = self._decode_image(id_bytes)

        # Analyze
        result = self.face_analyzer.analyze(selfie_img, id_img)

        # Get embedding for storage
        embedding = None
        if result.selfie_faces:
            embedding = result.selfie_faces[0].embedding.tolist()

        return FaceResult(
            selfie_detected=len(result.selfie_faces) > 0,
            id_detected=len(result.id_faces) > 0,
            similarity=result.similarity or 0.0,
            embedding=embedding,
            pad_score=0.0,  # PAD is computed separately for video
        )

    async def analyze_document(self, session_id: str, id_key: str) -> DocResult:
        """Analyze document for OCR and tampering."""
        id_bytes = await self.storage.download_file(id_key)
        id_img = self._decode_image(id_bytes)

        result = self.doc_analyzer.analyze(id_img)

        return DocResult(
            detected=result.detected,
            ocr_confidence=result.avg_confidence,
            template_match=result.template_match_score,
            doc_score=result.doc_score,
        )

    async def process_session(self, session_id: str) -> None:
        """
        Full session processing pipeline.
        Called as a background task after upload finalization.
        """
        from sqlalchemy import select

        from app.database import async_session_maker
        from app.models.session import KYCSession
        from app.models.result import KYCResult
        from app.models.reason import KYCReason
        from app.models.frame_metric import KYCFrameMetric
        from app.services.scoring import compute_risk_score
        from app.services.reason_codes import ReasonCode, REASON_MESSAGES, get_reason_severity

        async with async_session_maker() as db:
            session = await db.get(KYCSession, session_id)
            if not session:
                return

            try:
                session.status = "processing"
                await db.commit()

                # Download assets
                selfie_bytes = await self.storage.download_file(session.selfie_asset_key)
                id_bytes = await self.storage.download_file(session.id_asset_key)

                # Decode images
                selfie_img = self._decode_image(selfie_bytes)
                id_img = self._decode_image(id_bytes)

                # Check if selfie is video
                frames: list[np.ndarray] = []
                is_video = self._is_video(session.selfie_asset_key or "")

                if is_video:
                    extraction = self.frame_extractor.extract_from_bytes(selfie_bytes)
                    frames = extraction.frames
                    # Use middle frame as "selfie" for face matching
                    if frames:
                        selfie_img = frames[len(frames) // 2]

                # Run detection modules
                face_result = self.face_analyzer.analyze(selfie_img, id_img)
                doc_result = self.doc_analyzer.analyze(id_img)

                pad_result = None
                if frames:
                    pad_result = self.pad_analyzer.analyze_frames(frames)

                # Collect all reason codes
                reasons: list[KYCReason] = []

                # Face analysis reasons
                for code in face_result.reason_codes:
                    reason_code = getattr(ReasonCode, code, None)
                    if reason_code:
                        message = REASON_MESSAGES.get(reason_code, code)
                        # Format message with evidence
                        if "{similarity" in message and face_result.evidence.get("face_similarity"):
                            message = message.format(similarity=face_result.evidence["face_similarity"])
                        
                        reasons.append(
                            KYCReason(
                                session_id=session.id,
                                code=code,
                                severity=get_reason_severity(reason_code),
                                message=message,
                                evidence=face_result.evidence,
                            )
                        )

                # Document analysis reasons
                for code in doc_result.reason_codes:
                    reason_code = getattr(ReasonCode, code, None)
                    if reason_code:
                        message = REASON_MESSAGES.get(reason_code, code)
                        if "{confidence" in message and doc_result.evidence.get("avg_ocr_confidence"):
                            message = message.format(confidence=doc_result.evidence["avg_ocr_confidence"])
                        
                        reasons.append(
                            KYCReason(
                                session_id=session.id,
                                code=code,
                                severity=get_reason_severity(reason_code),
                                message=message,
                                evidence=doc_result.evidence,
                            )
                        )

                # PAD analysis reasons
                if pad_result:
                    for code in pad_result.reason_codes:
                        reason_code = getattr(ReasonCode, code, None)
                        if reason_code:
                            reasons.append(
                                KYCReason(
                                    session_id=session.id,
                                    code=code,
                                    severity=get_reason_severity(reason_code),
                                    message=REASON_MESSAGES.get(reason_code, code),
                                    evidence=pad_result.evidence,
                                )
                            )

                # Compute scores
                face_similarity = face_result.similarity or 0.0
                pad_score = pad_result.overall_pad_score if pad_result else 0.0
                doc_score = doc_result.doc_score

                risk_score, decision = compute_risk_score(
                    face_similarity=face_similarity,
                    pad_score=pad_score,
                    doc_score=doc_score,
                    reasons=reasons,
                )

                # Save result
                result = KYCResult(
                    session_id=session.id,
                    risk_score=risk_score,
                    decision=decision,
                    face_similarity=face_similarity,
                    pad_score=pad_score,
                    doc_score=doc_score,
                )
                db.add(result)

                # Save reasons
                for reason in reasons:
                    db.add(reason)

                # Save frame metrics if available
                if pad_result:
                    for fm in pad_result.frame_metrics:
                        db.add(
                            KYCFrameMetric(
                                session_id=session.id,
                                frame_idx=fm.frame_idx,
                                motion_entropy=fm.motion_entropy,
                                sharpness=fm.sharpness,
                                noise_score=fm.noise_level,
                                color_shift=fm.color_temperature,
                                pad_flag=len(fm.pad_flags) > 0,
                            )
                        )

                # Store face embedding for pgvector search
                if face_result.selfie_faces:
                    session.face_embedding = face_result.selfie_faces[0].embedding.tolist()

                # Upload face crops for evidence
                if face_result.selfie_faces:
                    crop_key = f"crops/{session.id}/selfie_face.jpg"
                    _, crop_encoded = cv2.imencode('.jpg', face_result.selfie_faces[0].crop)
                    await self.storage.upload_file(
                        crop_key, crop_encoded.tobytes(), "image/jpeg"
                    )

                if face_result.id_faces:
                    crop_key = f"crops/{session.id}/id_face.jpg"
                    _, crop_encoded = cv2.imencode('.jpg', face_result.id_faces[0].crop)
                    await self.storage.upload_file(
                        crop_key, crop_encoded.tobytes(), "image/jpeg"
                    )

                session.status = "completed"
                await db.commit()

            except Exception as e:
                session.status = "failed"
                await db.commit()
                raise

    async def generate_synthetic_session(
        self,
        session_id: str,
        attack_family: str,
        attack_severity: str,
    ) -> None:
        """Generate a synthetic session with attack artifacts.
        
        TODO: Implement with simulator module for generating attack samples.
        For now, processes the session normally.
        """
        await self.process_session(session_id)


class ModalBackend:
    """Modal serverless GPU implementation for production scale.
    
    Uses a hybrid storage access pattern:
    - Direct R2 access for video extraction (efficient byte-range requests)
    - Presigned URLs for face/document analysis (stateless, secure)
    """

    def __init__(self) -> None:
        """Initialize the Modal backend."""
        self._storage = None
        self._modal_functions = None

    @property
    def storage(self):
        """Get storage service (lazy load)."""
        if self._storage is None:
            from app.services.storage import get_storage_service
            self._storage = get_storage_service()
        return self._storage

    def _get_modal_functions(self):
        """Import Modal functions lazily to avoid import errors when Modal not installed."""
        if self._modal_functions is None:
            import modal_app
            self._modal_functions = {
                "extract_frames": modal_app.extract_frames,
                "analyze_face": modal_app.analyze_face,
                "analyze_document": modal_app.analyze_document,
            }
        return self._modal_functions

    def _is_video(self, asset_key: str) -> bool:
        """Check if asset is a video based on extension."""
        video_extensions = {'.mp4', '.mov', '.webm', '.avi', '.mkv'}
        return any(asset_key.lower().endswith(ext) for ext in video_extensions)

    async def extract_frames(self, session_id: str, video_key: str) -> FrameResult:
        """Extract frames from video using Modal GPU with direct R2 access."""
        funcs = self._get_modal_functions()
        
        # Call Modal function remotely (it has R2 credentials)
        result = funcs["extract_frames"].remote(
            session_id=session_id,
            video_key=video_key,
            max_frames=30,
        )
        
        # Store frames to storage for later access
        frame_keys = []
        for idx, frame_b64 in enumerate(result["frames"]):
            import base64
            key = f"sessions/{session_id}/frames/{idx:04d}.jpg"
            frame_bytes = base64.b64decode(frame_b64)
            await self.storage.upload_file(key, frame_bytes, "image/jpeg")
            frame_keys.append(key)
        
        metadata = result["metadata"]
        return FrameResult(
            frame_count=len(frame_keys),
            frame_keys=frame_keys,
            fps=metadata["fps"],
            resolution=f"{metadata['width']}x{metadata['height']}",
        )

    async def analyze_face(
        self, session_id: str, selfie_key: str, id_key: str
    ) -> FaceResult:
        """Analyze faces using Modal GPU with presigned URLs."""
        funcs = self._get_modal_functions()
        
        # Generate presigned download URLs (5 min expiry)
        selfie_url = await self.storage.generate_presigned_download_url(
            selfie_key, expiration=300
        )
        id_url = await self.storage.generate_presigned_download_url(
            id_key, expiration=300
        )
        
        # Call Modal function remotely with presigned URLs
        result = funcs["analyze_face"].remote(
            selfie_url=selfie_url,
            id_url=id_url,
            similarity_threshold=0.45,
        )
        
        return FaceResult(
            selfie_detected=result["evidence"].get("selfie_face_count", 0) > 0,
            id_detected=result["evidence"].get("id_face_count", 0) > 0,
            similarity=result["similarity"] or 0.0,
            embedding=result.get("selfie_embedding"),
            pad_score=0.0,  # PAD computed separately
        )

    async def analyze_document(self, session_id: str, id_key: str) -> DocResult:
        """Analyze document using Modal GPU with presigned URL."""
        funcs = self._get_modal_functions()
        
        # Generate presigned download URL
        id_url = await self.storage.generate_presigned_download_url(
            id_key, expiration=300
        )
        
        # Call Modal function remotely
        result = funcs["analyze_document"].remote(
            id_url=id_url,
        )
        
        return DocResult(
            detected=result["avg_confidence"] > 0,
            ocr_confidence=result["avg_confidence"],
            template_match=0.85,  # Placeholder
            doc_score=result["doc_score"],
        )

    async def process_session(self, session_id: str) -> None:
        """
        Full session processing pipeline using Modal GPU functions.
        
        Orchestrates:
        1. Generate presigned URLs for assets
        2. Call Modal functions in parallel where possible
        3. Aggregate results and save to database
        """
        import asyncio
        import base64
        
        from app.database import async_session_maker
        from app.models.session import KYCSession
        from app.models.result import KYCResult
        from app.models.reason import KYCReason
        from app.services.scoring import compute_risk_score
        from app.services.reason_codes import ReasonCode, REASON_MESSAGES, get_reason_severity

        funcs = self._get_modal_functions()

        async with async_session_maker() as db:
            session = await db.get(KYCSession, session_id)
            if not session:
                return

            try:
                session.status = "processing"
                await db.commit()

                # Generate presigned URLs for the analysis workers
                selfie_url = await self.storage.generate_presigned_download_url(
                    session.selfie_asset_key, expiration=300
                )
                id_url = await self.storage.generate_presigned_download_url(
                    session.id_asset_key, expiration=300
                )

                # Check if selfie is video (requires different handling)
                is_video = self._is_video(session.selfie_asset_key or "")

                if is_video:
                    # Video processing: extract frames first, then analyze
                    frame_result = await self.extract_frames(
                        session_id, session.selfie_asset_key
                    )
                    # Use middle frame for face matching
                    if frame_result.frame_keys:
                        middle_idx = len(frame_result.frame_keys) // 2
                        selfie_url = await self.storage.generate_presigned_download_url(
                            frame_result.frame_keys[middle_idx], expiration=300
                        )

                # Call Modal functions remotely
                # These run on GPU containers in parallel
                face_result_raw = funcs["analyze_face"].remote(
                    selfie_url=selfie_url,
                    id_url=id_url,
                    similarity_threshold=0.45,
                )
                doc_result_raw = funcs["analyze_document"].remote(
                    id_url=id_url,
                )

                # Collect reason codes
                reasons: list[KYCReason] = []

                # Face analysis reasons
                for code in face_result_raw.get("reason_codes", []):
                    reason_code = getattr(ReasonCode, code, None)
                    if reason_code:
                        message = REASON_MESSAGES.get(reason_code, code)
                        evidence = face_result_raw.get("evidence", {})
                        if "{similarity" in message and evidence.get("face_similarity"):
                            message = message.format(similarity=evidence["face_similarity"])
                        
                        reasons.append(
                            KYCReason(
                                session_id=session.id,
                                code=code,
                                severity=get_reason_severity(reason_code),
                                message=message,
                                evidence=evidence,
                            )
                        )

                # Document analysis reasons
                for code in doc_result_raw.get("reason_codes", []):
                    reason_code = getattr(ReasonCode, code, None)
                    if reason_code:
                        message = REASON_MESSAGES.get(reason_code, code)
                        evidence = doc_result_raw.get("evidence", {})
                        if "{confidence" in message and evidence.get("avg_ocr_confidence"):
                            message = message.format(confidence=evidence["avg_ocr_confidence"])
                        
                        reasons.append(
                            KYCReason(
                                session_id=session.id,
                                code=code,
                                severity=get_reason_severity(reason_code),
                                message=message,
                                evidence=evidence,
                            )
                        )

                # Compute scores
                face_similarity = face_result_raw.get("similarity") or 0.0
                pad_score = 0.0  # PAD would need video frame analysis
                doc_score = doc_result_raw.get("doc_score", 0.0)

                risk_score, decision = compute_risk_score(
                    face_similarity=face_similarity,
                    pad_score=pad_score,
                    doc_score=doc_score,
                    reasons=reasons,
                )

                # Save result
                result = KYCResult(
                    session_id=session.id,
                    risk_score=risk_score,
                    decision=decision,
                    face_similarity=face_similarity,
                    pad_score=pad_score,
                    doc_score=doc_score,
                )
                db.add(result)

                # Save reasons
                for reason in reasons:
                    db.add(reason)

                # Store face embedding for pgvector search
                if face_result_raw.get("selfie_embedding"):
                    session.face_embedding = face_result_raw["selfie_embedding"]

                # Upload face crops from Modal results
                if face_result_raw.get("selfie_crop_b64"):
                    crop_key = f"crops/{session.id}/selfie_face.jpg"
                    crop_bytes = base64.b64decode(face_result_raw["selfie_crop_b64"])
                    await self.storage.upload_file(
                        crop_key, crop_bytes, "image/jpeg"
                    )

                if face_result_raw.get("id_crop_b64"):
                    crop_key = f"crops/{session.id}/id_face.jpg"
                    crop_bytes = base64.b64decode(face_result_raw["id_crop_b64"])
                    await self.storage.upload_file(
                        crop_key, crop_bytes, "image/jpeg"
                    )

                session.status = "completed"
                await db.commit()

            except Exception as e:
                session.status = "failed"
                await db.commit()
                raise

    async def generate_synthetic_session(
        self,
        session_id: str,
        attack_family: str,
        attack_severity: str,
    ) -> None:
        """Generate a synthetic session with attack artifacts.
        
        TODO: Implement with simulator module for generating attack samples.
        For now, processes the session normally.
        """
        await self.process_session(session_id)


def get_processing_backend() -> ProcessingBackend:
    """Get the configured processing backend."""
    if settings.use_modal:
        return ModalBackend()
    return LocalBackend()
