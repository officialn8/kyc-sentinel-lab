"""Modal serverless GPU processing backend for production scale.

Uses a hybrid storage access pattern:
- Direct R2 access for video extraction (efficient byte-range requests)
- Presigned URLs for face/document analysis (stateless, secure)
"""

import asyncio
import base64
import logging

import cv2
import numpy as np

from app.services.processing.protocol import FrameResult, FaceResult, DocResult
from app.services.processing.helpers import is_video, build_reasons

logger = logging.getLogger(__name__)


class ModalBackend:
    """Modal serverless GPU implementation for production scale."""

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
        """Look up deployed Modal functions by name."""
        if self._modal_functions is None:
            import modal
            self._modal_functions = {
                "extract_frames": modal.Function.from_name("kyc-sentinel", "extract_frames"),
                "analyze_face": modal.Function.from_name("kyc-sentinel", "analyze_face"),
                "analyze_document": modal.Function.from_name("kyc-sentinel", "analyze_document"),
            }
        return self._modal_functions

    async def extract_frames(self, session_id: str, video_key: str) -> FrameResult:
        """Extract frames from video using Modal GPU with direct R2 access."""
        funcs = self._get_modal_functions()
        
        result = funcs["extract_frames"].remote(
            session_id=session_id,
            video_key=video_key,
            max_frames=30,
        )
        
        frame_keys = []
        for idx, frame_b64 in enumerate(result["frames"]):
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
        
        selfie_url = await self.storage.generate_presigned_download_url(
            selfie_key, expiration=300
        )
        id_url = await self.storage.generate_presigned_download_url(
            id_key, expiration=300
        )
        
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
            pad_score=0.0,
        )

    async def analyze_document(self, session_id: str, id_key: str) -> DocResult:
        """Analyze document using Modal GPU with presigned URL."""
        funcs = self._get_modal_functions()
        
        id_url = await self.storage.generate_presigned_download_url(
            id_key, expiration=300
        )
        
        result = funcs["analyze_document"].remote(id_url=id_url)
        
        return DocResult(
            detected=result["avg_confidence"] > 0,
            ocr_confidence=result["avg_confidence"],
            template_match=0.85,
            doc_score=result["doc_score"],
        )

    async def process_session(self, session_id: str) -> None:
        """Full session processing pipeline using Modal GPU functions."""
        from sqlalchemy import select
        from app.database import async_session_maker
        from app.models.session import KYCSession
        from app.models.result import KYCResult
        from app.models.reason import KYCReason
        from app.models.frame_metric import KYCFrameMetric
        from app.services.scoring import compute_risk_score
        from app.services.reason_codes import ReasonCode, REASON_MESSAGES, get_reason_severity
        from app.config import settings as app_settings
        from app.api.websocket import emit_progress

        funcs = self._get_modal_functions()

        async with async_session_maker() as db:
            session = await db.get(KYCSession, session_id)
            if not session:
                return

            try:
                # Idempotency check
                existing_result = await db.scalar(
                    select(KYCResult).where(KYCResult.session_id == session.id)
                )
                if existing_result is not None:
                    logger.info(f"Session {session_id} already processed, skipping")
                    if session.status != "completed":
                        session.status = "completed"
                        await db.commit()
                    return

                session.status = "processing"
                await db.commit()

                # Emit processing started event
                await emit_progress(session_id, "started", message="Processing started")

                # Generate presigned URLs
                selfie_url = await self.storage.generate_presigned_download_url(
                    session.selfie_asset_key, expiration=300
                )
                id_url = await self.storage.generate_presigned_download_url(
                    session.id_asset_key, expiration=300
                )

                session_is_video = is_video(session.selfie_asset_key or "")
                frame_result = None

                if session_is_video:
                    frame_result = await self.extract_frames(
                        session_id, session.selfie_asset_key
                    )
                    if frame_result.frame_keys:
                        middle_idx = len(frame_result.frame_keys) // 2
                        selfie_url = await self.storage.generate_presigned_download_url(
                            frame_result.frame_keys[middle_idx], expiration=300
                        )

                # Call Modal functions with progress updates
                await emit_progress(session_id, "progress", step="face_analysis", progress=0.2, message="Analyzing faces...")
                face_result_raw = funcs["analyze_face"].remote(
                    selfie_url=selfie_url,
                    id_url=id_url,
                    similarity_threshold=0.45,
                )

                await emit_progress(session_id, "progress", step="document_analysis", progress=0.4, message="Analyzing document...")
                doc_result_raw = funcs["analyze_document"].remote(id_url=id_url)

                # Build reasons using helper
                reasons: list[KYCReason] = []

                reasons.extend(build_reasons(
                    session.id,
                    face_result_raw.get("reason_codes", []),
                    face_result_raw.get("evidence", {}),
                ))

                reasons.extend(build_reasons(
                    session.id,
                    doc_result_raw.get("reason_codes", []),
                    doc_result_raw.get("evidence", {}),
                ))

                # Compute scores
                face_similarity = face_result_raw.get("similarity") or 0.0
                doc_score = doc_result_raw.get("doc_score", 0.0)
                
                # PAD analysis for video
                pad_score = 0.0
                if session_is_video and frame_result and frame_result.frame_keys:
                    await emit_progress(session_id, "progress", step="pad_analysis", progress=0.6, message="Running presentation attack detection...")
                elif not session_is_video:
                    # Skip PAD for images, move to next step
                    await emit_progress(session_id, "progress", step="pad_analysis", progress=0.6, message="Skipping PAD (image upload)")

                if session_is_video and frame_result and frame_result.frame_keys:
                    from app.detection.pad_heuristics import get_pad_analyzer
                    
                    pad_analyzer = get_pad_analyzer()
                    MAX_CONCURRENT_DOWNLOADS = 10
                    semaphore = asyncio.Semaphore(MAX_CONCURRENT_DOWNLOADS)

                    async def download_frame(frame_key: str) -> np.ndarray | None:
                        async with semaphore:
                            try:
                                frame_bytes = await self.storage.download_file(frame_key)
                                return cv2.imdecode(
                                    np.frombuffer(frame_bytes, np.uint8),
                                    cv2.IMREAD_COLOR
                                )
                            except Exception:
                                return None

                    frame_results = await asyncio.gather(
                        *[download_frame(key) for key in frame_result.frame_keys],
                        return_exceptions=True
                    )

                    frames = [f for f in frame_results if f is not None and not isinstance(f, Exception)]
                    
                    if frames:
                        pad_result = pad_analyzer.analyze_frames(frames)
                        pad_score = pad_result.overall_pad_score
                        
                        reasons.extend(build_reasons(
                            session.id,
                            pad_result.reason_codes,
                            pad_result.evidence,
                        ))
                        
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

                # Scoring
                await emit_progress(session_id, "progress", step="scoring", progress=0.8, message="Computing risk score...")

                from app.services.scoring import get_scoring_config
                scoring_config = get_scoring_config(app_settings.scoring_profile)

                risk_score, decision = compute_risk_score(
                    face_similarity=face_similarity,
                    pad_score=pad_score,
                    doc_score=doc_score,
                    reasons=reasons,
                    config=scoring_config,
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

                for reason in reasons:
                    db.add(reason)

                # Store face embedding
                if face_result_raw.get("selfie_embedding"):
                    session.face_embedding = face_result_raw["selfie_embedding"]

                # Upload face crops
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

                # Emit completion event
                await emit_progress(
                    session_id, "completed", progress=1.0,
                    message=f"Processing complete. Decision: {decision}"
                )

                # Dispatch webhooks for session completion
                from app.services.webhook_service import dispatch_webhooks
                await dispatch_webhooks("session.completed", {
                    "session_id": session_id,
                    "decision": decision,
                    "risk_score": risk_score,
                })

            except Exception as e:
                session.status = "failed"
                await db.commit()
                # Emit failure event
                await emit_progress(session_id, "failed", error=str(e))

                # Dispatch webhooks for session failure
                from app.services.webhook_service import dispatch_webhooks
                await dispatch_webhooks("session.failed", {
                    "session_id": session_id,
                    "error": str(e),
                })
                raise

    async def generate_synthetic_session(
        self,
        session_id: str,
        attack_family: str,
        attack_severity: str,
    ) -> None:
        """Generate a synthetic session with attack artifacts."""
        from simulator.base_images import (
            generate_synthetic_selfie,
            generate_synthetic_id_document,
            encode_image_to_jpeg,
        )
        from simulator.generator import ArtifactGenerator, AttackFamily, AttackSeverity

        from app.database import async_session_maker
        from app.models.session import KYCSession

        rng = np.random.default_rng()
        
        selfie_img = generate_synthetic_selfie(rng)
        id_img = generate_synthetic_id_document(rng)
        
        generator = ArtifactGenerator(seed=rng.integers(0, 2**31))
        
        if attack_family != "benign":
            severity = AttackSeverity(attack_severity)
            
            if attack_family in ("replay", "injection", "face_swap"):
                artifact = generator.generate(
                    selfie_img,
                    AttackFamily(attack_family),
                    severity,
                )
                selfie_img = artifact.image
            
            if attack_family == "doc_tamper":
                artifact = generator.generate(
                    id_img,
                    AttackFamily.DOC_TAMPER,
                    severity,
                )
                id_img = artifact.image
        
        selfie_key = f"sessions/{session_id}/selfie.jpg"
        id_key = f"sessions/{session_id}/id.jpg"
        
        selfie_bytes = encode_image_to_jpeg(selfie_img)
        id_bytes = encode_image_to_jpeg(id_img)
        
        await self.storage.upload_file(selfie_key, selfie_bytes, "image/jpeg")
        await self.storage.upload_file(id_key, id_bytes, "image/jpeg")
        
        async with async_session_maker() as db:
            session = await db.get(KYCSession, session_id)
            if session:
                session.selfie_asset_key = selfie_key
                session.id_asset_key = id_key
                await db.commit()
        
        await self.process_session(session_id)
