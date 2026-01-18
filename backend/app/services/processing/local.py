"""Local processing backend using real ML/CV models.

Orchestrates the detection pipeline for local/development use:
1. Download assets from storage
2. Run face analysis with InsightFace
3. Run document analysis with PaddleOCR
4. Run PAD heuristics with OpenCV (if video)
5. Compute risk scores and save results
"""

import logging

import cv2
import numpy as np

from app.services.processing.protocol import FrameResult, FaceResult, DocResult
from app.services.processing.helpers import decode_image, is_video, build_reasons

logger = logging.getLogger(__name__)


class LocalBackend:
    """FastAPI BackgroundTasks implementation using real ML/CV models."""

    def __init__(self) -> None:
        """Initialize the local backend with lazy-loaded dependencies."""
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
        selfie_bytes = await self.storage.download_file(selfie_key)
        id_bytes = await self.storage.download_file(id_key)

        selfie_img = decode_image(selfie_bytes)
        id_img = decode_image(id_bytes)

        result = self.face_analyzer.analyze(selfie_img, id_img)

        embedding = None
        if result.selfie_faces:
            embedding = result.selfie_faces[0].embedding.tolist()

        return FaceResult(
            selfie_detected=len(result.selfie_faces) > 0,
            id_detected=len(result.id_faces) > 0,
            similarity=result.similarity or 0.0,
            embedding=embedding,
            pad_score=0.0,
        )

    async def analyze_document(self, session_id: str, id_key: str) -> DocResult:
        """Analyze document for OCR and tampering."""
        id_bytes = await self.storage.download_file(id_key)
        id_img = decode_image(id_bytes)

        result = self.doc_analyzer.analyze(id_img)

        return DocResult(
            detected=result.detected,
            ocr_confidence=result.avg_confidence,
            template_match=result.template_match_score,
            doc_score=result.doc_score,
        )

    async def process_session(self, session_id: str) -> None:
        """Full session processing pipeline."""
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

        async with async_session_maker() as db:
            session = await db.get(KYCSession, session_id)
            if not session:
                return

            try:
                # Idempotency: skip if session already has results
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

                # SECURITY: Validate file sizes before downloading
                max_size = app_settings.max_upload_size_bytes
                
                selfie_size = await self.storage.get_object_size(session.selfie_asset_key)
                if selfie_size > max_size:
                    raise ValueError(f"Selfie too large: {selfie_size} bytes")
                
                id_size = await self.storage.get_object_size(session.id_asset_key)
                if id_size > max_size:
                    raise ValueError(f"ID too large: {id_size} bytes")

                # Download assets
                selfie_bytes = await self.storage.download_file(session.selfie_asset_key)
                id_bytes = await self.storage.download_file(session.id_asset_key)

                selfie_img = decode_image(selfie_bytes)
                id_img = decode_image(id_bytes)

                # Check if selfie is video
                frames: list[np.ndarray] = []
                session_is_video = is_video(session.selfie_asset_key or "")

                if session_is_video:
                    extraction = self.frame_extractor.extract_from_bytes(selfie_bytes)
                    frames = extraction.frames
                    if frames:
                        selfie_img = frames[len(frames) // 2]

                # Run detection modules with progress updates
                await emit_progress(session_id, "progress", step="face_analysis", progress=0.2, message="Analyzing faces...")
                face_result = self.face_analyzer.analyze(selfie_img, id_img)

                await emit_progress(session_id, "progress", step="document_analysis", progress=0.4, message="Analyzing document...")
                doc_result = self.doc_analyzer.analyze(id_img)

                pad_result = None
                if frames:
                    await emit_progress(session_id, "progress", step="pad_analysis", progress=0.6, message="Running presentation attack detection...")
                    pad_result = self.pad_analyzer.analyze_frames(frames)

                # Build reason codes using helper
                reasons: list[KYCReason] = []

                # Face analysis reasons
                reasons.extend(build_reasons(
                    session.id,
                    face_result.reason_codes,
                    face_result.evidence,
                ))

                # Document analysis reasons
                reasons.extend(build_reasons(
                    session.id,
                    doc_result.reason_codes,
                    doc_result.evidence,
                ))

                # PAD analysis reasons
                if pad_result:
                    reasons.extend(build_reasons(
                        session.id,
                        pad_result.reason_codes,
                        pad_result.evidence,
                    ))

                # Fraud detection
                from app.services.fraud_detection import compute_fraud_signals

                document_country = doc_result.evidence.get("mrz_country") or doc_result.evidence.get("ocr_country")

                fraud_signals = await compute_fraud_signals(
                    db=db,
                    device_fingerprint=session.device_fingerprint,
                    client_ip=session.client_ip,
                    document_country=document_country,
                    device_country=session.ip_country,
                    device_timezone=session.device_timezone,
                    exclude_session_id=str(session.id),
                )

                # Add fraud reason codes
                for code in fraud_signals.reason_codes:
                    reason_code = getattr(ReasonCode, code, None)
                    if reason_code:
                        evidence = {}
                        if fraud_signals.velocity_check:
                            evidence["velocity"] = {
                                "device_1h": fraud_signals.velocity_check.device_submissions_1h,
                                "device_24h": fraud_signals.velocity_check.device_submissions_24h,
                                "ip_1h": fraud_signals.velocity_check.ip_submissions_1h,
                                "ip_24h": fraud_signals.velocity_check.ip_submissions_24h,
                                "flags": fraud_signals.velocity_check.flags,
                            }
                        if fraud_signals.geo_anomaly:
                            evidence["geo"] = {
                                "document_country": fraud_signals.geo_anomaly.document_country,
                                "device_country": fraud_signals.geo_anomaly.device_country,
                                "anomaly_type": fraud_signals.geo_anomaly.anomaly_type,
                                "confidence": fraud_signals.geo_anomaly.confidence,
                            }
                        reasons.append(
                            KYCReason(
                                session_id=session.id,
                                code=code,
                                severity=get_reason_severity(reason_code),
                                message=REASON_MESSAGES.get(reason_code, code),
                                evidence=evidence,
                            )
                        )

                # Compute scores with progress update
                await emit_progress(session_id, "progress", step="scoring", progress=0.8, message="Computing risk score...")
                face_similarity = face_result.similarity or 0.0
                pad_score = pad_result.overall_pad_score if pad_result else 0.0
                doc_score = doc_result.doc_score

                from app.services.scoring import get_scoring_config
                scoring_config = get_scoring_config(app_settings.scoring_profile)

                risk_score, decision = compute_risk_score(
                    face_similarity=face_similarity,
                    pad_score=pad_score,
                    doc_score=doc_score,
                    reasons=reasons,
                    config=scoring_config,
                )

                # Save result with version tracking
                result = KYCResult(
                    session_id=session.id,
                    risk_score=risk_score,
                    decision=decision,
                    face_similarity=face_similarity,
                    pad_score=pad_score,
                    doc_score=doc_score,
                    model_version=f"{self.face_analyzer.version},{self.doc_analyzer.version}",
                    rules_version=app_settings.scoring_profile,
                )
                db.add(result)

                for reason in reasons:
                    db.add(reason)

                # Save frame metrics
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

                # Store face embedding
                if face_result.selfie_faces:
                    session.face_embedding = face_result.selfie_faces[0].embedding.tolist()

                # Upload face crops
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
