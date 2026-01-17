"""
Presentation Attack Detection heuristics using OpenCV.

Analyzes video frames for signs of:
- Replay attacks (screen capture artifacts)
- Injection attacks (unnatural image properties)
- Face boundary anomalies
- Lack of liveness signals (no blinks, static texture)
- Printed photo detection (LBP texture analysis)

These are heuristic-based signals, not learned models.
They provide interpretable reason codes.
"""

import numpy as np
from dataclasses import dataclass, field
from typing import Optional
from statistics import mean, stdev

import cv2
from scipy import fftpack
from skimage.feature import local_binary_pattern


@dataclass
class FrameMetrics:
    """Metrics extracted from a single frame."""

    frame_idx: int
    sharpness: float
    noise_level: float
    motion_entropy: float
    color_temperature: float
    moire_score: float
    pad_flags: list[str]
    
    # Liveness detection fields
    eye_aspect_ratio: Optional[float] = None  # EAR for blink detection
    blink_detected: bool = False
    lbp_texture_score: float = 0.0  # 0-1, higher = more like real skin
    
    # Backward compatible fields
    noise_score: float = 0.0
    color_shift: float = 0.0
    moire_detected: bool = False
    screen_glare_detected: bool = False

    def __post_init__(self):
        # Set backward-compatible fields
        self.noise_score = self.noise_level
        self.color_shift = self.color_temperature
        self.moire_detected = "moire_detected" in self.pad_flags


@dataclass
class BlinkAnalysis:
    """Result of blink detection analysis across frames."""
    
    blink_count: int
    ear_values: list[float]  # Eye aspect ratios per frame
    blink_frames: list[int]  # Frame indices where blinks detected
    avg_ear: float
    min_ear: float
    has_natural_blinks: bool  # True if natural blink pattern detected


@dataclass
class PADResult:
    """Result of PAD analysis."""

    frame_metrics: list[FrameMetrics]
    overall_pad_score: float  # 0-1, higher = more suspicious
    reason_codes: list[str]
    evidence: dict
    flagged_frames: list[int]
    
    # Liveness analysis results
    blink_analysis: Optional[BlinkAnalysis] = None
    texture_analysis: Optional[dict] = None
    
    # Backward compatible fields
    pad_score: float = 0.0
    replay_suspected: bool = False
    injection_suspected: bool = False
    low_motion: bool = False
    frame_stutter: bool = False
    face_boundary_mismatch: bool = False
    no_blinks_detected: bool = False
    printed_texture_detected: bool = False
    suspicious_frames: list[int] = None
    artifacts_detected: list[str] = None

    def __post_init__(self):
        # Set backward-compatible fields
        self.pad_score = self.overall_pad_score
        self.suspicious_frames = self.flagged_frames
        self.artifacts_detected = self.reason_codes
        self.replay_suspected = "PAD_SUSPECT_REPLAY" in self.reason_codes or "PAD_SCREEN_ARTIFACTS" in self.reason_codes
        self.injection_suspected = "PAD_INJECTION_ARTIFACTS" in self.reason_codes
        self.low_motion = "PAD_LOW_MOTION_ENTROPY" in self.reason_codes
        self.frame_stutter = "PAD_FRAME_STUTTER" in self.reason_codes
        self.no_blinks_detected = "PAD_NO_BLINK_DETECTED" in self.reason_codes
        self.printed_texture_detected = "PAD_PRINTED_TEXTURE" in self.reason_codes


class PADAnalyzer:
    """Presentation Attack Detection analyzer using OpenCV heuristics.
    
    Implements detection of:
    - Replay attacks via moiré pattern detection in FFT domain
    - Injection attacks via unnatural sharpness boundaries
    - Static image attacks via motion entropy analysis
    - Video replay via frame stutter detection
    - Liveness via blink detection using Eye Aspect Ratio (EAR)
    - Printed photo detection via LBP texture analysis
    """

    def __init__(
        self,
        sharpness_threshold: float = 100.0,
        noise_floor: float = 5.0,
        noise_ceiling: float = 50.0,
        motion_entropy_min: float = 0.1,
        moire_threshold: float = 0.15,
        color_temp_variance_max: float = 0.2,
        face_boundary_threshold: float = 0.4,
        # Liveness detection thresholds
        ear_blink_threshold: float = 0.21,
        min_blinks_expected: int = 1,
        min_video_frames_for_blink: int = 30,
        lbp_texture_threshold: float = 0.4,
    ) -> None:
        """
        Initialize PAD analyzer with configurable thresholds.
        
        Args:
            sharpness_threshold: Laplacian variance threshold
            noise_floor: Minimum expected noise level
            noise_ceiling: Maximum acceptable noise level
            motion_entropy_min: Minimum motion entropy for liveness
            moire_threshold: Threshold for moiré pattern detection
            color_temp_variance_max: Max acceptable color temp variance
            face_boundary_threshold: Threshold for face boundary mismatch detection
            ear_blink_threshold: Eye Aspect Ratio below which a blink is detected
            min_blinks_expected: Minimum blinks expected in a video (for liveness)
            min_video_frames_for_blink: Minimum frames required to check for blinks
            lbp_texture_threshold: Score below which texture is suspected printed
        """
        self.sharpness_threshold = sharpness_threshold
        self.noise_floor = noise_floor
        self.noise_ceiling = noise_ceiling
        self.motion_entropy_min = motion_entropy_min
        self.moire_threshold = moire_threshold
        self.color_temp_variance_max = color_temp_variance_max
        self.face_boundary_threshold = face_boundary_threshold
        self.ear_blink_threshold = ear_blink_threshold
        self.min_blinks_expected = min_blinks_expected
        self.min_video_frames_for_blink = min_video_frames_for_blink
        self.lbp_texture_threshold = lbp_texture_threshold
        
        # Lazy-load face detector for landmark extraction
        self._face_detector = None
        self._landmark_predictor = None

    def analyze_frames(
        self,
        frames: list[np.ndarray],
        facial_landmarks: Optional[list[np.ndarray]] = None,
    ) -> PADResult:
        """
        Analyze video frames for presentation attack indicators.
        
        Args:
            frames: List of BGR frame arrays
            facial_landmarks: Optional list of 68-point facial landmarks per frame
                             (for blink detection). If not provided, will attempt
                             to detect landmarks using dlib if available.
        
        Returns:
            PADResult with per-frame metrics and overall assessment
        """
        if not frames:
            return PADResult(
                frame_metrics=[],
                overall_pad_score=0.5,
                reason_codes=["PAD_NO_FRAMES"],
                evidence={"frame_count": 0},
                flagged_frames=[],
            )

        frame_metrics: list[FrameMetrics] = []
        prev_frame: Optional[np.ndarray] = None
        motion_scores: list[float] = []

        for idx, frame in enumerate(frames):
            # Get landmarks for this frame if provided
            frame_landmarks = None
            if facial_landmarks and idx < len(facial_landmarks):
                frame_landmarks = facial_landmarks[idx]
            
            metrics = self._analyze_single_frame(frame, prev_frame, idx, frame_landmarks)
            frame_metrics.append(metrics)

            if metrics.motion_entropy > 0:
                motion_scores.append(metrics.motion_entropy)

            prev_frame = frame

        # Aggregate analysis
        reason_codes: list[str] = []
        evidence: dict = {}
        flagged_frames: list[int] = []
        blink_analysis: Optional[BlinkAnalysis] = None
        texture_analysis: Optional[dict] = None

        # Check for replay indicators (screen artifacts)
        moire_frames = [
            m for m in frame_metrics if m.moire_score > self.moire_threshold
        ]
        if len(moire_frames) > len(frames) * 0.3:
            reason_codes.append("PAD_SCREEN_ARTIFACTS")
            evidence["moire_frame_ratio"] = len(moire_frames) / len(frames)
            flagged_frames.extend([m.frame_idx for m in moire_frames])

        # Check for low motion (static image / replay)
        if motion_scores:
            avg_motion = mean(motion_scores)
            evidence["avg_motion_entropy"] = avg_motion

            if avg_motion < self.motion_entropy_min:
                reason_codes.append("PAD_LOW_MOTION_ENTROPY")
                evidence["motion_threshold"] = self.motion_entropy_min

        # Check for frame stutter (duplicate frames)
        stutter_count = self._detect_frame_stutter(frames)
        if stutter_count > len(frames) * 0.2:
            reason_codes.append("PAD_FRAME_STUTTER")
            evidence["stutter_ratio"] = stutter_count / len(frames)

        # Check for injection artifacts (unnatural sharpness boundaries)
        injection_frames = self._detect_injection_artifacts(frames)
        if injection_frames:
            reason_codes.append("PAD_INJECTION_ARTIFACTS")
            evidence["injection_frame_count"] = len(injection_frames)
            flagged_frames.extend(injection_frames)

        # Check color temperature consistency
        color_temps = [m.color_temperature for m in frame_metrics]
        if len(color_temps) > 1:
            temp_std = stdev(color_temps)
            if temp_std > self.color_temp_variance_max:
                reason_codes.append("PAD_SUSPECT_REPLAY")
                evidence["color_temp_variance"] = temp_std

        # Check for face boundary mismatch (face swap indicator)
        face_boundary_detected, face_boundary_evidence = self._detect_face_boundary_mismatch(frames)
        if face_boundary_detected:
            reason_codes.append("PAD_FACE_BOUNDARY_MISMATCH")
            evidence.update(face_boundary_evidence)

        # === NEW: Blink Detection Analysis ===
        # Only check for blinks if we have enough frames (video, not single image)
        if len(frames) >= self.min_video_frames_for_blink:
            blink_analysis = self._analyze_blinks(frame_metrics)
            evidence["blink_count"] = blink_analysis.blink_count
            evidence["avg_ear"] = blink_analysis.avg_ear
            evidence["has_natural_blinks"] = blink_analysis.has_natural_blinks
            
            if not blink_analysis.has_natural_blinks:
                reason_codes.append("PAD_NO_BLINK_DETECTED")
                evidence["expected_blinks"] = self.min_blinks_expected

        # === NEW: LBP Texture Analysis ===
        texture_analysis = self._analyze_texture(frame_metrics, frames)
        evidence["avg_texture_score"] = texture_analysis.get("avg_score", 0)
        evidence["texture_verdict"] = texture_analysis.get("verdict", "unknown")
        
        if texture_analysis.get("is_printed", False):
            reason_codes.append("PAD_PRINTED_TEXTURE")

        # Compute overall PAD score
        overall_score = self._compute_pad_score(frame_metrics, reason_codes)

        return PADResult(
            frame_metrics=frame_metrics,
            overall_pad_score=overall_score,
            reason_codes=reason_codes,
            evidence=evidence,
            flagged_frames=list(set(flagged_frames)),
            blink_analysis=blink_analysis,
            texture_analysis=texture_analysis,
        )

    def _analyze_single_frame(
        self,
        frame: np.ndarray,
        prev_frame: Optional[np.ndarray],
        idx: int,
        landmarks: Optional[np.ndarray] = None,
    ) -> FrameMetrics:
        """Compute metrics for a single frame."""
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

        # Sharpness (Laplacian variance)
        laplacian = cv2.Laplacian(gray, cv2.CV_64F)
        sharpness = float(laplacian.var())

        # Noise level estimation
        noise_level = self._estimate_noise(gray)

        # Motion entropy (optical flow magnitude variance)
        motion_entropy = 0.0
        if prev_frame is not None:
            motion_entropy = self._compute_motion_entropy(prev_frame, frame)

        # Color temperature
        color_temperature = self._estimate_color_temperature(frame)

        # Moiré pattern detection
        moire_score = self._detect_moire(gray)

        # Eye Aspect Ratio for blink detection
        ear: Optional[float] = None
        blink_detected = False
        if landmarks is not None and len(landmarks) >= 68:
            ear = self._compute_eye_aspect_ratio(landmarks)
            blink_detected = ear < self.ear_blink_threshold if ear else False
        
        # LBP texture score for skin vs printed photo
        lbp_texture_score = self._compute_lbp_texture_score(frame)

        # Per-frame flags
        pad_flags: list[str] = []
        if sharpness < self.sharpness_threshold * 0.5:
            pad_flags.append("low_sharpness")
        if noise_level > self.noise_ceiling:
            pad_flags.append("high_noise")
        if moire_score > self.moire_threshold:
            pad_flags.append("moire_detected")
        if blink_detected:
            pad_flags.append("blink_detected")
        if lbp_texture_score < self.lbp_texture_threshold:
            pad_flags.append("printed_texture")

        return FrameMetrics(
            frame_idx=idx,
            sharpness=sharpness,
            noise_level=noise_level,
            motion_entropy=motion_entropy,
            color_temperature=color_temperature,
            moire_score=moire_score,
            pad_flags=pad_flags,
            eye_aspect_ratio=ear,
            blink_detected=blink_detected,
            lbp_texture_score=lbp_texture_score,
        )

    def _estimate_noise(self, gray: np.ndarray) -> float:
        """Estimate image noise level using median absolute deviation."""
        # High-pass filter to isolate noise
        kernel = np.array([[-1, -1, -1], [-1, 8, -1], [-1, -1, -1]])
        filtered = cv2.filter2D(gray, -1, kernel)

        # MAD-based noise estimation
        median = np.median(np.abs(filtered))
        return float(median)

    def _compute_motion_entropy(self, prev: np.ndarray, curr: np.ndarray) -> float:
        """Compute motion entropy using optical flow."""
        prev_gray = cv2.cvtColor(prev, cv2.COLOR_BGR2GRAY)
        curr_gray = cv2.cvtColor(curr, cv2.COLOR_BGR2GRAY)

        # Dense optical flow
        flow = cv2.calcOpticalFlowFarneback(
            prev_gray,
            curr_gray,
            None,
            pyr_scale=0.5,
            levels=3,
            winsize=15,
            iterations=3,
            poly_n=5,
            poly_sigma=1.2,
            flags=0,
        )

        # Magnitude
        mag, _ = cv2.cartToPolar(flow[..., 0], flow[..., 1])

        # Entropy of magnitude distribution
        hist, _ = np.histogram(mag.flatten(), bins=50, range=(0, 20))
        hist = hist / hist.sum()
        hist = hist[hist > 0]  # Remove zeros for log

        if len(hist) == 0:
            return 0.0
            
        entropy = -np.sum(hist * np.log2(hist))
        return float(entropy)

    def _estimate_color_temperature(self, frame: np.ndarray) -> float:
        """Estimate color temperature as B/R ratio."""
        b, g, r = cv2.split(frame)
        r_mean = np.mean(r)
        b_mean = np.mean(b)

        if r_mean > 0:
            return float(b_mean / r_mean)
        return 1.0

    def _detect_moire(self, gray: np.ndarray) -> float:
        """
        Detect moiré patterns using FFT analysis.
        Screen captures often have periodic patterns from pixel grids.
        """
        # FFT
        f = fftpack.fft2(gray.astype(float))
        fshift = fftpack.fftshift(f)
        magnitude = np.abs(fshift)

        # Normalize
        magnitude = np.log1p(magnitude)

        # Look for periodic peaks (excluding DC component)
        h, w = magnitude.shape
        center_y, center_x = h // 2, w // 2

        # Mask out center (low frequencies)
        mask = np.ones_like(magnitude)
        cv2.circle(mask, (center_x, center_y), min(h, w) // 10, 0, -1)

        masked_mag = magnitude * mask

        # Score based on peak-to-average ratio
        peak = np.max(masked_mag)
        mask_nonzero = mask > 0
        avg = np.mean(masked_mag[mask_nonzero]) if mask_nonzero.any() else 0

        if avg > 0:
            return float((peak - avg) / avg)
        return 0.0

    def _detect_frame_stutter(self, frames: list[np.ndarray]) -> int:
        """Count duplicate/near-duplicate consecutive frames."""
        stutter_count = 0

        for i in range(1, len(frames)):
            diff = cv2.absdiff(frames[i - 1], frames[i])
            diff_score = np.mean(diff)

            # Very low difference = likely duplicate
            if diff_score < 2.0:
                stutter_count += 1

        return stutter_count

    def _detect_injection_artifacts(self, frames: list[np.ndarray]) -> list[int]:
        """
        Detect frames with unnatural sharpness boundaries.
        Injection attacks often have unnaturally sharp face regions
        against softer backgrounds.
        """
        flagged: list[int] = []

        for idx, frame in enumerate(frames):
            gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

            # Divide into regions and compare sharpness
            h, w = gray.shape
            regions = [
                gray[0 : h // 2, 0 : w // 2],  # top-left
                gray[0 : h // 2, w // 2 : w],  # top-right
                gray[h // 2 : h, 0 : w // 2],  # bottom-left
                gray[h // 2 : h, w // 2 : w],  # bottom-right
                gray[h // 4 : 3 * h // 4, w // 4 : 3 * w // 4],  # center
            ]

            sharpnesses: list[float] = []
            for region in regions:
                if region.size > 0:
                    lap_var = cv2.Laplacian(region, cv2.CV_64F).var()
                    sharpnesses.append(lap_var)

            if len(sharpnesses) >= 2:
                max_sharp = max(sharpnesses)
                min_sharp = min(sharpnesses)

                # Extreme ratio between regions is suspicious
                if min_sharp > 0 and max_sharp / min_sharp > 10:
                    flagged.append(idx)

        return flagged

    def _compute_pad_score(
        self,
        frame_metrics: list[FrameMetrics],
        reason_codes: list[str],
    ) -> float:
        """Compute overall PAD suspicion score."""
        if not frame_metrics:
            return 0.5

        # Base score from flagged frames
        total_flags = sum(len(m.pad_flags) for m in frame_metrics)
        flag_ratio = total_flags / (len(frame_metrics) * 3)  # Normalize by max flags per frame

        # Penalty for each reason code
        reason_penalty = len(reason_codes) * 0.15

        # Combine
        score = flag_ratio * 0.6 + reason_penalty
        return min(max(score, 0.0), 1.0)

    def calculate_sharpness(self, image: np.ndarray) -> float:
        """Calculate image sharpness using Laplacian variance."""
        gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        return float(cv2.Laplacian(gray, cv2.CV_64F).var())

    def detect_moire(self, image: np.ndarray) -> bool:
        """Detect moiré patterns using FFT analysis."""
        gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        score = self._detect_moire(gray)
        return score > self.moire_threshold

    # === LIVENESS DETECTION METHODS ===

    def _compute_eye_aspect_ratio(self, landmarks: np.ndarray) -> float:
        """Compute Eye Aspect Ratio (EAR) for blink detection.
        
        EAR is computed from 68-point facial landmarks.
        When eyes are open, EAR is relatively constant (~0.3).
        When eyes close (blink), EAR drops rapidly (<0.21).
        
        Args:
            landmarks: 68x2 array of facial landmark coordinates
            
        Returns:
            Average EAR of both eyes (0-0.5 typical range)
        """
        def eye_aspect_ratio(eye_points: np.ndarray) -> float:
            """Compute EAR for a single eye (6 points)."""
            # Vertical distances
            v1 = np.linalg.norm(eye_points[1] - eye_points[5])
            v2 = np.linalg.norm(eye_points[2] - eye_points[4])
            # Horizontal distance
            h = np.linalg.norm(eye_points[0] - eye_points[3])
            
            if h == 0:
                return 0.0
            
            ear = (v1 + v2) / (2.0 * h)
            return ear
        
        # 68-point landmark indices for eyes:
        # Left eye: 36-41 (indices 36, 37, 38, 39, 40, 41)
        # Right eye: 42-47 (indices 42, 43, 44, 45, 46, 47)
        left_eye = landmarks[36:42]
        right_eye = landmarks[42:48]
        
        left_ear = eye_aspect_ratio(left_eye)
        right_ear = eye_aspect_ratio(right_eye)
        
        # Average of both eyes
        return (left_ear + right_ear) / 2.0

    def _analyze_blinks(self, frame_metrics: list[FrameMetrics]) -> BlinkAnalysis:
        """Analyze blink patterns across frames.
        
        Detects natural blink patterns which indicate liveness.
        Printed photos and static images won't have blinks.
        
        Args:
            frame_metrics: Per-frame metrics including EAR values
            
        Returns:
            BlinkAnalysis with blink count and pattern assessment
        """
        ear_values = [
            m.eye_aspect_ratio for m in frame_metrics 
            if m.eye_aspect_ratio is not None
        ]
        
        if not ear_values:
            # No EAR data available (no landmarks detected)
            return BlinkAnalysis(
                blink_count=0,
                ear_values=[],
                blink_frames=[],
                avg_ear=0.0,
                min_ear=0.0,
                has_natural_blinks=False,
            )
        
        blink_frames = [
            m.frame_idx for m in frame_metrics 
            if m.blink_detected
        ]
        
        # Detect blink sequences (consecutive low-EAR frames)
        # A natural blink is typically 3-5 consecutive low-EAR frames
        blink_count = 0
        in_blink = False
        blink_length = 0
        
        for m in frame_metrics:
            if m.eye_aspect_ratio is None:
                continue
                
            if m.eye_aspect_ratio < self.ear_blink_threshold:
                if not in_blink:
                    in_blink = True
                    blink_length = 1
                else:
                    blink_length += 1
            else:
                if in_blink:
                    # End of blink - check if it was a natural blink (2-8 frames)
                    if 2 <= blink_length <= 8:
                        blink_count += 1
                    in_blink = False
                    blink_length = 0
        
        # Check if still in blink at end
        if in_blink and 2 <= blink_length <= 8:
            blink_count += 1
        
        avg_ear = mean(ear_values) if ear_values else 0.0
        min_ear = min(ear_values) if ear_values else 0.0
        
        # Natural blinks expected: at least min_blinks_expected in the video
        # Also check that EAR variance exists (not constant)
        has_variance = len(ear_values) >= 5 and stdev(ear_values) > 0.02 if len(ear_values) >= 5 else False
        has_natural_blinks = blink_count >= self.min_blinks_expected and has_variance
        
        return BlinkAnalysis(
            blink_count=blink_count,
            ear_values=ear_values,
            blink_frames=blink_frames,
            avg_ear=avg_ear,
            min_ear=min_ear,
            has_natural_blinks=has_natural_blinks,
        )

    def _compute_lbp_texture_score(self, frame: np.ndarray) -> float:
        """Compute LBP texture score to distinguish real skin from printed photos.
        
        Local Binary Patterns (LBP) capture micro-texture patterns.
        Real skin has characteristic texture from pores, fine lines.
        Printed photos have different texture from paper/ink patterns.
        
        Args:
            frame: BGR image
            
        Returns:
            Score from 0-1 where higher = more like real skin texture
        """
        # Convert to grayscale
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        
        # Detect face region for texture analysis
        skin_mask = self._segment_skin_region(frame)
        
        if skin_mask is None:
            # No skin region detected, analyze whole image
            roi = gray
        else:
            # Apply mask to focus on skin regions
            roi = cv2.bitwise_and(gray, gray, mask=skin_mask)
            # Crop to bounding box of skin region
            coords = np.column_stack(np.where(skin_mask > 0))
            if len(coords) > 100:
                y1, x1 = coords.min(axis=0)
                y2, x2 = coords.max(axis=0)
                roi = gray[y1:y2, x1:x2]
            else:
                roi = gray
        
        if roi.size < 100:
            return 0.5  # Not enough data
        
        # Compute LBP
        # radius=1, n_points=8 is classic LBP
        # radius=2, n_points=16 captures more spatial structure
        radius = 2
        n_points = 8 * radius
        
        try:
            lbp = local_binary_pattern(roi, n_points, radius, method='uniform')
        except Exception:
            return 0.5  # LBP computation failed
        
        # Compute histogram of LBP values
        n_bins = n_points + 2  # Uniform LBP has n_points + 2 patterns
        hist, _ = np.histogram(lbp.ravel(), bins=n_bins, range=(0, n_bins))
        hist = hist.astype(float)
        hist /= (hist.sum() + 1e-7)
        
        # Real skin texture characteristics:
        # - Higher entropy (more varied patterns)
        # - Specific distribution of uniform patterns
        # Printed photos tend to have:
        # - Lower entropy (more uniform patterns from printing)
        # - Different pattern distribution (paper/ink artifacts)
        
        # Compute histogram entropy
        hist_nonzero = hist[hist > 0]
        entropy = -np.sum(hist_nonzero * np.log2(hist_nonzero))
        max_entropy = np.log2(n_bins)
        normalized_entropy = entropy / max_entropy if max_entropy > 0 else 0
        
        # Check for printing artifacts (periodic patterns)
        # Printed photos often have regular patterns from halftone/dithering
        fft = np.fft.fft(hist)
        fft_mag = np.abs(fft[1:len(fft)//2])  # Exclude DC component
        if len(fft_mag) > 2:
            peak_ratio = np.max(fft_mag) / (np.mean(fft_mag) + 1e-7)
            # High peak ratio suggests periodic patterns (printed)
            periodicity_score = 1.0 - min(peak_ratio / 10, 1.0)
        else:
            periodicity_score = 0.5
        
        # Combine scores
        # Higher entropy and lower periodicity = more like real skin
        score = normalized_entropy * 0.6 + periodicity_score * 0.4
        
        return float(max(0, min(1, score)))

    def _analyze_texture(
        self,
        frame_metrics: list[FrameMetrics],
        frames: list[np.ndarray],
    ) -> dict:
        """Aggregate texture analysis across frames.
        
        Args:
            frame_metrics: Per-frame metrics with LBP scores
            frames: Original frames
            
        Returns:
            Dictionary with texture analysis results
        """
        lbp_scores = [m.lbp_texture_score for m in frame_metrics]
        
        if not lbp_scores:
            return {
                "avg_score": 0.0,
                "min_score": 0.0,
                "verdict": "no_data",
                "is_printed": False,
            }
        
        avg_score = mean(lbp_scores)
        min_score = min(lbp_scores)
        
        # If multiple frames consistently show printed texture, flag it
        low_score_ratio = sum(1 for s in lbp_scores if s < self.lbp_texture_threshold) / len(lbp_scores)
        
        is_printed = (
            avg_score < self.lbp_texture_threshold or
            low_score_ratio > 0.5
        )
        
        if avg_score >= 0.7:
            verdict = "real_skin"
        elif avg_score >= self.lbp_texture_threshold:
            verdict = "likely_real"
        elif avg_score >= 0.3:
            verdict = "suspicious"
        else:
            verdict = "likely_printed"
        
        return {
            "avg_score": avg_score,
            "min_score": min_score,
            "low_score_ratio": low_score_ratio,
            "verdict": verdict,
            "is_printed": is_printed,
        }

    def _detect_face_boundary_mismatch(
        self,
        frames: list[np.ndarray],
    ) -> tuple[bool, dict]:
        """Detect face boundary inconsistencies suggesting face swap.
        
        Face swaps typically exhibit:
        - Color/lighting discontinuities at face boundary
        - Texture differences between face and background
        - Unnatural edge sharpness at face contour
        - Noise level mismatches between regions
        
        Args:
            frames: List of BGR frame arrays
            
        Returns:
            Tuple of (detected: bool, evidence: dict)
        """
        if not frames:
            return False, {}
        
        evidence = {}
        suspicious_count = 0
        total_analyzed = 0
        
        # Sample frames (don't need to analyze every frame)
        sample_indices = list(range(0, len(frames), max(1, len(frames) // 5)))[:5]
        
        for idx in sample_indices:
            frame = frames[idx]
            
            # Segment skin region as proxy for face
            skin_mask = self._segment_skin_region(frame)
            if skin_mask is None:
                continue
            
            total_analyzed += 1
            
            # Get boundary region around skin/face
            boundary_mask = self._get_boundary_region(skin_mask, width=15)
            if boundary_mask is None:
                continue
            
            # Analyze boundary characteristics
            color_score = self._analyze_boundary_color(frame, skin_mask, boundary_mask)
            texture_score = self._analyze_boundary_texture(frame, skin_mask)
            edge_score = self._analyze_boundary_edges(frame, boundary_mask)
            
            # Combine scores (higher = more suspicious)
            combined_score = (color_score * 0.4 + texture_score * 0.3 + edge_score * 0.3)
            
            if combined_score > self.face_boundary_threshold:
                suspicious_count += 1
                evidence[f"frame_{idx}_color_score"] = color_score
                evidence[f"frame_{idx}_texture_score"] = texture_score
                evidence[f"frame_{idx}_edge_score"] = edge_score
        
        # If more than 40% of analyzed frames are suspicious, flag it
        if total_analyzed > 0:
            suspicious_ratio = suspicious_count / total_analyzed
            evidence["suspicious_ratio"] = suspicious_ratio
            evidence["frames_analyzed"] = total_analyzed
            
            if suspicious_ratio > 0.4:
                return True, evidence
        
        return False, evidence

    def _segment_skin_region(self, frame: np.ndarray) -> Optional[np.ndarray]:
        """Segment skin regions using color-based detection.
        
        Args:
            frame: BGR image array
            
        Returns:
            Binary mask of skin regions, or None if no significant skin found
        """
        # Convert to HSV and YCrCb for better skin detection
        hsv = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)
        ycrcb = cv2.cvtColor(frame, cv2.COLOR_BGR2YCrCb)
        
        # HSV skin range
        lower_hsv = np.array([0, 20, 70], dtype=np.uint8)
        upper_hsv = np.array([25, 255, 255], dtype=np.uint8)
        mask_hsv = cv2.inRange(hsv, lower_hsv, upper_hsv)
        
        # YCrCb skin range (more robust across skin tones)
        lower_ycrcb = np.array([0, 133, 77], dtype=np.uint8)
        upper_ycrcb = np.array([255, 173, 127], dtype=np.uint8)
        mask_ycrcb = cv2.inRange(ycrcb, lower_ycrcb, upper_ycrcb)
        
        # Combine masks
        skin_mask = cv2.bitwise_and(mask_hsv, mask_ycrcb)
        
        # Clean up mask
        kernel = np.ones((5, 5), np.uint8)
        skin_mask = cv2.morphologyEx(skin_mask, cv2.MORPH_CLOSE, kernel)
        skin_mask = cv2.morphologyEx(skin_mask, cv2.MORPH_OPEN, kernel)
        
        # Check if we have a significant skin region
        skin_ratio = np.sum(skin_mask > 0) / skin_mask.size
        if skin_ratio < 0.05 or skin_ratio > 0.7:
            # Too little or too much skin detected
            return None
        
        return skin_mask

    def _get_boundary_region(
        self,
        mask: np.ndarray,
        width: int = 10,
    ) -> Optional[np.ndarray]:
        """Extract boundary region around a mask.
        
        Args:
            mask: Binary mask
            width: Width of boundary region in pixels
            
        Returns:
            Binary mask of boundary region
        """
        kernel = np.ones((width, width), np.uint8)
        
        # Dilate to expand mask
        dilated = cv2.dilate(mask, kernel, iterations=1)
        # Erode to shrink mask
        eroded = cv2.erode(mask, kernel, iterations=1)
        
        # Boundary is the ring between dilated and eroded
        boundary = cv2.subtract(dilated, eroded)
        
        if np.sum(boundary > 0) < 100:
            return None
        
        return boundary

    def _analyze_boundary_color(
        self,
        frame: np.ndarray,
        skin_mask: np.ndarray,
        boundary_mask: np.ndarray,
    ) -> float:
        """Analyze color discontinuity at face boundary.
        
        Face swaps often have visible color transitions at the boundary
        between the swapped face and original background/skin.
        
        Returns:
            Score from 0-1 where 1 = high color discontinuity (suspicious)
        """
        # Get color values inside and outside the skin region
        inside_mask = cv2.erode(skin_mask, np.ones((10, 10), np.uint8), iterations=1)
        outside_mask = cv2.subtract(
            cv2.dilate(skin_mask, np.ones((25, 25), np.uint8), iterations=1),
            cv2.dilate(skin_mask, np.ones((10, 10), np.uint8), iterations=1)
        )
        
        # Calculate mean colors
        if np.sum(inside_mask > 0) < 100 or np.sum(outside_mask > 0) < 100:
            return 0.0
        
        inside_mean = cv2.mean(frame, mask=inside_mask)[:3]
        outside_mean = cv2.mean(frame, mask=outside_mask)[:3]
        boundary_mean = cv2.mean(frame, mask=boundary_mask)[:3]
        
        # Calculate color differences
        inside_to_boundary = np.sqrt(sum((a - b) ** 2 for a, b in zip(inside_mean, boundary_mean)))
        outside_to_boundary = np.sqrt(sum((a - b) ** 2 for a, b in zip(outside_mean, boundary_mean)))
        inside_to_outside = np.sqrt(sum((a - b) ** 2 for a, b in zip(inside_mean, outside_mean)))
        
        # Suspicious if boundary has very different characteristics from both regions
        # (indicates a visible seam)
        if inside_to_outside < 10:
            # Colors too similar, no clear boundary
            return 0.0
        
        # Check for sharp transition (boundary color very different from expectation)
        expected_boundary = [(a + b) / 2 for a, b in zip(inside_mean, outside_mean)]
        actual_boundary = boundary_mean
        boundary_deviation = np.sqrt(sum((a - b) ** 2 for a, b in zip(expected_boundary, actual_boundary)))
        
        # Normalize to 0-1 range
        score = min(boundary_deviation / 50, 1.0)
        
        return score

    def _analyze_boundary_texture(
        self,
        frame: np.ndarray,
        skin_mask: np.ndarray,
    ) -> float:
        """Analyze texture/noise mismatch between face and background.
        
        Face swaps often have different noise characteristics in the
        swapped region vs the original image.
        
        Returns:
            Score from 0-1 where 1 = high texture mismatch (suspicious)
        """
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        
        # Get inside and outside regions
        inside_mask = cv2.erode(skin_mask, np.ones((15, 15), np.uint8), iterations=1)
        outside_mask = cv2.subtract(
            255 - skin_mask,
            cv2.dilate(skin_mask, np.ones((15, 15), np.uint8), iterations=1) - skin_mask
        )
        outside_mask = np.clip(outside_mask, 0, 255).astype(np.uint8)
        
        # Estimate noise in each region using Laplacian variance
        if np.sum(inside_mask > 0) < 500 or np.sum(outside_mask > 0) < 500:
            return 0.0
        
        # Apply high-pass filter to isolate noise
        kernel = np.array([[-1, -1, -1], [-1, 8, -1], [-1, -1, -1]])
        filtered = cv2.filter2D(gray, -1, kernel)
        
        # Calculate noise variance in each region
        inside_pixels = filtered[inside_mask > 0]
        outside_pixels = filtered[outside_mask > 0]
        
        inside_var = np.var(inside_pixels)
        outside_var = np.var(outside_pixels)
        
        # Large difference in noise levels is suspicious
        if max(inside_var, outside_var) < 1:
            return 0.0
        
        noise_ratio = max(inside_var, outside_var) / (min(inside_var, outside_var) + 1)
        
        # Ratio > 2 is suspicious (face swaps often have different noise profiles)
        score = min((noise_ratio - 1) / 3, 1.0)
        
        return max(0.0, score)

    def _analyze_boundary_edges(
        self,
        frame: np.ndarray,
        boundary_mask: np.ndarray,
    ) -> float:
        """Analyze edge sharpness at face boundary.
        
        Face swaps often have unnaturally sharp edges at the blend boundary
        or visible seams from imperfect blending.
        
        Returns:
            Score from 0-1 where 1 = unnatural edge sharpness (suspicious)
        """
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        
        # Calculate gradient magnitude at boundary
        sobelx = cv2.Sobel(gray, cv2.CV_64F, 1, 0, ksize=3)
        sobely = cv2.Sobel(gray, cv2.CV_64F, 0, 1, ksize=3)
        gradient_mag = np.sqrt(sobelx ** 2 + sobely ** 2)
        
        # Get gradient values at boundary
        boundary_gradients = gradient_mag[boundary_mask > 0]
        
        if len(boundary_gradients) < 100:
            return 0.0
        
        # Calculate statistics
        mean_gradient = np.mean(boundary_gradients)
        max_gradient = np.percentile(boundary_gradients, 95)
        
        # Very high gradients at the boundary are suspicious
        # (natural faces have gradual color transitions)
        
        # Compare to overall image gradient
        overall_mean = np.mean(gradient_mag)
        
        if overall_mean < 1:
            return 0.0
        
        # Boundary should not be much sharper than the overall image
        sharpness_ratio = mean_gradient / overall_mean
        
        # Ratio > 2.5 is suspicious
        score = min((sharpness_ratio - 1.5) / 2, 1.0)
        
        return max(0.0, score)


# Singleton
_pad_analyzer: Optional[PADAnalyzer] = None


def get_pad_analyzer() -> PADAnalyzer:
    """Get cached PAD analyzer instance."""
    global _pad_analyzer
    if _pad_analyzer is None:
        _pad_analyzer = PADAnalyzer()
    return _pad_analyzer
