"""
Presentation Attack Detection heuristics using OpenCV.

Analyzes video frames for signs of:
- Replay attacks (screen capture artifacts)
- Injection attacks (unnatural image properties)
- Face boundary anomalies

These are heuristic-based signals, not learned models.
They provide interpretable reason codes.
"""

import numpy as np
from dataclasses import dataclass
from typing import Optional
from statistics import mean, stdev

import cv2
from scipy import fftpack


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
class PADResult:
    """Result of PAD analysis."""

    frame_metrics: list[FrameMetrics]
    overall_pad_score: float  # 0-1, higher = more suspicious
    reason_codes: list[str]
    evidence: dict
    flagged_frames: list[int]
    
    # Backward compatible fields
    pad_score: float = 0.0
    replay_suspected: bool = False
    injection_suspected: bool = False
    low_motion: bool = False
    frame_stutter: bool = False
    face_boundary_mismatch: bool = False
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


class PADAnalyzer:
    """Presentation Attack Detection analyzer using OpenCV heuristics.
    
    Implements detection of:
    - Replay attacks via moiré pattern detection in FFT domain
    - Injection attacks via unnatural sharpness boundaries
    - Static image attacks via motion entropy analysis
    - Video replay via frame stutter detection
    """

    def __init__(
        self,
        sharpness_threshold: float = 100.0,
        noise_floor: float = 5.0,
        noise_ceiling: float = 50.0,
        motion_entropy_min: float = 0.1,
        moire_threshold: float = 0.15,
        color_temp_variance_max: float = 0.2,
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
        """
        self.sharpness_threshold = sharpness_threshold
        self.noise_floor = noise_floor
        self.noise_ceiling = noise_ceiling
        self.motion_entropy_min = motion_entropy_min
        self.moire_threshold = moire_threshold
        self.color_temp_variance_max = color_temp_variance_max

    def analyze_frames(self, frames: list[np.ndarray]) -> PADResult:
        """
        Analyze video frames for presentation attack indicators.
        
        Args:
            frames: List of BGR frame arrays
        
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
            metrics = self._analyze_single_frame(frame, prev_frame, idx)
            frame_metrics.append(metrics)

            if metrics.motion_entropy > 0:
                motion_scores.append(metrics.motion_entropy)

            prev_frame = frame

        # Aggregate analysis
        reason_codes: list[str] = []
        evidence: dict = {}
        flagged_frames: list[int] = []

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

        # Compute overall PAD score
        overall_score = self._compute_pad_score(frame_metrics, reason_codes)

        return PADResult(
            frame_metrics=frame_metrics,
            overall_pad_score=overall_score,
            reason_codes=reason_codes,
            evidence=evidence,
            flagged_frames=list(set(flagged_frames)),
        )

    def _analyze_single_frame(
        self,
        frame: np.ndarray,
        prev_frame: Optional[np.ndarray],
        idx: int,
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

        # Per-frame flags
        pad_flags: list[str] = []
        if sharpness < self.sharpness_threshold * 0.5:
            pad_flags.append("low_sharpness")
        if noise_level > self.noise_ceiling:
            pad_flags.append("high_noise")
        if moire_score > self.moire_threshold:
            pad_flags.append("moire_detected")

        return FrameMetrics(
            frame_idx=idx,
            sharpness=sharpness,
            noise_level=noise_level,
            motion_entropy=motion_entropy,
            color_temperature=color_temperature,
            moire_score=moire_score,
            pad_flags=pad_flags,
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


# Singleton
_pad_analyzer: Optional[PADAnalyzer] = None


def get_pad_analyzer() -> PADAnalyzer:
    """Get cached PAD analyzer instance."""
    global _pad_analyzer
    if _pad_analyzer is None:
        _pad_analyzer = PADAnalyzer()
    return _pad_analyzer
