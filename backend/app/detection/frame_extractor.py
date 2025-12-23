"""
Video frame extraction using OpenCV.

Provides efficient frame sampling from video files for PAD analysis.
"""

import cv2
import numpy as np
from pathlib import Path
import tempfile
from dataclasses import dataclass
from typing import Optional


@dataclass
class VideoMetadata:
    """Metadata extracted from video."""

    duration: float  # seconds
    fps: float
    width: int
    height: int
    codec: str
    frame_count: int


@dataclass
class ExtractionResult:
    """Result of frame extraction."""

    frames: list[np.ndarray]
    fps: float
    total_frames: int
    duration_seconds: float
    resolution: tuple[int, int]
    
    # Backward compatible
    metadata: Optional[VideoMetadata] = None
    timestamps: list[float] = None

    def __post_init__(self):
        if self.metadata is None:
            self.metadata = VideoMetadata(
                duration=self.duration_seconds,
                fps=self.fps,
                width=self.resolution[0],
                height=self.resolution[1],
                codec="unknown",
                frame_count=self.total_frames,
            )
        if self.timestamps is None:
            self.timestamps = [
                i / self.fps if self.fps > 0 else 0.0
                for i in range(len(self.frames))
            ]


class FrameExtractor:
    """Video frame extraction using OpenCV.
    
    Supports uniform sampling or first-N frame extraction strategies.
    """

    def __init__(
        self,
        max_frames: int = 30,
        sample_strategy: str = "uniform",
        target_fps: float = 5.0,
    ) -> None:
        """
        Initialize frame extractor.
        
        Args:
            max_frames: Maximum frames to extract
            sample_strategy: "uniform" (evenly spaced) or "first_n" (first N frames)
            target_fps: Target frames per second for extraction
        """
        self.max_frames = max_frames
        self.sample_strategy = sample_strategy
        self.target_fps = target_fps

    def extract_from_bytes(
        self,
        video_bytes: bytes,
        max_frames: Optional[int] = None,
    ) -> ExtractionResult:
        """
        Extract frames from video bytes.
        
        Args:
            video_bytes: Raw video data
            max_frames: Override max_frames for this extraction
        
        Returns:
            ExtractionResult with frames and metadata
        """
        # Write to temp file (OpenCV needs file path)
        with tempfile.NamedTemporaryFile(suffix=".mp4", delete=False) as f:
            f.write(video_bytes)
            temp_path = f.name

        try:
            return self.extract_from_path(temp_path, max_frames)
        finally:
            Path(temp_path).unlink(missing_ok=True)

    def extract_from_path(
        self,
        video_path: str,
        max_frames: Optional[int] = None,
    ) -> ExtractionResult:
        """
        Extract frames from video file path.
        
        Args:
            video_path: Path to video file
            max_frames: Override max_frames for this extraction
            
        Returns:
            ExtractionResult with frames and metadata
        """
        cap = cv2.VideoCapture(video_path)

        if not cap.isOpened():
            raise ValueError(f"Could not open video: {video_path}")

        fps = cap.get(cv2.CAP_PROP_FPS)
        total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
        width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
        fourcc = int(cap.get(cv2.CAP_PROP_FOURCC))
        codec = "".join([chr((fourcc >> 8 * i) & 0xFF) for i in range(4)])
        duration = total_frames / fps if fps > 0 else 0

        # Use override or default max_frames
        n_frames = max_frames if max_frames is not None else self.max_frames

        # Determine which frames to extract
        if self.sample_strategy == "uniform":
            indices = self._uniform_sample(total_frames, n_frames)
        else:
            indices = list(range(min(total_frames, n_frames)))

        frames: list[np.ndarray] = []
        timestamps: list[float] = []
        
        for idx in indices:
            cap.set(cv2.CAP_PROP_POS_FRAMES, idx)
            ret, frame = cap.read()
            if ret:
                frames.append(frame)
                timestamps.append(idx / fps if fps > 0 else 0.0)

        cap.release()

        metadata = VideoMetadata(
            duration=duration,
            fps=fps,
            width=width,
            height=height,
            codec=codec,
            frame_count=total_frames,
        )

        return ExtractionResult(
            frames=frames,
            fps=fps,
            total_frames=total_frames,
            duration_seconds=duration,
            resolution=(width, height),
            metadata=metadata,
            timestamps=timestamps,
        )

    def extract_from_file(
        self,
        video_path: str | Path,
        max_frames: int = 100,
    ) -> ExtractionResult:
        """
        Extract frames from a video file.
        
        Alias for extract_from_path for backward compatibility.
        """
        return self.extract_from_path(str(video_path), max_frames)

    def _uniform_sample(self, total: int, n: int) -> list[int]:
        """Get uniformly spaced frame indices."""
        if total <= n:
            return list(range(total))

        step = total / n
        return [int(i * step) for i in range(n)]

    def get_video_metadata(self, video_path: str | Path) -> VideoMetadata:
        """
        Get metadata without extracting frames.
        
        Args:
            video_path: Path to video file
            
        Returns:
            Video metadata
        """
        cap = cv2.VideoCapture(str(video_path))

        if not cap.isOpened():
            raise ValueError(f"Could not open video: {video_path}")

        fps = cap.get(cv2.CAP_PROP_FPS)
        total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
        width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
        fourcc = int(cap.get(cv2.CAP_PROP_FOURCC))
        codec = "".join([chr((fourcc >> 8 * i) & 0xFF) for i in range(4)])
        duration = total_frames / fps if fps > 0 else 0

        cap.release()

        return VideoMetadata(
            duration=duration,
            fps=fps,
            width=width,
            height=height,
            codec=codec,
            frame_count=total_frames,
        )

    def extract_keyframes(
        self,
        video_path: str | Path,
        max_frames: int = 20,
    ) -> ExtractionResult:
        """
        Extract only keyframes (I-frames) from video.
        
        Note: This is a simplified implementation that uses scene change
        detection rather than actual I-frame detection, which would require
        ffprobe. For actual keyframe extraction, consider using ffmpeg directly.
        
        Args:
            video_path: Path to video file
            max_frames: Maximum number of keyframes to extract
            
        Returns:
            Extraction result with keyframes only
        """
        cap = cv2.VideoCapture(str(video_path))

        if not cap.isOpened():
            raise ValueError(f"Could not open video: {video_path}")

        fps = cap.get(cv2.CAP_PROP_FPS)
        total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
        width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
        fourcc = int(cap.get(cv2.CAP_PROP_FOURCC))
        codec = "".join([chr((fourcc >> 8 * i) & 0xFF) for i in range(4)])
        duration = total_frames / fps if fps > 0 else 0

        frames: list[np.ndarray] = []
        timestamps: list[float] = []
        prev_frame = None
        frame_idx = 0
        scene_threshold = 30.0  # Threshold for scene change detection

        while len(frames) < max_frames and cap.isOpened():
            ret, frame = cap.read()
            if not ret:
                break

            # Detect scene changes using frame difference
            if prev_frame is not None:
                diff = cv2.absdiff(
                    cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY),
                    cv2.cvtColor(prev_frame, cv2.COLOR_BGR2GRAY),
                )
                mean_diff = np.mean(diff)
                
                if mean_diff > scene_threshold:
                    frames.append(frame)
                    timestamps.append(frame_idx / fps if fps > 0 else 0.0)
            else:
                # Always include first frame
                frames.append(frame)
                timestamps.append(0.0)

            prev_frame = frame
            frame_idx += 1

        cap.release()

        metadata = VideoMetadata(
            duration=duration,
            fps=fps,
            width=width,
            height=height,
            codec=codec,
            frame_count=total_frames,
        )

        return ExtractionResult(
            frames=frames,
            fps=fps,
            total_frames=total_frames,
            duration_seconds=duration,
            resolution=(width, height),
            metadata=metadata,
            timestamps=timestamps,
        )


def get_frame_extractor(max_frames: int = 30) -> FrameExtractor:
    """Get a frame extractor instance."""
    return FrameExtractor(max_frames=max_frames)
