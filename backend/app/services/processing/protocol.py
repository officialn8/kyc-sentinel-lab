"""Processing backend protocol and shared data structures.

Defines the interface for processing backends and the result dataclasses
used across local and Modal implementations.
"""

from abc import abstractmethod
from dataclasses import dataclass
from typing import Protocol


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
    """Protocol for processing backend implementations.
    
    Both LocalBackend and ModalBackend must implement this interface.
    """

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
