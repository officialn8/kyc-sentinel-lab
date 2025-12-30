"""Main artifact generator class."""

from dataclasses import dataclass
from enum import Enum
from typing import Optional

import numpy as np


class AttackFamily(str, Enum):
    """Attack family enumeration."""

    REPLAY = "replay"
    INJECTION = "injection"
    FACE_SWAP = "face_swap"
    DOC_TAMPER = "doc_tamper"
    BENIGN = "benign"


class AttackSeverity(str, Enum):
    """Attack severity levels."""

    LOW = "low"
    MEDIUM = "medium"
    HIGH = "high"


@dataclass
class GeneratedArtifact:
    """Result of artifact generation."""

    image: np.ndarray
    attack_family: AttackFamily
    attack_severity: AttackSeverity
    artifacts_applied: list[str]
    metadata: dict


class ArtifactGenerator:
    """Generator for synthetic KYC attack artifacts.
    
    This class creates realistic-looking attack artifacts for testing
    the detection pipeline without using actual deepfake technology.
    """

    def __init__(self, seed: Optional[int] = None) -> None:
        """Initialize the generator.
        
        Args:
            seed: Random seed for reproducibility
        """
        self.rng = np.random.default_rng(seed)

    def generate(
        self,
        base_image: np.ndarray,
        attack_family: AttackFamily,
        severity: AttackSeverity = AttackSeverity.MEDIUM,
    ) -> GeneratedArtifact:
        """Generate attack artifacts on a base image.
        
        Args:
            base_image: Input image to apply artifacts to
            attack_family: Type of attack to simulate
            severity: Intensity of the artifacts
            
        Returns:
            Generated artifact with modified image
        """
        if attack_family == AttackFamily.BENIGN:
            return GeneratedArtifact(
                image=base_image.copy(),
                attack_family=attack_family,
                attack_severity=severity,
                artifacts_applied=[],
                metadata={},
            )

        generators = {
            AttackFamily.REPLAY: self._generate_replay_artifacts,
            AttackFamily.INJECTION: self._generate_injection_artifacts,
            AttackFamily.FACE_SWAP: self._generate_face_swap_artifacts,
            AttackFamily.DOC_TAMPER: self._generate_doc_tamper_artifacts,
        }

        generator = generators.get(attack_family)
        if generator:
            return generator(base_image, severity)

        raise ValueError(f"Unknown attack family: {attack_family}")

    def _generate_replay_artifacts(
        self,
        image: np.ndarray,
        severity: AttackSeverity,
    ) -> GeneratedArtifact:
        """Generate replay attack artifacts (screen display simulation).
        
        Simulates artifacts from displaying an image on a screen:
        - Moiré patterns
        - Screen glare
        - Color banding
        - Reduced dynamic range
        """
        from simulator.artifacts.replay import apply_replay_artifacts
        
        result_image, artifacts = apply_replay_artifacts(image, severity, self.rng)
        
        return GeneratedArtifact(
            image=result_image,
            attack_family=AttackFamily.REPLAY,
            attack_severity=severity,
            artifacts_applied=artifacts,
            metadata={"type": "screen_replay"},
        )

    def _generate_injection_artifacts(
        self,
        image: np.ndarray,
        severity: AttackSeverity,
    ) -> GeneratedArtifact:
        """Generate injection attack artifacts (virtual camera simulation).
        
        Simulates artifacts from virtual camera injection:
        - Unnatural sharpness boundaries
        - Noise level mismatches
        - Compression artifacts
        """
        from simulator.artifacts.injection import apply_injection_artifacts
        
        result_image, artifacts = apply_injection_artifacts(image, severity, self.rng)
        
        return GeneratedArtifact(
            image=result_image,
            attack_family=AttackFamily.INJECTION,
            attack_severity=severity,
            artifacts_applied=artifacts,
            metadata={"type": "virtual_camera"},
        )

    def _generate_face_swap_artifacts(
        self,
        image: np.ndarray,
        severity: AttackSeverity,
    ) -> GeneratedArtifact:
        """Generate face swap artifacts (blending seams simulation).
        
        Simulates artifacts from face swapping without actual GAN:
        - Blending boundaries
        - Color mismatches at edges
        - Texture inconsistencies
        """
        from simulator.artifacts.face_swap import apply_face_swap_artifacts
        
        result_image, artifacts = apply_face_swap_artifacts(image, severity, self.rng)
        
        return GeneratedArtifact(
            image=result_image,
            attack_family=AttackFamily.FACE_SWAP,
            attack_severity=severity,
            artifacts_applied=artifacts,
            metadata={"type": "face_swap_simulation"},
        )

    def _generate_doc_tamper_artifacts(
        self,
        image: np.ndarray,
        severity: AttackSeverity,
    ) -> GeneratedArtifact:
        """Generate document tampering artifacts.
        
        Simulates document manipulation:
        - Text misalignment
        - Font inconsistencies
        - Edge artifacts
        """
        from simulator.artifacts.doc_tamper import apply_doc_tamper_artifacts
        
        result_image, artifacts = apply_doc_tamper_artifacts(image, severity, self.rng)
        
        return GeneratedArtifact(
            image=result_image,
            attack_family=AttackFamily.DOC_TAMPER,
            attack_severity=severity,
            artifacts_applied=artifacts,
            metadata={"type": "document_manipulation"},
        )

    def generate_batch(
        self,
        base_images: list[np.ndarray],
        attack_family: AttackFamily,
        severity: AttackSeverity = AttackSeverity.MEDIUM,
    ) -> list[GeneratedArtifact]:
        """Generate artifacts for a batch of images.
        
        Args:
            base_images: List of input images
            attack_family: Type of attack to simulate
            severity: Intensity of the artifacts
            
        Returns:
            List of generated artifacts
        """
        return [
            self.generate(img, attack_family, severity)
            for img in base_images
        ]











