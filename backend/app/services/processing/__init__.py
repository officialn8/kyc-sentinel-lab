"""Processing package - modular backend implementations.

This package provides processing backends for KYC session analysis.
Exports maintain backward compatibility with the original processing.py module.
"""

from app.config import settings
from app.services.processing.protocol import (
    ProcessingBackend,
    FrameResult,
    FaceResult,
    DocResult,
)
from app.services.processing.local import LocalBackend
from app.services.processing.modal import ModalBackend


def get_processing_backend() -> ProcessingBackend:
    """Get the configured processing backend.
    
    Returns LocalBackend for development, ModalBackend for production.
    """
    if settings.use_modal:
        return ModalBackend()
    return LocalBackend()


# Backward compatibility exports
__all__ = [
    "ProcessingBackend",
    "FrameResult",
    "FaceResult",
    "DocResult",
    "LocalBackend",
    "ModalBackend",
    "get_processing_backend",
]
