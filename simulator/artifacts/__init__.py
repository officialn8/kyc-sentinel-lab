"""Artifact generation modules."""

from simulator.artifacts.replay import apply_replay_artifacts
from simulator.artifacts.injection import apply_injection_artifacts
from simulator.artifacts.face_swap import apply_face_swap_artifacts
from simulator.artifacts.doc_tamper import apply_doc_tamper_artifacts

__all__ = [
    "apply_replay_artifacts",
    "apply_injection_artifacts",
    "apply_face_swap_artifacts",
    "apply_doc_tamper_artifacts",
]




