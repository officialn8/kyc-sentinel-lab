"""Replay attack artifact generation.

Simulates artifacts from screen capture/replay attacks:
- Moiré patterns from screen pixel grid
- Screen glare and reflections
- Color banding from limited display gamut
- Frame rate artifacts
"""

import numpy as np

from simulator.generator import AttackSeverity


def apply_replay_artifacts(
    image: np.ndarray,
    severity: AttackSeverity,
    rng: np.random.Generator,
) -> tuple[np.ndarray, list[str]]:
    """Apply replay attack artifacts to an image.
    
    Args:
        image: Input image (HWC, uint8)
        severity: Artifact intensity
        rng: Random number generator
        
    Returns:
        Tuple of (modified image, list of applied artifacts)
    """
    result = image.copy().astype(np.float32)
    artifacts = []
    
    # Severity multiplier
    mult = {"low": 0.3, "medium": 0.6, "high": 1.0}[severity.value]
    
    # Apply moiré pattern
    if rng.random() < 0.7 * mult:
        result = _add_moire_pattern(result, mult, rng)
        artifacts.append("moire_pattern")
    
    # Apply screen glare
    if rng.random() < 0.5 * mult:
        result = _add_screen_glare(result, mult, rng)
        artifacts.append("screen_glare")
    
    # Reduce color depth (banding)
    if rng.random() < 0.6 * mult:
        result = _add_color_banding(result, mult)
        artifacts.append("color_banding")
    
    # Reduce dynamic range
    if rng.random() < 0.4 * mult:
        result = _reduce_dynamic_range(result, mult)
        artifacts.append("reduced_dynamic_range")
    
    return np.clip(result, 0, 255).astype(np.uint8), artifacts


def _add_moire_pattern(
    image: np.ndarray,
    intensity: float,
    rng: np.random.Generator,
) -> np.ndarray:
    """Add moiré interference pattern.
    
    Creates a pattern that simulates the interference between
    camera sensor and screen pixel grid.
    """
    h, w = image.shape[:2]
    
    # Create interference pattern
    freq = rng.uniform(0.05, 0.15)
    angle = rng.uniform(0, np.pi)
    
    y, x = np.ogrid[:h, :w]
    pattern = np.sin(2 * np.pi * freq * (x * np.cos(angle) + y * np.sin(angle)))
    pattern = pattern * 20 * intensity
    
    # Apply pattern to all channels
    for c in range(3):
        image[:, :, c] += pattern
    
    return image


def _add_screen_glare(
    image: np.ndarray,
    intensity: float,
    rng: np.random.Generator,
) -> np.ndarray:
    """Add screen glare/reflection artifact.
    
    Creates a bright spot simulating light reflection on screen.
    """
    h, w = image.shape[:2]
    
    # Random glare position
    cx = rng.integers(w // 4, 3 * w // 4)
    cy = rng.integers(h // 4, 3 * h // 4)
    
    # Create Gaussian glare
    y, x = np.ogrid[:h, :w]
    sigma = rng.uniform(50, 150)
    glare = np.exp(-((x - cx) ** 2 + (y - cy) ** 2) / (2 * sigma ** 2))
    glare = glare * 100 * intensity
    
    # Add to image
    for c in range(3):
        image[:, :, c] += glare
    
    return image


def _add_color_banding(
    image: np.ndarray,
    intensity: float,
) -> np.ndarray:
    """Add color banding from limited color depth.
    
    Simulates the effect of screen display with fewer color levels.
    """
    # Reduce color levels
    levels = int(32 - 16 * intensity)  # 32 levels at low, 16 at high
    image = (image / 255 * levels).astype(np.int32)
    image = (image / levels * 255).astype(np.float32)
    
    return image


def _reduce_dynamic_range(
    image: np.ndarray,
    intensity: float,
) -> np.ndarray:
    """Reduce dynamic range (screens can't display full range).
    
    Compresses highlights and lifts shadows.
    """
    # Compress range
    black_point = 20 * intensity
    white_point = 255 - 30 * intensity
    
    image = np.clip(image, black_point, white_point)
    image = (image - black_point) / (white_point - black_point) * 255
    
    return image




