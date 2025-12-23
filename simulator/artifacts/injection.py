"""Injection attack artifact generation.

Simulates artifacts from virtual camera/video injection:
- Unnatural sharpness boundaries
- Noise level mismatches
- Compression artifacts
- Perfect temporal consistency (too perfect)
"""

import numpy as np

from simulator.generator import AttackSeverity


def apply_injection_artifacts(
    image: np.ndarray,
    severity: AttackSeverity,
    rng: np.random.Generator,
) -> tuple[np.ndarray, list[str]]:
    """Apply injection attack artifacts to an image.
    
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
    
    # Over-sharpening (virtual cameras often over-sharpen)
    if rng.random() < 0.7 * mult:
        result = _add_oversharpening(result, mult)
        artifacts.append("over_sharpening")
    
    # Noise mismatch (different noise characteristics)
    if rng.random() < 0.6 * mult:
        result = _add_noise_mismatch(result, mult, rng)
        artifacts.append("noise_mismatch")
    
    # Compression artifacts (double compression)
    if rng.random() < 0.5 * mult:
        result = _add_compression_artifacts(result, mult, rng)
        artifacts.append("compression_artifacts")
    
    # Color shift (virtual camera color processing)
    if rng.random() < 0.4 * mult:
        result = _add_color_shift(result, mult, rng)
        artifacts.append("color_shift")
    
    return np.clip(result, 0, 255).astype(np.uint8), artifacts


def _add_oversharpening(
    image: np.ndarray,
    intensity: float,
) -> np.ndarray:
    """Add over-sharpening artifacts.
    
    Creates unnaturally sharp edges typical of virtual cameras.
    """
    # Simple unsharp mask simulation
    kernel_size = 5
    
    # Create blurred version (approximate with averaging)
    blurred = image.copy()
    for c in range(3):
        for _ in range(2):
            blurred[:, :, c] = np.roll(blurred[:, :, c], 1, axis=0) * 0.25 + \
                               np.roll(blurred[:, :, c], -1, axis=0) * 0.25 + \
                               np.roll(blurred[:, :, c], 1, axis=1) * 0.25 + \
                               np.roll(blurred[:, :, c], -1, axis=1) * 0.25
    
    # Enhance edges
    amount = 1.0 + intensity * 2.0
    result = image + (image - blurred) * amount
    
    return result


def _add_noise_mismatch(
    image: np.ndarray,
    intensity: float,
    rng: np.random.Generator,
) -> np.ndarray:
    """Add noise level mismatch.
    
    Creates regions with different noise characteristics.
    """
    h, w = image.shape[:2]
    
    # Add uniform synthetic noise (lacks natural camera noise texture)
    noise_level = 3 + 7 * intensity
    noise = rng.normal(0, noise_level, image.shape)
    
    # Make noise suspiciously uniform
    image = image + noise
    
    return image


def _add_compression_artifacts(
    image: np.ndarray,
    intensity: float,
    rng: np.random.Generator,
) -> np.ndarray:
    """Add compression artifacts.
    
    Simulates double/triple compression from video processing.
    """
    # Block-based quantization artifacts
    block_size = 8
    h, w = image.shape[:2]
    
    # Quantization strength
    q_strength = int(5 + 15 * intensity)
    
    for y in range(0, h - block_size, block_size):
        for x in range(0, w - block_size, block_size):
            block = image[y:y+block_size, x:x+block_size]
            # Quantize
            block = (block / q_strength).astype(np.int32) * q_strength
            image[y:y+block_size, x:x+block_size] = block.astype(np.float32)
    
    return image


def _add_color_shift(
    image: np.ndarray,
    intensity: float,
    rng: np.random.Generator,
) -> np.ndarray:
    """Add color shift from virtual camera processing.
    
    Virtual cameras often have slightly off color reproduction.
    """
    # Shift each channel slightly
    shifts = rng.uniform(-10 * intensity, 10 * intensity, 3)
    
    for c in range(3):
        image[:, :, c] += shifts[c]
    
    return image



