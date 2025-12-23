"""Document tampering artifact generation.

Simulates artifacts from document manipulation:
- Text misalignment
- Font inconsistencies
- Edge/cutout artifacts
- Background inconsistencies
"""

import numpy as np

from simulator.generator import AttackSeverity


def apply_doc_tamper_artifacts(
    image: np.ndarray,
    severity: AttackSeverity,
    rng: np.random.Generator,
) -> tuple[np.ndarray, list[str]]:
    """Apply document tampering artifacts to an image.
    
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
    
    # Text region artifacts
    if rng.random() < 0.7 * mult:
        result = _add_text_region_artifacts(result, mult, rng)
        artifacts.append("text_region_artifacts")
    
    # Background inconsistency
    if rng.random() < 0.6 * mult:
        result = _add_background_inconsistency(result, mult, rng)
        artifacts.append("background_inconsistency")
    
    # Edge/cutout artifacts
    if rng.random() < 0.5 * mult:
        result = _add_cutout_artifacts(result, mult, rng)
        artifacts.append("cutout_artifacts")
    
    # JPEG artifacts in specific regions
    if rng.random() < 0.4 * mult:
        result = _add_localized_compression(result, mult, rng)
        artifacts.append("localized_compression")
    
    return np.clip(result, 0, 255).astype(np.uint8), artifacts


def _add_text_region_artifacts(
    image: np.ndarray,
    intensity: float,
    rng: np.random.Generator,
) -> np.ndarray:
    """Add artifacts in simulated text regions.
    
    Creates regions that look like tampered text areas.
    """
    h, w = image.shape[:2]
    
    # Create multiple text-like regions
    num_regions = rng.integers(2, 5)
    
    for _ in range(num_regions):
        # Random rectangular region
        x1 = rng.integers(w // 4, w // 2)
        y1 = rng.integers(h // 4, 3 * h // 4)
        rw = rng.integers(50, 150)
        rh = rng.integers(15, 30)
        
        x2, y2 = min(x1 + rw, w), min(y1 + rh, h)
        
        # Slightly different characteristics in region
        region = image[y1:y2, x1:x2]
        
        # Brightness shift
        shift = rng.uniform(-15, 15) * intensity
        region = region + shift
        
        # Different sharpness
        if rng.random() > 0.5:
            kernel = np.array([[0, -1, 0], [-1, 5, -1], [0, -1, 0]]) * intensity * 0.3
            for c in range(3):
                # Simple convolution
                region[:, :, c] = region[:, :, c] * (1 + intensity * 0.2)
        
        image[y1:y2, x1:x2] = region
    
    return image


def _add_background_inconsistency(
    image: np.ndarray,
    intensity: float,
    rng: np.random.Generator,
) -> np.ndarray:
    """Add background pattern inconsistency.
    
    Creates subtle variations in document background.
    """
    h, w = image.shape[:2]
    
    # Create patches with different noise levels
    patch_size = 50
    
    for y in range(0, h - patch_size, patch_size):
        for x in range(0, w - patch_size, patch_size):
            if rng.random() < 0.3 * intensity:
                patch = image[y:y+patch_size, x:x+patch_size]
                
                # Different noise in this patch
                noise_level = rng.uniform(2, 8) * intensity
                noise = rng.normal(0, noise_level, patch.shape)
                
                image[y:y+patch_size, x:x+patch_size] = patch + noise
    
    return image


def _add_cutout_artifacts(
    image: np.ndarray,
    intensity: float,
    rng: np.random.Generator,
) -> np.ndarray:
    """Add cutout/paste artifacts.
    
    Creates visible boundaries where content was cut and pasted.
    """
    h, w = image.shape[:2]
    
    # Random rectangular cutout region
    x1 = rng.integers(w // 4, w // 2)
    y1 = rng.integers(h // 4, 3 * h // 4)
    rw = rng.integers(80, 200)
    rh = rng.integers(40, 80)
    
    x2, y2 = min(x1 + rw, w - 1), min(y1 + rh, h - 1)
    
    # Add visible border
    border_width = int(2 + 3 * intensity)
    border_intensity = 20 * intensity
    
    # Top and bottom borders
    image[y1:y1+border_width, x1:x2] += border_intensity
    image[y2-border_width:y2, x1:x2] += border_intensity
    
    # Left and right borders
    image[y1:y2, x1:x1+border_width] += border_intensity
    image[y1:y2, x2-border_width:x2] += border_intensity
    
    return image


def _add_localized_compression(
    image: np.ndarray,
    intensity: float,
    rng: np.random.Generator,
) -> np.ndarray:
    """Add compression artifacts in specific regions.
    
    Simulates re-saving parts of document at different quality.
    """
    h, w = image.shape[:2]
    
    # Random region
    x1 = rng.integers(w // 4, w // 2)
    y1 = rng.integers(h // 4, 3 * h // 4)
    rw = rng.integers(100, 200)
    rh = rng.integers(50, 100)
    
    x2, y2 = min(x1 + rw, w), min(y1 + rh, h)
    
    # Apply block-based quantization
    block_size = 8
    q_strength = int(10 + 20 * intensity)
    
    region = image[y1:y2, x1:x2]
    rh_r, rw_r = region.shape[:2]
    
    for by in range(0, rh_r - block_size, block_size):
        for bx in range(0, rw_r - block_size, block_size):
            block = region[by:by+block_size, bx:bx+block_size]
            block = (block / q_strength).astype(np.int32) * q_strength
            region[by:by+block_size, bx:bx+block_size] = block.astype(np.float32)
    
    image[y1:y2, x1:x2] = region
    
    return image




