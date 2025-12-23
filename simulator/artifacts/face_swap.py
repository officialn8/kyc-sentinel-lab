"""Face swap artifact generation.

Simulates artifacts from face swapping without using actual deepfakes:
- Blending boundaries around face region
- Color/lighting mismatches
- Texture inconsistencies
- Edge artifacts
"""

import numpy as np

from simulator.generator import AttackSeverity


def apply_face_swap_artifacts(
    image: np.ndarray,
    severity: AttackSeverity,
    rng: np.random.Generator,
) -> tuple[np.ndarray, list[str]]:
    """Apply face swap artifacts to an image.
    
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
    
    # Create a face-like region mask (ellipse in center)
    h, w = image.shape[:2]
    mask = _create_face_mask(h, w, rng)
    
    # Blending boundary artifacts
    if rng.random() < 0.8 * mult:
        result = _add_blending_artifacts(result, mask, mult, rng)
        artifacts.append("blending_boundary")
    
    # Color mismatch in face region
    if rng.random() < 0.6 * mult:
        result = _add_color_mismatch(result, mask, mult, rng)
        artifacts.append("color_mismatch")
    
    # Texture inconsistency
    if rng.random() < 0.5 * mult:
        result = _add_texture_inconsistency(result, mask, mult, rng)
        artifacts.append("texture_inconsistency")
    
    # Edge artifacts
    if rng.random() < 0.4 * mult:
        result = _add_edge_artifacts(result, mask, mult, rng)
        artifacts.append("edge_artifacts")
    
    return np.clip(result, 0, 255).astype(np.uint8), artifacts


def _create_face_mask(
    h: int,
    w: int,
    rng: np.random.Generator,
) -> np.ndarray:
    """Create an elliptical mask simulating face region."""
    # Face-like ellipse in center of image
    cy, cx = h // 2 + rng.integers(-h // 8, h // 8), w // 2 + rng.integers(-w // 8, w // 8)
    ry, rx = h // 4 + rng.integers(-20, 20), w // 5 + rng.integers(-15, 15)
    
    y, x = np.ogrid[:h, :w]
    mask = ((x - cx) ** 2 / rx ** 2 + (y - cy) ** 2 / ry ** 2) <= 1
    
    return mask.astype(np.float32)


def _add_blending_artifacts(
    image: np.ndarray,
    mask: np.ndarray,
    intensity: float,
    rng: np.random.Generator,
) -> np.ndarray:
    """Add visible blending boundary around face region.
    
    Creates a visible seam where the face meets the background.
    """
    # Find boundary by dilating and subtracting
    from scipy import ndimage
    
    # Create boundary mask
    dilated = ndimage.binary_dilation(mask, iterations=int(3 + 5 * intensity))
    boundary = dilated.astype(np.float32) - mask
    
    # Add brightness shift at boundary
    brightness_shift = rng.uniform(15, 40) * intensity
    for c in range(3):
        image[:, :, c] += boundary * brightness_shift * (1 if rng.random() > 0.5 else -1)
    
    return image


def _add_color_mismatch(
    image: np.ndarray,
    mask: np.ndarray,
    intensity: float,
    rng: np.random.Generator,
) -> np.ndarray:
    """Add color/lighting mismatch in face region.
    
    Face region has slightly different color temperature or saturation.
    """
    # Color shift for face region
    shifts = rng.uniform(-20 * intensity, 20 * intensity, 3)
    
    mask_3d = mask[:, :, np.newaxis]
    color_shift = np.ones_like(image) * shifts
    
    image = image + mask_3d * color_shift
    
    return image


def _add_texture_inconsistency(
    image: np.ndarray,
    mask: np.ndarray,
    intensity: float,
    rng: np.random.Generator,
) -> np.ndarray:
    """Add texture inconsistency in face region.
    
    Face region has different noise/texture characteristics.
    """
    # Different noise level in face region
    face_noise = rng.normal(0, 5 * intensity, image.shape)
    bg_noise = rng.normal(0, 2 * intensity, image.shape)
    
    mask_3d = mask[:, :, np.newaxis]
    combined_noise = mask_3d * face_noise + (1 - mask_3d) * bg_noise
    
    image = image + combined_noise
    
    return image


def _add_edge_artifacts(
    image: np.ndarray,
    mask: np.ndarray,
    intensity: float,
    rng: np.random.Generator,
) -> np.ndarray:
    """Add edge artifacts around face boundary.
    
    Creates visible edges from imperfect face cutout.
    """
    from scipy import ndimage
    
    # Find edges
    edges = ndimage.sobel(mask)
    edges = np.abs(edges)
    edges = edges / (edges.max() + 1e-6)
    
    # Enhance edges
    edge_strength = 30 * intensity
    for c in range(3):
        image[:, :, c] += edges * edge_strength
    
    return image




