"""Synthetic base image generation for simulator.

Generates synthetic selfie and ID document images that have enough structure
for the detection pipeline to analyze, but don't contain real personal data.
"""

import numpy as np
import cv2
from typing import Optional


def generate_synthetic_selfie(
    rng: Optional[np.random.Generator] = None,
    width: int = 640,
    height: int = 480,
) -> np.ndarray:
    """Generate a synthetic selfie-like image.
    
    Creates an image with:
    - A face-like oval shape with skin tone
    - Simple eye, nose, and mouth features
    - Background with subtle gradient
    
    The goal is to produce an image that face detectors can process,
    not to create a photorealistic face.
    
    Args:
        rng: Random number generator for reproducibility
        width: Image width
        height: Image height
        
    Returns:
        BGR image as numpy array
    """
    if rng is None:
        rng = np.random.default_rng()
    
    # Create background with subtle gradient
    image = np.zeros((height, width, 3), dtype=np.uint8)
    
    # Random background color (muted indoor colors)
    bg_base = rng.integers(120, 200, 3)
    for y in range(height):
        gradient_factor = 1.0 - (y / height) * 0.15
        for c in range(3):
            image[y, :, c] = int(bg_base[c] * gradient_factor)
    
    # Add some noise to background
    noise = rng.integers(-10, 10, image.shape, dtype=np.int16)
    image = np.clip(image.astype(np.int16) + noise, 0, 255).astype(np.uint8)
    
    # Face parameters
    face_center_x = width // 2 + rng.integers(-20, 20)
    face_center_y = height // 2 + rng.integers(-30, 10)
    face_width = int(width * 0.35) + rng.integers(-20, 20)
    face_height = int(face_width * 1.3) + rng.integers(-10, 10)
    
    # Skin tone variations (natural range)
    skin_tones = [
        (200, 180, 160),  # Light
        (180, 150, 130),  # Medium light
        (150, 120, 100),  # Medium
        (120, 90, 70),    # Medium dark
        (90, 70, 50),     # Dark
    ]
    skin_base = skin_tones[rng.integers(0, len(skin_tones))]
    skin_variation = rng.integers(-15, 15, 3)
    # Convert to Python ints (OpenCV requires native ints, not numpy ints)
    skin_color = tuple(int(c) for c in np.clip(np.array(skin_base) + skin_variation, 0, 255))
    
    # Draw face oval
    cv2.ellipse(
        image,
        (face_center_x, face_center_y),
        (face_width // 2, face_height // 2),
        0, 0, 360,
        skin_color[::-1],  # BGR
        -1
    )
    
    # Add face shading for depth
    darker_skin = tuple(max(0, c - 30) for c in skin_color)
    cv2.ellipse(
        image,
        (face_center_x - face_width // 6, face_center_y),
        (face_width // 4, face_height // 3),
        0, 0, 360,
        darker_skin[::-1],
        -1
    )
    
    # Smooth the face region
    face_mask = np.zeros((height, width), dtype=np.uint8)
    cv2.ellipse(face_mask, (face_center_x, face_center_y), 
                (face_width // 2, face_height // 2), 0, 0, 360, 255, -1)
    image = cv2.GaussianBlur(image, (15, 15), 0)
    
    # Eye parameters
    eye_y = face_center_y - face_height // 5
    eye_spacing = face_width // 3
    eye_width = face_width // 6
    eye_height = eye_width // 2
    
    # Draw eyes (white part)
    for eye_x in [face_center_x - eye_spacing, face_center_x + eye_spacing]:
        cv2.ellipse(image, (eye_x, eye_y), (eye_width, eye_height), 
                    0, 0, 360, (255, 255, 255), -1)
        # Iris (brown/blue/green)
        iris_colors = [(50, 80, 100), (100, 80, 50), (80, 100, 60)]
        iris_color = iris_colors[rng.integers(0, len(iris_colors))]
        cv2.circle(image, (eye_x, eye_y), eye_height - 2, iris_color, -1)
        # Pupil
        cv2.circle(image, (eye_x, eye_y), eye_height // 2, (20, 20, 20), -1)
        # Highlight
        cv2.circle(image, (eye_x - 2, eye_y - 2), 2, (220, 220, 220), -1)
    
    # Eyebrows
    brow_y = eye_y - eye_height - 8
    brow_color = tuple(max(0, c - 60) for c in skin_color)
    for brow_x in [face_center_x - eye_spacing, face_center_x + eye_spacing]:
        pts = np.array([
            [brow_x - eye_width, brow_y + 3],
            [brow_x, brow_y - 3],
            [brow_x + eye_width, brow_y + 2],
            [brow_x + eye_width, brow_y + 5],
            [brow_x, brow_y],
            [brow_x - eye_width, brow_y + 6],
        ], np.int32)
        cv2.fillPoly(image, [pts], brow_color[::-1])
    
    # Nose
    nose_y = face_center_y + face_height // 10
    nose_color = tuple(max(0, c - 15) for c in skin_color)
    pts = np.array([
        [face_center_x, eye_y + eye_height + 5],
        [face_center_x - 12, nose_y + 15],
        [face_center_x + 12, nose_y + 15],
    ], np.int32)
    cv2.fillPoly(image, [pts], nose_color[::-1])
    
    # Nostrils
    cv2.ellipse(image, (face_center_x - 8, nose_y + 12), (5, 3), 
                0, 0, 360, tuple(max(0, c - 40) for c in skin_color)[::-1], -1)
    cv2.ellipse(image, (face_center_x + 8, nose_y + 12), (5, 3), 
                0, 0, 360, tuple(max(0, c - 40) for c in skin_color)[::-1], -1)
    
    # Mouth
    mouth_y = face_center_y + face_height // 3
    mouth_width = face_width // 4
    lip_color = (100, 120, 180)  # Pinkish
    
    # Upper lip
    cv2.ellipse(image, (face_center_x, mouth_y - 3), (mouth_width, 8), 
                0, 180, 360, lip_color, -1)
    # Lower lip
    cv2.ellipse(image, (face_center_x, mouth_y + 3), (mouth_width, 10), 
                0, 0, 180, lip_color, -1)
    # Mouth line
    cv2.line(image, (face_center_x - mouth_width, mouth_y), 
             (face_center_x + mouth_width, mouth_y), (80, 80, 100), 2)
    
    # Hair (simple arc on top)
    hair_colors = [(30, 30, 30), (50, 40, 30), (80, 60, 40), (150, 120, 80)]
    hair_color = hair_colors[rng.integers(0, len(hair_colors))]
    hair_top = face_center_y - face_height // 2 - 10
    cv2.ellipse(image, (face_center_x, hair_top + 30), 
                (face_width // 2 + 20, face_height // 3),
                0, 180, 360, hair_color, -1)
    
    # Neck
    neck_top = face_center_y + face_height // 2 - 20
    neck_width = face_width // 3
    cv2.rectangle(image, 
                  (face_center_x - neck_width, neck_top),
                  (face_center_x + neck_width, height),
                  skin_color[::-1], -1)
    
    # Shoulders/shirt hint
    shirt_colors = [(150, 100, 80), (80, 80, 120), (60, 60, 60), (200, 200, 200)]
    shirt_color = shirt_colors[rng.integers(0, len(shirt_colors))]
    pts = np.array([
        [0, height],
        [face_center_x - neck_width - 50, neck_top + 40],
        [face_center_x - neck_width, neck_top + 60],
        [face_center_x + neck_width, neck_top + 60],
        [face_center_x + neck_width + 50, neck_top + 40],
        [width, height],
    ], np.int32)
    cv2.fillPoly(image, [pts], shirt_color)
    
    # Final blur for smoothness
    image = cv2.GaussianBlur(image, (3, 3), 0)
    
    return image


def generate_synthetic_id_document(
    rng: Optional[np.random.Generator] = None,
    width: int = 640,
    height: int = 400,
) -> np.ndarray:
    """Generate a synthetic ID document-like image.
    
    Creates an image with:
    - Card-like rectangular boundary
    - Photo region on one side
    - Text-like regions (horizontal lines)
    - Background pattern
    
    The goal is to produce an image that document analyzers can process,
    not to create a real-looking ID.
    
    Args:
        rng: Random number generator for reproducibility
        width: Image width
        height: Image height
        
    Returns:
        BGR image as numpy array
    """
    if rng is None:
        rng = np.random.default_rng()
    
    # Create card background
    bg_colors = [
        (240, 235, 220),  # Off-white
        (220, 230, 240),  # Light blue
        (230, 240, 230),  # Light green
        (240, 230, 220),  # Light tan
    ]
    bg_color = bg_colors[rng.integers(0, len(bg_colors))]
    image = np.full((height, width, 3), bg_color, dtype=np.uint8)
    
    # Add subtle pattern to background
    for _ in range(50):
        px = rng.integers(0, width)
        py = rng.integers(0, height)
        pattern_color = tuple(max(0, c - 15) for c in bg_color)
        cv2.circle(image, (px, py), 2, pattern_color, -1)
    
    # Card margin
    margin = 15
    
    # Card border
    cv2.rectangle(image, (margin, margin), (width - margin, height - margin),
                  (150, 150, 150), 2)
    
    # Add subtle gradient overlay
    for y in range(height):
        alpha = 0.02 * (1 - y / height)
        image[y] = np.clip(image[y].astype(float) * (1 + alpha), 0, 255).astype(np.uint8)
    
    # Photo region (left side)
    photo_x = margin + 20
    photo_y = margin + 30
    photo_w = int(width * 0.25)
    photo_h = int(photo_w * 1.3)
    
    # Photo background
    cv2.rectangle(image, (photo_x, photo_y), (photo_x + photo_w, photo_y + photo_h),
                  (200, 200, 200), -1)
    cv2.rectangle(image, (photo_x, photo_y), (photo_x + photo_w, photo_y + photo_h),
                  (100, 100, 100), 2)
    
    # Mini face in photo (simplified)
    face_cx = photo_x + photo_w // 2
    face_cy = photo_y + photo_h // 2 - 10
    face_rx = photo_w // 3
    face_ry = int(face_rx * 1.3)
    
    # Skin tone
    skin_tones = [(180, 160, 140), (160, 130, 110), (130, 100, 80)]
    skin_color = skin_tones[rng.integers(0, len(skin_tones))]
    cv2.ellipse(image, (face_cx, face_cy), (face_rx, face_ry), 0, 0, 360, skin_color[::-1], -1)
    
    # Eyes
    eye_y = face_cy - face_ry // 4
    for ex in [face_cx - face_rx // 2, face_cx + face_rx // 2]:
        cv2.circle(image, (ex, eye_y), 4, (50, 50, 50), -1)
    
    # Mouth
    cv2.line(image, (face_cx - face_rx // 3, face_cy + face_ry // 3),
             (face_cx + face_rx // 3, face_cy + face_ry // 3), (100, 100, 120), 2)
    
    # Hair
    hair_color = (40, 40, 40)
    cv2.ellipse(image, (face_cx, face_cy - face_ry + 10), (face_rx + 5, face_ry // 2),
                0, 180, 360, hair_color, -1)
    
    # Text regions (right side)
    text_x = photo_x + photo_w + 30
    text_y = margin + 40
    text_width = width - text_x - margin - 20
    
    # Header text (title like "IDENTIFICATION CARD")
    header_color = (80, 80, 80)
    cv2.rectangle(image, (text_x, text_y), (text_x + text_width, text_y + 15),
                  header_color, -1)
    
    # Field labels and values
    field_y = text_y + 40
    fields = [
        ("NAME:", 0.7),
        ("DOB:", 0.4),
        ("ID NO:", 0.5),
        ("ADDRESS:", 0.8),
        ("", 0.6),  # Second line of address
        ("EXPIRY:", 0.35),
    ]
    
    label_color = (120, 120, 120)
    value_color = (60, 60, 60)
    
    for label, value_ratio in fields:
        if label:
            # Label
            label_width = int(text_width * 0.25)
            cv2.rectangle(image, (text_x, field_y), (text_x + label_width, field_y + 8),
                          label_color, -1)
        
        # Value (simulated text with varying width)
        value_x = text_x + (int(text_width * 0.28) if label else 0)
        value_width = int((text_width - (int(text_width * 0.28) if label else 0)) * value_ratio)
        value_width += rng.integers(-10, 10)
        cv2.rectangle(image, (value_x, field_y), (value_x + max(20, value_width), field_y + 10),
                      value_color, -1)
        
        field_y += 22
    
    # Add barcode region at bottom
    barcode_y = height - margin - 35
    barcode_x = text_x
    barcode_w = text_width
    barcode_h = 25
    
    # Barcode lines
    x = barcode_x
    while x < barcode_x + barcode_w:
        bar_width = rng.integers(1, 4)
        if rng.random() > 0.3:
            cv2.rectangle(image, (x, barcode_y), (x + bar_width, barcode_y + barcode_h),
                          (30, 30, 30), -1)
        x += bar_width + rng.integers(1, 3)
    
    # MRZ-like region at very bottom (optional)
    if rng.random() > 0.5:
        mrz_y = height - margin - 8
        mrz_color = (80, 80, 80)
        cv2.rectangle(image, (margin + 10, mrz_y), (width - margin - 10, mrz_y + 5),
                      mrz_color, -1)
    
    # Add subtle security pattern (diagonal lines)
    overlay = image.copy()
    for i in range(-height, width, 40):
        cv2.line(overlay, (i, 0), (i + height, height), 
                 tuple(max(0, c - 5) for c in bg_color), 1)
    image = cv2.addWeighted(image, 0.95, overlay, 0.05, 0)
    
    # Slight blur for realism
    image = cv2.GaussianBlur(image, (3, 3), 0)
    
    return image


def encode_image_to_jpeg(image: np.ndarray, quality: int = 90) -> bytes:
    """Encode numpy image to JPEG bytes.
    
    Args:
        image: BGR image as numpy array
        quality: JPEG quality (0-100)
        
    Returns:
        JPEG encoded bytes
    """
    _, encoded = cv2.imencode('.jpg', image, [cv2.IMWRITE_JPEG_QUALITY, quality])
    return encoded.tobytes()
