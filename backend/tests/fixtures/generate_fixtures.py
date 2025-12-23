"""
Generate synthetic test fixtures for detection pipeline testing.
Creates face images, ID documents, and video frames with various characteristics.
"""

import cv2
import numpy as np
from pathlib import Path
import random

FIXTURES_DIR = Path(__file__).parent
FIXTURES_DIR.mkdir(exist_ok=True)


def generate_realistic_face(
    width: int = 640,
    height: int = 480,
    face_offset: tuple[int, int] = (0, 0),
    skin_tone: tuple[int, int, int] = (180, 160, 150),
    add_noise: bool = False,
    blur_amount: int = 0
) -> np.ndarray:
    """
    Generate a more realistic synthetic face image.
    Note: InsightFace may not detect these as real faces,
    but they're useful for testing the pipeline flow.
    """
    img = np.ones((height, width, 3), dtype=np.uint8) * 220
    
    center_x = width // 2 + face_offset[0]
    center_y = height // 2 + face_offset[1]
    
    # Face oval with gradient
    face_mask = np.zeros((height, width), dtype=np.uint8)
    cv2.ellipse(face_mask, (center_x, center_y), (90, 120), 0, 0, 360, 255, -1)
    
    # Apply skin tone
    img[face_mask > 0] = skin_tone
    
    # Add subtle shading
    for i in range(height):
        for j in range(width):
            if face_mask[i, j] > 0:
                # Darken edges
                dist_from_center = np.sqrt((i - center_y)**2 + (j - center_x)**2)
                shade_factor = max(0.7, 1 - dist_from_center / 200)
                img[i, j] = np.clip(img[i, j] * shade_factor, 0, 255).astype(np.uint8)
    
    # Eyes
    eye_y = center_y - 30
    left_eye_x = center_x - 35
    right_eye_x = center_x + 35
    
    # Eye whites
    cv2.ellipse(img, (left_eye_x, eye_y), (18, 10), 0, 0, 360, (255, 255, 255), -1)
    cv2.ellipse(img, (right_eye_x, eye_y), (18, 10), 0, 0, 360, (255, 255, 255), -1)
    
    # Irises
    cv2.circle(img, (left_eye_x, eye_y), 8, (80, 60, 40), -1)
    cv2.circle(img, (right_eye_x, eye_y), 8, (80, 60, 40), -1)
    
    # Pupils
    cv2.circle(img, (left_eye_x, eye_y), 4, (20, 20, 20), -1)
    cv2.circle(img, (right_eye_x, eye_y), 4, (20, 20, 20), -1)
    
    # Eye highlights
    cv2.circle(img, (left_eye_x + 2, eye_y - 2), 2, (255, 255, 255), -1)
    cv2.circle(img, (right_eye_x + 2, eye_y - 2), 2, (255, 255, 255), -1)
    
    # Eyebrows
    cv2.ellipse(img, (left_eye_x, eye_y - 20), (22, 4), 0, 0, 180, (60, 40, 30), -1)
    cv2.ellipse(img, (right_eye_x, eye_y - 20), (22, 4), 0, 0, 180, (60, 40, 30), -1)
    
    # Nose
    nose_pts = np.array([
        [center_x, center_y - 10],
        [center_x - 12, center_y + 30],
        [center_x + 12, center_y + 30]
    ], np.int32)
    cv2.polylines(img, [nose_pts], False, (150, 130, 120), 2)
    
    # Nostrils
    cv2.ellipse(img, (center_x - 8, center_y + 28), (5, 3), 0, 0, 360, (120, 100, 90), -1)
    cv2.ellipse(img, (center_x + 8, center_y + 28), (5, 3), 0, 0, 360, (120, 100, 90), -1)
    
    # Mouth
    cv2.ellipse(img, (center_x, center_y + 55), (25, 8), 0, 0, 180, (150, 100, 100), 2)
    cv2.ellipse(img, (center_x, center_y + 52), (20, 5), 0, 180, 360, (180, 120, 120), -1)
    
    # Add noise if requested
    if add_noise:
        noise = np.random.normal(0, 10, img.shape).astype(np.int16)
        img = np.clip(img.astype(np.int16) + noise, 0, 255).astype(np.uint8)
    
    # Apply blur if requested
    if blur_amount > 0:
        img = cv2.GaussianBlur(img, (blur_amount * 2 + 1, blur_amount * 2 + 1), 0)
    
    return img


def generate_id_document(
    name: str = "JOHN DOE",
    dob: str = "01/15/1990",
    id_number: str = "1234567890",
    include_face: bool = True,
    tamper_text: bool = False,
    misalign_text: bool = False
) -> np.ndarray:
    """Generate a synthetic ID document image."""
    width, height = 600, 400
    img = np.ones((height, width, 3), dtype=np.uint8) * 245
    
    # Card border
    cv2.rectangle(img, (5, 5), (width-5, height-5), (180, 180, 180), 1)
    cv2.rectangle(img, (10, 10), (width-10, height-10), (100, 100, 100), 2)
    
    # Header with background
    cv2.rectangle(img, (10, 10), (width-10, 55), (70, 100, 140), -1)
    cv2.putText(img, "SAMPLE IDENTIFICATION CARD", (120, 42),
                cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)
    
    # Photo area
    photo_x, photo_y = 25, 70
    photo_w, photo_h = 140, 180
    cv2.rectangle(img, (photo_x, photo_y), (photo_x + photo_w, photo_y + photo_h),
                  (200, 200, 200), -1)
    cv2.rectangle(img, (photo_x, photo_y), (photo_x + photo_w, photo_y + photo_h),
                  (100, 100, 100), 2)
    
    # Add face to photo area
    if include_face:
        face = generate_realistic_face(photo_w, photo_h)
        img[photo_y:photo_y+photo_h, photo_x:photo_x+photo_w] = face
    
    # Text fields
    text_x = 180
    text_start_y = 90
    line_height = 40
    
    fields = [
        ("Name:", name),
        ("DOB:", dob),
        ("ID No:", id_number),
        ("Exp:", "12/31/2030"),
    ]
    
    for i, (label, value) in enumerate(fields):
        y = text_start_y + i * line_height
        
        # Add misalignment if requested
        x_offset = random.randint(-15, 15) if misalign_text else 0
        
        # Different font for tampered text
        if tamper_text and i == 0:
            font = cv2.FONT_HERSHEY_DUPLEX
            font_scale = 0.55
        else:
            font = cv2.FONT_HERSHEY_SIMPLEX
            font_scale = 0.6
        
        cv2.putText(img, label, (text_x + x_offset, y),
                    font, font_scale, (80, 80, 80), 1)
        cv2.putText(img, value, (text_x + 80 + x_offset, y),
                    font, font_scale, (0, 0, 0), 2)
    
    # Add some security-like features
    # Hologram-ish area
    cv2.rectangle(img, (400, 280), (570, 370), (200, 220, 240), -1)
    cv2.putText(img, "SECURE", (430, 320), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (150, 150, 200), 1)
    cv2.putText(img, "ELEMENT", (425, 350), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (150, 150, 200), 1)
    
    # Barcode-like element
    for x in range(30, 160, 3):
        bar_height = random.randint(20, 40)
        cv2.line(img, (x, height - 50), (x, height - 50 + bar_height), (0, 0, 0), 2)
    
    return img


def generate_video_frames(
    n_frames: int = 30,
    fps: float = 30.0,
    motion_type: str = "natural",  # "natural", "static", "replay"
    add_moire: bool = False,
    add_stutter: bool = False
) -> list[np.ndarray]:
    """
    Generate synthetic video frames.
    
    motion_type:
        - "natural": Slight head movements, blinks
        - "static": No movement (suspicious)
        - "replay": Screen-like artifacts
    """
    frames = []
    width, height = 640, 480
    base_frame = None
    
    for i in range(n_frames):
        if motion_type == "natural":
            # Natural micro-movements
            offset_x = int(8 * np.sin(i * 0.15))
            offset_y = int(4 * np.cos(i * 0.2))
            
            frame = generate_realistic_face(
                width, height,
                face_offset=(offset_x, offset_y),
                add_noise=True
            )
            
            # Add occasional blink
            if i % 25 in [10, 11]:
                # Draw closed eyes
                eye_y = height // 2 - 30 + offset_y
                left_x = width // 2 - 35 + offset_x
                right_x = width // 2 + 35 + offset_x
                cv2.line(frame, (left_x - 15, eye_y), (left_x + 15, eye_y), (60, 40, 30), 2)
                cv2.line(frame, (right_x - 15, eye_y), (right_x + 15, eye_y), (60, 40, 30), 2)
        
        elif motion_type == "static":
            # No movement at all - suspicious
            if base_frame is None:
                base_frame = generate_realistic_face(width, height)
            frame = base_frame.copy()
        
        elif motion_type == "replay":
            # Screen capture artifacts
            frame = generate_realistic_face(width, height, add_noise=True)
            
            # Add moiré pattern
            if add_moire:
                for y in range(0, height, 3):
                    frame[y, :] = np.clip(frame[y, :].astype(int) - 20, 0, 255).astype(np.uint8)
            
            # Add slight color shift (screen warmth)
            frame[:, :, 2] = np.clip(frame[:, :, 2].astype(int) + 10, 0, 255).astype(np.uint8)
        
        else:
            frame = generate_realistic_face(width, height)
        
        # Add frame stutter (duplicate frames)
        if add_stutter and i > 0 and i % 5 == 0:
            frame = frames[-1].copy()
        
        frames.append(frame)
    
    return frames


def generate_attack_samples():
    """Generate samples for different attack types."""
    
    # 1. Genuine session
    print("Generating genuine samples...")
    selfie = generate_realistic_face()
    id_doc = generate_id_document(include_face=True)
    cv2.imwrite(str(FIXTURES_DIR / "genuine_selfie.jpg"), selfie)
    cv2.imwrite(str(FIXTURES_DIR / "genuine_id.jpg"), id_doc)
    
    # Genuine video frames
    frames = generate_video_frames(motion_type="natural")
    for i, frame in enumerate(frames):
        cv2.imwrite(str(FIXTURES_DIR / f"genuine_frame_{i:03d}.jpg"), frame)
    
    # 2. Face mismatch (different person)
    print("Generating mismatch samples...")
    selfie_mismatch = generate_realistic_face(skin_tone=(160, 140, 130))
    id_mismatch = generate_id_document(name="JANE SMITH", include_face=True)
    cv2.imwrite(str(FIXTURES_DIR / "mismatch_selfie.jpg"), selfie_mismatch)
    cv2.imwrite(str(FIXTURES_DIR / "mismatch_id.jpg"), id_mismatch)
    
    # 3. Replay attack (static + moire)
    print("Generating replay attack samples...")
    replay_frames = generate_video_frames(motion_type="replay", add_moire=True)
    for i, frame in enumerate(replay_frames):
        cv2.imwrite(str(FIXTURES_DIR / f"replay_frame_{i:03d}.jpg"), frame)
    
    # 4. Static attack (no movement)
    print("Generating static attack samples...")
    static_frames = generate_video_frames(motion_type="static", add_stutter=True)
    for i, frame in enumerate(static_frames):
        cv2.imwrite(str(FIXTURES_DIR / f"static_frame_{i:03d}.jpg"), frame)
    
    # 5. Document tampering
    print("Generating tampered document samples...")
    tampered_id = generate_id_document(
        name="JOHN DOE",
        tamper_text=True,
        misalign_text=True
    )
    cv2.imwrite(str(FIXTURES_DIR / "tampered_id.jpg"), tampered_id)
    
    # 6. No face in selfie
    print("Generating no-face samples...")
    no_face = np.ones((480, 640, 3), dtype=np.uint8) * 200
    cv2.putText(no_face, "NO FACE HERE", (180, 250),
                cv2.FONT_HERSHEY_SIMPLEX, 1.2, (100, 100, 100), 2)
    cv2.imwrite(str(FIXTURES_DIR / "no_face_selfie.jpg"), no_face)
    
    # 7. Multiple faces
    print("Generating multiple faces sample...")
    multi_face = np.ones((480, 800, 3), dtype=np.uint8) * 220
    face1 = generate_realistic_face(300, 400)
    face2 = generate_realistic_face(300, 400, skin_tone=(170, 150, 140))
    multi_face[40:440, 50:350] = face1
    multi_face[40:440, 450:750] = face2
    cv2.imwrite(str(FIXTURES_DIR / "multiple_faces_selfie.jpg"), multi_face)
    
    print(f"\nFixtures generated in {FIXTURES_DIR}")
    print("Files created:")
    for f in sorted(FIXTURES_DIR.glob("*.jpg")):
        print(f"  - {f.name}")


if __name__ == "__main__":
    generate_attack_samples()

