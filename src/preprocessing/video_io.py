import cv2
import numpy as np
from typing import List, Tuple, Dict


def load_video(
    video_path: str, target_fps: int = 60
) -> Tuple[List[np.ndarray], float, Dict]:
    """
    Load video and extract frames at target frame rate.

    Args:
        video_path: Path to video file
        target_fps: Target frame rate (downsample if source is higher)

    Returns:
        frames: List of RGB frame arrays (H, W, 3)
        effective_fps: Actual frame rate after downsampling
        metadata: Video metadata dictionary

    Raises:
        ValueError: If video cannot be opened
        ValueError: If video duration < 1.5 second
    """
    cap = cv2.VideoCapture(video_path)

    if not cap.isOpened():
        raise ValueError(f"Cannot open video: {video_path}")

    # Extract metadata
    source_fps = cap.get(cv2.CAP_PROP_FPS)
    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    duration = total_frames / source_fps

    # Absolute minimum: 1.5s for any processing (inference minimum is 1.5s)
    # Note: Training requires 1.8s minimum — enforced by normalize_training_video()
    if duration < 1.5:
        raise ValueError(f"Video too short: {duration:.2f}s (absolute minimum 1.5s)")

    # Calculate downsampling
    skip_rate = max(1, int(source_fps / target_fps))
    effective_fps = source_fps / skip_rate

    frames = []
    frame_idx = 0

    while True:
        ret, frame = cap.read()
        if not ret:
            break

        if frame_idx % skip_rate == 0:
            # Convert BGR (OpenCV) to RGB
            frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            frames.append(frame_rgb)

        frame_idx += 1

    cap.release()

    metadata = {
        "source_path": video_path,
        "source_fps": source_fps,
        "effective_fps": effective_fps,
        "width": width,
        "height": height,
        "total_frames": total_frames,
        "extracted_frames": len(frames),
        "duration_seconds": duration,
        "skip_rate": skip_rate,
    }

    return frames, effective_fps, metadata


def validate_frame_quality(frames: List[np.ndarray], metadata: Dict) -> Dict:
    """
    Validate extracted frames meet quality requirements.

    Returns:
        validation: Dictionary with pass/fail status and diagnostics
    """
    issues = []

    # Check resolution
    height, width = frames[0].shape[:2]
    if width < 1280 or height < 720:
        issues.append(f"Resolution too low: {width}x{height} (minimum 1280x720)")

    # Check frame count (90 = inference minimum; training requires 108)
    if len(frames) < 90:
        issues.append(
            f"Too few frames: {len(frames)} "
            f"(minimum 90 for inference, 108 for training)"
        )

    # Check for consistent dimensions
    for i, frame in enumerate(frames):
        if frame.shape[:2] != (height, width):
            issues.append(f"Inconsistent dimensions at frame {i}")
            break

    # Check brightness (detect underexposure)
    mean_brightness = np.mean([np.mean(f) for f in frames])
    if mean_brightness < 30:
        issues.append(f"Video too dark: mean brightness {mean_brightness:.1f}")
    elif mean_brightness > 240:
        issues.append(f"Video overexposed: mean brightness {mean_brightness:.1f}")

    return {
        "valid": len(issues) == 0,
        "issues": issues,
        "resolution": (width, height),
        "frame_count": len(frames),
        "mean_brightness": mean_brightness,
    }
