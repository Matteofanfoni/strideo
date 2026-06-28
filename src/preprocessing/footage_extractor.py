from dataclasses import dataclass
from typing import List, Tuple

import numpy as np


@dataclass
class UsableFootageResult:
    """Result of usable footage extraction."""

    start_frame: int  # First frame of usable segment (in raw video)
    end_frame: int  # Last frame of usable segment (inclusive)
    usable_frames: int  # Number of frames in segment
    usable_duration_s: float  # Duration in seconds
    extraction_confidence: float  # Mean hip visibility across segment
    raw_video_frames: int  # Total frames in raw video
    utilisation_ratio: float  # usable_frames / raw_video_frames
    near_side: str  # 'left' or 'right'

    @property
    def is_valid_for_training(self) -> bool:
        """Check if segment meets training requirements (≥108 frames)."""
        return self.usable_frames >= 108

    @property
    def is_valid_for_inference(self) -> bool:
        """Check if segment meets inference requirements (≥90 frames)."""
        return self.usable_frames >= 90


# MediaPipe landmark indices
LANDMARKS = {
    # Head
    "nose": 0,
    "left_ear": 7,
    "right_ear": 8,
    # Shoulders
    "left_shoulder": 11,
    "right_shoulder": 12,
    # Arms
    "left_elbow": 13,
    "right_elbow": 14,
    "left_wrist": 15,
    "right_wrist": 16,
    # Hips
    "left_hip": 23,
    "right_hip": 24,
    # Legs
    "left_ankle": 27,
    "right_ankle": 28,
    "left_heel": 29,
    "right_heel": 30,
    "left_foot_index": 31,
    "right_foot_index": 32,
}

# Frame margin ratios
MARGIN_X_RATIO = 0.05  # 5% horizontal margin
MARGIN_Y_RATIO = 0.08  # 8% vertical margin

# Visibility thresholds
VIS_THRESHOLD_CORE = 0.5  # Head, shoulder, hip
VIS_THRESHOLD_EXTREMITY = 0.3  # Arms, legs


def detect_near_side(visibilities: np.ndarray) -> str:
    """
    Detect which side of the body is closer to the camera.

    Args:
        visibilities: (T, 33) visibility scores from MediaPipe

    Returns:
        'left' or 'right' indicating the near side
    """
    # Compare visibility of key left vs right landmarks
    left_indices = [
        LANDMARKS["left_shoulder"],
        LANDMARKS["left_hip"],
        LANDMARKS["left_ankle"],
    ]
    right_indices = [
        LANDMARKS["right_shoulder"],
        LANDMARKS["right_hip"],
        LANDMARKS["right_ankle"],
    ]

    left_vis = np.mean(visibilities[:, left_indices])
    right_vis = np.mean(visibilities[:, right_indices])

    return "left" if left_vis > right_vis else "right"


def get_landmarks_to_check(near_side: str) -> Tuple[List[int], List[int]]:
    """
    Get landmark indices to check based on near side.

    Args:
        near_side: 'left' or 'right'

    Returns:
        core_landmarks: Indices for head/torso (near-side only)
        extremity_landmarks: Indices for arms/legs (both sides)
    """
    if near_side == "left":
        core_landmarks = [
            LANDMARKS["nose"],
            LANDMARKS["left_ear"],
            LANDMARKS["left_shoulder"],
            LANDMARKS["left_hip"],
        ]
    else:
        core_landmarks = [
            LANDMARKS["nose"],
            LANDMARKS["right_ear"],
            LANDMARKS["right_shoulder"],
            LANDMARKS["right_hip"],
        ]

    # Extremities: check both sides (arms and legs swing through)
    extremity_landmarks = [
        LANDMARKS["left_elbow"],
        LANDMARKS["right_elbow"],
        LANDMARKS["left_wrist"],
        LANDMARKS["right_wrist"],
        LANDMARKS["left_ankle"],
        LANDMARKS["right_ankle"],
        LANDMARKS["left_heel"],
        LANDMARKS["right_heel"],
        LANDMARKS["left_foot_index"],
        LANDMARKS["right_foot_index"],
    ]

    return core_landmarks, extremity_landmarks


def is_frame_usable(
    landmarks: np.ndarray,
    visibilities: np.ndarray,
    frame_idx: int,
    frame_width: int,
    frame_height: int,
    core_landmarks: List[int],
    extremity_landmarks: List[int],
) -> bool:
    """
    Check if a single frame is usable.

    Args:
        landmarks: (T, 33, 2) pixel coordinates
        visibilities: (T, 33) visibility scores
        frame_idx: Frame index to check
        frame_width: Video frame width
        frame_height: Video frame height
        core_landmarks: Landmark indices for head/torso
        extremity_landmarks: Landmark indices for arms/legs

    Returns:
        True if frame is usable, False otherwise
    """
    margin_x = MARGIN_X_RATIO * frame_width
    margin_y = MARGIN_Y_RATIO * frame_height

    # Check core landmarks (near-side head, shoulder, hip)
    for lm_idx in core_landmarks:
        vis = visibilities[frame_idx, lm_idx]
        if vis < VIS_THRESHOLD_CORE:
            return False

        x, y = landmarks[frame_idx, lm_idx]
        if np.isnan(x) or np.isnan(y):
            return False

        # Check margins
        if x < margin_x or x > (frame_width - margin_x):
            return False
        if y < margin_y or y > (frame_height - margin_y):
            return False

    # Check extremity landmarks (both sides)
    for lm_idx in extremity_landmarks:
        vis = visibilities[frame_idx, lm_idx]
        if vis < VIS_THRESHOLD_EXTREMITY:
            return False

        x, y = landmarks[frame_idx, lm_idx]
        if np.isnan(x) or np.isnan(y):
            return False

        # Check margins
        if x < margin_x or x > (frame_width - margin_x):
            return False
        if y < margin_y or y > (frame_height - margin_y):
            return False

    return True


def extract_usable_footage(
    landmarks: np.ndarray,
    visibilities: np.ndarray,
    fps: float,
    frame_width: int,
    frame_height: int,
    max_gap: int = 2,
    min_frames_training: int = 108,
    min_frames_inference: int = 90,
) -> UsableFootageResult:
    """
    Extract usable footage using entry-to-exit forward scan.

    The algorithm:
    1. Detect near-side (higher visibility side)
    2. Scan forward to find ENTRY (first usable frame)
    3. Continue forward to find EXIT (sustained non-usable frames)
    4. Allow small gaps (≤ max_gap frames) for robustness

    Args:
        landmarks: (T, 33, 2) pixel coordinates from MediaPipe
        visibilities: (T, 33) visibility scores from MediaPipe
        fps: Video frame rate
        frame_width: Video frame width in pixels
        frame_height: Video frame height in pixels
        max_gap: Maximum gap (non-usable frames) to bridge (default 2)
        min_frames_training: Minimum frames for training (default 108)
        min_frames_inference: Minimum frames for inference (default 90)

    Returns:
        UsableFootageResult with extraction details

    Raises:
        ValueError: If no usable segment found or segment too short
    """
    T = len(landmarks)

    # Step 1: Detect near-side
    near_side = detect_near_side(visibilities)
    core_landmarks, extremity_landmarks = get_landmarks_to_check(near_side)

    # Step 2: Score each frame
    usable = np.zeros(T, dtype=bool)
    for t in range(T):
        usable[t] = is_frame_usable(
            landmarks,
            visibilities,
            t,
            frame_width,
            frame_height,
            core_landmarks,
            extremity_landmarks,
        )

    # Step 3: Find ENTRY (first usable frame)
    usable_start = None
    for t in range(T):
        if usable[t]:
            usable_start = t
            break

    if usable_start is None:
        raise ValueError(
            "No usable frames found. "
            "Ensure the full runner figure is visible in frame at some point."
        )

    # Step 4: Find EXIT (scan forward, allowing small gaps)
    usable_end = usable_start
    gap_count = 0

    for t in range(usable_start + 1, T):
        if usable[t]:
            # Usable frame: extend segment, reset gap counter
            usable_end = t
            gap_count = 0
        else:
            # Non-usable frame: count gap
            gap_count += 1
            if gap_count > max_gap:
                # Gap too large: runner has exited
                break

    # Calculate metrics
    usable_frames = usable_end - usable_start + 1
    usable_duration = usable_frames / fps

    # Mean hip visibility across segment (near-side hip)
    near_hip_idx = (
        LANDMARKS["left_hip"] if near_side == "left" else LANDMARKS["right_hip"]
    )
    segment_hip_vis = visibilities[usable_start : usable_end + 1, near_hip_idx]
    extraction_confidence = float(np.mean(segment_hip_vis))

    utilisation_ratio = usable_frames / T

    result = UsableFootageResult(
        start_frame=usable_start,
        end_frame=usable_end,
        usable_frames=usable_frames,
        usable_duration_s=usable_duration,
        extraction_confidence=extraction_confidence,
        raw_video_frames=T,
        utilisation_ratio=utilisation_ratio,
        near_side=near_side,
    )

    # Validate minimum duration
    if usable_frames < min_frames_inference:
        raise ValueError(
            f"Usable segment too short: {usable_frames} frames "
            f"({usable_duration:.2f}s). "
            f"Minimum for inference: {min_frames_inference} "
            f"frames ({min_frames_inference/fps:.2f}s). "
            f"Re-film required."
        )

    return result


# extract_usable_frames() was removed (commit e07fe15 → this commit).
# It was a lightweight-MediaPipe integration wrapper around extract_usable_footage()
# intended as a Streamlit quality gate, but was never wired into the production
# pipeline (which uses normalize_training_video + extract_pose_landmarks instead).
# Single-pass detection proved unreliable on 4K race footage (runner small in frame).
# If a Streamlit quality gate is needed in future, wire extract_usable_footage()
# directly against the v1.9 bidirectional pose extraction output.
