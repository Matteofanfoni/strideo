from enum import Enum
from dataclasses import dataclass
from typing import List
import numpy as np


class VideoStatus(Enum):
    """Status of video after frame normalization."""

    REJECTED = "rejected"  # < 108 frames, cannot use
    PADDED = "padded"  # 108-119 frames, padded to 120
    CROPPED = "cropped"  # ≥ 120 frames, center-cropped to 120


@dataclass
class NormalizationResult:
    """Result of training video normalization."""

    frames: np.ndarray  # (120, H, W, 3) normalized frames
    status: VideoStatus  # How normalization was achieved
    original_count: int  # Original frame count
    start_frame: int  # First frame used (in original video)
    end_frame: int  # Last frame used (in original video)
    pad_start: int  # Frames padded at start (0 if none)
    pad_end: int  # Frames padded at end (0 if none)


def normalize_training_video(
    frames: List[np.ndarray], target_frames: int = 120, min_frames: int = 108
) -> NormalizationResult:
    """
    Normalize video to exactly target_frames for NN training.

    Strategy:
    - < min_frames: REJECT (insufficient data)
    - min_frames to target_frames-1: PAD with edge frames
    - ≥ target_frames: CENTER CROP to target_frames

    Args:
        frames: List of RGB frames from video
        target_frames: Exact number of frames needed (default: 120)
        min_frames: Minimum acceptable frames (default: 108 = 1.8s)

    Returns:
        NormalizationResult with exactly target_frames

    Raises:
        ValueError: If video has fewer than min_frames
    """
    n_frames = len(frames)
    frames_array = np.array(frames)  # (N, H, W, 3)

    # Case 1: REJECT — insufficient frames
    if n_frames < min_frames:
        raise ValueError(
            f"Video too short for training: {n_frames} frames "
            f"(minimum {min_frames} = {min_frames/60:.1f}s). "
            f"Re-film required."
        )

    # Case 2: PAD — slightly short, pad with edge frames
    if n_frames < target_frames:
        pad_needed = target_frames - n_frames
        pad_start = pad_needed // 2
        pad_end = pad_needed - pad_start

        # Pad by repeating edge frames
        normalized = np.concatenate(
            [
                np.repeat(frames_array[:1], pad_start, axis=0),  # Repeat first frame
                frames_array,
                np.repeat(frames_array[-1:], pad_end, axis=0),  # Repeat last frame
            ],
            axis=0,
        )

        return NormalizationResult(
            frames=normalized,
            status=VideoStatus.PADDED,
            original_count=n_frames,
            start_frame=0,
            end_frame=n_frames - 1,
            pad_start=pad_start,
            pad_end=pad_end,
        )

    # Case 3: CROP — at or above target, center crop to exactly target_frames
    excess = n_frames - target_frames
    start = excess // 2
    end = start + target_frames

    return NormalizationResult(
        frames=frames_array[start:end],
        status=VideoStatus.CROPPED,
        original_count=n_frames,
        start_frame=start,
        end_frame=end - 1,
        pad_start=0,
        pad_end=0,
    )
