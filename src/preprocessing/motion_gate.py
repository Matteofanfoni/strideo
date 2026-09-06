"""DORMANT — v1.1 experiment (M10), superseded by v1.2 motion_filter.

Tested on PV_800m_1_Victory in April 2026. Motion gating correctly
identified the runner's active window (frames 47-186, 140 frames) but
did NOT prevent ghost-lock inside that window: MediaPipe's VIDEO-mode
tracker still latched onto fence-shaped detections at frame 62, 88, 98,
108 before eventually snapping to the runner at frame 135. Detection
rate rose from 0.414 (v1.0) to 0.557 but visual inspection showed
messier tracking overall — downstream metrics (cadence, stride) worsened.

The v1.2 default pipeline does not use this module. It remains available
via its public API in case a future iteration wants to combine motion
gating with a smarter seed-frame strategy (e.g., delay tracker start
until the frame with peak motion energy). For reference, 01c §10.6
M10 is marked "Tested — failed" with the frame-level QA above.

Motion-gated frame windowing for pose extraction (Phase 2 / M10).

Context: MediaPipe Pose in VIDEO mode locks onto whichever person-shaped
thing exceeds the detection threshold in the first frame it runs on. When
the clip starts with the runner not yet in frame, the detector can latch
onto a static background structure (fence posts, signs) and the tracker
follows the ghost for the rest of the clip.

The fix: run pose extraction only on frames where the runner is actually
moving. A cheap frame-differencing pre-pass identifies the "motion
window" — the contiguous span where pixel activity is well above the
static-background baseline. The pose tracker's first frame in that
window is guaranteed to contain a moving subject, so it locks onto the
runner.

The module also exposes ``find_best_visibility_window`` which scans a
pose-visibility series for the 120-frame (or arbitrary-size) sub-window
where the runner's critical landmarks are most visible. This picks the
training window automatically instead of blindly centre-cropping.

Typical use:

    from src.preprocessing.motion_gate import compute_motion_signal, \\
        find_active_window, find_best_visibility_window

    motion = compute_motion_signal(video_path)
    t_start, t_end = find_active_window(motion)
    # ... run pose extraction on [t_start, t_end] ...
    t0, t1 = find_best_visibility_window(pose.visibilities, [23, 24, 27, 28])
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import List, Optional, Tuple

import cv2
import numpy as np


@dataclass
class MotionWindow:
    """Result of motion gating."""

    start_frame: int  # inclusive, source-video frame index
    end_frame: int  # inclusive
    signal: np.ndarray  # per-frame motion intensity, full length
    threshold: float  # threshold used to classify active vs idle
    source_fps: float  # for converting frames <-> seconds

    @property
    def length(self) -> int:
        return self.end_frame - self.start_frame + 1

    @property
    def duration_s(self) -> float:
        return self.length / max(self.source_fps, 1e-6)


def compute_motion_signal(
    video_path: str, downsample_factor: int = 4
) -> Tuple[np.ndarray, float]:
    """Compute per-frame motion intensity via grayscale frame-differencing.

    Frames are downsampled by ``downsample_factor`` before differencing so
    the pass is fast even on 4K. Returns the raw per-frame signal; frame 0
    is forced to 0 (no previous frame to diff against).

    Args:
        video_path: Path to the CFR video file.
        downsample_factor: Integer divisor for the smaller dimension. 4
            typically yields ~0.5 s per clip at 4K 60 fps.

    Returns:
        signal: (T,) float array of mean absolute frame-to-frame
            grayscale pixel differences.
        source_fps: Frame rate read from the container (for window-to-time
            conversion later).
    """
    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        raise ValueError(f"Cannot open video: {video_path}")

    source_fps = cap.get(cv2.CAP_PROP_FPS)
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    small_w = max(1, width // downsample_factor)
    small_h = max(1, height // downsample_factor)

    signal = np.zeros(total_frames, dtype=np.float32)
    prev_gray: Optional[np.ndarray] = None
    idx = 0

    try:
        while True:
            ret, frame = cap.read()
            if not ret:
                break
            if idx >= total_frames:
                break
            small = cv2.resize(frame, (small_w, small_h), interpolation=cv2.INTER_AREA)
            gray = cv2.cvtColor(small, cv2.COLOR_BGR2GRAY)
            if prev_gray is not None:
                diff = cv2.absdiff(gray, prev_gray)
                signal[idx] = float(diff.mean())
            prev_gray = gray
            idx += 1
    finally:
        cap.release()

    return signal[:idx], float(source_fps)


def _longest_run(mask: np.ndarray) -> Tuple[int, int, int]:
    """Return (start, end_inclusive, length) of the longest contiguous True run."""
    if not mask.any():
        return 0, -1, 0
    best_start = 0
    best_len = 0
    cur_start = -1
    for i, v in enumerate(mask):
        if v and cur_start < 0:
            cur_start = i
        elif not v and cur_start >= 0:
            run_len = i - cur_start
            if run_len > best_len:
                best_len = run_len
                best_start = cur_start
            cur_start = -1
    if cur_start >= 0:
        run_len = len(mask) - cur_start
        if run_len > best_len:
            best_len = run_len
            best_start = cur_start
    return best_start, best_start + best_len - 1, best_len


def find_active_window(
    motion_signal: np.ndarray,
    source_fps: float,
    threshold: Optional[float] = None,
    threshold_k: float = 0.0,
    min_length_s: float = 0.5,
    pad_start: int = 15,
    pad_end: int = 15,
    smooth_window: int = 21,
) -> MotionWindow:
    """Locate the runner's active window in the motion signal.

    Algorithm:
      1. Smooth the motion signal with a moving-average of ``smooth_window``
         frames — kills single-frame spikes (shadow flicker, bird, etc.).
      2. If ``threshold`` is not given, auto-compute as
         ``mean(signal) + threshold_k * std(signal)``. This adapts to the
         static-background baseline of the clip.
      3. Find the longest contiguous run of smoothed-signal values above
         threshold — that's the motion window.
      4. Pad the window by ``pad_start`` / ``pad_end`` frames so the pose
         tracker has a few landing frames before the runner's first clear
         appearance and a few trailing frames after they leave.
      5. Require at least ``min_length_s`` seconds of activity; otherwise
         fall back to the full clip (the motion gate's best-effort).

    Args:
        motion_signal: output of compute_motion_signal.
        source_fps: video frame rate (for min_length_s conversion).
        threshold: absolute threshold in motion units; None = auto.
        threshold_k: k factor for auto-threshold (mean + k*std).
        min_length_s: minimum active-window duration in seconds; below
            this the function returns the entire clip.
        pad_start/pad_end: padding in frames.
        smooth_window: moving-average window (odd int recommended).

    Returns:
        MotionWindow with clipped [start_frame, end_frame].
    """
    T = len(motion_signal)
    if T == 0:
        return MotionWindow(0, -1, motion_signal, 0.0, source_fps)

    # Smooth
    kernel = np.ones(max(1, smooth_window)) / max(1, smooth_window)
    smoothed = np.convolve(motion_signal, kernel, mode="same")

    if threshold is None:
        base = float(smoothed.mean()) + threshold_k * float(smoothed.std())
        threshold = max(base, 1e-6)

    active = smoothed > threshold
    start, end, length = _longest_run(active)

    min_frames = int(min_length_s * source_fps)
    if length < min_frames:
        # Fall back: use the full clip so downstream still runs; caller
        # can inspect .length and decide whether to warn.
        return MotionWindow(0, T - 1, motion_signal, float(threshold), source_fps)

    start = max(0, start - pad_start)
    end = min(T - 1, end + pad_end)
    return MotionWindow(start, end, motion_signal, float(threshold), source_fps)


def find_best_visibility_window(
    visibilities: np.ndarray,
    key_indices: List[int],
    window_size: int = 120,
) -> Tuple[int, int, float]:
    """Within a pose output, find the sub-window of ``window_size`` frames
    whose mean key-landmark visibility is highest.

    Key landmarks are usually the body features the downstream metrics
    rely on (hip L/R, ankle L/R, shoulder L/R, foot index L/R). A
    sub-window where all these are high means the *full figure* is in
    frame — no partial entry/exit.

    Args:
        visibilities: (T, 33) visibility scores.
        key_indices: Landmark indices (0-32) that must be visible.
        window_size: Sub-window length in frames (default 120 to match the
            training pipeline's normalization target).

    Returns:
        start, end_inclusive, mean_score. If ``T < window_size`` the entire
        range is returned.
    """
    T = len(visibilities)
    if T == 0 or not key_indices:
        return 0, T - 1, 0.0
    if T <= window_size:
        score = float(visibilities[:, key_indices].mean()) if T else 0.0
        return 0, T - 1, score

    # Per-frame score = mean visibility across key landmarks.
    per_frame = visibilities[:, key_indices].mean(axis=1)

    # Sliding-window sum via cumulative sum (O(T)).
    cumsum = np.concatenate([[0.0], np.cumsum(per_frame)])
    window_sums = cumsum[window_size:] - cumsum[:-window_size]
    best_start = int(np.argmax(window_sums))
    best_end = best_start + window_size - 1
    best_score = float(window_sums[best_start] / window_size)
    return best_start, best_end, best_score
