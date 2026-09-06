"""Pace estimation utilities for Phase 6d (Fix A2 spurious-contact pruning).

Provides a velocity estimate from the post-pipeline hip-x trajectory and a
leg-length-aware cadence band used to define the plausible inter-contact
interval that gates the sliding-window prune in
``src.preprocessing.ground_contact.prune_spurious_contacts``.

Design notes
------------
``estimate_velocity_from_hipx`` uses the longest consecutive non-NaN run of
the mean(LEFT_HIP, RIGHT_HIP) x-coordinate.  A linear fit over that run gives
a slope in px/frame; converting by fps and pixels_per_m yields m/s.  Hip
oscillation (±1–2 cm) is negligible over a 50-frame baseline, so the linear
fit is appropriate without RANSAC.

``expected_cadence_band`` uses the empirical step-length predictor::

    step_length_pred (m) = leg_length_m × (1.4 + 0.1 × velocity_ms)

Validated against the 5 annotated clips:

* PV_800m_1 (v=7.27, leg=0.97 m): step_pred=2.06 m → cadence_pred=212 spm
  (truth 212.4 spm, −0.2 %)
* PV_Steady_1 (v=4.62, leg=0.93 m): step_pred=1.74 m → cadence_pred=159 spm
  (truth 161.5 spm, −1.6 %)
* PV_Steady_2 (v=4.14, leg=0.95 m): step_pred=1.73 m → cadence_pred=144 spm
  (truth 156.1 spm, −7.8 %; within the ±25 % band)
* PV_800m_2 (v=7.74, leg=0.97 m): step_pred=2.12 m → cadence_pred=219 spm
  (truth 227.4 spm, −3.7 %)

A generous ±25 % band absorbs the residual prediction error.  The absolute
floor (14 frames at 60 fps ≡ 257 spm cap) prevents the band from going below
what is biomechanically possible even at sprint pace.
"""

from __future__ import annotations

import numpy as np
from typing import Tuple

from src.preprocessing.pose_estimator import LandmarkIndex

# Absolute floor for the minimum alternating-step interval regardless of the
# pace-derived prediction.  At 60 fps, 10 frames ≡ a cap of 360 spm —
# biomechanically impossible for any human runner at any distance.  Set low
# so the pace-derived threshold (more accurate for typical estimates) is the
# effective gate; the floor only catches truly catastrophic mis-estimates.
ABSOLUTE_FLOOR_FRAMES = 10

# Minimum finite frames required in the longest hip-x run to attempt a
# velocity estimate.  Below this the linear fit is unreliable.
MIN_RUN_FRAMES = 10


def estimate_velocity_from_hipx(
    landmarks: np.ndarray,
    visibilities: np.ndarray,
    pixels_per_m: float,
    fps: float,
    min_vis: float = 0.4,
) -> float:
    """Estimate runner velocity (m/s) from the mean hip-x trajectory.

    Computes mean(LEFT_HIP.x, RIGHT_HIP.x) per frame, finds the longest
    consecutive run where both hips are visible, and fits a line to get
    the horizontal velocity in px/frame, then converts to m/s.

    Args:
        landmarks: (T, 33, 2) post-pipeline landmark tensor.  NaN frames
            have already been masked by the motion filter / combiner.
        visibilities: (T, 33) per-(frame, landmark) visibility in [0, 1].
        pixels_per_m: spatial calibration scale (px/m).
        fps: video frame rate.
        min_vis: minimum hip visibility to include a frame in the run.

    Returns:
        Estimated velocity in m/s (always positive — runner moves in one
        direction across the FOV).

    Raises:
        ValueError: if no run of ``MIN_RUN_FRAMES`` consecutive valid
            frames is found.
    """
    if pixels_per_m <= 0:
        raise ValueError(f"pixels_per_m must be > 0, got {pixels_per_m}")
    if fps <= 0:
        raise ValueError(f"fps must be > 0, got {fps}")

    T = len(landmarks)
    l_hip = LandmarkIndex.LEFT_HIP
    r_hip = LandmarkIndex.RIGHT_HIP

    l_x = landmarks[:, l_hip, 0]
    r_x = landmarks[:, r_hip, 0]
    l_vis = visibilities[:, l_hip]
    r_vis = visibilities[:, r_hip]

    valid = ~np.isnan(l_x) & ~np.isnan(r_x) & (l_vis >= min_vis) & (r_vis >= min_vis)
    hip_x = np.where(valid, 0.5 * (l_x + r_x), np.nan)

    # Find the longest consecutive non-NaN run.
    best_start, best_len = 0, 0
    cur_start, cur_len = 0, 0
    for t in range(T):
        if not np.isnan(hip_x[t]):
            if cur_len == 0:
                cur_start = t
            cur_len += 1
            if cur_len > best_len:
                best_start, best_len = cur_start, cur_len
        else:
            cur_len = 0

    if best_len < MIN_RUN_FRAMES:
        raise ValueError(
            f"Longest valid hip-x run is {best_len} frames "
            f"(minimum {MIN_RUN_FRAMES} required for pace estimate)."
        )

    run_idx = np.arange(best_start, best_start + best_len, dtype=float)
    run_x = hip_x[best_start : best_start + best_len]

    slope_px_per_frame = float(np.polyfit(run_idx, run_x, 1)[0])
    velocity_ms = abs(slope_px_per_frame) * fps / pixels_per_m
    return velocity_ms


def expected_cadence_band(
    velocity_ms: float,
    leg_length_m: float,
    band_frac: float = 0.25,
) -> Tuple[float, float]:
    """Return a plausible cadence band (spm) given pace and leg length.

    Uses the empirical predictor::

        step_length_pred = leg_length_m × (1.4 + 0.1 × velocity_ms)
        cadence_pred     = 60 × velocity_ms / step_length_pred

    The ±``band_frac`` band (default ±25 %) absorbs individual variability
    and predictor inaccuracy at slower paces.

    Args:
        velocity_ms: estimated runner velocity in m/s.
        leg_length_m: runner leg length in metres.
        band_frac: fractional half-width of the cadence band. Default 0.25.

    Returns:
        (cadence_min_spm, cadence_max_spm) — the lower and upper bounds.

    Raises:
        ValueError: if inputs are out of plausible range.
    """
    if velocity_ms <= 0:
        raise ValueError(f"velocity_ms must be > 0, got {velocity_ms}")
    if leg_length_m <= 0:
        raise ValueError(f"leg_length_m must be > 0, got {leg_length_m}")
    if not 0 < band_frac < 1:
        raise ValueError(f"band_frac must be in (0, 1), got {band_frac}")

    step_length_pred = leg_length_m * (1.4 + 0.1 * velocity_ms)
    cadence_pred = 60.0 * velocity_ms / step_length_pred
    cadence_min = cadence_pred * (1.0 - band_frac)
    cadence_max = cadence_pred * (1.0 + band_frac)
    return cadence_min, cadence_max


def min_step_interval_frames(
    velocity_ms: float,
    leg_length_m: float,
    fps: float,
    band_frac: float = 0.25,
    absolute_floor: int = ABSOLUTE_FLOOR_FRAMES,
) -> float:
    """Minimum plausible alternating-step interval in frames.

    Converts the upper end of the cadence band to a frame count, then
    applies the absolute floor so the result is never below
    ``absolute_floor`` frames regardless of the pace estimate.

    Args:
        velocity_ms: runner velocity in m/s.
        leg_length_m: runner leg length in metres.
        fps: video frame rate.
        band_frac: cadence band half-width (passed to
            ``expected_cadence_band``).
        absolute_floor: hard minimum in frames (default 14 ≡ 257 spm cap
            at 60 fps).

    Returns:
        Minimum plausible alternating-step interval in frames (float).
    """
    _, cadence_max = expected_cadence_band(velocity_ms, leg_length_m, band_frac)
    pace_derived = 60.0 * fps / cadence_max
    return max(float(absolute_floor), pace_derived)
