"""Per-landmark temporal gap-fill for the post-combiner / post-motion-filter
landmark stream (Phase 6a, ships as pipeline v1.7).

The v1.6 bidirectional combiner + motion gate produces a landmark tensor
with NaN gaps wherever both pose passes failed anatomy ratification or
the motion gate rejected a frame. On the detection-side clips
(Recovery_2 / Steady_1 / Steady_2 / 800m_1 / 800m_2) those NaN gaps are
typically 1–4 frames long and sit at gait-cycle extremes (full leg
extension at toe-off, mid-flight knee-bent foreshortening) — short,
biomechanically interpolable, and located *between* contact moments
where downstream contact detection cares about ankle-y velocity. Filling
them stabilises the longest non-NaN run so the 120-frame NN training
window is achievable and gives contact detection a smoother input
signal.

Phase 6a is *NN-input quality* AND *validation-pipeline cleanup*. The
preprocessing pipeline's per-clip metrics (stride length, cadence, GCT,
vertical oscillation) are the FiLM-CNN training labels; gap-fill in the
validation pipeline directly improves training-label quality. See
`docs/technical/01c_pre_validation_session_report.md` §10.16 for the
results record.

Per-landmark independence is the design choice: a frame where one
landmark is missing and another is present has the missing one filled
(if eligible) without disturbing the present one. This avoids dropping
whole frames over single-joint occlusions.

Three interpolation methods are supported and the choice is exposed at
the runner-script CLI:

- ``"pchip"`` (default): Piecewise Cubic Hermite Interpolating Polynomial,
  monotone-preserving. Won't overshoot the data envelope — kinematically
  safest, no fabricated sub-ground ankle positions.
- ``"cubic"``: natural cubic spline. Smoothest fit, but can overshoot
  bouncy signals. Risky on ankle-y near touchdown.
- ``"linear"``: piecewise linear. No overshoot, simplest. Worst fit on
  curved trajectories.

Typical use::

    from src.preprocessing.landmarks_cleanup import fill_gaps

    cleaned = fill_gaps(
        mf.landmarks, mf.visibilities, mf.world_landmarks,
        max_gap_frames=4,
        min_anchor_visibility=0.5,
        method="pchip",
    )
    # cleaned.landmarks / .visibilities have interpolated values
    # at previously-NaN frames whose flanking anchors were trusted.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional, Tuple

import numpy as np
from scipy.interpolate import CubicSpline, PchipInterpolator, interp1d

# Defaults chosen from the Phase 6a design (01c §10.16). 4 frames at
# 60 fps = 67 ms — biomechanically plausible to interpolate (well
# inside a stride). 0.5 visibility threshold matches the floor used by
# motion_filter elsewhere in the pipeline.
DEFAULT_MAX_GAP_FRAMES = 4
DEFAULT_MIN_ANCHOR_VISIBILITY = 0.5
DEFAULT_METHOD = "pchip"


@dataclass
class CleanupResult:
    """Output of fill_gaps. Mirrors MotionFilterResult /
    BidirectionalCombineResult contract: same ndarray shapes, NaN-on-
    unfilled, plus a per-frame log of fill activity for downstream
    diagnostics and CSV export."""

    landmarks: np.ndarray  # (T, 33, 2) — interpolated where possible
    visibilities: np.ndarray  # (T, 33) — linearly-interpolated for filled frames
    world_landmarks: Optional[np.ndarray]  # (T, 33, 3) — None if input was None
    log: List[Dict[str, Any]] = field(default_factory=list)
    method: str = DEFAULT_METHOD
    max_gap_frames: int = DEFAULT_MAX_GAP_FRAMES
    min_anchor_visibility: float = DEFAULT_MIN_ANCHOR_VISIBILITY

    @property
    def n_filled_total(self) -> int:
        """Total per-(frame, landmark) cells the function filled."""
        return sum(int(e["n_filled"]) for e in self.log)

    @property
    def n_left_nan_long_total(self) -> int:
        return sum(int(e["n_left_nan_long"]) for e in self.log)

    @property
    def n_left_nan_edge_total(self) -> int:
        return sum(int(e["n_left_nan_edge"]) for e in self.log)

    @property
    def n_frames_with_fill(self) -> int:
        """Number of frames where at least one landmark was filled."""
        return sum(1 for e in self.log if int(e["n_filled"]) > 0)


def _make_interpolator(method: str, anchor_t: np.ndarray, anchor_y: np.ndarray) -> Any:
    """Build a 1D interpolator over (anchor_t, anchor_y) for the chosen method.

    anchor_y can be (N,) or (N, D). Returned callable accepts (M,) eval
    indices and returns (M,) or (M, D) values, matching anchor_y shape.
    """
    if method == "pchip":
        return PchipInterpolator(anchor_t, anchor_y, axis=0, extrapolate=False)
    if method == "cubic":
        return CubicSpline(anchor_t, anchor_y, axis=0, extrapolate=False)
    if method == "linear":
        return interp1d(
            anchor_t,
            anchor_y,
            kind="linear",
            axis=0,
            bounds_error=False,
            fill_value=np.nan,
        )
    raise ValueError(
        f"Unknown interpolation method: {method!r}; "
        "expected 'pchip' / 'cubic' / 'linear'"
    )


def _find_nan_runs(is_finite: np.ndarray) -> List[Tuple[int, int]]:
    """Return (start, end) inclusive indices of contiguous False runs in
    is_finite (per-landmark NaN runs to candidate-fill)."""
    runs: List[Tuple[int, int]] = []
    in_run = False
    run_start = -1
    for i, m in enumerate(is_finite):
        if not m:
            if not in_run:
                run_start = i
                in_run = True
        elif in_run:
            runs.append((run_start, i - 1))
            in_run = False
    if in_run:
        runs.append((run_start, len(is_finite) - 1))
    return runs


def fill_gaps(
    landmarks: np.ndarray,
    visibilities: np.ndarray,
    world_landmarks: Optional[np.ndarray] = None,
    max_gap_frames: int = DEFAULT_MAX_GAP_FRAMES,
    min_anchor_visibility: float = DEFAULT_MIN_ANCHOR_VISIBILITY,
    method: str = DEFAULT_METHOD,
) -> CleanupResult:
    """Fill short NaN gaps in per-landmark trajectories via temporal
    interpolation. Per-landmark independent — a frame with one missing
    and one present landmark has the missing one filled (if eligible)
    keeping the present one untouched.

    Eligibility per gap (contiguous NaN run for that landmark):
      - Gap length ≤ ``max_gap_frames``.
      - Both flanking *anchors* exist (the gap is not at the array edge).
        An anchor is a frame where the landmark is finite AND its
        visibility ≥ ``min_anchor_visibility``.
      - At least 2 anchors exist for the landmark across the array
        (interpolators need 2+ control points).

    Frames where the landmark is finite but visibility is below
    threshold are *not* anchors but are also *not* filled — they keep
    their original (low-confidence) values. Gap-fill is a fill operation,
    not a smoothing operation.

    Visibilities for filled frames are linearly interpolated between the
    flanking anchors and clamped to [0, 1].

    World landmarks (3D metric space) are interpolated with the same
    method when they are non-None and the anchor world landmarks are all
    finite.

    Args:
        landmarks: (T, 33, 2) pixel landmarks; NaN means missing.
        visibilities: (T, 33) per-(frame, landmark) confidence in [0, 1];
            zero on motion-filter-rejected / combiner-dropped frames.
        world_landmarks: optional (T, 33, 3) metric-space landmarks.
        max_gap_frames: maximum NaN-run length to attempt fill on.
        min_anchor_visibility: visibility floor below which a finite
            landmark frame is not eligible as an anchor.
        method: ``"pchip"`` / ``"cubic"`` / ``"linear"``. PCHIP is
            kinematically safest (monotone, no overshoot); cubic spline
            risks fabricating overshoot artefacts on bouncy ankle-y;
            linear is the simplest fallback.

    Returns:
        CleanupResult. The landmarks / visibilities / world_landmarks
        ndarrays have interpolated values at filled frames; gaps that
        failed eligibility are unchanged (still NaN). The per-frame log
        records how many landmarks were anchored / filled / left-NaN
        for each frame.

    Raises:
        ValueError: on unknown ``method``.
    """
    if method not in ("pchip", "cubic", "linear"):
        raise ValueError(f"method must be 'pchip' / 'cubic' / 'linear', got {method!r}")

    T, n_landmarks, _ = landmarks.shape
    out_lm = landmarks.copy()
    out_vis = visibilities.copy()
    out_world = None if world_landmarks is None else world_landmarks.copy()

    # Per-frame log skeleton. Entries are mutated as we walk landmarks.
    log: List[Dict[str, Any]] = [
        {
            "frame": int(t),
            "n_anchor": 0,
            "n_filled": 0,
            "n_left_nan_long": 0,
            "n_left_nan_edge": 0,
        }
        for t in range(T)
    ]

    for lm_idx in range(n_landmarks):
        lm_xy = landmarks[:, lm_idx, :]
        lm_vis = visibilities[:, lm_idx]

        is_finite = ~np.isnan(lm_xy).any(axis=-1)
        # Anchor = finite landmark AND visibility ≥ threshold. Frames with
        # finite but low-vis landmarks are NOT anchors but remain in
        # is_finite (they're not gap candidates either).
        is_anchor = is_finite & (lm_vis >= min_anchor_visibility)

        # Tally anchors-as-data (the n_anchor counter records how many
        # landmarks were already present per frame, regardless of vis).
        for t in range(T):
            if bool(is_finite[t]):
                log[t]["n_anchor"] += 1

        if int(is_anchor.sum()) < 2:
            # Not enough anchor points to build any interpolator on this
            # landmark — every NaN run is structurally an edge case.
            for start, end in _find_nan_runs(is_finite):
                for t in range(start, end + 1):
                    log[t]["n_left_nan_edge"] += 1
            continue

        anchor_t = np.where(is_anchor)[0]
        anchor_xy = lm_xy[anchor_t]
        try:
            interp_xy = _make_interpolator(method, anchor_t, anchor_xy)
        except Exception:
            for start, end in _find_nan_runs(is_finite):
                for t in range(start, end + 1):
                    log[t]["n_left_nan_edge"] += 1
            continue

        interp_world: Any = None
        if out_world is not None and world_landmarks is not None:
            world_anchor = world_landmarks[anchor_t, lm_idx, :]
            if not np.isnan(world_anchor).any():
                try:
                    interp_world = _make_interpolator(method, anchor_t, world_anchor)
                except Exception:
                    interp_world = None

        for start, end in _find_nan_runs(is_finite):
            gap_length = end - start + 1

            if start == 0 or end == T - 1:
                for t in range(start, end + 1):
                    log[t]["n_left_nan_edge"] += 1
                continue
            if gap_length > max_gap_frames:
                for t in range(start, end + 1):
                    log[t]["n_left_nan_long"] += 1
                continue

            # Find the bracketing anchor frames (skip low-vis between).
            left_anchor_idx: Optional[int] = None
            for cand in range(start - 1, -1, -1):
                if bool(is_anchor[cand]):
                    left_anchor_idx = cand
                    break
            right_anchor_idx: Optional[int] = None
            for cand in range(end + 1, T):
                if bool(is_anchor[cand]):
                    right_anchor_idx = cand
                    break
            if left_anchor_idx is None or right_anchor_idx is None:
                for t in range(start, end + 1):
                    log[t]["n_left_nan_edge"] += 1
                continue

            t_to_fill = np.arange(start, end + 1)
            xy_filled = interp_xy(t_to_fill)
            if np.isnan(xy_filled).any():
                # Defensive — we already excluded edge cases, so this
                # only fires if the interpolator returned NaN inside its
                # anchor range (shouldn't happen with PCHIP/cubic/linear).
                for t in range(start, end + 1):
                    log[t]["n_left_nan_edge"] += 1
                continue

            world_filled: Optional[np.ndarray] = None
            if interp_world is not None:
                candidate = interp_world(t_to_fill)
                if not np.isnan(candidate).any():
                    world_filled = candidate

            span = float(right_anchor_idx - left_anchor_idx)
            for i, t_fill_np in enumerate(t_to_fill):
                t_fill = int(t_fill_np)
                out_lm[t_fill, lm_idx, 0] = float(xy_filled[i, 0])
                out_lm[t_fill, lm_idx, 1] = float(xy_filled[i, 1])
                alpha = (
                    (float(t_fill) - float(left_anchor_idx)) / span if span > 0 else 0.0
                )
                vis_filled = (1.0 - alpha) * float(
                    lm_vis[left_anchor_idx]
                ) + alpha * float(lm_vis[right_anchor_idx])
                out_vis[t_fill, lm_idx] = float(np.clip(vis_filled, 0.0, 1.0))
                if (
                    out_world is not None
                    and world_filled is not None
                    and world_landmarks is not None
                ):
                    out_world[t_fill, lm_idx, :] = world_filled[i]
                log[t_fill]["n_filled"] += 1

    return CleanupResult(
        landmarks=out_lm,
        visibilities=out_vis,
        world_landmarks=out_world,
        log=log,
        method=method,
        max_gap_frames=max_gap_frames,
        min_anchor_visibility=min_anchor_visibility,
    )


# ---------------------------------------------------------------------------
# Phase 6b: position-quality flagging via RTS Kalman smoother
# ---------------------------------------------------------------------------
#
# Motivation (01c §10.15.6): the per-frame anatomy ratifier in the v1.6
# bidirectional combiner is a *shape*-quality gate (torso/thigh/shank
# ratio bands) — it cannot catch positional drift. PV_800m_1 frames
# 125–127 had skeletons drawn with feet at/below track level despite
# passing the asymmetric anatomy ratios. v1.7 gap-fill recovers
# non-contact-moment dropouts but doesn't touch the contact-moment
# anchor frames §6.4 GCT consumes. Phase 6b operates *at* contact
# moments by flagging landmark observations that deviate from a
# temporally-smoothed trajectory and replacing them with the smoother
# estimate.
#
# Algorithm: per-(landmark, axis) constant-velocity Kalman filter with
# RTS backward pass. dt = 1 frame, fixed (Q is in px²/frame² units, so
# the parameters are frame-rate independent). State [position, velocity]
# evolves under F = [[1, 1], [0, 1]] with process noise Q = diag(Q_pos,
# Q_vel); observations z_t = position + N(0, R). NaN observations skip
# the update step (predict-only) — the RTS pass natively handles missing
# data. v1.7 gap-fill runs first so most short gaps are already filled
# before the smoother sees them.
#
# Output mode is **flag-and-replace**: clean-frame observations pass
# through unchanged; only frames where |obs - smoothed| > flag_sigma *
# sqrt(R) get replaced. This preserves the v1.7-passing verdict cells
# byte-for-byte on clean frames (the "no regression" criterion is
# binary) while directly attacking the §10.15.6 mis-anchored case.

# Q_vel = 2.8 px²/frame² with R = 25 px² gives a smoothing scale
# τ ≈ √(R / Q_vel) ≈ 3 frames (~50 ms at 60 fps) — wide enough to
# damp single-frame mis-anchoring, narrow enough to preserve the
# touchdown / toe-off velocity transition contact detection reads.
# Q_pos is a small acceleration-leakage term (essentially zero in a
# clean constant-velocity regime).
DEFAULT_Q_POS = 0.1
DEFAULT_Q_VEL = 2.8
# R = 25 px² (~5 px MediaPipe per-frame jitter at 4K). Sets the
# observation noise floor and defines the flag threshold scale.
DEFAULT_R = 25.0
# 3-σ rule on observation noise: |obs - smoothed| > 3 * 5 px = 15 px
# at default R. At Kinovea 244.61 px/m calibration that's ~6 cm — the
# magnitude of the §10.15.6 frames-125–127 mis-anchoring.
DEFAULT_FLAG_SIGMA = 3.0


@dataclass
class SmoothResult:
    """Output of flag_and_smooth. Mirrors CleanupResult contract: same
    ndarray shapes, NaN-on-pre-init / unobservable, plus a per-frame
    log of flag / replace activity for downstream diagnostics."""

    landmarks: np.ndarray  # (T, 33, 2) — flagged values replaced with smoothed
    visibilities: np.ndarray  # (T, 33) — passthrough (smoother does not modify)
    world_landmarks: Optional[np.ndarray]  # (T, 33, 3) — passthrough
    log: List[Dict[str, Any]] = field(default_factory=list)
    Q_pos: float = DEFAULT_Q_POS
    Q_vel: float = DEFAULT_Q_VEL
    R: float = DEFAULT_R
    flag_sigma: float = DEFAULT_FLAG_SIGMA

    @property
    def n_flagged_total(self) -> int:
        return sum(int(e["n_flagged"]) for e in self.log)

    @property
    def n_replaced_total(self) -> int:
        # In flag-only mode, n_replaced == n_flagged. Kept as a separate
        # counter so a future full-pass mode can diverge them.
        return sum(int(e["n_replaced"]) for e in self.log)

    @property
    def n_frames_with_flag(self) -> int:
        return sum(1 for e in self.log if int(e["n_flagged"]) > 0)


def _rts_smooth_1d(
    obs: np.ndarray,
    Q_pos: float,
    Q_vel: float,
    R: float,
) -> np.ndarray:
    """Forward Kalman + backward RTS pass on a 1-D observation series.

    State: [position, velocity]. Constant-velocity model, dt = 1 frame.
    NaN observations skip the update step (predict-only). Returns
    smoothed positions; NaN before the first finite observation and
    when the stream has fewer than 2 finite observations (cannot
    establish velocity estimate).
    """
    T = len(obs)
    F = np.array([[1.0, 1.0], [0.0, 1.0]])
    H = np.array([[1.0, 0.0]])
    Q = np.array([[Q_pos, 0.0], [0.0, Q_vel]])
    R_mat = np.array([[R]])
    eye2 = np.eye(2)

    finite_idx = np.where(~np.isnan(obs))[0]
    if len(finite_idx) < 2:
        return np.full(T, np.nan)

    t0 = int(finite_idx[0])

    x_pred = np.zeros((T, 2))
    P_pred = np.zeros((T, 2, 2))
    x_filt = np.zeros((T, 2))
    P_filt = np.zeros((T, 2, 2))

    # Initialise at t0: position seeded from obs, velocity prior weak.
    x_filt[t0, 0] = float(obs[t0])
    x_filt[t0, 1] = 0.0
    P_filt[t0] = np.diag([R, 10.0 * Q_vel])

    for t in range(t0 + 1, T):
        x_pred[t] = F @ x_filt[t - 1]
        P_pred[t] = F @ P_filt[t - 1] @ F.T + Q
        if not np.isnan(obs[t]):
            S = H @ P_pred[t] @ H.T + R_mat
            K = P_pred[t] @ H.T @ np.linalg.inv(S)
            innov = float(obs[t] - (H @ x_pred[t])[0])
            x_filt[t] = x_pred[t] + (K.flatten() * innov)
            P_filt[t] = (eye2 - K @ H) @ P_pred[t]
        else:
            x_filt[t] = x_pred[t]
            P_filt[t] = P_pred[t]

    x_smooth = x_filt.copy()
    P_smooth = P_filt.copy()
    for t in range(T - 2, t0 - 1, -1):
        # Smoother gain: G_t = P_filt[t] F^T inv(P_pred[t+1])
        try:
            P_pred_inv = np.linalg.inv(P_pred[t + 1])
        except np.linalg.LinAlgError:
            continue
        G = P_filt[t] @ F.T @ P_pred_inv
        x_smooth[t] = x_filt[t] + G @ (x_smooth[t + 1] - x_pred[t + 1])
        P_smooth[t] = P_filt[t] + G @ (P_smooth[t + 1] - P_pred[t + 1]) @ G.T

    out = np.full(T, np.nan)
    out[t0:] = x_smooth[t0:, 0]
    return out


def flag_and_smooth(
    landmarks: np.ndarray,
    visibilities: np.ndarray,
    world_landmarks: Optional[np.ndarray] = None,
    Q_pos: float = DEFAULT_Q_POS,
    Q_vel: float = DEFAULT_Q_VEL,
    R: float = DEFAULT_R,
    flag_sigma: float = DEFAULT_FLAG_SIGMA,
) -> SmoothResult:
    """Flag-and-replace position-quality cleanup via RTS Kalman smoother.

    For each (landmark, axis) trajectory: run a constant-velocity Kalman
    filter forward then RTS-smooth backward. Compute residual
    ``|obs[t] - smoothed[t]|`` for each finite observation; replace
    observations where the residual exceeds ``flag_sigma * sqrt(R)``
    with the smoother estimate. Observations within the threshold pass
    through unchanged — this preserves clean-frame inputs byte-identical
    to the upstream pipeline (motion_filter / combiner / fill_gaps),
    minimising downstream regression risk.

    Per-(landmark, axis) independence is intentional: the x and y
    components of a running landmark are approximately independent in
    the sagittal-plane image projection, and per-axis filtering halves
    the state-space dimension. Per-landmark independence is the same
    rationale as ``fill_gaps``.

    World landmarks (3D) are *not* smoothed — the smoother operates on
    pixel coordinates and the noise floor in MediaPipe's world frame is
    a different beast. World landmarks pass through unchanged.

    Visibilities are *not* modified — a smoother that replaces a
    landmark position should not invent confidence; the visibility was
    whatever MediaPipe / fill_gaps produced for that frame.

    Args:
        landmarks: (T, 33, 2) pixel landmarks. NaN means missing /
            previously-rejected.
        visibilities: (T, 33) per-(frame, landmark) confidence. Used
            only for the per-frame log; not modified.
        world_landmarks: optional (T, 33, 3) metric-space landmarks.
            Passed through unchanged.
        Q_pos: process-noise variance on position state (px²).
        Q_vel: process-noise variance on velocity state (px²/frame²).
        R: observation-noise variance (px²). Default 25 ≈ (5 px)²
            MediaPipe per-frame jitter at 4K.
        flag_sigma: residual threshold in units of sqrt(R). 3.0 ≈ 15 px
            at default R, ≈ 6 cm at 4K Kinovea calibration.

    Returns:
        SmoothResult. The landmarks ndarray has flagged values replaced
        with their smoother estimates; non-flagged values are unchanged.
        visibilities and world_landmarks pass through (copied).
    """
    if Q_pos < 0 or Q_vel < 0 or R <= 0:
        raise ValueError(
            f"Q_pos / Q_vel must be ≥ 0 and R > 0, got "
            f"Q_pos={Q_pos}, Q_vel={Q_vel}, R={R}"
        )
    if flag_sigma <= 0:
        raise ValueError(f"flag_sigma must be > 0, got {flag_sigma}")

    T, n_landmarks, _ = landmarks.shape
    out_lm = landmarks.copy()
    out_vis = visibilities.copy()
    out_world = None if world_landmarks is None else world_landmarks.copy()

    flag_threshold = flag_sigma * float(np.sqrt(R))

    log: List[Dict[str, Any]] = [
        {
            "frame": int(t),
            "n_observed": 0,
            "n_flagged": 0,
            "n_replaced": 0,
            "max_residual_px": 0.0,
        }
        for t in range(T)
    ]

    for lm_idx in range(n_landmarks):
        for ax in range(2):
            obs = landmarks[:, lm_idx, ax]
            smoothed = _rts_smooth_1d(obs, Q_pos, Q_vel, R)
            for t in range(T):
                obs_t = float(obs[t]) if not np.isnan(obs[t]) else np.nan
                sm_t = float(smoothed[t]) if not np.isnan(smoothed[t]) else np.nan
                if np.isnan(obs_t):
                    continue
                # Only the x-axis pass increments n_observed (axes are
                # paired per landmark — counting both would double-count).
                if ax == 0:
                    log[t]["n_observed"] += 1
                if np.isnan(sm_t):
                    continue
                residual = abs(obs_t - sm_t)
                if residual > log[t]["max_residual_px"]:
                    log[t]["max_residual_px"] = float(residual)
                if residual > flag_threshold:
                    out_lm[t, lm_idx, ax] = sm_t
                    # n_flagged / n_replaced count per-(landmark, axis)
                    # flag events — both x and y axes contribute. A
                    # landmark with only y flagged increments by 1; a
                    # landmark with both axes flagged increments by 2.
                    log[t]["n_flagged"] += 1
                    log[t]["n_replaced"] += 1

    return SmoothResult(
        landmarks=out_lm,
        visibilities=out_vis,
        world_landmarks=out_world,
        log=log,
        Q_pos=Q_pos,
        Q_vel=Q_vel,
        R=R,
        flag_sigma=flag_sigma,
    )


# ---------------------------------------------------------------------------
# Phase 6b iteration #2: per-landmark velocity ceiling
# ---------------------------------------------------------------------------
#
# Iteration #1 (RTS Kalman in flag_and_smooth above) failed pilot on
# PV_800m_1: at any parameter setting the smoother either over-flagged
# real contact-moment biomechanics (regressing §6.4 GCT) or
# under-flagged the §10.15.6 mis-anchored case. Root cause: the
# constant-velocity prior cannot distinguish "ankle decelerating at
# touchdown" from "ankle drawn at wrong height" — both produce large
# residuals.
#
# Iteration #2 uses a different mechanism: hard biomechanical velocity
# ceiling. A frame is flagged as an isolated outlier ONLY when:
#
#   |Δposition[t-1 → t]|   > ceiling, AND
#   |Δposition[t   → t+1]| > ceiling, AND
#   |Δposition[t-1 → t+1]| ≤ ceiling  (the neighbours are mutually
#                                       consistent — the outlier is t)
#
# This pattern detects single-frame spikes (the §10.15.6 frame-126
# case: ankle drops 100+ px to track level then snaps back) but does
# NOT trigger on contact-moment biomechanics (where exactly one of the
# Δs is high — touchdown deceleration, toe-off acceleration — but the
# OTHER is small because the ankle is briefly stationary on the ground).
# Because real contact transitions only flag one neighbour, not both,
# the contact-moment-replacement failure mode of iteration #1 is
# structurally avoided.
#
# Default ceiling 80 px/frame at 4K / 60 fps ≈ 33 cm/frame ≈ 19.6 m/s.
# Real ankle vertical velocity peaks at toe-off ≈ 4–6 m/s (~16–24
# px/frame); horizontal swing-leg velocity peaks ≈ 14 m/s (~57 px/frame
# in body frame). The 80 px/frame ceiling leaves a comfortable margin
# above real motion while catching the §10.15.6 single-frame mis-
# anchoring (~70–120 px Δy in one frame).

DEFAULT_VELOCITY_CEILING_PX_PER_FRAME = 80.0


@dataclass
class VelocityOutlierResult:
    """Output of flag_velocity_outliers. Mirrors SmoothResult contract:
    same ndarray shapes, isolated outliers replaced with neighbour
    midpoint, plus a per-frame log of flag activity."""

    landmarks: np.ndarray  # (T, 33, 2) — flagged values replaced
    visibilities: np.ndarray  # (T, 33) — passthrough
    world_landmarks: Optional[np.ndarray]  # (T, 33, 3) — passthrough
    log: List[Dict[str, Any]] = field(default_factory=list)
    ceiling_px_per_frame: float = DEFAULT_VELOCITY_CEILING_PX_PER_FRAME

    @property
    def n_flagged_total(self) -> int:
        return sum(int(e["n_flagged"]) for e in self.log)

    @property
    def n_frames_with_flag(self) -> int:
        return sum(1 for e in self.log if int(e["n_flagged"]) > 0)


def flag_velocity_outliers(
    landmarks: np.ndarray,
    visibilities: np.ndarray,
    world_landmarks: Optional[np.ndarray] = None,
    ceiling_px_per_frame: float = DEFAULT_VELOCITY_CEILING_PX_PER_FRAME,
) -> VelocityOutlierResult:
    """Flag isolated single-frame outliers via per-(landmark, axis)
    velocity ceiling and replace with neighbour midpoint.

    A frame ``t`` is flagged on a given (landmark, axis) iff:

      - ``landmarks[t-1, t, t+1, lm_idx, ax]`` are all finite, AND
      - ``|obs[t] - obs[t-1]| > ceiling``, AND
      - ``|obs[t+1] - obs[t]| > ceiling``, AND
      - ``|obs[t+1] - obs[t-1]| ≤ ceiling`` (neighbours mutually
        consistent — confirms ``t`` is the outlier, not ``t-1`` or
        ``t+1``).

    Flagged values are replaced with the simple midpoint
    ``(obs[t-1] + obs[t+1]) / 2``. No interpolation across multi-frame
    gaps — those are gap-fill's territory; this function only handles
    single-frame spikes.

    The pattern naturally avoids contact-moment biomechanics: at
    touchdown only ``|obs[t-1] - obs[t]|`` is large (foot landing); at
    toe-off only ``|obs[t] - obs[t+1]|`` is large (foot lifting); in
    neither case are BOTH transitions large, so neither triggers a
    flag. This is the structural fix for the iteration-#1 RTS failure
    mode (where every contact moment was over-flagged).

    Args:
        landmarks: (T, 33, 2) pixel landmarks. NaN means missing.
        visibilities: (T, 33) per-(frame, landmark) confidence.
            Passthrough — not modified.
        world_landmarks: optional (T, 33, 3) metric landmarks.
            Passthrough — not modified.
        ceiling_px_per_frame: per-frame inter-pixel velocity threshold.
            Default 80 px/frame ≈ 33 cm/frame at 4K Kinovea calibration
            (244.61 px/m). A real ankle peaks ≈ 24 px/frame vertical at
            toe-off, ≈ 57 px/frame horizontal during swing — the
            default leaves a 1.4–3× headroom above real motion.

    Returns:
        VelocityOutlierResult. landmarks has flagged values replaced
        with neighbour midpoints; non-flagged values pass through. log
        records the per-frame flag count and max observed velocity.

    Raises:
        ValueError: on non-positive ceiling.
    """
    if ceiling_px_per_frame <= 0:
        raise ValueError(
            f"ceiling_px_per_frame must be > 0, got {ceiling_px_per_frame}"
        )

    T, n_landmarks, _ = landmarks.shape
    out_lm = landmarks.copy()
    out_vis = visibilities.copy()
    out_world = None if world_landmarks is None else world_landmarks.copy()

    log: List[Dict[str, Any]] = [
        {
            "frame": int(t),
            "n_observed": 0,
            "n_flagged": 0,
            "max_velocity_px_per_frame": 0.0,
        }
        for t in range(T)
    ]

    for lm_idx in range(n_landmarks):
        # n_observed is per-(landmark) — count via x-axis only (paired).
        x_finite = ~np.isnan(landmarks[:, lm_idx, 0])
        for t in range(T):
            if bool(x_finite[t]):
                log[t]["n_observed"] += 1

        for ax in range(2):
            obs = landmarks[:, lm_idx, ax]
            for t in range(1, T - 1):
                if np.isnan(obs[t]) or np.isnan(obs[t - 1]) or np.isnan(obs[t + 1]):
                    continue
                a, b, c = float(obs[t - 1]), float(obs[t]), float(obs[t + 1])
                d_prev = abs(b - a)
                d_next = abs(c - b)
                d_skip = abs(c - a)
                local_max = max(d_prev, d_next)
                if local_max > log[t]["max_velocity_px_per_frame"]:
                    log[t]["max_velocity_px_per_frame"] = float(local_max)
                if (
                    d_prev > ceiling_px_per_frame
                    and d_next > ceiling_px_per_frame
                    and d_skip <= ceiling_px_per_frame
                ):
                    out_lm[t, lm_idx, ax] = 0.5 * (a + c)
                    log[t]["n_flagged"] += 1

    return VelocityOutlierResult(
        landmarks=out_lm,
        visibilities=out_vis,
        world_landmarks=out_world,
        log=log,
        ceiling_px_per_frame=ceiling_px_per_frame,
    )
