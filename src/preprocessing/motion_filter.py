"""Post-hoc motion-consistency filter for single-candidate pose output.

Original v1.2 design: flag frames where the tracked "person" is not
actually moving horizontally. Premise: a background ghost (construction
fence, poster, crane arm) that ghost-locks MediaPipe's tracker has ~0
horizontal hip-x velocity, while the real runner has 3-8 m/s. Filtering
on that single property lets us keep v1.0's simple pipeline while
discarding ghost-contaminated frames before they poison downstream
metrics.

v1.4 anatomy-gate experiment (TESTED — DEFERRED, see 01c §10.12):
an optional anatomy gate runs alongside the motion gate as a strict-AND
combiner. The 2026-04-25 Phase 4.5 measurement
(``scripts/measure_anatomy_violations.py``) showed that motion_filter
passes 8.8-22.1 % of ghost frames whose landmarks are clearly
non-anatomical (FAIL torso medians 0.53-0.68 vs NASA 0.30 — visually
confirmed ghost-lock). The v1.4 strict-AND ship correctly rejected
those frames but introduced NaN gaps in the landmark stream that broke
ground-contact temporal continuity (800m_1 lost 2 of 3 refined contacts;
Recovery_2 stride regressed PASS → FAIL). §6.4 GCT was unchanged on
every clip regardless (refined-only mean — confirms anatomy gating
cannot help §6.4). Decision: rolled back as production default. The
gate code is retained as **dormant** capability —
``apply_anatomy_gate`` defaults to ``False`` so the production path is
v1.2-equivalent; the runner script's ``--anatomy-gate`` flag is kept
for future re-experimentation. Re-design options recorded in 01c §10.12
(mark-don't-filter, asymmetric tolerance, or Fix-C-coupled re-detection
trigger) — none should be pursued before Fix B (v1.5) ships.

This module is the final surviving piece of the M7/M9/M10 mitigation
chain described in ``docs/technical/01c_pre_validation_session_report.md``
§10.6 — the multi-candidate/motion-gated/composite approaches all failed
because MediaPipe's detector itself, not the tracker, is the bottleneck
for fast runners at 4K. A post-hoc filter sidesteps that problem: we
don't try to make the detector better, we just drop its output when it's
obviously wrong.

Typical use:

    from src.preprocessing.motion_filter import filter_landmarks_by_motion

    result = filter_landmarks_by_motion(
        landmarks, visibilities, world_landmarks,
        fps=fps, pixels_per_cm=calibration.pixels_per_cm,
        body_height_cm=runner_height + sole_cm,  # v1.4 anatomy gate
        apply_anatomy_gate=True,
    )
    # result.landmarks / .visibilities have NaN/0 on rejected frames
    # result.log has per-frame decisions (motion + anatomy)
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional, Tuple

import numpy as np

from src.preprocessing.pose_estimator import LandmarkIndex

# Default threshold in metres/second. Below this apparent hip-x speed
# the frame is considered static (ghost). 0.5 m/s is well below any
# plausible running speed (recovery pace floor ~2.5 m/s) and well above
# pose-jitter noise on a static subject. Tunable by caller.
DEFAULT_MIN_VELOCITY_MS = 0.5

# Rolling window used to smooth the velocity signal. Stride-cycle hip
# motion naturally dips to near zero at mid-stance even for a real
# runner; taking the max over a small window keeps those frames.
ROLLING_WINDOW = 5

# v1.4 anatomy gate. NASA body-height proportions and the per-segment
# tolerance bands the gate enforces per frame. Asymmetric (Phase 5
# 2026-04-25): torso stays at ±20 % because torso ratio is essentially
# rigid across the gait cycle and was the strongest discriminator in the
# §10.12 measurement (PASS torso 0.276-0.328 vs FAIL torso medians
# 0.53-0.68). Thigh / shank widened to ±30 % because those segments
# legitimately stretch / contract through the stride cycle (apparent
# pixel length grows toward full extension at toe-off) — the §10.12
# measurement showed PASS shanks reaching 0.198, exactly at the old
# ±20 % lower bound (0.197), causing single-frame dropouts mid-stride.
# The first Phase 5 pilot on PV_800m_1_Victory showed every visible-
# window gap was a shank-ratio failure on a runner-tracking frame.
NASA_PROPORTIONS = {"torso": 0.30, "thigh": 0.245, "shank": 0.246}
ANATOMY_TOLERANCE = {"torso": 0.20, "thigh": 0.30, "shank": 0.30}


@dataclass
class MotionFilterResult:
    """Output of filter_landmarks_by_motion."""

    landmarks: np.ndarray  # (T, 33, 2) — NaN where rejected
    visibilities: np.ndarray  # (T, 33) — 0 where rejected
    world_landmarks: Optional[np.ndarray]  # (T, 33, 3) — NaN where rejected
    log: List[Dict[str, Any]] = field(default_factory=list)
    threshold_m_s: float = DEFAULT_MIN_VELOCITY_MS

    @property
    def n_accepted(self) -> int:
        return sum(1 for entry in self.log if entry["accepted"])

    @property
    def n_rejected(self) -> int:
        return sum(1 for entry in self.log if not entry["accepted"])


def _hip_x(landmarks: np.ndarray) -> np.ndarray:
    """Per-frame mean hip X position. Returns (T,) with NaN where missing."""
    hip_l = landmarks[:, LandmarkIndex.LEFT_HIP, 0]
    hip_r = landmarks[:, LandmarkIndex.RIGHT_HIP, 0]
    with np.errstate(invalid="ignore"):
        out: np.ndarray = np.nanmean(np.stack([hip_l, hip_r]), axis=0)
        return out


def _segment_lengths_2d(
    L: np.ndarray,
) -> Optional[Tuple[float, float, float, float, float, float, float]]:
    """For a single frame's (33, 2) landmarks, return key vertical positions
    and 2D pixel segment lengths used by the anatomy gate.

    Returns None if any required landmark is NaN.
    Returns (y_nose, y_mid_sh, y_mid_hip, y_mid_knee, torso_px, thigh_px,
    shank_px). thigh_px / shank_px take the larger of left vs right per leg
    (the more-extended leg, less affected by foreshortening).
    """
    NOSE = LandmarkIndex.NOSE
    LSH, RSH = LandmarkIndex.LEFT_SHOULDER, LandmarkIndex.RIGHT_SHOULDER
    LH, RH = LandmarkIndex.LEFT_HIP, LandmarkIndex.RIGHT_HIP
    LK, RK = LandmarkIndex.LEFT_KNEE, LandmarkIndex.RIGHT_KNEE
    LA, RA = LandmarkIndex.LEFT_ANKLE, LandmarkIndex.RIGHT_ANKLE
    required = (NOSE, LSH, RSH, LH, RH, LK, RK, LA, RA)
    for i in required:
        if np.any(np.isnan(L[i])):
            return None

    y_nose = float(L[NOSE, 1])
    y_mid_sh = float((L[LSH, 1] + L[RSH, 1]) / 2)
    y_mid_hip = float((L[LH, 1] + L[RH, 1]) / 2)
    y_mid_knee = float((L[LK, 1] + L[RK, 1]) / 2)

    mid_sh = (L[LSH] + L[RSH]) / 2
    mid_hip = (L[LH] + L[RH]) / 2
    torso_px = float(np.linalg.norm(mid_sh - mid_hip))
    thigh_l = float(np.linalg.norm(L[LH] - L[LK]))
    thigh_r = float(np.linalg.norm(L[RH] - L[RK]))
    shank_l = float(np.linalg.norm(L[LK] - L[LA]))
    shank_r = float(np.linalg.norm(L[RK] - L[RA]))
    return (
        y_nose,
        y_mid_sh,
        y_mid_hip,
        y_mid_knee,
        torso_px,
        max(thigh_l, thigh_r),
        max(shank_l, shank_r),
    )


def _anatomy_pass_per_frame(
    landmarks: np.ndarray, body_height_px: float
) -> Tuple[np.ndarray, List[str]]:
    """Compute per-frame anatomy-gate decisions.

    Returns (anatomy_pass, reasons). anatomy_pass[t] is True iff the frame
    has all required landmarks AND vertical-stacking is correct AND each
    of {torso, thigh, shank} pixel length divided by body_height_px lands
    within ±ANATOMY_TOLERANCE[segment] of its NASA proportion (torso
    ±20 %, thigh / shank ±30 %; see the constant block above for the
    measurement that motivated the asymmetric bands). reasons[t] is a
    short string describing the failure mode (or "anatomy_pass" /
    "missing_landmarks").
    """
    T = len(landmarks)
    anatomy_pass = np.zeros(T, dtype=bool)
    reasons: List[str] = []
    if body_height_px <= 0:
        return anatomy_pass, ["bad_body_height_px"] * T

    for t in range(T):
        seg = _segment_lengths_2d(landmarks[t])
        if seg is None:
            reasons.append("missing_landmarks")
            continue
        y_nose, y_sh, y_hip, y_knee, torso_px, thigh_px, shank_px = seg
        v_ok = y_nose < y_sh < y_hip < y_knee
        torso_r = torso_px / body_height_px
        thigh_r = thigh_px / body_height_px
        shank_r = shank_px / body_height_px
        torso_ok = (
            abs(torso_r - NASA_PROPORTIONS["torso"]) / NASA_PROPORTIONS["torso"]
            <= ANATOMY_TOLERANCE["torso"]
        )
        thigh_ok = (
            abs(thigh_r - NASA_PROPORTIONS["thigh"]) / NASA_PROPORTIONS["thigh"]
            <= ANATOMY_TOLERANCE["thigh"]
        )
        shank_ok = (
            abs(shank_r - NASA_PROPORTIONS["shank"]) / NASA_PROPORTIONS["shank"]
            <= ANATOMY_TOLERANCE["shank"]
        )
        r_ok = torso_ok and thigh_ok and shank_ok
        if v_ok and r_ok:
            anatomy_pass[t] = True
            reasons.append("anatomy_pass")
        elif not v_ok and not r_ok:
            reasons.append("vertical_AND_ratios")
        elif not v_ok:
            reasons.append("vertical_only")
        else:
            failing = []
            if not torso_ok:
                failing.append("torso")
            if not thigh_ok:
                failing.append("thigh")
            if not shank_ok:
                failing.append("shank")
            reasons.append("ratios:" + "+".join(failing))
    return anatomy_pass, reasons


def _rolling_max(values: np.ndarray, window: int) -> np.ndarray:
    """Windowed max over ±window//2 frames around each sample."""
    T = len(values)
    half = window // 2
    out = np.full(T, np.nan)
    for t in range(T):
        lo = max(0, t - half)
        hi = min(T, t + half + 1)
        chunk = values[lo:hi]
        finite = chunk[np.isfinite(chunk)]
        if finite.size:
            out[t] = float(finite.max())
    return out


def filter_landmarks_by_motion(
    landmarks: np.ndarray,
    visibilities: np.ndarray,
    world_landmarks: Optional[np.ndarray],
    fps: float,
    pixels_per_cm: float,
    min_velocity_ms: float = DEFAULT_MIN_VELOCITY_MS,
    rolling_window: int = ROLLING_WINDOW,
    body_height_cm: Optional[float] = None,
    apply_anatomy_gate: bool = False,
) -> MotionFilterResult:
    """Flag static-hip frames (and, optionally, anatomically-implausible
    frames) in a v1.0 pose output.

    Motion gate (always on):
      1. Compute per-frame hip X position from LEFT_HIP + RIGHT_HIP.
      2. Frame-to-frame velocity = |Δhip_x| (px/frame). Frames where
         MediaPipe detected nothing carry NaN and propagate.
      3. Smooth with a rolling-max over ``rolling_window`` frames so
         mid-stance hip pauses don't falsely trigger rejection on a real
         runner (the runner's hip moves fast *somewhere* in any small
         window; a static ghost does not).
      4. Threshold at ``min_velocity_ms`` (converted to px/frame via the
         calibration's pixels_per_cm). Frames below threshold → rejected.

    Anatomy gate (v1.4, opt-in via apply_anatomy_gate=True):
      5. For each frame, compute 2D pixel segment lengths (torso, thigh,
         shank) and check that each, divided by body_height_px =
         body_height_cm × pixels_per_cm, lands within
         ±ANATOMY_TOLERANCE[segment] of its NASA proportion (torso
         ±20 %, thigh / shank ±30 %). Vertical-stacking
         (NOSE < shoulder < hip < knee in pixel-y) also checked as a
         zero-cost belt-and-braces signal. A frame is kept iff motion AND
         anatomy both pass (strict-AND).

    Args:
        landmarks: (T, 33, 2) v1.0 single-candidate landmarks in pixels.
        visibilities: (T, 33) visibility scores.
        world_landmarks: Optional (T, 33, 3) metric-space landmarks.
        fps: Video frame rate.
        pixels_per_cm: From the anatomical calibration.
        min_velocity_ms: Velocity threshold in metres/second.
        rolling_window: Frames over which to take rolling max.
        body_height_cm: Runner standing height in cm (with shoes). Required
            when apply_anatomy_gate=True; otherwise ignored.
        apply_anatomy_gate: When True, also runs the anatomy gate and
            combines it with the motion gate via strict-AND.

    Returns:
        MotionFilterResult. Rejected frames carry NaN landmarks and zero
        visibilities; the log records both motion and anatomy decisions
        per frame plus a unified ``reason``.
    """
    if apply_anatomy_gate and body_height_cm is None:
        raise ValueError("apply_anatomy_gate=True requires body_height_cm to be set.")

    T = len(landmarks)
    if T == 0:
        return MotionFilterResult(
            landmarks=landmarks.copy(),
            visibilities=visibilities.copy(),
            world_landmarks=None if world_landmarks is None else world_landmarks.copy(),
            log=[],
            threshold_m_s=float(min_velocity_ms),
        )

    hip_x = _hip_x(landmarks)
    diffs = np.abs(np.diff(hip_x))
    # Prepend NaN so velocity[t] means "displacement from frame t-1 to t".
    velocity_pxf = np.concatenate([[np.nan], diffs])
    smoothed = _rolling_max(velocity_pxf, rolling_window)

    ms_to_pxf = (pixels_per_cm * 100.0) / fps
    threshold_pxf = float(min_velocity_ms) * ms_to_pxf

    if apply_anatomy_gate:
        assert body_height_cm is not None  # guarded above
        body_height_px = float(body_height_cm) * float(pixels_per_cm)
        anatomy_pass, anatomy_reasons = _anatomy_pass_per_frame(
            landmarks, body_height_px
        )
    else:
        anatomy_pass = np.ones(T, dtype=bool)
        anatomy_reasons = ["disabled"] * T

    out_lm = landmarks.copy()
    out_vis = visibilities.copy()
    out_world = None if world_landmarks is None else world_landmarks.copy()

    log: List[Dict[str, Any]] = []
    for t in range(T):
        frame_missing = not np.isfinite(hip_x[t])
        smooth_v = smoothed[t]
        has_signal = np.isfinite(smooth_v)
        if frame_missing:
            motion_reason = "no_detection"
            motion_accepted = False
        elif not has_signal:
            motion_reason = "no_velocity_context"
            motion_accepted = False
        elif smooth_v < threshold_pxf:
            motion_reason = "below_velocity_threshold"
            motion_accepted = False
        else:
            motion_reason = "moving"
            motion_accepted = True

        anat_accepted = bool(anatomy_pass[t])
        anat_reason = anatomy_reasons[t]
        accepted = motion_accepted and anat_accepted

        if motion_accepted and not anat_accepted:
            unified_reason = f"anatomy:{anat_reason}"
        elif not motion_accepted and not anat_accepted and apply_anatomy_gate:
            unified_reason = f"{motion_reason}+anatomy:{anat_reason}"
        else:
            unified_reason = motion_reason

        if not accepted:
            out_lm[t] = np.nan
            out_vis[t] = 0.0
            if out_world is not None:
                out_world[t] = np.nan

        log.append(
            {
                "frame": int(t),
                "accepted": bool(accepted),
                "reason": unified_reason,
                "motion_accepted": bool(motion_accepted),
                "motion_reason": motion_reason,
                "anatomy_accepted": anat_accepted,
                "anatomy_reason": anat_reason,
                "hip_x_px": float(hip_x[t]) if np.isfinite(hip_x[t]) else float("nan"),
                "velocity_px_per_frame": (
                    float(smooth_v) if np.isfinite(smooth_v) else float("nan")
                ),
                "velocity_m_s": (
                    float(smooth_v / ms_to_pxf)
                    if np.isfinite(smooth_v) and ms_to_pxf > 0
                    else float("nan")
                ),
            }
        )

    return MotionFilterResult(
        landmarks=out_lm,
        visibilities=out_vis,
        world_landmarks=out_world,
        log=log,
        threshold_m_s=float(min_velocity_ms),
    )


# Default fraction of body height (in pixels) under which two anatomy-passing
# pose candidates are deemed "essentially the same pose" and averaged. At
# 4K with ~600 px body height that's ~18 px — well inside frame-to-frame
# pose-jitter noise but well below the 50-200 px gap a runner-vs-ghost
# disagreement produces.
BIDIR_AVG_DISTANCE_FRAC = 0.03


@dataclass
class BidirectionalCombineResult:
    """Output of combine_bidirectional_pose."""

    landmarks: np.ndarray  # (T, 33, 2) — NaN on frames where neither pass passes
    visibilities: np.ndarray  # (T, 33) — 0 where neither pass passes
    world_landmarks: Optional[np.ndarray]  # (T, 33, 3) — NaN where neither passes
    log: List[Dict[str, Any]] = field(default_factory=list)

    @property
    def n_chosen_fwd(self) -> int:
        return sum(1 for entry in self.log if entry["chosen_pass"] == "fwd")

    @property
    def n_chosen_rev(self) -> int:
        return sum(1 for entry in self.log if entry["chosen_pass"] == "rev")

    @property
    def n_chosen_avg(self) -> int:
        return sum(1 for entry in self.log if entry["chosen_pass"] == "avg")

    @property
    def n_dropped(self) -> int:
        return sum(1 for entry in self.log if entry["chosen_pass"] == "none")

    @property
    def n_chosen_seed_bwd(self) -> int:
        return sum(1 for entry in self.log if entry["chosen_pass"] == "seed_bwd")

    @property
    def n_chosen_seed_fwd(self) -> int:
        return sum(1 for entry in self.log if entry["chosen_pass"] == "seed_fwd")


def combine_bidirectional_pose(
    lm_fwd: np.ndarray,
    vis_fwd: np.ndarray,
    world_fwd: Optional[np.ndarray],
    lm_rev: np.ndarray,
    vis_rev: np.ndarray,
    world_rev: Optional[np.ndarray],
    body_height_px: float,
    avg_distance_threshold_frac: float = BIDIR_AVG_DISTANCE_FRAC,
) -> BidirectionalCombineResult:
    """Per-frame combiner for forward + reverse pose passes (v1.6).

    Anatomy is used here as a per-frame **ratifier** between two candidate
    landmark sets, NOT as a strict-AND filter on a single candidate (the
    v1.4 mistake — see 01c §10.12). Reuses ``_anatomy_pass_per_frame``
    from this module, which Phase 4.5 measurement showed cleanly
    discriminates ghost-locked frames (FAIL torso medians 0.5-0.7) from
    runner-locked frames (PASS torso ranges 0.276-0.328).
    Per-frame decision tree:
      - Forward passes anatomy and reverse fails  → use forward.
      - Reverse passes and forward fails          → use reverse.
      - Both pass and inter-pass landmark
        distance ≤ avg_distance_threshold_frac
        × body_height_px                          → average them.
      - Both pass but disagree spatially          → use the higher
                                                    mean-hip-visibility
                                                    candidate.
      - Neither passes                            → NaN.

    A single-candidate fallback (use a detected-but-anatomy-failing pose
    when the other side has missing_landmarks) was tested as Phase 5
    Option B (2026-04-25) and reverted: the fallback frames carried
    enough hip / ankle position noise at gait-cycle extremes to perturb
    contact-detection refinement, regressing 3 v1.5-passing verdict cells
    (Steady_2 §6.2, 800m_1 §6.2, 800m_1 §6.4) for a single new PASS
    (Steady_1 §6.4) — net −2 PASSes vs the asymmetric-only baseline. See
    01c §10.15 for the full measurement. Re-design before re-attempting.

    Args:
        lm_fwd / lm_rev: (T, 33, 2) pixel landmarks, indexed in source-frame
            order. The reverse pass must already be flipped back to
            forward-time before being passed here (see
            ``extract_pose_landmarks_streaming_reverse``).
        vis_fwd / vis_rev: (T, 33) visibility scores.
        world_fwd / world_rev: Optional (T, 33, 3) metric-space landmarks.
            Either may be None; combined world_landmarks is None iff both
            are None.
        body_height_px: Runner standing height in pixels (runner_height_cm
            × pixels_per_cm). Required for anatomy ratification — without
            it, every frame falls into the "both fail" branch and emits NaN.
        avg_distance_threshold_frac: Fraction of body_height_px below which
            two anatomy-passing candidates are deemed identical and
            averaged. Defaults to 3 %.

    Returns:
        BidirectionalCombineResult with a per-frame log carrying
        ``chosen_pass``, ``anatomy_pass_fwd``, ``anatomy_pass_rev``,
        ``mean_hip_vis_fwd``, ``mean_hip_vis_rev``,
        ``inter_pass_distance_px``, and a human-readable ``reason``.
    """
    T = min(len(lm_fwd), len(lm_rev))

    pass_fwd, reasons_fwd = _anatomy_pass_per_frame(lm_fwd[:T], body_height_px)
    pass_rev, reasons_rev = _anatomy_pass_per_frame(lm_rev[:T], body_height_px)

    out_lm = np.full((T, 33, 2), np.nan)
    out_vis = np.zeros((T, 33))
    out_world: Optional[np.ndarray] = None
    if world_fwd is not None or world_rev is not None:
        out_world = np.full((T, 33, 3), np.nan)

    avg_distance_threshold_px = float(avg_distance_threshold_frac) * float(
        body_height_px
    )

    L_HIP = LandmarkIndex.LEFT_HIP
    R_HIP = LandmarkIndex.RIGHT_HIP

    log: List[Dict[str, Any]] = []
    for t in range(T):
        with np.errstate(invalid="ignore"):
            hip_vis_fwd = float(np.nanmean([vis_fwd[t, L_HIP], vis_fwd[t, R_HIP]]))
            hip_vis_rev = float(np.nanmean([vis_rev[t, L_HIP], vis_rev[t, R_HIP]]))

        # Inter-pass landmark distance (mean Euclidean over landmarks
        # finite in both passes). NaN if no landmark is finite in both.
        both_valid = ~(
            np.isnan(lm_fwd[t]).any(axis=-1) | np.isnan(lm_rev[t]).any(axis=-1)
        )
        if both_valid.any():
            diffs = lm_fwd[t, both_valid] - lm_rev[t, both_valid]
            mean_dist_px = float(np.linalg.norm(diffs, axis=-1).mean())
        else:
            mean_dist_px = float("nan")

        if pass_fwd[t] and pass_rev[t]:
            if np.isfinite(mean_dist_px) and mean_dist_px <= avg_distance_threshold_px:
                out_lm[t] = (lm_fwd[t] + lm_rev[t]) / 2
                out_vis[t] = (vis_fwd[t] + vis_rev[t]) / 2
                if out_world is not None:
                    if world_fwd is not None and world_rev is not None:
                        out_world[t] = (world_fwd[t] + world_rev[t]) / 2
                    elif world_fwd is not None:
                        out_world[t] = world_fwd[t]
                    elif world_rev is not None:
                        out_world[t] = world_rev[t]
                chosen = "avg"
                reason = (
                    f"both_pass:avg(d={mean_dist_px:.1f}px"
                    f"<={avg_distance_threshold_px:.1f})"
                )
            elif hip_vis_fwd >= hip_vis_rev:
                out_lm[t] = lm_fwd[t]
                out_vis[t] = vis_fwd[t]
                if out_world is not None and world_fwd is not None:
                    out_world[t] = world_fwd[t]
                chosen = "fwd"
                reason = (
                    f"both_pass:fwd_higher_vis({hip_vis_fwd:.2f}"
                    f">={hip_vis_rev:.2f},d={mean_dist_px:.1f}px)"
                )
            else:
                out_lm[t] = lm_rev[t]
                out_vis[t] = vis_rev[t]
                if out_world is not None and world_rev is not None:
                    out_world[t] = world_rev[t]
                chosen = "rev"
                reason = (
                    f"both_pass:rev_higher_vis({hip_vis_rev:.2f}"
                    f">{hip_vis_fwd:.2f},d={mean_dist_px:.1f}px)"
                )
        elif pass_fwd[t]:
            out_lm[t] = lm_fwd[t]
            out_vis[t] = vis_fwd[t]
            if out_world is not None and world_fwd is not None:
                out_world[t] = world_fwd[t]
            chosen = "fwd"
            reason = f"fwd_only(rev:{reasons_rev[t]})"
        elif pass_rev[t]:
            out_lm[t] = lm_rev[t]
            out_vis[t] = vis_rev[t]
            if out_world is not None and world_rev is not None:
                out_world[t] = world_rev[t]
            chosen = "rev"
            reason = f"rev_only(fwd:{reasons_fwd[t]})"
        else:
            chosen = "none"
            reason = f"both_fail(fwd:{reasons_fwd[t]},rev:{reasons_rev[t]})"

        log.append(
            {
                "frame": int(t),
                "chosen_pass": chosen,
                "anatomy_pass_fwd": bool(pass_fwd[t]),
                "anatomy_pass_rev": bool(pass_rev[t]),
                "mean_hip_vis_fwd": (
                    hip_vis_fwd if np.isfinite(hip_vis_fwd) else float("nan")
                ),
                "mean_hip_vis_rev": (
                    hip_vis_rev if np.isfinite(hip_vis_rev) else float("nan")
                ),
                "inter_pass_distance_px": (
                    mean_dist_px if np.isfinite(mean_dist_px) else float("nan")
                ),
                "reason": reason,
            }
        )

    return BidirectionalCombineResult(
        landmarks=out_lm,
        visibilities=out_vis,
        world_landmarks=out_world,
        log=log,
    )


def combine_three_pass_pose(
    lm_fwd: np.ndarray,
    vis_fwd: np.ndarray,
    world_fwd: Optional[np.ndarray],
    lm_rev: np.ndarray,
    vis_rev: np.ndarray,
    world_rev: Optional[np.ndarray],
    lm_seed_bwd: np.ndarray,
    vis_seed_bwd: np.ndarray,
    world_seed_bwd: Optional[np.ndarray],
    lm_seed_fwd: np.ndarray,
    vis_seed_fwd: np.ndarray,
    world_seed_fwd: Optional[np.ndarray],
    anchor_frame: int,
    body_height_px: float,
) -> BidirectionalCombineResult:
    """Per-frame best-of-3 combiner for forward + reverse + seeded passes (v1.11).

    Extends the v1.6 bidirectional combiner with two seeded passes that start
    from a well-detected mid-clip anchor frame:
      - seed_bwd covers frames [0, anchor_frame]
      - seed_fwd covers frames [anchor_frame, T-1]

    Per-frame scoring (score-based, not tournament):
      score_c = mean_hip_visibility_c  if anatomy_pass_c  else float("-inf")
    Winner = argmax(scores) over the applicable candidate set:
      frames [0, anchor_frame]  → {fwd, rev, seed_bwd}
      frames (anchor_frame, T-1] → {fwd, rev, seed_fwd}
    All candidates fail anatomy → NaN output ("none").

    Does NOT average candidates (unlike the v1.6 two-pass combiner) — with
    three candidates a score-based pick is cleaner and avoids averaging a
    good detection with a degraded one.

    Args:
        lm_fwd / lm_rev: (T, 33, 2) forward and reverse pass landmarks in
            source-frame order (reverse already flipped by its extractor).
        vis_fwd / vis_rev: (T, 33) visibility scores.
        world_fwd / world_rev: Optional (T, 33, 3) metric-space landmarks.
        lm_seed_bwd / lm_seed_fwd: (T, 33, 2) seeded pass landmarks.
            seed_bwd frames (anchor_frame, T-1] are NaN (not covered).
            seed_fwd frames [0, anchor_frame) are NaN (not covered).
        vis_seed_bwd / vis_seed_fwd: (T, 33) visibility scores.
        world_seed_bwd / world_seed_fwd: Optional (T, 33, 3).
        anchor_frame: Source-frame index of the seeded-pass anchor.
        body_height_px: Runner standing height in pixels (for anatomy gate).

    Returns:
        BidirectionalCombineResult with chosen_pass ∈
        {"fwd", "rev", "seed_bwd", "seed_fwd", "none"}.
        n_chosen_seed_bwd / n_chosen_seed_fwd properties carry seeded counts.
    """
    T = min(len(lm_fwd), len(lm_rev), len(lm_seed_bwd), len(lm_seed_fwd))
    anchor_frame = int(np.clip(anchor_frame, 0, T - 1))

    pass_fwd, reasons_fwd = _anatomy_pass_per_frame(lm_fwd[:T], body_height_px)
    pass_rev, reasons_rev = _anatomy_pass_per_frame(lm_rev[:T], body_height_px)
    pass_seed_bwd, reasons_seed_bwd = _anatomy_pass_per_frame(
        lm_seed_bwd[:T], body_height_px
    )
    pass_seed_fwd, reasons_seed_fwd = _anatomy_pass_per_frame(
        lm_seed_fwd[:T], body_height_px
    )

    out_lm = np.full((T, 33, 2), np.nan)
    out_vis = np.zeros((T, 33))
    out_world: Optional[np.ndarray] = None
    if any(
        w is not None for w in [world_fwd, world_rev, world_seed_bwd, world_seed_fwd]
    ):
        out_world = np.full((T, 33, 3), np.nan)

    L_HIP = LandmarkIndex.LEFT_HIP
    R_HIP = LandmarkIndex.RIGHT_HIP

    log: List[Dict[str, Any]] = []
    for t in range(T):
        # Select the applicable seeded candidate for this frame
        if t <= anchor_frame:
            seed_pass = pass_seed_bwd[t]
            seed_reason = reasons_seed_bwd[t]
            seed_lm = lm_seed_bwd[t]
            seed_vis = vis_seed_bwd[t]
            seed_world = world_seed_bwd
            seed_tag = "seed_bwd"
        else:
            seed_pass = pass_seed_fwd[t]
            seed_reason = reasons_seed_fwd[t]
            seed_lm = lm_seed_fwd[t]
            seed_vis = vis_seed_fwd[t]
            seed_world = world_seed_fwd
            seed_tag = "seed_fwd"

        with np.errstate(invalid="ignore"):
            hip_vis_fwd = float(np.nanmean([vis_fwd[t, L_HIP], vis_fwd[t, R_HIP]]))
            hip_vis_rev = float(np.nanmean([vis_rev[t, L_HIP], vis_rev[t, R_HIP]]))
            hip_vis_seed = float(np.nanmean([seed_vis[L_HIP], seed_vis[R_HIP]]))

        # Score: hip visibility if anatomy passes, else -inf
        score_fwd = hip_vis_fwd if pass_fwd[t] else float("-inf")
        score_rev = hip_vis_rev if pass_rev[t] else float("-inf")
        score_seed = hip_vis_seed if seed_pass else float("-inf")

        scores = [score_fwd, score_rev, score_seed]
        tags = ["fwd", "rev", seed_tag]
        lms = [lm_fwd[t], lm_rev[t], seed_lm]
        viss = [vis_fwd[t], vis_rev[t], seed_vis]
        worlds = [world_fwd, world_rev, seed_world]
        reasons_list = [reasons_fwd[t], reasons_rev[t], seed_reason]

        best = int(np.argmax(scores))
        if scores[best] == float("-inf"):
            chosen = "none"
            reason = (
                f"all_fail(fwd:{reasons_fwd[t]},"
                f"rev:{reasons_rev[t]},"
                f"{seed_tag}:{seed_reason})"
            )
        else:
            chosen = tags[best]
            out_lm[t] = lms[best]
            out_vis[t] = viss[best]
            world_chosen = worlds[best]
            if out_world is not None and world_chosen is not None:
                out_world[t] = world_chosen[t]
            reason = (
                f"{chosen}(score={scores[best]:.3f},"
                f"fwd={score_fwd:.3f},"
                f"rev={score_rev:.3f},"
                f"{seed_tag}={score_seed:.3f})"
            )

        log.append(
            {
                "frame": int(t),
                "chosen_pass": chosen,
                "anatomy_pass_fwd": bool(pass_fwd[t]),
                "anatomy_pass_rev": bool(pass_rev[t]),
                "anatomy_pass_seed": bool(seed_pass),
                "mean_hip_vis_fwd": (
                    hip_vis_fwd if np.isfinite(hip_vis_fwd) else float("nan")
                ),
                "mean_hip_vis_rev": (
                    hip_vis_rev if np.isfinite(hip_vis_rev) else float("nan")
                ),
                "mean_hip_vis_seed": (
                    hip_vis_seed if np.isfinite(hip_vis_seed) else float("nan")
                ),
                "inter_pass_distance_px": float("nan"),
                "reason": reason,
                # Extra fields silently dropped by write_bidirectional_log
                # (extrasaction="ignore") but available for direct inspection.
                "seed_tag": seed_tag,
                "anatomy_reason_fwd": reasons_list[0],
                "anatomy_reason_rev": reasons_list[1],
                "anatomy_reason_seed": reasons_list[2],
            }
        )

    return BidirectionalCombineResult(
        landmarks=out_lm,
        visibilities=out_vis,
        world_landmarks=out_world,
        log=log,
    )
