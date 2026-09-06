import numpy as np
from dataclasses import dataclass
from typing import Dict, List, Optional, Sequence, Tuple

from src.preprocessing.pose_estimator import LandmarkIndex
from src.utils.metrics import BiomechanicalMetrics

# Anchors for the reliability composite.
# An n=45 diagnosis found the old `overall_score` composite scored AUC 0.588
# against known badly-measured clips while its own best input
# (detection_rate) scored 0.850 alone, diluted by calibration confidence
# (AUC 0.399, near chance) and a metric-warning count that turned out to be
# significantly ANTI-informative (AUC 0.234). stride_cv (not previously
# computed) scored 0.878 alone -- the strongest single signal found.
# Highest stride_cv among clips with no other evidence of a measurement problem
_STRIDE_CV_CONSISTENCY_SCALE = 0.60
# Empty gap in the sorted S1 data (0.343 -> 0.380); catches 14/14 known-bad clips
_STRIDE_CV_UNRELIABLE_THRESHOLD = 0.36

# Reference-free per-clip presence-window gate. Replaces an earlier
# entry/mid/exit zone-thirds mechanism entirely.
#
# The zone split was retired rather than retuned: its single shipped
# catch (`S1_01_threshold_1`) was boundary-placement luck, not signal --
# the identical rule flags zero clips at 4/5/6/8/10 zones, and shifting the
# two internal cut points by +/-5% of clip length releases the clip at every
# tested offset. That clip's real defect (spatial-scale inflation from
# running off the calibrated pole line) is invisible to any
# visibility gate by construction, so nothing protective was lost, and this
# gate is deliberately NOT scored against reproducing it.
#
# What survived that testing is interior dropout: a contiguous run of
# frames inside the runner's own presence window with no hip/ankle detected
# at all. A *mean* over that window cannot see this -- `nanmean` skips
# absent frames, so it measures visibility-quality-when-detected and flags
# 0 of 51 clips at any floor -- whereas a mid-transit detection loss is a
# genuine defect rather than the post-exit recording padding found on
# nearly every clip.
#
# The threshold is anchored to the pipeline, not fitted to the cohort. The
# cohort cannot discriminate: only `PV_Steady_1_Glycerin` has any interior
# dropout at all (27 frames), the other 50 clips have exactly 0, so every
# threshold from 1 to 27 gives an identical partition. `pose_quality` is
# built AFTER gap-fill, so an interior gap that survives and is longer than
# `gap_max_frames` -- the longest gap the pipeline itself will repair -- is
# by the pipeline's own declared standard an unrecoverable tracking loss.
# Keep this in step with `run_clip_pipeline`'s `gap_max_frames` default.
# In frames, matching gap-fill's own units; at 60 fps this is ~0.083 s,
# still shorter than a single stance phase.
_INTERIOR_DROPOUT_MAX_FRAMES = 4

# Expected ranges for middle-distance runners
# NOTE: stride_length is FULL gait cycle (2 steps), step_length is HALF gait cycle
METRIC_RANGES = {
    "cadence_spm": (160, 220),
    "gct_ms": (120, 280),
    "flight_time_ms": (80, 200),
    # Stride-based (contact / stride time). NOT a rescale of the
    # old (0.25, 0.55): that band was mis-centred as well as on the wrong
    # scale, flagging 9 of the 45 S1 clips at v1.31, and halving it to
    # (0.125, 0.275) flags the same 9. This range spans the audited literature
    # (0.216 for an elite 800m individual up to 0.41 for recreational women at
    # ~10 km/h) with margin, flags 0 of 45, and still catches walking (0.6-0.7).
    "duty_factor": (0.18, 0.45),
    "stride_length_m": (2.4, 5.0),  # Full gait cycle (2 steps) — middle-distance range
    "step_length_m": (1.2, 2.5),  # Half gait cycle (1 step)
    "oscillation_cm": (4, 14),
    "stride_leg_ratio": (2.4, 5.5),  # Stride / leg_length (stride ≈ 2.6-4.5× leg)
    "oscillation_leg_ratio": (0.03, 0.10),
    # oscillation / step, so twice the old per-stride band of (1, 6) — the
    # metric was redefined 2026-08-31, not re-measured
    "vertical_oscillation_ratio": (2, 12),
    "velocity_ms": (2.5, 8.5),
}


def validate_metrics(metrics: BiomechanicalMetrics) -> Tuple[bool, List[str]]:
    """
    Validate extracted metrics against expected ranges.

    Args:
        metrics: BiomechanicalMetrics object

    Returns:
        valid: True if all metrics within expected ranges
        warnings: List of warning messages
    """
    warnings = []

    checks = [
        ("cadence_spm", metrics.cadence_spm),
        ("gct_ms", metrics.gct_ms),
        ("flight_time_ms", metrics.flight_time_ms),
        ("duty_factor", metrics.duty_factor),
        ("stride_length_m", metrics.stride_length_m),
        ("step_length_m", metrics.step_length_m),
        ("oscillation_cm", metrics.oscillation_cm),
        ("stride_leg_ratio", metrics.stride_leg_ratio),
        ("oscillation_leg_ratio", metrics.oscillation_leg_ratio),
        ("vertical_oscillation_ratio", metrics.vertical_oscillation_ratio),
        ("velocity_ms", metrics.velocity_ms),
    ]

    for name, value in checks:
        if value is None or np.isnan(value):
            warnings.append(f"{name}: could not be calculated")
        elif name in METRIC_RANGES:
            low, high = METRIC_RANGES[name]
            if value < low:
                warnings.append(
                    f"{name}: {value:.2f} below expected range [{low}, {high}]"
                )
            elif value > high:
                warnings.append(
                    f"{name}: {value:.2f} above expected range [{low}, {high}]"
                )

    return len(warnings) == 0, warnings


def _presence_mask(landmarks_px: np.ndarray) -> np.ndarray:
    """Frames where a hip AND an ankle are *present* (non-NaN pixel coords).

    "At least one of the L/R pair" on each side, mirroring the ``nanmean``
    tolerance the visibility average applies to a one-side-missing frame.
    Deliberately a coarse *detection* check on pixel coordinates and never a
    threshold on a visibility score: defining the window with the same
    signal the gate then judges would make the gate circular.

    In practice the strict all-four variant is identical on the S1 and
    pre_validation cohorts -- BlazePose's NaN-fill is all-or-nothing per
    frame, so no frame anywhere has some but not all of landmarks
    23/24/27/28 missing (measured).
    """
    if len(landmarks_px) == 0:
        return np.zeros(0, dtype=bool)
    ok = ~np.isnan(landmarks_px[:, :, 0])
    hip = ok[:, [LandmarkIndex.LEFT_HIP, LandmarkIndex.RIGHT_HIP]]
    ankle = ok[:, [LandmarkIndex.LEFT_ANKLE, LandmarkIndex.RIGHT_ANKLE]]
    return np.asarray(hip.any(axis=1) & ankle.any(axis=1), dtype=bool)


def compute_presence_quality(
    selected_landmarks: np.ndarray,
) -> Dict[str, Optional[float]]:
    """Presence-window shape and interior dropout.

    Finds ``[first, last]`` over frames where a hip and an ankle are
    detected, then measures detection loss *inside* that window. Bounding
    by the runner's own transit is what makes an absent frame meaningful:
    The camera keeps rolling ~1.5 s after the runner exits on
    nearly every S1 clip, so absent frames outside the window are recording
    padding, the cohort norm rather than a defect.

    Returns JSON-serialisable values; ``window_start``/``window_end`` are
    ``None`` when no frame has a hip and an ankle at all. That case is
    *worse* than uncomputable, not merely unknown, and the gate in
    ``assess_clip_reliability`` excludes on it -- ``window_n_frames`` is 0
    there, never ``None``, so a caller cannot mistake it for missing data.
    """
    present = _presence_mask(selected_landmarks)
    idx = np.flatnonzero(present)
    if idx.size == 0:
        return {
            "window_start": None,
            "window_end": None,
            "window_n_frames": 0.0,
            "longest_interior_dropout": 0.0,
            "n_interior_absent_frames": 0.0,
        }
    lo, hi = int(idx[0]), int(idx[-1])
    interior = present[lo : hi + 1]
    longest = 0
    run = 0
    for ok in interior:
        run = 0 if ok else run + 1
        longest = max(longest, run)
    return {
        "window_start": float(lo),
        "window_end": float(hi),
        "window_n_frames": float(interior.size),
        "longest_interior_dropout": float(longest),
        "n_interior_absent_frames": float(np.count_nonzero(~interior)),
    }


def compute_detected_mean_visibility(
    selected_landmarks: np.ndarray, selected_visibilities: np.ndarray
) -> Dict[str, float]:
    """Mean hip/ankle visibility over *detected* frames only.

    ``selected_visibilities`` is never NaN on an undetected frame -
    ``pose_estimator.py`` pre-fills it with ``0.0`` (unlike landmarks, which
    correctly use NaN), so a plain ``np.nanmean`` silently counts every
    undetected frame as "visibility exactly 0" instead of excluding it.
    Measured impact: across all 51 certified clips, this collapsed the
    reported mean toward the detection rate itself (they were within 1e-4 of
    each other on every clip) rather than reflecting genuine per-landmark
    confidence - the true detected-frame mean was ~0.44-0.45 higher on
    average, per clip. This affects only the derived summary statistic, not
    ``selected_visibilities`` itself, so it changes no threshold comparison
    anywhere else in the pipeline that reads the raw array.

    A frame counts as detected for a pair (hip or ankle) when at least one
    of its L/R landmarks has a non-NaN pixel coordinate, matching
    ``_presence_mask``'s one-side-missing tolerance above.
    """
    hip_idx = [LandmarkIndex.LEFT_HIP, LandmarkIndex.RIGHT_HIP]
    ankle_idx = [LandmarkIndex.LEFT_ANKLE, LandmarkIndex.RIGHT_ANKLE]

    def _masked_mean(idx: Sequence[int]) -> float:
        vis = selected_visibilities[:, idx]
        detected = ~np.isnan(selected_landmarks[:, idx, 0])
        keep = detected.any(axis=1)
        if not np.any(keep):
            return float("nan")
        row_vis = np.where(detected[keep], vis[keep], np.nan)
        with np.errstate(invalid="ignore"):
            per_frame = np.nanmean(row_vis, axis=1)
        return float(np.nanmean(per_frame))

    return {
        "mean_hip_visibility": _masked_mean(hip_idx),
        "mean_ankle_visibility": _masked_mean(ankle_idx),
    }


@dataclass
class ReliabilityAssessment:
    """Per-clip reliability signal, replacing the old ``overall_score``.

    ``calibration.confidence``, the metric-warning count and ``n_contacts``
    all carry zero weight here, on purpose: they measured at
    AUC 0.399/0.234/untested-but-saturated respectively against known
    badly-measured clips, versus 0.878/0.850 for ``stride_cv``/
    ``detection_rate``.

    ``presence_quality_ok``/``excluded`` are a second, independent gate
    (it replaced an earlier ``zone_visibility_ok``,
    whose entry/mid/exit mechanism was retired outright): ``None`` when
    ``pose_quality`` carries no ``presence_quality`` (older callers), else
    ``False`` when the runner's presence window is undefined or contains an
    interior detection dropout longer than gap-fill will repair -- the
    case ``is_reliable`` alone was found to miss entirely. ``excluded`` is
    the authoritative "do not use this clip's output" signal; it is
    ``True`` only when ``presence_quality_ok is False``, kept distinct from
    ``is_reliable`` so existing ``is_reliable`` consumers are not silently
    reinterpreted.
    """

    score: float
    is_reliable: Optional[bool]
    reasons: List[str]
    components: Dict[str, Optional[float]]
    presence_quality_ok: Optional[bool] = None
    excluded: bool = False


def assess_clip_reliability(
    pose_quality: Dict,
    metrics: BiomechanicalMetrics,
) -> ReliabilityAssessment:
    """
    Assess how much a clip's measurements can be trusted.

    Args:
        pose_quality: Quality metrics from pose extraction
        metrics: Extracted biomechanical metrics

    Returns:
        A ``ReliabilityAssessment`` — a 0-1 ``score`` (0.50 stride
        self-consistency + 0.50 detection rate), a boolean ``is_reliable``
        gate (``None`` if stride_cv could not be computed — never asserted
        True/False on missing data), human-readable ``reasons``, and the
        raw component values for display/diagnostics. If
        ``pose_quality["presence_quality"]`` is present,
        ``is_reliable`` is additionally forced ``False`` — never loosened
        back to ``True`` — when the presence window is undefined or carries
        an over-long interior dropout, and ``excluded`` is set as the
        dedicated signal for that case.
    """
    detection_component = min(1.0, pose_quality["detection_rate"] / 0.95)

    stride_cv: Optional[float] = None
    if metrics.stride_length_m and not np.isnan(metrics.stride_length_m):
        candidate = abs(metrics.stride_length_std / metrics.stride_length_m)
        if not np.isnan(candidate):
            stride_cv = candidate

    reasons: List[str] = []
    if stride_cv is None:
        consistency_component = None
        is_reliable = None
        score = detection_component
        reasons.append("stride_cv unavailable; score reflects detection rate only")
    else:
        consistency_component = min(
            1.0, max(0.0, 1.0 - stride_cv / _STRIDE_CV_CONSISTENCY_SCALE)
        )
        is_reliable = stride_cv <= _STRIDE_CV_UNRELIABLE_THRESHOLD
        score = 0.5 * consistency_component + 0.5 * detection_component
        if not is_reliable:
            reasons.append(
                f"stride_cv {stride_cv:.3f} exceeds the "
                f"{_STRIDE_CV_UNRELIABLE_THRESHOLD} unreliable threshold"
            )

    components: Dict[str, Optional[float]] = {
        "detection": detection_component,
        "consistency": consistency_component,
        "stride_cv": stride_cv,
    }

    presence_quality = pose_quality.get("presence_quality")
    presence_quality_ok: Optional[bool] = None
    excluded = False
    if presence_quality:
        for label, value in presence_quality.items():
            components[f"presence_{label}"] = value
        window_n_frames = presence_quality.get("window_n_frames") or 0.0
        dropout = presence_quality.get("longest_interior_dropout") or 0.0
        if window_n_frames <= 0:
            presence_quality_ok = False
            reasons.append(
                "no frame in the clip has both a hip and an ankle detected, "
                "so the runner's presence window is undefined "
                "-- clip excluded"
            )
        elif dropout > _INTERIOR_DROPOUT_MAX_FRAMES:
            presence_quality_ok = False
            reasons.append(
                f"tracking is lost for {int(dropout)} consecutive frames "
                "mid-transit, longer than the "
                f"{_INTERIOR_DROPOUT_MAX_FRAMES}-frame gap the pipeline can "
                "repair -- clip excluded"
            )
        else:
            presence_quality_ok = True
        if not presence_quality_ok:
            excluded = True
            is_reliable = False

    return ReliabilityAssessment(
        score=score,
        is_reliable=is_reliable,
        reasons=reasons,
        components=components,
        presence_quality_ok=presence_quality_ok,
        excluded=excluded,
    )
