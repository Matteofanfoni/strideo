import math

import numpy as np
from dataclasses import dataclass
from typing import Dict, List, Optional, Tuple
from scipy.ndimage import gaussian_filter1d

from src.preprocessing.ground_contact import GroundContact, detect_ground_contacts
from src.preprocessing.calibration import SpatialCalibration, create_spatial_calibration
from src.preprocessing.pose_estimator import LandmarkIndex


def calculate_cadence(contacts: List[GroundContact], fps: float) -> Tuple[float, float]:
    """
    Calculate cadence (steps per minute) from ground contacts.

    Uses sub-frame contact timing when available (from toe refinement).

    Args:
        contacts: List of ground contact events
        fps: Video frame rate

    Returns:
        cadence_spm: Mean cadence in steps per minute
        cadence_std: Standard deviation of cadence
    """
    if len(contacts) < 2:
        return np.nan, np.nan

    # Use refined contact_frame if available, otherwise fall back to coarse frame
    def get_contact_time(c: GroundContact) -> float:
        if c.contact_frame is not None:
            return c.contact_frame
        return float(c.frame)

    # Calculate inter-contact intervals using sub-frame timing
    intervals_frames = np.array(
        [
            get_contact_time(contacts[i + 1]) - get_contact_time(contacts[i])
            for i in range(len(contacts) - 1)
        ]
    )

    # Filter out any invalid intervals (too short or too long)
    min_interval = 0.15 * fps  # ~150ms minimum (200 spm)
    max_interval = 0.50 * fps  # ~500ms maximum (120 spm)
    valid_mask = (intervals_frames > min_interval) & (intervals_frames < max_interval)

    if not np.any(valid_mask):
        return np.nan, np.nan

    intervals_frames = intervals_frames[valid_mask]

    # Exclude outlier intervals (entry/exit zone coarse contacts that pass
    # the 150–500ms gate but are anomalously short or long vs the clip median).
    # Uses the same 1.5× median window applied in calculate_stride_length.
    if len(intervals_frames) >= 3:
        median_interval = float(np.median(intervals_frames))
        outlier_mask = (intervals_frames >= median_interval / 1.5) & (
            intervals_frames <= median_interval * 1.5
        )
        if np.any(outlier_mask):
            intervals_frames = intervals_frames[outlier_mask]
    intervals_seconds = intervals_frames / fps

    # Cadence = steps per minute
    steps_per_second = 1.0 / intervals_seconds
    cadence_spm = float(np.mean(steps_per_second) * 60)
    cadence_std = float(np.std(steps_per_second) * 60)

    return cadence_spm, cadence_std


def calculate_ground_contact_time(
    contacts: List[GroundContact], refined_only: bool = True
) -> Tuple[float, float, List[float]]:
    """
    Extract ground contact time from pre-computed contact data.

    GCT is computed during ground contact detection (Part 3) using the
    two-pass coarse/fine method. This function aggregates those values.

    Args:
        contacts: Ground contact events (with gct_ms pre-computed)
        refined_only: If True, only use contacts with 'refined' detection method

    Returns:
        mean_gct_ms: Mean GCT in milliseconds
        std_gct_ms: Standard deviation of GCT
        gct_values: List of individual GCT measurements
    """
    # Filter contacts based on detection method and valid GCT
    if refined_only:
        valid_contacts = [
            c
            for c in contacts
            if c.detection_method == "refined" and c.gct_ms is not None
        ]
    else:
        valid_contacts = [c for c in contacts if c.gct_ms is not None]

    if len(valid_contacts) == 0:
        # Fall back to coarse if no refined contacts available
        if refined_only:
            return calculate_ground_contact_time(contacts, refined_only=False)
        return np.nan, np.nan, []

    gct_values: List[float] = [c.gct_ms for c in valid_contacts if c.gct_ms is not None]

    return float(np.mean(gct_values)), float(np.std(gct_values)), gct_values


def calculate_ground_contact_time_by_leg(
    contacts: List[GroundContact], refined_only: bool = True
) -> Dict[str, Tuple[float, float, List[float]]]:
    """
    Calculate GCT separately for each leg (for asymmetry analysis).

    Args:
        contacts: Ground contact events
        refined_only: If True, only use refined contacts

    Returns:
        Dictionary with 'L' and 'R' keys, each containing (mean, std, values)
    """
    result = {}

    for leg in ["L", "R"]:
        leg_contacts = [c for c in contacts if c.leg == leg]
        mean_gct, std_gct, values = calculate_ground_contact_time(
            leg_contacts, refined_only=refined_only
        )
        result[leg] = (mean_gct, std_gct, values)

    return result


def calculate_flight_time(cadence_spm: float, gct_ms: float) -> float:
    """
    Calculate flight time from cadence and GCT.

    Flight time = Step time - GCT
    Step time = 60000 / cadence (ms)

    Args:
        cadence_spm: Cadence in steps per minute
        gct_ms: Ground contact time in milliseconds

    Returns:
        flight_time_ms: Flight time in milliseconds
    """
    if np.isnan(cadence_spm) or np.isnan(gct_ms):
        return np.nan

    step_time_ms = 60000 / cadence_spm
    flight_time_ms = step_time_ms - gct_ms

    # Sanity check: flight time should be positive
    if flight_time_ms < 0:
        return np.nan

    return flight_time_ms


def calculate_duty_factor(gct_ms: float, flight_time_ms: float) -> float:
    """
    Calculate duty factor (proportion of STRIDE time spent in stance).

    Duty factor = contact_time / stride_time
                = GCT / (2 * (GCT + flight_time))

    The denominator is the stride, one full gait cycle for ONE limb, not the
    step. A stride is two steps, so stride_time = 2 * step_time. Nothing
    is double-counted by this: the limb whose contact is in the numerator
    touches down exactly once per stride, so numerator and denominator both
    describe the same limb.

    Do NOT substitute swing time for flight time when reading the literature's
    equivalent form, `contact / (contact + swing)`. They are different
    intervals:

        flight = toe-off of one foot -> initial contact of the OPPOSITE foot
        swing  = toe-off of one foot -> initial contact of the SAME foot

    Running has no double support, so `swing = contact + 2 * flight`, and the
    literature's form expands to exactly the expression above.

    CORRECTED 2026-09-05. This function previously returned
    `GCT / (GCT + flight_time)`, contact over STEP time, which is exactly twice
    the standard value, and this docstring wrongly asserted that step time was
    "the standard biomechanics definition". All 30 published observations
    behind the elite reference ranges use the stride
    denominator, verified by audit rather than assumed,
    so the app's elite-range comparison had been scoring every runner against a
    band on half its scale since 2026-08-28. The old value is still available
    as `flight_ratio`'s complement, which is a per-step quantity by design.

    Lower values indicate more time airborne (faster running).
    Typical values (stride-based):
      - Walking: 0.6-0.7 (double support; the identity above does not hold)
      - Recreational running, ~10 km/h: 0.39-0.41
      - Elite threshold pace: 0.26-0.29
      - Elite 1500m race pace: 0.245-0.250
      - Elite 800m race pace: 0.22-0.26

    Args:
        gct_ms: Ground contact time in ms
        flight_time_ms: Flight time in ms (the aerial phase, i.e. toe-off of
            one foot to initial contact of the opposite foot)

    Returns:
        duty_factor: Ratio (typically 0.22-0.41 for running)
    """
    if np.isnan(gct_ms) or np.isnan(flight_time_ms):
        return np.nan

    step_time = gct_ms + flight_time_ms
    if step_time <= 0:
        return np.nan

    stride_time = 2.0 * step_time
    return gct_ms / stride_time


def calculate_stride_length(
    landmarks: np.ndarray,
    contacts: List[GroundContact],
    calibration: SpatialCalibration,
) -> Tuple[float, float, List[float]]:
    """
    Calculate stride length from hip displacement between same-leg contacts.

    Stride = horizontal displacement of pelvis over one complete gait cycle
    (two steps, returning to same leg).

    Args:
        landmarks: (T, 33, 2) pose landmarks
        contacts: Ground contact events
        calibration: Spatial calibration for unit conversion

    Returns:
        mean_stride_m: Mean stride length in metres
        std_stride_m: Standard deviation in metres
        stride_values_m: List of individual stride lengths
    """
    # Calculate hip midpoint trajectory
    left_hip_x = landmarks[:, LandmarkIndex.LEFT_HIP, 0]
    right_hip_x = landmarks[:, LandmarkIndex.RIGHT_HIP, 0]
    hip_x = np.nanmean([left_hip_x, right_hip_x], axis=0)
    # A frame with no detected hip landmark gets a *fabricated* position at
    # the clip mean below (needed so gaussian_filter1d has no NaNs to
    # propagate) -- any pair anchored there measures a distance to that
    # constant, not to the runner. Track which frames are real so
    # such pairs can be excluded below rather than measured.
    hip_x_missing = np.isnan(hip_x)

    # Smooth to reduce noise
    hip_x_smooth = gaussian_filter1d(
        np.nan_to_num(hip_x, nan=float(np.nanmean(hip_x))), sigma=2
    )

    # Collect same-leg (i, i+2) pairs with their frame intervals.
    # Cross-leg pairs are skipped: a stride is one complete same-leg cycle.
    # Pairs with a fabricated endpoint (real detection missing at s or e) are
    # skipped too -- keeping them would measure displacement to a
    # constant clip-mean position rather than the runner's actual hip.
    candidate_intervals: List[int] = []
    candidate_pairs: List[tuple] = []
    for i in range(len(contacts) - 2):
        if contacts[i].leg != contacts[i + 2].leg:
            continue
        s = contacts[i].frame
        e = contacts[i + 2].frame
        if s < len(hip_x_smooth) and e < len(hip_x_smooth):
            if hip_x_missing[s] or hip_x_missing[e]:
                continue
            candidate_intervals.append(e - s)
            candidate_pairs.append((s, e))

    if len(candidate_pairs) == 0:
        return np.nan, np.nan, []

    # Exclude outlier-interval pairs (entry/exit zone artifacts where one
    # endpoint has unreliable hip data produce stride intervals > 1.5× median).
    median_interval = float(np.median(candidate_intervals))
    interval_ceil = 1.5 * median_interval

    stride_lengths_px = []
    for (s, e), interval in zip(candidate_pairs, candidate_intervals):
        if interval > interval_ceil:
            continue
        stride_lengths_px.append(abs(hip_x_smooth[e] - hip_x_smooth[s]))

    if len(stride_lengths_px) == 0:
        return np.nan, np.nan, []

    # Convert to metres
    stride_lengths_m = [calibration.px_to_m(s) for s in stride_lengths_px]

    return (
        float(np.mean(stride_lengths_m)),
        float(np.std(stride_lengths_m)),
        stride_lengths_m,
    )


def calculate_step_length(
    landmarks: np.ndarray,
    contacts: List[GroundContact],
    calibration: SpatialCalibration,
) -> Tuple[float, float, List[float]]:
    """
    Calculate step length from hip displacement between consecutive contacts.

    Step = horizontal displacement over one step (half stride).

    Returns:
        mean_step_m, std_step_m, step_values_m
    """
    left_hip_x = landmarks[:, LandmarkIndex.LEFT_HIP, 0]
    right_hip_x = landmarks[:, LandmarkIndex.RIGHT_HIP, 0]
    hip_x = np.nanmean([left_hip_x, right_hip_x], axis=0)
    # See calculate_stride_length for why fabricated (nan_to_num-filled)
    # endpoints must be excluded rather than measured.
    hip_x_missing = np.isnan(hip_x)
    hip_x_smooth = gaussian_filter1d(
        np.nan_to_num(hip_x, nan=float(np.nanmean(hip_x))), sigma=2
    )

    step_lengths_px = []

    for i in range(len(contacts) - 1):
        start_frame = contacts[i].frame
        end_frame = contacts[i + 1].frame

        if start_frame < len(hip_x_smooth) and end_frame < len(hip_x_smooth):
            if hip_x_missing[start_frame] or hip_x_missing[end_frame]:
                continue
            displacement = abs(hip_x_smooth[end_frame] - hip_x_smooth[start_frame])
            step_lengths_px.append(displacement)

    if len(step_lengths_px) == 0:
        return np.nan, np.nan, []

    step_lengths_m = [calibration.px_to_m(s) for s in step_lengths_px]

    return float(np.mean(step_lengths_m)), float(np.std(step_lengths_m)), step_lengths_m


#: Legacy defaults, used when `fps`/`cadence_spm` are unavailable so
#: older call sites see byte-identical behaviour to before the time-aware fix.
_VO_LEGACY_SIGMA = 2.0
_VO_LEGACY_DISTANCE = 10
#: Half-cycle frame count the legacy constants were implicitly tuned for
#: (fps=60, cadence=180 spm -> 30/180*60 = 10 frames). Used as the anchor for
#: the fps/cadence-derived cutoff below.
_VO_REFERENCE_HALF_CYCLE_FRAMES = 10.0
#: Fraction of the current half-stride period the smoothing sigma is allowed
#: to span. The fixed sigma=2 costs ~4% of amplitude at
#: recovery/steady cadence (half-cycle ~11 frames) but 19-22% at 800m-pace
#: cadence (half-cycle ~8 frames), because the same absolute smoothing width
#: eats a larger share of a shorter stride period. Tying sigma to the period
#: instead of a frame count keeps the (small) smoothing loss roughly pace-
#: independent. Value chosen empirically against the 5 pre_validation clips
#: to land close to the diagnosis's U1
#: candidate numbers without the interpolation it also flags as harmful.
_VO_SIGMA_TO_HALF_CYCLE = 0.05
#: Fraction of the half-stride period used as the minimum peak/trough
#: separation, replacing the fixed `distance=10` (itself only correct at the
#: reference cadence above).
_VO_DISTANCE_TO_HALF_CYCLE = 0.8
_VO_MIN_DISTANCE_FRAMES = 2
_VO_MIN_RUN_FRAMES = 6


def _vo_contiguous_valid_runs(signal: np.ndarray, min_len: int) -> List[np.ndarray]:
    """Split ``signal`` into its contiguous non-NaN runs of length >= min_len.

    Unlike ``signal[~np.isnan(signal)]``, this never concatenates samples
    across a dropout gap, so a run's internal frame spacing stays the true
    one (the old drop-then-concatenate step silently compressed time,
    which the smoothing sigma and peak-distance below assume is real time).
    """
    runs: List[np.ndarray] = []
    valid = ~np.isnan(signal)
    start: Optional[int] = None
    n = len(signal)
    for i, ok in enumerate(valid):
        if ok and start is None:
            start = i
        if start is not None and (not ok or i == n - 1):
            end = i if not ok else i + 1
            if end - start >= min_len:
                runs.append(signal[start:end])
            start = None
    return runs


def _vo_smoothing_params(
    fps: Optional[float], cadence_spm: Optional[float]
) -> Tuple[float, int]:
    """fps/cadence-derived (sigma, distance), falling back to the legacy
    constants when either input is unavailable, non-positive, or NaN.

    **NaN is checked explicitly because it passes both other predicates**
    (``not nan`` is False and ``nan <= 0`` is False), so a NaN once
    cadence reached ``round()`` below and raised
    ``ValueError: cannot convert float NaN to integer``, failing the whole
    analysis instead of degrading it. ``calculate_cadence`` returns
    ``np.nan`` by design whenever no step interval clears its 150-500 ms gate,
    which needs only a clip with too few detected contacts -- so this was
    reachable at any frame rate, not just the 30 fps ingest case that exposed
    it.
    """
    if fps is None or cadence_spm is None:
        return _VO_LEGACY_SIGMA, _VO_LEGACY_DISTANCE
    if math.isnan(fps) or math.isnan(cadence_spm):
        return _VO_LEGACY_SIGMA, _VO_LEGACY_DISTANCE
    if fps <= 0 or cadence_spm <= 0:
        return _VO_LEGACY_SIGMA, _VO_LEGACY_DISTANCE
    half_cycle_frames = (30.0 / cadence_spm) * fps
    sigma = _VO_SIGMA_TO_HALF_CYCLE * half_cycle_frames
    distance = max(
        _VO_MIN_DISTANCE_FRAMES, round(_VO_DISTANCE_TO_HALF_CYCLE * half_cycle_frames)
    )
    return sigma, distance


def calculate_vertical_oscillation(
    landmarks: np.ndarray,
    calibration: SpatialCalibration,
    fps: Optional[float] = None,
    cadence_spm: Optional[float] = None,
) -> Tuple[float, float]:
    """
    Calculate vertical oscillation (peak-to-peak hip displacement).

    Time-aware: the smoothing width and the minimum
    peak/trough separation are derived from `fps`/`cadence_spm` instead of
    fixed frame counts, and gaps in landmark detection are handled by
    measuring within each contiguous valid run on the true frame index
    rather than deleting NaN frames and concatenating what's left (which
    silently compresses time — see `_vo_contiguous_valid_runs`). A fixed
    `sigma=2` cost ~4% of amplitude at recovery/steady pace but 19-22% at
    800m pace, where the oscillation period is shortest — a systematic bias
    against this project's target population.

    Args:
        landmarks: (T, 33, 2) pose landmarks
        calibration: Spatial calibration
        fps: Video frame rate. Optional for backward compatibility; when
            omitted (or `cadence_spm` is), falls back to the legacy
            fixed sigma=2 / distance=10.
        cadence_spm: Runner's cadence in steps per minute, used with `fps`
            to derive the smoothing/peak-separation cutoff.

    Returns:
        oscillation_cm: Vertical oscillation in centimetres
        oscillation_leg_ratio: Oscillation as fraction of leg length
    """
    from scipy.signal import find_peaks

    # Hip Y position (average of left and right)
    left_hip_y = landmarks[:, LandmarkIndex.LEFT_HIP, 1]
    right_hip_y = landmarks[:, LandmarkIndex.RIGHT_HIP, 1]
    hip_y = np.nanmean([left_hip_y, right_hip_y], axis=0)

    n_valid = int(np.sum(~np.isnan(hip_y)))
    if n_valid < 20:
        return np.nan, np.nan

    sigma, distance = _vo_smoothing_params(fps, cadence_spm)
    min_run_len = max(_VO_MIN_RUN_FRAMES, 2 * distance)

    # Note: in image coordinates, lower Y = higher position
    # So peaks in the signal are the LOW points of oscillation
    cycle_amps_px: List[float] = []
    for run in _vo_contiguous_valid_runs(hip_y, min_run_len):
        run_smooth = gaussian_filter1d(run, sigma=sigma) if sigma > 0 else run
        peaks, _ = find_peaks(run_smooth, distance=distance)
        troughs, _ = find_peaks(-run_smooth, distance=distance)
        marked = [(int(i), "p") for i in peaks] + [(int(i), "t") for i in troughs]
        marked.sort(key=lambda x: x[0])
        for (i0, k0), (i1, k1) in zip(marked, marked[1:]):
            if k0 != k1:
                cycle_amps_px.append(float(abs(run_smooth[i1] - run_smooth[i0])))

    if not cycle_amps_px:
        # Fall back to simple range over all valid samples (no per-cycle
        # extrema found — e.g. too few valid frames or a monotonic signal).
        hip_y_valid = hip_y[~np.isnan(hip_y)]
        oscillation_px = float(np.max(hip_y_valid) - np.min(hip_y_valid))
    else:
        oscillation_px = float(np.median(cycle_amps_px))

    oscillation_px = abs(oscillation_px)
    oscillation_cm = calibration.px_to_cm(oscillation_px)
    oscillation_leg_ratio = calibration.to_leg_ratio(oscillation_px)

    return oscillation_cm, oscillation_leg_ratio


def calculate_velocity(
    stride_length_m: float, cadence_spm: float
) -> Tuple[float, float, str]:
    """
    Calculate running velocity from stride length and cadence.

    Velocity = stride_length × (cadence / 60) / 2
    Note: cadence is steps/min, stride is two steps, so divide by 2

    Or equivalently:
    Velocity = step_length × cadence / 60

    Using stride:
    Velocity = stride_length × stride_frequency
             = stride_length × (cadence / 2) / 60
             = stride_length × cadence / 120

    Args:
        stride_length_m: Stride length in metres
        cadence_spm: Cadence in steps per minute

    Returns:
        velocity_ms: Velocity in metres per second
        velocity_kmh: Velocity in kilometres per hour
        pace_per_km: Pace as string (e.g., "3:45")
    """
    if np.isnan(stride_length_m) or np.isnan(cadence_spm):
        return np.nan, np.nan, "N/A"

    # Stride frequency = cadence / 2 (strides per minute)
    stride_frequency = cadence_spm / 2 / 60  # strides per second

    velocity_ms = stride_length_m * stride_frequency
    velocity_kmh = velocity_ms * 3.6

    # Calculate pace. Rounds the seconds rather than truncating them, and
    # carries a 59.5+ remainder into the next minute, so this agrees with
    # src.utils.pace_predictor.format_pace_per_km on identical input. It used
    # to truncate (`int(pace % 60)`), which put the Results page's two pace
    # readings up to 1 s/km apart for no reason beyond the mismatch:
    # found that, parked it at the user's instruction, and this closes it.
    if velocity_ms > 0:
        pace_seconds_per_km = 1000 / velocity_ms
        pace_minutes = int(pace_seconds_per_km // 60)
        pace_seconds = int(round(pace_seconds_per_km - pace_minutes * 60))
        if pace_seconds == 60:
            pace_minutes, pace_seconds = pace_minutes + 1, 0
        pace_per_km = f"{pace_minutes}:{pace_seconds:02d}"
    else:
        pace_per_km = "N/A"

    return velocity_ms, velocity_kmh, pace_per_km


def calculate_vertical_oscillation_ratio(
    oscillation_cm: float, stride_length_m: float
) -> float:
    """
    Vertical oscillation as a percentage of step length.

    ratio = (vertical oscillation / step length) × 100, where step length is
    half a stride — a stride is one complete same-leg cycle, so two steps.

    The two terms are perpendicular: how far the runner rises and falls
    against how far they travel forward over the same step. Lower means less
    of the motion is vertical.

    Takes ``stride_length_m`` rather than a step length because that is the
    quantity both pipelines actually hold. The classical path does measure a
    step of its own (``calculate_step_length``, hip displacement between
    consecutive contacts), but feeding it here would make the displayed ratio
    irreproducible from the oscillation and stride shown beside it, and the
    fast path has no measured step at all (no contact detection, so its
    ``step_length_m`` is NaN). One definition, stride/2, keeps both paths and
    the Results card's own arithmetic consistent.

    Was ``calculate_running_economy_index`` over a full-stride denominator
    until 2026-08-31. Running economy is submaximal oxygen cost measured on a
    metabolic cart, which no video can produce; the denominator moved
    to a step in the same change, per-step being the convention this ratio is
    normally quoted in, so values are twice the old ones.

    Args:
        oscillation_cm: Vertical oscillation in cm
        stride_length_m: Stride length (two steps) in metres

    Returns:
        ratio: Vertical oscillation ratio (percentage of step length)
    """
    if np.isnan(oscillation_cm) or np.isnan(stride_length_m) or stride_length_m <= 0:
        return np.nan

    step_length_cm = stride_length_m * 100 / 2
    ratio = (oscillation_cm / step_length_cm) * 100

    return ratio


@dataclass
class BiomechanicalMetrics:
    """Complete biomechanical metrics output."""

    # Temporal metrics
    cadence_spm: float
    cadence_std: float
    gct_ms: float
    gct_std: float
    flight_time_ms: float
    duty_factor: float

    # Spatial metrics (absolute)
    stride_length_m: float
    stride_length_std: float
    step_length_m: float
    oscillation_cm: float
    leg_length_cm: float

    # Spatial metrics (normalised)
    stride_leg_ratio: float
    oscillation_leg_ratio: float

    # Efficiency metrics
    vertical_oscillation_ratio: float
    flight_ratio: float

    # Derived metrics
    velocity_ms: float
    velocity_kmh: float
    pace_per_km: str

    # Quality indicators
    n_contacts: int
    n_refined_contacts: int  # Contacts with toe-refined timing
    calibration_confidence: float


def extract_all_metrics(
    landmarks: np.ndarray,
    visibilities: np.ndarray,
    fps: float,
    runner_height_cm: float,
    shoe_sole_cm: float = 2.5,
    shoe_type: Optional[str] = None,
    resolution_height: int = 2160,
    toe_off_method: str = "ankle_y_lift",
    body_height_px: Optional[float] = None,
    tau_strike_frac: float = 0.01,
    delta_lift_frac: float = 0.02,
    delta_lift_frac_forefoot: float = 0.01,
    ankle_horiz_vel_gate_frac: float = 0.0,
    clip_strike_pattern: Optional[str] = None,
    prune_spurious: bool = False,
    velocity_ms: Optional[float] = None,
    leg_length_m: Optional[float] = None,
    cadence_band_frac: float = 0.25,
    interpolate_missing: bool = False,
    gate_landmark_bounds: bool = False,
    resolution_width: Optional[float] = None,
    rtm_landmarks: Optional[np.ndarray] = None,
    rtm_scores: Optional[np.ndarray] = None,
    unknown_as_forefoot: bool = False,
    contacts: Optional[List[GroundContact]] = None,
) -> BiomechanicalMetrics:
    """
    Extract complete biomechanical metrics from pose sequence.

    Args:
        landmarks: (T, 33, 2) pose landmarks
        visibilities: (T, 33) visibility scores
        fps: Video frame rate
        runner_height_cm: Runner's known height in cm
        shoe_sole_cm: Shoe sole thickness in cm (default 2.5)
        shoe_type: Key from SHOE_TYPES dict (overrides shoe_sole_cm)
        resolution_height: Video height in pixels for adaptive thresholds
        toe_off_method: Phase 6c — toe-off detection method. Default
            ``"ankle_y_lift"`` reproduces v1.5 / v1.7 byte-identical.
            See ``ground_contact.refine_contacts_with_toe`` for the
            other options (foot_index_position_lift / per_strike_pattern).
        body_height_px: Body height in pixels for Fix B2c / B3 thresholds.
        tau_strike_frac: Strike-pattern detection threshold (Fix B3).
        delta_lift_frac: Foot-index toe-off lift threshold (Fix B2c).
        gate_landmark_bounds: Drop contacts whose hip/
            ankle landmarks are out of frame at their own frame (entry/
            exit fabrication). Requires ``resolution_width``.
        resolution_width: Video width in pixels, required when
            ``gate_landmark_bounds=True``.
        contacts: An already-detected contact list to compute
            metrics from. When given, this function does **not** call
            ``detect_ground_contacts`` and every detection-only argument
            above is ignored, so the caller's contacts and the caller's
            metrics are guaranteed to describe the same detection run.
            Callers that write a contact list alongside these metrics
            (``run_prevalidation_single.compute_metrics_on_window``,
            ``pipeline.run_clip_pipeline``) must pass it; leaving it None
            re-detects, which is what let a per-contact parameter reach a
            written ``contacts.csv`` and never reach ``result_full.json``.

    Returns:
        BiomechanicalMetrics object with all metrics
    """
    # Create spatial calibration (no ground contact dependency)
    _cal_args = (landmarks, visibilities, runner_height_cm, shoe_sole_cm, shoe_type)
    try:
        calibration = create_spatial_calibration(*_cal_args)
    except ValueError:
        try:
            calibration = create_spatial_calibration(
                *_cal_args, min_visibility=0.35, min_samples=5
            )
        except ValueError:
            calibration = create_spatial_calibration(
                *_cal_args, min_visibility=0.0, min_samples=3
            )

    # Detect ground contacts using two-pass coarse/fine method, unless the
    # caller already did (see the ``contacts`` argument).
    if contacts is None:
        contacts, _contact_summary = detect_ground_contacts(
            landmarks,
            visibilities,
            fps,
            resolution_height=resolution_height,
            toe_off_method=toe_off_method,
            body_height_px=body_height_px,
            tau_strike_frac=tau_strike_frac,
            delta_lift_frac=delta_lift_frac,
            delta_lift_frac_forefoot=delta_lift_frac_forefoot,
            ankle_horiz_vel_gate_frac=ankle_horiz_vel_gate_frac,
            clip_strike_pattern=clip_strike_pattern,
            unknown_as_forefoot=unknown_as_forefoot,
            prune_spurious=prune_spurious,
            velocity_ms=velocity_ms,
            leg_length_m=leg_length_m,
            cadence_band_frac=cadence_band_frac,
            interpolate_missing=interpolate_missing,
            gate_landmark_bounds=gate_landmark_bounds,
            resolution_width=resolution_width,
            rtm_landmarks=rtm_landmarks,
            rtm_scores=rtm_scores,
        )

    if len(contacts) < 3:
        raise ValueError(f"Insufficient ground contacts detected: {len(contacts)}")

    # Temporal metrics (GCT already computed during detection)
    cadence_spm, cadence_std = calculate_cadence(contacts, fps)
    gct_ms, gct_std, _ = calculate_ground_contact_time(contacts, refined_only=True)
    flight_time_ms = calculate_flight_time(cadence_spm, gct_ms)
    duty_factor = calculate_duty_factor(gct_ms, flight_time_ms)

    # Spatial metrics
    stride_length_m, stride_std, _ = calculate_stride_length(
        landmarks, contacts, calibration
    )
    step_length_m, _, _ = calculate_step_length(landmarks, contacts, calibration)
    oscillation_cm, oscillation_leg_ratio = calculate_vertical_oscillation(
        landmarks, calibration, fps=fps, cadence_spm=cadence_spm
    )

    # Normalised metrics
    stride_leg_ratio = stride_length_m / (calibration.leg_length_cm / 100)

    # Efficiency metrics
    vo_ratio = calculate_vertical_oscillation_ratio(oscillation_cm, stride_length_m)
    # Flight ratio = flight_time / step_time (NOT stride_time). Deliberately a
    # per-STEP quantity, unlike duty_factor, which is per-stride.
    # The identity is therefore flight_ratio == 1 - 2 * duty_factor, not
    # 1 - duty_factor as this comment claimed before that correction.
    flight_ratio = (
        flight_time_ms / (gct_ms + flight_time_ms)
        if not np.isnan(flight_time_ms)
        else np.nan
    )

    # Derived metrics
    velocity_ms, velocity_kmh, pace_per_km = calculate_velocity(
        stride_length_m, cadence_spm
    )

    return BiomechanicalMetrics(
        cadence_spm=cadence_spm,
        cadence_std=cadence_std,
        gct_ms=gct_ms,
        gct_std=gct_std,
        flight_time_ms=flight_time_ms,
        duty_factor=duty_factor,
        stride_length_m=stride_length_m,
        stride_length_std=stride_std,
        step_length_m=step_length_m,
        oscillation_cm=oscillation_cm,
        leg_length_cm=calibration.leg_length_cm,
        stride_leg_ratio=stride_leg_ratio,
        oscillation_leg_ratio=oscillation_leg_ratio,
        vertical_oscillation_ratio=vo_ratio,
        flight_ratio=flight_ratio,
        velocity_ms=velocity_ms,
        velocity_kmh=velocity_kmh,
        pace_per_km=pace_per_km,
        n_contacts=len(contacts),
        n_refined_contacts=sum(1 for c in contacts if c.detection_method == "refined"),
        calibration_confidence=calibration.confidence,
    )
