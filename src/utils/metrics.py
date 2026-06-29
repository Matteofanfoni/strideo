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
    Calculate duty factor (proportion of step time spent in stance).

    Duty factor = GCT / step_time
                = GCT / (GCT + flight_time)

    IMPORTANT: Uses step_time (one foot cycle), NOT stride_time (full gait cycle).
    This is the standard biomechanics definition.

    Lower values indicate more time airborne (faster running).
    Typical values:
      - Walking: 0.6-0.7
      - Jogging: 0.5-0.6
      - Fast running: 0.4-0.5
      - Sprinting: 0.3-0.4

    Args:
        gct_ms: Ground contact time in ms
        flight_time_ms: Flight time in ms

    Returns:
        duty_factor: Ratio (typically 0.3-0.5 for running)
    """
    if np.isnan(gct_ms) or np.isnan(flight_time_ms):
        return np.nan

    step_time = gct_ms + flight_time_ms
    if step_time <= 0:
        return np.nan

    return gct_ms / step_time


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

    # Smooth to reduce noise
    hip_x_smooth = gaussian_filter1d(
        np.nan_to_num(hip_x, nan=float(np.nanmean(hip_x))), sigma=2
    )

    # Collect same-leg (i, i+2) pairs with their frame intervals.
    # Cross-leg pairs are skipped: a stride is one complete same-leg cycle.
    candidate_intervals: List[int] = []
    candidate_pairs: List[tuple] = []
    for i in range(len(contacts) - 2):
        if contacts[i].leg != contacts[i + 2].leg:
            continue
        s = contacts[i].frame
        e = contacts[i + 2].frame
        if s < len(hip_x_smooth) and e < len(hip_x_smooth):
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
    hip_x_smooth = gaussian_filter1d(
        np.nan_to_num(hip_x, nan=float(np.nanmean(hip_x))), sigma=2
    )

    step_lengths_px = []

    for i in range(len(contacts) - 1):
        start_frame = contacts[i].frame
        end_frame = contacts[i + 1].frame

        if start_frame < len(hip_x_smooth) and end_frame < len(hip_x_smooth):
            displacement = abs(hip_x_smooth[end_frame] - hip_x_smooth[start_frame])
            step_lengths_px.append(displacement)

    if len(step_lengths_px) == 0:
        return np.nan, np.nan, []

    step_lengths_m = [calibration.px_to_m(s) for s in step_lengths_px]

    return float(np.mean(step_lengths_m)), float(np.std(step_lengths_m)), step_lengths_m


def calculate_vertical_oscillation(
    landmarks: np.ndarray, calibration: SpatialCalibration
) -> Tuple[float, float]:
    """
    Calculate vertical oscillation (peak-to-peak hip displacement).

    Args:
        landmarks: (T, 33, 2) pose landmarks
        calibration: Spatial calibration

    Returns:
        oscillation_cm: Vertical oscillation in centimetres
        oscillation_leg_ratio: Oscillation as fraction of leg length
    """
    from scipy.signal import find_peaks

    # Hip Y position (average of left and right)
    left_hip_y = landmarks[:, LandmarkIndex.LEFT_HIP, 1]
    right_hip_y = landmarks[:, LandmarkIndex.RIGHT_HIP, 1]
    hip_y = np.nanmean([left_hip_y, right_hip_y], axis=0)

    # Remove NaN and smooth
    hip_y_valid = hip_y[~np.isnan(hip_y)]
    if len(hip_y_valid) < 20:
        return np.nan, np.nan

    hip_y_smooth = gaussian_filter1d(hip_y_valid, sigma=2)

    # Find peaks and troughs
    # Note: in image coordinates, lower Y = higher position
    # So peaks in the signal are the LOW points of oscillation
    peaks, _ = find_peaks(hip_y_smooth, distance=10)
    troughs, _ = find_peaks(-hip_y_smooth, distance=10)

    if len(peaks) < 2 or len(troughs) < 2:
        # Fall back to simple range
        oscillation_px = np.max(hip_y_smooth) - np.min(hip_y_smooth)
    else:
        # Average peak-to-trough amplitude
        peak_values = hip_y_smooth[peaks]
        trough_values = hip_y_smooth[troughs]
        oscillation_px = np.mean(peak_values) - np.mean(trough_values)

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

    # Calculate pace
    if velocity_ms > 0:
        pace_seconds_per_km = 1000 / velocity_ms
        pace_minutes = int(pace_seconds_per_km // 60)
        pace_seconds = int(pace_seconds_per_km % 60)
        pace_per_km = f"{pace_minutes}:{pace_seconds:02d}"
    else:
        pace_per_km = "N/A"

    return velocity_ms, velocity_kmh, pace_per_km


def calculate_running_economy_index(
    oscillation_cm: float, stride_length_m: float
) -> float:
    """
    Calculate Running Economy Index (REI).

    REI = (vertical oscillation / stride length) × 100

    Lower values indicate more efficient forward propulsion
    (less energy wasted on vertical motion).

    Args:
        oscillation_cm: Vertical oscillation in cm
        stride_length_m: Stride length in metres

    Returns:
        rei: Running Economy Index (percentage)
    """
    if np.isnan(oscillation_cm) or np.isnan(stride_length_m) or stride_length_m <= 0:
        return np.nan

    stride_length_cm = stride_length_m * 100
    rei = (oscillation_cm / stride_length_cm) * 100

    return rei


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
    running_economy_index: float
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
    rtm_landmarks: Optional[np.ndarray] = None,
    rtm_scores: Optional[np.ndarray] = None,
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

    # Detect ground contacts using two-pass coarse/fine method
    contacts, contact_summary = detect_ground_contacts(
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
        prune_spurious=prune_spurious,
        velocity_ms=velocity_ms,
        leg_length_m=leg_length_m,
        cadence_band_frac=cadence_band_frac,
        interpolate_missing=interpolate_missing,
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
        landmarks, calibration
    )

    # Normalised metrics
    stride_leg_ratio = stride_length_m / (calibration.leg_length_cm / 100)

    # Efficiency metrics
    rei = calculate_running_economy_index(oscillation_cm, stride_length_m)
    # Flight ratio = flight_time / step_time (NOT stride_time)
    # This equals 1 - duty_factor
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
        running_economy_index=rei,
        flight_ratio=flight_ratio,
        velocity_ms=velocity_ms,
        velocity_kmh=velocity_kmh,
        pace_per_km=pace_per_km,
        n_contacts=len(contacts),
        n_refined_contacts=sum(1 for c in contacts if c.detection_method == "refined"),
        calibration_confidence=calibration.confidence,
    )
