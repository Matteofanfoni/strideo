import numpy as np
from enum import IntEnum
from typing import Tuple, Dict

# MediaPipe landmark pairs (left_idx, right_idx)
BILATERAL_PAIRS = [
    (11, 12),  # shoulders
    (13, 14),  # elbows
    (15, 16),  # wrists
    (17, 18),  # pinkies
    (19, 20),  # index fingers
    (21, 22),  # thumbs
    (23, 24),  # hips
    (25, 26),  # knees
    (27, 28),  # ankles
    (29, 30),  # heels
    (31, 32),  # foot indices
]


def determine_camera_side(visibilities: np.ndarray) -> str:
    """
    Determine which side of the body faces the camera.

    Args:
        visibilities: (T, 33) visibility scores

    Returns:
        'L' if left side faces camera, 'R' if right side
    """
    # Key landmarks for side determination (upper body most reliable)
    left_indices = [11, 23, 25, 27]  # L shoulder, hip, knee, ankle
    right_indices = [12, 24, 26, 28]  # R shoulder, hip, knee, ankle

    left_mean_vis = np.nanmean(visibilities[:, left_indices])
    right_mean_vis = np.nanmean(visibilities[:, right_indices])

    return "L" if left_mean_vis >= right_mean_vis else "R"


def detect_running_direction(landmarks: np.ndarray) -> str:
    """
    Detect running direction from hip trajectory.

    Args:
        landmarks: (T, 33, 2) pose landmarks

    Returns:
        'LR' if left-to-right (X increasing), 'RL' if right-to-left
    """
    LEFT_HIP, RIGHT_HIP = 23, 24

    # Average hip X position over time
    hip_x = np.nanmean([landmarks[:, LEFT_HIP, 0], landmarks[:, RIGHT_HIP, 0]], axis=0)

    # Remove NaN for regression
    valid_frames = ~np.isnan(hip_x)
    if valid_frames.sum() < 10:
        # Insufficient data — use visibility to infer direction
        # (LEFT near implies R→L, RIGHT near implies L→R)
        return "RL"  # Default assumption

    times = np.arange(len(hip_x))[valid_frames]
    positions = hip_x[valid_frames]

    # Linear regression slope indicates direction
    slope = np.polyfit(times, positions, 1)[0]

    return "LR" if slope > 0 else "RL"


def swap_bilateral_labels(
    landmarks: np.ndarray, visibilities: np.ndarray
) -> Tuple[np.ndarray, np.ndarray]:
    """
    Swap LEFT and RIGHT landmark labels (indices), not coordinates.

    After this operation:
    - Indices that were "LEFT" now contain coordinates from "RIGHT"
    - Indices that were "RIGHT" now contain coordinates from "LEFT"

    Args:
        landmarks: (T, 33, 2) landmark coordinates
        visibilities: (T, 33) visibility scores

    Returns:
        swapped_landmarks: (T, 33, 2)
        swapped_visibilities: (T, 33)
    """
    swapped_lm = landmarks.copy()
    swapped_vis = visibilities.copy()

    for left_idx, right_idx in BILATERAL_PAIRS:
        # Swap coordinates between left and right indices
        swapped_lm[:, [left_idx, right_idx]] = landmarks[:, [right_idx, left_idx]]
        # Swap visibilities
        swapped_vis[:, [left_idx, right_idx]] = visibilities[:, [right_idx, left_idx]]

    return swapped_lm, swapped_vis


def mirror_x_coordinates(landmarks: np.ndarray, frame_width: int) -> np.ndarray:
    """
    Mirror X coordinates horizontally (reverse running direction).

    NOTE: This does NOT swap labels — it only flips X values.

    Args:
        landmarks: (T, 33, 2) landmark coordinates
        frame_width: Width of video frame in pixels

    Returns:
        mirrored_landmarks: (T, 33, 2) with X coordinates mirrored
    """
    mirrored = landmarks.copy()
    mirrored[:, :, 0] = frame_width - mirrored[:, :, 0]
    return mirrored


def canonicalise_pose_sequence(
    landmarks: np.ndarray, visibilities: np.ndarray, frame_width: int
) -> Tuple[np.ndarray, np.ndarray, Dict]:
    """
    Canonicalise pose sequence to standard form.

    Canonical form:
    - LEFT landmarks = near side (higher visibility)
    - Running direction = right-to-left (X decreasing)

    The natural relationship is:
    - L→R direction ↔ RIGHT near (State A)
    - R→L direction ↔ LEFT near (State B)

    State B is canonical; State A requires both swap and mirror.

    Args:
        landmarks: (T, 33, 2) raw landmarks
        visibilities: (T, 33) visibility scores
        frame_width: Video frame width

    Returns:
        canonical_landmarks: (T, 33, 2)
        canonical_visibilities: (T, 33)
        transforms: Dict describing what was applied
    """
    # Step 1: Detect current state
    near_side = determine_camera_side(visibilities)
    direction = detect_running_direction(landmarks)

    transforms = {
        "original_near_side": near_side,
        "original_direction": direction,
        "swapped_labels": False,
        "mirrored_x": False,
    }

    # Work with copies
    canon_lm = landmarks.copy()
    canon_vis = visibilities.copy()

    # Step 2: Transform to canonical form (LEFT near, R→L direction)

    # If RIGHT is near side, swap labels so LEFT indices contain near-side data
    if near_side == "R":
        canon_lm, canon_vis = swap_bilateral_labels(canon_lm, canon_vis)
        transforms["swapped_labels"] = True

    # If direction is L→R, mirror X to make it R→L
    if direction == "LR":
        canon_lm = mirror_x_coordinates(canon_lm, frame_width)
        transforms["mirrored_x"] = True

    return canon_lm, canon_vis, transforms


def verify_canonical_form(landmarks: np.ndarray, visibilities: np.ndarray) -> Dict:
    """
    Verify that a pose sequence is in canonical form.

    Returns diagnostics for validation.
    """
    near_side = determine_camera_side(visibilities)
    direction = detect_running_direction(landmarks)

    is_canonical = (near_side == "L") and (direction == "RL")

    return {
        "is_canonical": is_canonical,
        "near_side": near_side,
        "direction": direction,
        "expected_near_side": "L",
        "expected_direction": "RL",
    }


class NNLandmarkIndex(IntEnum):
    """
    Neural network landmark indices (17 landmarks, U-shape ordering).

    After canonicalisation, LEFT = near-side (higher visibility).
    Ordering follows body contour: Head → L arm → L leg → R leg → R arm
    """

    # Head
    NOSE = 0

    # Left side (near after canonicalisation)
    LEFT_WRIST = 1
    LEFT_ELBOW = 2
    LEFT_SHOULDER = 3
    LEFT_HIP = 4
    LEFT_KNEE = 5
    LEFT_ANKLE = 6
    LEFT_HEEL = 7
    LEFT_FOOT_INDEX = 8

    # Right side (far after canonicalisation)
    RIGHT_FOOT_INDEX = 9
    RIGHT_HEEL = 10
    RIGHT_ANKLE = 11
    RIGHT_KNEE = 12
    RIGHT_HIP = 13
    RIGHT_SHOULDER = 14
    RIGHT_ELBOW = 15
    RIGHT_WRIST = 16


# Mapping from MediaPipe indices to NN indices
MP_TO_NN_INDEX = {
    0: NNLandmarkIndex.NOSE,
    15: NNLandmarkIndex.LEFT_WRIST,
    13: NNLandmarkIndex.LEFT_ELBOW,
    11: NNLandmarkIndex.LEFT_SHOULDER,
    23: NNLandmarkIndex.LEFT_HIP,
    25: NNLandmarkIndex.LEFT_KNEE,
    27: NNLandmarkIndex.LEFT_ANKLE,
    29: NNLandmarkIndex.LEFT_HEEL,
    31: NNLandmarkIndex.LEFT_FOOT_INDEX,
    32: NNLandmarkIndex.RIGHT_FOOT_INDEX,
    30: NNLandmarkIndex.RIGHT_HEEL,
    28: NNLandmarkIndex.RIGHT_ANKLE,
    26: NNLandmarkIndex.RIGHT_KNEE,
    24: NNLandmarkIndex.RIGHT_HIP,
    12: NNLandmarkIndex.RIGHT_SHOULDER,
    14: NNLandmarkIndex.RIGHT_ELBOW,
    16: NNLandmarkIndex.RIGHT_WRIST,
}

# Reverse mapping for visualisation
NN_TO_MP_INDEX = {v: k for k, v in MP_TO_NN_INDEX.items()}


def nn_to_mp_landmarks(
    nn_landmarks: np.ndarray, nn_visibilities: np.ndarray
) -> Tuple[np.ndarray, np.ndarray]:
    """
    Convert neural network landmarks back to MediaPipe indexing.

    Useful for visualisation with standard MediaPipe tools.
    Note: Only the 17 selected landmarks are populated; others are NaN.

    Args:
        nn_landmarks: (T, 17, 2) landmarks in U-shape order
        nn_visibilities: (T, 17) visibilities

    Returns:
        mp_landmarks: (T, 33, 2) with selected landmarks filled
        mp_visibilities: (T, 33) with selected visibilities filled
    """
    T = nn_landmarks.shape[0]
    mp_landmarks = np.full((T, 33, 2), np.nan)
    mp_visibilities = np.zeros((T, 33))

    for mp_idx, nn_idx in MP_TO_NN_INDEX.items():
        mp_landmarks[:, mp_idx] = nn_landmarks[:, nn_idx]
        mp_visibilities[:, mp_idx] = nn_visibilities[:, nn_idx]

    return mp_landmarks, mp_visibilities


def extract_nn_landmarks_raw(
    canon_landmarks: np.ndarray, canon_visibilities: np.ndarray
) -> Tuple[np.ndarray, np.ndarray]:
    """
    Extract 17-landmark subset from canonicalised MediaPipe output.

    Returns raw pixel coordinates (no normalization) for subsequent
    interpolation and hip-centered normalization.

    Args:
        canon_landmarks: (T, 33, 2) canonicalised landmarks (pixel coords)
        canon_visibilities: (T, 33) visibility scores

    Returns:
        nn_landmarks: (T, 17, 2) landmarks in U-shape order (pixels)
        nn_visibilities: (T, 17) visibility scores
    """
    T = len(canon_landmarks)

    nn_landmarks = np.full((T, 17, 2), np.nan)
    nn_visibilities = np.zeros((T, 17))

    for mp_idx, nn_idx in MP_TO_NN_INDEX.items():
        nn_landmarks[:, nn_idx] = canon_landmarks[:, mp_idx]
        nn_visibilities[:, nn_idx] = canon_visibilities[:, mp_idx]

    return nn_landmarks, nn_visibilities


def filter_invalid_coordinates(
    landmarks: np.ndarray, visibilities: np.ndarray, threshold: float = 1e-6
) -> Tuple[np.ndarray, np.ndarray]:
    """
    Convert (0,0) coordinates (MediaPipe sentinel) to NaN.

    Must be called early in the pipeline, before normalization or interpolation.

    Args:
        landmarks: (T, 33, 2) or (T, 17, 2) landmarks
        visibilities: (T, 33) or (T, 17) visibility scores
        threshold: Coordinates below this are considered invalid

    Returns:
        landmarks: With invalid coordinates set to NaN
        visibilities: With invalid landmarks set to 0
    """
    landmarks = landmarks.copy()
    visibilities = visibilities.copy()

    # Identify (0,0) or near-zero coordinates
    invalid_mask = (np.abs(landmarks[:, :, 0]) < threshold) & (
        np.abs(landmarks[:, :, 1]) < threshold
    )

    # Set coordinates to NaN
    landmarks[invalid_mask, 0] = np.nan
    landmarks[invalid_mask, 1] = np.nan

    # Set visibility to 0 for invalid landmarks
    visibilities[invalid_mask] = 0.0

    return landmarks, visibilities


class LandmarkStatus(IntEnum):
    """Status of each landmark at each frame."""

    MEASURED = 0  # Original MediaPipe detection
    INTERPOLATED = 1  # Filled by linear interpolation
    MISSING = 2  # Large gap, filled with placeholder


def interpolate_short_gaps(
    landmarks: np.ndarray,
    visibilities: np.ndarray,
    max_gap: int = 3,
    visibility_discount: float = 0.8,
) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """
    Linearly interpolate gaps of up to max_gap consecutive NaN frames.

    For interpolated frames:
    - Coordinates: linear interpolation between boundaries
    - Visibility: 0.8 × min(vis_before_gap, vis_after_gap)
    - Status: INTERPOLATED

    Gaps larger than max_gap are left as NaN (handled by fill_large_gaps).

    Args:
        landmarks: (T, L, 2) landmarks with NaN for missing values
        visibilities: (T, L) visibility scores
        max_gap: Maximum gap size to interpolate (default: 3 frames = 50ms)
        visibility_discount: Multiplier for interpolated visibility (default: 0.8)

    Returns:
        landmarks: (T, L, 2) with short gaps filled
        visibilities: (T, L) with interpolated visibilities
        status: (T, L) with LandmarkStatus values
    """
    T, L, _ = landmarks.shape
    landmarks = landmarks.copy()
    visibilities = visibilities.copy()

    # Initialize status: MEASURED for valid, will update for interpolated/missing
    status = np.zeros((T, L), dtype=np.int8)

    # Mark current NaN as potentially MISSING (will update to INTERPOLATED if filled)
    for lm in range(L):
        nan_mask = np.isnan(landmarks[:, lm, 0])
        status[nan_mask, lm] = LandmarkStatus.MISSING

    # Interpolate each landmark independently
    for lm in range(L):
        for coord in range(2):  # x, y
            values = landmarks[:, lm, coord]
            vis = visibilities[:, lm]

            # Find NaN segments
            is_nan = np.isnan(values)
            if not is_nan.any():
                continue

            # Find gap boundaries using diff
            nan_int = is_nan.astype(int)
            nan_changes = np.diff(nan_int)

            # gap_starts: indices where NaN begins
            # gap_ends: indices where NaN ends (first valid after gap)
            gap_starts = np.where(nan_changes == 1)[0] + 1
            gap_ends = np.where(nan_changes == -1)[0] + 1

            # Handle edge cases: gap at start or end of sequence
            if is_nan[0]:
                gap_starts = np.concatenate([[0], gap_starts])
            if is_nan[-1]:
                gap_ends = np.concatenate([gap_ends, [T]])

            # Interpolate each gap
            for start, end in zip(gap_starts, gap_ends):
                gap_length = end - start

                if gap_length <= max_gap:
                    # Get boundary indices
                    left_idx = start - 1 if start > 0 else None
                    right_idx = end if end < T else None

                    if left_idx is not None and right_idx is not None:
                        # Both boundaries available: linear interpolation
                        left_val = values[left_idx]
                        right_val = values[right_idx]
                        left_vis = vis[left_idx]
                        right_vis = vis[right_idx]

                        # Only interpolate if both boundaries are valid (not NaN)
                        if not (np.isnan(left_val) or np.isnan(right_val)):
                            for i, t in enumerate(range(start, end)):
                                alpha = (i + 1) / (gap_length + 1)
                                landmarks[t, lm, coord] = (
                                    1 - alpha
                                ) * left_val + alpha * right_val

                                # Update visibility (once per landmark,
                                # not per coord)
                                if coord == 0:  # Only on x pass
                                    interp_vis = visibility_discount * min(
                                        left_vis, right_vis
                                    )
                                    visibilities[t, lm] = interp_vis
                                    status[t, lm] = LandmarkStatus.INTERPOLATED

                    elif left_idx is not None and not np.isnan(values[left_idx]):
                        # Only left boundary: forward fill (edge case)
                        landmarks[start:end, lm, coord] = values[left_idx]
                        if coord == 0:
                            visibilities[start:end, lm] = (
                                visibility_discount * vis[left_idx]
                            )
                            status[start:end, lm] = LandmarkStatus.INTERPOLATED

                    elif right_idx is not None and not np.isnan(values[right_idx]):
                        # Only right boundary: backward fill (edge case)
                        landmarks[start:end, lm, coord] = values[right_idx]
                        if coord == 0:
                            visibilities[start:end, lm] = (
                                visibility_discount * vis[right_idx]
                            )
                            status[start:end, lm] = LandmarkStatus.INTERPOLATED

    return landmarks, visibilities, status


def hip_centered_normalization(
    landmarks: np.ndarray,
    visibilities: np.ndarray,
    L_HIP: int = 4,  # NNLandmarkIndex.LEFT_HIP
    R_HIP: int = 13,  # NNLandmarkIndex.RIGHT_HIP
    L_SHOULDER: int = 3,  # NNLandmarkIndex.LEFT_SHOULDER
    min_visibility: float = 0.5,
) -> Tuple[np.ndarray, float]:
    """
    Center landmarks on hip and scale by torso length.

    Torso length (hip → shoulder) is used for scaling because:
    - It's gait-invariant (doesn't change with knee flexion)
    - Both landmarks have high visibility (>0.80) in sagittal view
    - It's proportional to body size

    Args:
        landmarks: (T, 17, 2) landmarks in U-shape order (pixels, after interpolation)
        visibilities: (T, 17) visibility scores
        L_HIP, R_HIP, L_SHOULDER: Landmark indices
        min_visibility: Minimum visibility to include frame in calculations

    Returns:
        normalized: (T, 17, 2) hip-centered, torso-scaled landmarks
        torso_length: The torso length used for scaling (for metadata)
    """
    # Calculate hip center per frame
    hip_x = (landmarks[:, L_HIP, 0] + landmarks[:, R_HIP, 0]) / 2
    hip_y = (landmarks[:, L_HIP, 1] + landmarks[:, R_HIP, 1]) / 2

    # Use MEAN hip position across valid frames (preserves vertical oscillation)
    hip_vis = np.minimum(visibilities[:, L_HIP], visibilities[:, R_HIP])
    valid_hip = (hip_vis >= min_visibility) & ~np.isnan(hip_x) & ~np.isnan(hip_y)

    if valid_hip.sum() > 0:
        mean_hip_x = np.nanmean(hip_x[valid_hip])
        mean_hip_y = np.nanmean(hip_y[valid_hip])
    else:
        # Fallback: use median of all frames
        mean_hip_x = np.nanmedian(hip_x)
        mean_hip_y = np.nanmedian(hip_y)

    # Calculate torso length for scaling (near-side = left after canonicalization)
    torso_dx = landmarks[:, L_SHOULDER, 0] - landmarks[:, L_HIP, 0]
    torso_dy = landmarks[:, L_SHOULDER, 1] - landmarks[:, L_HIP, 1]
    torso_length_per_frame = np.sqrt(torso_dx**2 + torso_dy**2)

    # Use median torso length from frames with good visibility
    shoulder_vis = visibilities[:, L_SHOULDER]
    torso_vis = np.minimum(hip_vis, shoulder_vis)
    valid_torso = (torso_vis >= min_visibility) & ~np.isnan(torso_length_per_frame)

    if valid_torso.sum() > 0:
        torso_length = np.median(torso_length_per_frame[valid_torso])
    else:
        # Fallback: use any valid measurement
        valid_any = ~np.isnan(torso_length_per_frame)
        if valid_any.sum() > 0:
            torso_length = np.median(torso_length_per_frame[valid_any])
        else:
            # Last resort: assume typical proportion (torso ≈ 30% of frame height)
            torso_length = 0.3 * np.nanmax(landmarks[:, :, 1])

    # Prevent division by zero
    torso_length = max(torso_length, 1e-6)

    # Center and scale
    normalized = np.zeros_like(landmarks)
    normalized[:, :, 0] = (landmarks[:, :, 0] - mean_hip_x) / torso_length
    normalized[:, :, 1] = (landmarks[:, :, 1] - mean_hip_y) / torso_length

    return normalized, torso_length


# Anthropometric conversion constants
class AnthropometricRatios:
    """Convert between torso-scaled and standard biomechanical units."""

    # Torso to other units
    TORSO_TO_HEIGHT = 0.26  # torso ≈ 26% of height
    TORSO_TO_LEG_LENGTH = 0.49  # torso ≈ 49% of leg length

    # Inverse conversions
    HEIGHT_TO_TORSO = 3.85  # height ≈ 3.85 × torso
    LEG_LENGTH_TO_TORSO = 2.04  # leg ≈ 2.04 × torso


def convert_to_height_scaled(coords_torso: np.ndarray) -> np.ndarray:
    """
    Convert torso-scaled coordinates to height-scaled.

    Args:
        coords_torso: Coordinates in torso-length units

    Returns:
        coords_height: Coordinates in height units (1 unit = full height)
    """
    return coords_torso * AnthropometricRatios.TORSO_TO_HEIGHT


def convert_to_leg_scaled(coords_torso: np.ndarray) -> np.ndarray:
    """
    Convert torso-scaled coordinates to leg-length-scaled.

    Args:
        coords_torso: Coordinates in torso-length units

    Returns:
        coords_leg: Coordinates in leg-length units (1 unit = hip to ankle)
    """
    return coords_torso * AnthropometricRatios.TORSO_TO_LEG_LENGTH


def to_meters(value_torso: float, torso_length_px: float, px_per_m: float) -> float:
    """Convert torso-scaled value to meters."""
    return value_torso * torso_length_px / px_per_m


def fill_large_gaps(
    landmarks: np.ndarray, visibilities: np.ndarray, status: np.ndarray
) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """
    Fill remaining NaN values with (0, 0) and visibility 0.

    After hip-centered normalization, (0, 0) means "at hip center",
    which is a neutral placeholder for missing data. The visibility
    of 0 tells the CNN to ignore these landmarks.

    Args:
        landmarks: (T, L, 2) with NaN for large gaps
        visibilities: (T, L) visibility scores
        status: (T, L) with LandmarkStatus values

    Returns:
        landmarks: With NaN replaced by (0, 0)
        visibilities: With corresponding entries set to 0
        status: With MISSING status confirmed
    """
    landmarks = landmarks.copy()
    visibilities = visibilities.copy()
    status = status.copy()

    # Find remaining NaN (large gaps that weren't interpolated)
    nan_mask = np.isnan(landmarks[:, :, 0]) | np.isnan(landmarks[:, :, 1])

    # Set coordinates to (0, 0) = hip center in normalized coords
    landmarks[nan_mask, 0] = 0.0
    landmarks[nan_mask, 1] = 0.0

    # Set visibility to 0
    visibilities[nan_mask] = 0.0

    # Confirm MISSING status
    status[nan_mask] = LandmarkStatus.MISSING

    return landmarks, visibilities, status


# Discrete nominal pace values for FiLM conditioning
# (see Neural Network Architecture §3.2.1–3.2.2)
PACE_LEVEL_NORMALIZED: Dict[str, float] = {
    "threshold": 0.33,
    "1500m": 0.55,
    "800m": 0.78,
}


def normalize_pace(pace_level: str) -> float:
    """
    Convert a pace_level string to a normalised scalar in [0, 1].

    Uses discrete nominal values (not per-runner measured speed) to
    prevent FiLM from conflating runner identity with pace effect.

    Args:
        pace_level: One of 'threshold', '1500m', '800m'

    Returns:
        Normalised pace scalar for FiLM conditioning

    Raises:
        ValueError: If pace_level is not recognised
    """
    if pace_level not in PACE_LEVEL_NORMALIZED:
        raise ValueError(
            f"Unknown pace_level '{pace_level}'. "
            f"Expected one of: {list(PACE_LEVEL_NORMALIZED.keys())}"
        )

    return PACE_LEVEL_NORMALIZED[pace_level]


def preprocess_for_nn(
    landmarks: np.ndarray,
    visibilities: np.ndarray,
    frame_width: int,
    max_interp_gap: int = 3,
    visibility_discount: float = 0.8,
) -> Tuple[np.ndarray, np.ndarray, Dict]:
    """
    Complete preprocessing pipeline for neural network input.

    Pipeline:
    1. Filter (0,0) → NaN (MediaPipe sentinel handling)
    2. Canonicalise (direction + side normalization)
    3. Extract 17 landmarks (U-shape ordering)
    4. Interpolate short gaps (≤3 frames)
    5. Hip-centered normalization (torso-length scaling)
    6. Fill large gaps with (0,0), visibility=0
    7. Combine into (T, 17, 3) tensor

    Args:
        landmarks: (T, 33, 2) raw MediaPipe landmarks
        visibilities: (T, 33) visibility scores
        frame_width: Video width in pixels
        max_interp_gap: Maximum gap to interpolate (frames)
        visibility_discount: Multiplier for interpolated visibility

    Returns:
        nn_input: (T, 17, 3) with (x, y, visibility) per landmark
        status: (T, 17) with LandmarkStatus values
        metadata: Dict with preprocessing info
    """
    metadata: Dict[str, object] = {}

    # Step 1: Filter invalid (0,0) coordinates
    landmarks, visibilities = filter_invalid_coordinates(landmarks, visibilities)
    metadata["invalid_filtered"] = True

    # Step 2: Canonicalise (direction + side normalization)
    canon_lm, canon_vis, transforms = canonicalise_pose_sequence(
        landmarks, visibilities, frame_width
    )
    metadata["transforms"] = transforms

    # Step 3: Extract 17 landmarks (U-shape order) - raw pixels
    nn_landmarks, nn_visibilities = extract_nn_landmarks_raw(canon_lm, canon_vis)

    # Step 4: Interpolate short gaps
    nn_landmarks, nn_visibilities, status = interpolate_short_gaps(
        nn_landmarks,
        nn_visibilities,
        max_gap=max_interp_gap,
        visibility_discount=visibility_discount,
    )

    # Count interpolated frames
    n_interpolated = np.sum(status == LandmarkStatus.INTERPOLATED)
    n_total = status.size
    metadata["interpolation"] = {
        "max_gap": max_interp_gap,
        "n_interpolated": int(n_interpolated),
        "interpolation_rate": n_interpolated / n_total,
    }

    # Step 5: Hip-centered normalization
    nn_landmarks, torso_length = hip_centered_normalization(
        nn_landmarks, nn_visibilities
    )
    metadata["normalization"] = {
        "method": "hip_centered",
        "scale": "torso_length",
        "torso_length_px": torso_length,
    }

    # Step 6: Fill large gaps
    nn_landmarks, nn_visibilities, status = fill_large_gaps(
        nn_landmarks, nn_visibilities, status
    )

    # Count missing frames
    n_missing = np.sum(status == LandmarkStatus.MISSING)
    metadata["missing"] = {
        "n_missing": int(n_missing),
        "missing_rate": n_missing / n_total,
    }

    # Step 7: Combine into (T, 17, 3)
    nn_input = np.concatenate(
        [nn_landmarks, nn_visibilities[:, :, np.newaxis]], axis=-1
    )

    return nn_input, status, metadata
