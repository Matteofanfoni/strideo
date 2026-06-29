import numpy as np
from dataclasses import dataclass
from typing import Dict, List, Optional, Tuple

from src.preprocessing.pose_estimator import LandmarkIndex


def determine_near_side(visibilities: np.ndarray) -> Tuple[str, float, Dict]:
    """
    Determine which side of the body faces the camera (near side).

    The near side has consistently higher visibility scores across
    key landmarks (shoulder, hip, knee, ankle).

    Args:
        visibilities: (T, 33) visibility scores

    Returns:
        side: 'L' or 'R' indicating near side
        confidence: How strongly one side dominates (0.5 = equal, 1.0 = fully dominant)
        diagnostics: Per-landmark visibility comparison
    """
    # Key landmarks for side determination
    landmark_pairs = {
        "shoulder": (LandmarkIndex.LEFT_SHOULDER, LandmarkIndex.RIGHT_SHOULDER),
        "hip": (LandmarkIndex.LEFT_HIP, LandmarkIndex.RIGHT_HIP),
        "knee": (LandmarkIndex.LEFT_KNEE, LandmarkIndex.RIGHT_KNEE),
        "ankle": (LandmarkIndex.LEFT_ANKLE, LandmarkIndex.RIGHT_ANKLE),
    }

    diagnostics = {}
    left_total = 0.0
    right_total = 0.0

    for name, (left_idx, right_idx) in landmark_pairs.items():
        left_vis = np.nanmean(visibilities[:, left_idx])
        right_vis = np.nanmean(visibilities[:, right_idx])

        diagnostics[f"{name}_left_vis"] = left_vis
        diagnostics[f"{name}_right_vis"] = right_vis

        left_total += left_vis
        right_total += right_vis

    # Determine dominant side
    total = left_total + right_total
    if total == 0:
        return "L", 0.5, diagnostics  # Default if no visibility

    if left_total >= right_total:
        side = "L"
        confidence = left_total / total
    else:
        side = "R"
        confidence = right_total / total

    diagnostics["near_side"] = side
    diagnostics["confidence"] = confidence

    return side, confidence, diagnostics


def get_near_side_indices(side: str) -> Dict[str, int]:
    """
    Get landmark indices for the near side.

    Args:
        side: 'L' or 'R'

    Returns:
        Dict mapping segment names to MediaPipe landmark indices
    """
    if side == "L":
        return {
            "ear": LandmarkIndex.LEFT_EAR,
            "shoulder": LandmarkIndex.LEFT_SHOULDER,
            "hip": LandmarkIndex.LEFT_HIP,
            "knee": LandmarkIndex.LEFT_KNEE,
            "ankle": LandmarkIndex.LEFT_ANKLE,
        }
    else:
        return {
            "ear": LandmarkIndex.RIGHT_EAR,
            "shoulder": LandmarkIndex.RIGHT_SHOULDER,
            "hip": LandmarkIndex.RIGHT_HIP,
            "knee": LandmarkIndex.RIGHT_KNEE,
            "ankle": LandmarkIndex.RIGHT_ANKLE,
        }


# Anatomical constants
HEAD_EXTRAPOLATION_FACTOR = 1.40  # shoulder-to-ear → shoulder-to-crown
FOOT_TO_SHANK_RATIO = 0.20  # Lateral malleolus height / shank length


@dataclass
class ShoeType:
    """Shoe sole thickness specifications."""

    name: str
    sole_cm: float
    description: str


# Common shoe types for middle-distance running
SHOE_TYPES = {
    "racing_flat": ShoeType(
        "racing_flat", 1.5, "Traditional racing flats (e.g., Nike Streak)"
    ),
    "super_shoe": ShoeType(
        "super_shoe", 3.5, "Carbon-plated super shoes (e.g., Nike Vaporfly)"
    ),
    "track_spike": ShoeType("track_spike", 1.0, "Track spikes with minimal sole"),
    "training_shoe": ShoeType("training_shoe", 2.5, "Daily training shoes"),
    "barefoot": ShoeType("barefoot", 0.0, "Barefoot or minimalist"),
}


@dataclass
class SegmentMeasurements:
    """Body segment measurements in pixels."""

    shank_px: float  # Ankle to knee (measured)
    thigh_px: float  # Knee to hip (measured)
    torso_px: float  # Hip to shoulder (measured)
    head_px: float  # Shoulder to crown (extrapolated from ear)
    foot_px: float  # Ankle to ground (estimated from shank + shoe)
    shoe_sole_cm: float  # Shoe sole thickness used

    @property
    def total_height_px(self) -> float:
        """Total body height in pixels."""
        return (
            self.foot_px + self.shank_px + self.thigh_px + self.torso_px + self.head_px
        )

    @property
    def leg_length_px(self) -> float:
        """Leg length (hip to ground) in pixels."""
        return self.foot_px + self.shank_px + self.thigh_px

    def to_proportions(self) -> Dict[str, float]:
        """Convert to proportions of total height."""
        total = self.total_height_px
        return {
            "foot": self.foot_px / total,
            "shank": self.shank_px / total,
            "thigh": self.thigh_px / total,
            "torso": self.torso_px / total,
            "head": self.head_px / total,
        }


def measure_body_segments(
    landmarks: np.ndarray,
    visibilities: np.ndarray,
    near_side: str,
    shoe_sole_cm: float = 2.5,
    min_visibility: float = 0.5,
    min_samples: int = 10,
) -> Tuple[SegmentMeasurements, Dict[str, List[float]], Dict]:
    """
    Measure body segments using near-side landmarks only.

    All measurements use the consistently-visible near side, avoiding
    the need for per-frame visibility selection.

    Args:
        landmarks: (T, 33, 2) pose landmarks in pixels
        visibilities: (T, 33) visibility scores
        near_side: 'L' or 'R' (from determine_near_side)
        shoe_sole_cm: Shoe sole thickness in centimetres
        min_visibility: Minimum visibility to include frame

    Returns:
        segments: SegmentMeasurements with median values
        all_measurements: Dict of all per-frame measurements
        diagnostics: Measurement statistics
    """
    T = len(landmarks)
    indices = get_near_side_indices(near_side)

    # Storage for per-frame measurements
    shank_measurements = []
    thigh_measurements = []
    torso_measurements = []
    head_measurements = []  # shoulder-to-ear (before extrapolation)

    diagnostics = {
        "frames_processed": T,
        "frames_with_full_body": 0,
        "near_side": near_side,
    }

    for t in range(T):
        frame = landmarks[t]
        vis = visibilities[t]

        # Extract near-side landmarks
        ear = frame[indices["ear"]]
        shoulder = frame[indices["shoulder"]]
        hip = frame[indices["hip"]]
        knee = frame[indices["knee"]]
        ankle = frame[indices["ankle"]]

        # Check visibility
        ear_vis = vis[indices["ear"]]
        shoulder_vis = vis[indices["shoulder"]]
        hip_vis = vis[indices["hip"]]
        knee_vis = vis[indices["knee"]]
        ankle_vis = vis[indices["ankle"]]

        # SHANK: ankle to knee
        if (
            ankle_vis >= min_visibility
            and knee_vis >= min_visibility
            and not np.any(np.isnan(ankle))
            and not np.any(np.isnan(knee))
        ):
            shank = np.linalg.norm(knee - ankle)
            if 30 < shank < 500:  # Sanity check (pixels)
                shank_measurements.append(shank)

        # THIGH: knee to hip
        if (
            knee_vis >= min_visibility
            and hip_vis >= min_visibility
            and not np.any(np.isnan(knee))
            and not np.any(np.isnan(hip))
        ):
            thigh = np.linalg.norm(hip - knee)
            if 30 < thigh < 500:
                thigh_measurements.append(thigh)

        # TORSO: hip to shoulder
        if (
            hip_vis >= min_visibility
            and shoulder_vis >= min_visibility
            and not np.any(np.isnan(hip))
            and not np.any(np.isnan(shoulder))
        ):
            torso = np.linalg.norm(shoulder - hip)
            if 30 < torso < 600:
                torso_measurements.append(torso)

        # HEAD (shoulder to ear): for extrapolation to crown
        if (
            shoulder_vis >= min_visibility
            and ear_vis >= min_visibility * 0.8
            and not np.any(np.isnan(shoulder))
            and not np.any(np.isnan(ear))
        ):
            shoulder_to_ear = np.linalg.norm(ear - shoulder)
            if 10 < shoulder_to_ear < 200:
                head_measurements.append(shoulder_to_ear)

        # Count frames with full body visibility
        if all(
            v >= min_visibility for v in [shoulder_vis, hip_vis, knee_vis, ankle_vis]
        ):
            diagnostics["frames_with_full_body"] += 1  # type: ignore[operator]

    # Compute medians
    shank_px = (
        np.median(shank_measurements)
        if len(shank_measurements) >= min_samples
        else np.nan
    )
    thigh_px = (
        np.median(thigh_measurements)
        if len(thigh_measurements) >= min_samples
        else np.nan
    )
    torso_px = (
        np.median(torso_measurements)
        if len(torso_measurements) >= min_samples
        else np.nan
    )

    # Head: extrapolate from shoulder-to-ear
    if len(head_measurements) >= min_samples:
        shoulder_to_ear_px = np.median(head_measurements)
        head_px = shoulder_to_ear_px * HEAD_EXTRAPOLATION_FACTOR
    else:
        head_px = np.nan

    # Foot: estimate from shank proportion + shoe sole
    if not np.isnan(shank_px):
        # Anatomical foot height (ankle to ground without shoe)
        anatomical_foot_px = shank_px * FOOT_TO_SHANK_RATIO
        # We'll add shoe sole after we know pixels_per_cm
        # For now, store anatomical only; shoe added in calibration step
        foot_px = (
            anatomical_foot_px  # Placeholder, adjusted in create_spatial_calibration
        )
    else:
        foot_px = np.nan

    # Store all measurements
    all_measurements = {
        "shank": shank_measurements,
        "thigh": thigh_measurements,
        "torso": torso_measurements,
        "head_raw": head_measurements,  # shoulder-to-ear before extrapolation
    }

    # Diagnostics
    diagnostics["n_shank"] = len(shank_measurements)
    diagnostics["n_thigh"] = len(thigh_measurements)
    diagnostics["n_torso"] = len(torso_measurements)
    diagnostics["n_head"] = len(head_measurements)

    segments = SegmentMeasurements(
        shank_px=shank_px,
        thigh_px=thigh_px,
        torso_px=torso_px,
        head_px=head_px,
        foot_px=foot_px,
        shoe_sole_cm=shoe_sole_cm,
    )

    all_measurements_float: Dict[str, List[float]] = {
        k: [float(v) for v in vals] for k, vals in all_measurements.items()
    }

    return segments, all_measurements_float, diagnostics


# Expected anatomical proportions (as fraction of total height)
# Based on anthropometric literature (Drillis & Contini, NASA standards)
ANATOMICAL_PROPORTIONS = {
    "foot": (0.04, 0.07),  # 4-7% (lateral malleolus height + shoe)
    "shank": (0.20, 0.26),  # 20-26%
    "thigh": (0.23, 0.29),  # 23-29%
    "torso": (0.28, 0.34),  # 28-34%
    "head": (0.12, 0.16),  # 12-16% (shoulder to crown)
}


def validate_segment_proportions(
    segments: SegmentMeasurements,
) -> Tuple[bool, List[str]]:
    """
    Validate that measured proportions are anatomically plausible.

    Args:
        segments: Measured body segments

    Returns:
        is_valid: True if all proportions within expected ranges
        warnings: List of warning messages for out-of-range segments
    """
    proportions = segments.to_proportions()
    warnings = []

    for segment, (min_prop, max_prop) in ANATOMICAL_PROPORTIONS.items():
        actual = proportions[segment]

        if actual < min_prop:
            warnings.append(
                f"{segment} proportion {actual:.1%} below expected"
                f" minimum {min_prop:.1%}"
            )
        elif actual > max_prop:
            warnings.append(
                f"{segment} proportion {actual:.1%} above expected"
                f" maximum {max_prop:.1%}"
            )

    return len(warnings) == 0, warnings


@dataclass
class SpatialCalibration:
    """Spatial calibration result."""

    runner_height_cm: float  # Input: known height
    shoe_sole_cm: float  # Input: shoe sole thickness
    measured_height_px: float  # Output: measured height in pixels
    pixels_per_cm: float  # Conversion factor
    segments: SegmentMeasurements  # Individual segment measurements
    proportion_warnings: List[str]  # Validation warnings
    confidence: float  # Calibration confidence (0-1)
    near_side: str  # Which side was used ('L' or 'R')

    def px_to_cm(self, value_px: float) -> float:
        """Convert pixel value to centimetres."""
        return value_px / self.pixels_per_cm

    def px_to_m(self, value_px: float) -> float:
        """Convert pixel value to metres."""
        return value_px / self.pixels_per_cm / 100

    def to_leg_ratio(self, value_px: float) -> float:
        """Convert pixel value to leg-length ratio."""
        return value_px / self.segments.leg_length_px

    @property
    def leg_length_cm(self) -> float:
        """Leg length in centimetres."""
        return self.px_to_cm(self.segments.leg_length_px)


def create_spatial_calibration(
    landmarks: np.ndarray,
    visibilities: np.ndarray,
    runner_height_cm: float,
    shoe_sole_cm: float = 2.5,
    shoe_type: Optional[str] = None,
    min_visibility: float = 0.5,
    min_samples: int = 10,
) -> SpatialCalibration:
    """
    Create spatial calibration from pose landmarks and known runner height.

    This revised approach:
    1. Detects the near (visible) side once
    2. Measures all segments from near-side landmarks only
    3. Uses ear for head estimation (stable, not affected by head tilt)
    4. Estimates foot height from shank proportion + shoe sole
    5. Does NOT require ground contact detection

    Args:
        landmarks: (T, 33, 2) pose landmarks
        visibilities: (T, 33) visibility scores
        runner_height_cm: Runner's known height in cm
        shoe_sole_cm: Shoe sole thickness in cm (ignored if shoe_type provided)
        shoe_type: Key from SHOE_TYPES dict (overrides shoe_sole_cm)

    Returns:
        SpatialCalibration object

    Raises:
        ValueError: If calibration fails (insufficient landmark visibility)
    """
    # Resolve shoe sole thickness
    if shoe_type is not None:
        if shoe_type not in SHOE_TYPES:
            raise ValueError(
                f"Unknown shoe type: {shoe_type}. Options: {list(SHOE_TYPES.keys())}"
            )
        shoe_sole_cm = SHOE_TYPES[shoe_type].sole_cm

    # Step 1: Determine near side
    near_side, side_confidence, side_diag = determine_near_side(visibilities)

    if side_confidence < 0.55:
        # Very balanced visibility — unusual for sagittal view
        # Could indicate non-sagittal filming angle
        pass  # Continue but note in diagnostics

    # Step 2: Measure body segments
    segments, all_measurements, measure_diag = measure_body_segments(
        landmarks,
        visibilities,
        near_side,
        shoe_sole_cm,
        min_visibility=min_visibility,
        min_samples=min_samples,
    )

    # Check for missing segments
    for segment_name in ["shank_px", "thigh_px", "torso_px", "head_px"]:
        value = getattr(segments, segment_name)
        if np.isnan(value):
            raise ValueError(f"Cannot measure {segment_name}: insufficient visibility")

    # Step 3: Calculate preliminary pixels_per_cm (without shoe correction)
    # Total measured = shank + thigh + torso + head + anatomical_foot
    anatomical_height_px = (
        segments.shank_px
        + segments.thigh_px
        + segments.torso_px
        + segments.head_px
        + segments.foot_px
    )

    # Runner height without shoes = runner_height_cm - shoe_sole_cm
    anatomical_height_cm = runner_height_cm - shoe_sole_cm

    # Pixels per cm
    pixels_per_cm = anatomical_height_px / anatomical_height_cm

    # Step 4: Add shoe sole to foot segment (in pixels)
    shoe_sole_px = shoe_sole_cm * pixels_per_cm
    final_foot_px = segments.foot_px + shoe_sole_px

    # Update segments with final foot value
    final_segments = SegmentMeasurements(
        shank_px=segments.shank_px,
        thigh_px=segments.thigh_px,
        torso_px=segments.torso_px,
        head_px=segments.head_px,
        foot_px=final_foot_px,
        shoe_sole_cm=shoe_sole_cm,
    )

    # Recalculate with shoe included
    total_height_px = final_segments.total_height_px
    pixels_per_cm = total_height_px / runner_height_cm

    # Step 5: Validate proportions
    proportions_valid, proportion_warnings = validate_segment_proportions(
        final_segments
    )

    # Step 6: Calculate confidence
    n_samples = min(
        measure_diag["n_shank"],
        measure_diag["n_thigh"],
        measure_diag["n_torso"],
        measure_diag["n_head"],
    )
    sample_factor = min(1.0, n_samples / 50)
    warning_factor = max(0.5, 1.0 - len(proportion_warnings) * 0.15)
    side_factor = min(1.0, side_confidence / 0.6)  # Penalize very balanced visibility

    confidence = sample_factor * warning_factor * side_factor

    return SpatialCalibration(
        runner_height_cm=runner_height_cm,
        shoe_sole_cm=shoe_sole_cm,
        measured_height_px=total_height_px,
        pixels_per_cm=pixels_per_cm,
        segments=final_segments,
        proportion_warnings=proportion_warnings,
        confidence=confidence,
        near_side=near_side,
    )
