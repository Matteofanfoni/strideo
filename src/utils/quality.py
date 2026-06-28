import numpy as np
from typing import Dict, List, Tuple

from src.preprocessing.calibration import SpatialCalibration
from src.utils.metrics import BiomechanicalMetrics

# Expected ranges for middle-distance runners
# NOTE: stride_length is FULL gait cycle (2 steps), step_length is HALF gait cycle
METRIC_RANGES = {
    "cadence_spm": (160, 220),
    "gct_ms": (120, 280),
    "flight_time_ms": (80, 200),
    "duty_factor": (0.25, 0.55),
    "stride_length_m": (2.4, 5.0),  # Full gait cycle (2 steps) — middle-distance range
    "step_length_m": (1.2, 2.5),  # Half gait cycle (1 step)
    "oscillation_cm": (4, 14),
    "stride_leg_ratio": (2.4, 5.5),  # Stride / leg_length (stride ≈ 2.6-4.5× leg)
    "oscillation_leg_ratio": (0.03, 0.10),
    "running_economy_index": (1, 6),  # oscillation / stride — lower with longer stride
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
        ("running_economy_index", metrics.running_economy_index),
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


def calculate_quality_score(
    pose_quality: Dict,
    calibration: SpatialCalibration,
    metrics: BiomechanicalMetrics,
    metric_warnings: List[str],
) -> Tuple[float, Dict]:
    """
    Calculate overall quality score for the analysis.

    Args:
        pose_quality: Quality metrics from pose extraction
        calibration: Spatial calibration result
        metrics: Extracted biomechanical metrics
        metric_warnings: Warnings from metric validation

    Returns:
        score: Overall quality score (0-1)
        breakdown: Component scores
    """
    # Detection quality (0-1)
    detection_score = min(1.0, pose_quality["detection_rate"] / 0.95)

    # Visibility quality (0-1)
    visibility_score = min(1.0, pose_quality["mean_hip_visibility"] / 0.85)

    # Calibration quality (0-1)
    calibration_score = calibration.confidence

    # Metric quality (0-1)
    metric_score = max(0, 1.0 - len(metric_warnings) * 0.1)

    # Contact detection quality (0-1)
    contact_score = min(1.0, metrics.n_contacts / 6)

    # Weighted combination
    weights = {
        "detection": 0.25,
        "visibility": 0.20,
        "calibration": 0.25,
        "metrics": 0.15,
        "contacts": 0.15,
    }

    overall_score = (
        weights["detection"] * detection_score
        + weights["visibility"] * visibility_score
        + weights["calibration"] * calibration_score
        + weights["metrics"] * metric_score
        + weights["contacts"] * contact_score
    )

    breakdown = {
        "detection": detection_score,
        "visibility": visibility_score,
        "calibration": calibration_score,
        "metrics": metric_score,
        "contacts": contact_score,
        "overall": overall_score,
    }

    return overall_score, breakdown
