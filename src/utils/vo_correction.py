"""Reference-fitted linear correction for pipeline vertical oscillation.

**Single source of truth for these coefficients.** They used to live in
``src/training/dataset.py`` alone, which made them unreachable from ``app/``:
that package imports torch and is not even copied to the public mirror. The
app needed them once the engine's displayed oscillation had to be put on the
ground-truth scale, and a second copy in ``app/`` would be
precisely the drift this module exists to prevent. ``dataset.py`` now
re-exports from here, so training labels and the app read the same numbers.

**What the correction is for.** The BlazePose hip trajectory under-reports
vertical oscillation badly and consistently: measured against Kinovea on the
15 annotated S1 clips, the raw pipeline value is **54.5% low on 15 of 15**,
measured rather than estimated. This linear fit maps it onto the annotation's
scale.

**What it is not.** It is fitted on those same 15 clips, one venue, one camera
geometry, so it is a calibration, not a validated estimator: it
puts its leave-one-out error at 14.65% against the ground truth's own 12.1%
per-cycle spread, and it is still open whether fitting a
correction to a 4-cycle annotation is sound at all. It is therefore applied
**at display time only** (the app), never inside the pipeline's stored result:
correcting the stored value would make ``report_validation_result.py`` score a
fit against the data it was fitted on, turning an honest raw measurement into
a self-scoring one. That hazard is the same one B1/B2 already document for
ground contact time.
"""

from __future__ import annotations

from typing import Any, Dict, Optional

import numpy as np

# Values copied from data/main_collection/analysis/vo_bias_fit_coefficients.json
# (gitignored data, not a runtime dependency), re-fitted 2026-08-26
# against pole-parallax depth-corrected Tier A labels. The full
# provenance, including the two superseded fits that must not be reinstated,
# is in src/training/dataset.py's own note beside the re-export.
VO_CORRECTION_A = 4.754363882633737
VO_CORRECTION_B = 1.0357032759204108

# Marker written into a corrected metrics dict so the correction cannot be
# applied twice -- to a dict already corrected this session, or to one coming
# back from a saved-results file that was exported after this shipped.
CORRECTED_FLAG = "oscillation_display_corrected"


def correct_oscillation_cm(raw_oscillation_cm: float) -> float:
    """Map a raw pipeline oscillation (cm) onto the ground-truth scale."""
    if raw_oscillation_cm is None or np.isnan(raw_oscillation_cm):
        return float("nan")
    return VO_CORRECTION_A + VO_CORRECTION_B * float(raw_oscillation_cm)


def correct_engine_metrics_for_display(
    metrics: Optional[Dict[str, Any]],
) -> Optional[Dict[str, Any]]:
    """Copy of ``metrics`` with oscillation, and all of it, on the GT scale.

    For the deterministic engine's metrics only. StrideoNet's are already on
    that scale: its oscillation target is the Kinovea annotation itself where
    one exists, and this same correction applied to the pipeline value where
    one does not (``src/training/dataset.py::_resolve_targets``), so applying
    this to a fast-path dict would correct twice.

    Everything derived from oscillation is recomputed rather than left alone,
    for a recorded reason: a card whose displayed ratio cannot be
    reproduced from the displayed numbers beside it is its own defect.

    Returns the input unchanged if it is None or already carries
    ``CORRECTED_FLAG``.
    """
    if not metrics or metrics.get(CORRECTED_FLAG):
        return metrics

    from src.utils.metrics import calculate_vertical_oscillation_ratio

    out = dict(metrics)
    raw = out.get("oscillation_cm")
    if raw is None or (isinstance(raw, float) and np.isnan(raw)):
        out[CORRECTED_FLAG] = True
        return out

    corrected = correct_oscillation_cm(float(raw))
    out["oscillation_cm"] = corrected
    out["oscillation_cm_raw"] = float(raw)

    leg_length_cm = out.get("leg_length_cm")
    if leg_length_cm:
        out["oscillation_leg_ratio"] = corrected / float(leg_length_cm)

    stride_length_m = out.get("stride_length_m")
    if stride_length_m is not None:
        out["vertical_oscillation_ratio"] = calculate_vertical_oscillation_ratio(
            corrected, float(stride_length_m)
        )

    out[CORRECTED_FLAG] = True
    return out
