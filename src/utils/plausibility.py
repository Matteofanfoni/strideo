"""A GT-free plausibility check on the deterministic engine's result.

The engine's velocity is stride x cadence / 120 (``calculate_velocity``), so
checking that identity against itself is circular. The hip-x velocity
(``estimate_velocity_from_hipx``) is independent of both: a line fitted to
hip x against time, with no contacts in it. Both share the px/m calibration,
so scale largely cancels and a disagreement points at the temporal side
(cadence, or stride's same-leg pairing).

A failure flags the result, it does not withhold it. Measured on the stored
cohort and app-path results before the threshold was set: the worst good
result disagrees by 6.0%, the one gross error found by 38%.
"""

import math
from typing import Optional

#: ``|ln(engine / hip-x)|`` above which the result is flagged, about a 15%
#: disagreement either way: past 2x the worst good result, under half the
#: gross one.
MAX_VELOCITY_ABS_LOG = 0.14


def velocity_disagreement(
    engine_velocity_ms: Optional[float], hipx_velocity_ms: Optional[float]
) -> Optional[float]:
    """``|ln(engine / hip-x)|``, or None when either value is missing or not
    positive."""
    if engine_velocity_ms is None or hipx_velocity_ms is None:
        return None
    e, h = float(engine_velocity_ms), float(hipx_velocity_ms)
    if not (math.isfinite(e) and math.isfinite(h)) or e <= 0 or h <= 0:
        return None
    return abs(math.log(e / h))


def is_implausible(
    engine_velocity_ms: Optional[float], hipx_velocity_ms: Optional[float]
) -> bool:
    """True when the two velocities disagree past ``MAX_VELOCITY_ABS_LOG``.

    A missing reference is not a failure: the check abstains rather than
    flagging a clip it cannot judge.
    """
    d = velocity_disagreement(engine_velocity_ms, hipx_velocity_ms)
    return d is not None and d > MAX_VELOCITY_ABS_LOG
