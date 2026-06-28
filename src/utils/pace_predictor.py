"""Predict equivalent race paces and a runner profile from personal bests.

This is the *predictive* counterpart to :mod:`src.utils.pace_estimator` — the
estimator measures velocity from the video (hip-x trajectory, the source of
truth), whereas this module turns one or two entered PBs into *suggested*
target paces for the upload clips (threshold, 1500 m, 800 m) plus a coarse
speed/endurance profile.

Model
-----
Equivalent paces use Jack Daniels' VDOT framework:

* ``%VO2max`` as a function of race duration *t* (minutes)::

      pct(t) = 0.8 + 0.1894393·e^(−0.012778·t) + 0.2989558·e^(−0.1932605·t)

* the oxygen cost of running at velocity *v* (m/min)::

      VO2(v) = −4.60 + 0.182258·v + 0.000104·v²

  so ``VDOT = VO2(v) / pct(t)`` for a race covering a known distance in time
  *t* at average velocity *v*.

Threshold (lactate-threshold) pace is taken at ``PCT_THRESHOLD ≈ 0.86`` of
VO2max — Daniels' "T pace" intensity (cross-checked: VDOT 70 → ~3:18/km).

With **two** PBs the runner's own Riegel exponent
``e = ln(t₂/t₁) / ln(d₂/d₁)`` is fitted and used to interpolate the 800 m /
1500 m equivalents (the exponent itself is the speed↔endurance profile). With
**one** PB the population VDOT model is used and no profile is inferred.

Accuracy caveat
---------------
Extrapolating an aerobic *threshold* pace from anaerobic middle-distance PBs
(800/1500) is the least reliable case; confidence is reported accordingly and
is only ``High`` when at least one PB is ≥ 3000 m. These are suggestions, never
ground truth — the measured hip velocity remains authoritative.
"""

from __future__ import annotations

import math
from dataclasses import dataclass, field
from typing import Dict, List, Optional, Tuple

# VO2(v) polynomial coefficients (Daniels & Gilbert), v in m/min.
_VO2_A = 0.000104
_VO2_B = 0.182258
_VO2_C = -4.60

# pct(t) coefficients, t in minutes.
_PCT_BASE = 0.8
_PCT_K1, _PCT_E1 = 0.1894393, -0.012778
_PCT_K2, _PCT_E2 = 0.2989558, -0.1932605

# Lactate-threshold intensity as a fraction of VO2max (Daniels' "T pace").
PCT_THRESHOLD = 0.86

# Population Riegel exponent (used only as context; the fitted value drives
# two-PB predictions).
RIEGEL_DEFAULT_EXP = 1.06

# Profile bands around the population exponent. Larger exponent ⇒ times rise
# faster with distance ⇒ relatively stronger at short distances (speed type).
_PROFILE_ENDURANCE_MAX = 1.04
_PROFILE_SPEED_MIN = 1.08

# Distance (m) at/above which a PB makes the aerobic threshold estimate
# trustworthy enough to call "High" confidence.
_LONG_EVENT_M = 3000


@dataclass
class PacePrediction:
    """Result of :func:`predict_paces`.

    Attributes:
        vdot: Representative VDOT (the best of the entered performances).
        profile: ``"Speed-leaning"`` / ``"Balanced"`` / ``"Endurance-leaning"``
            when two PBs are given, else ``None``.
        riegel_exponent: Fitted exponent when two PBs are given, else ``None``.
        confidence: ``"Low"`` / ``"Medium"`` / ``"High"``.
        clip_paces: Suggested pace per km (seconds) keyed by the upload clip
            labels ``"Threshold"``, ``"1500m"``, ``"800m"``.
        race_times: Predicted race times (seconds) keyed ``"800m"``/``"1500m"``.
    """

    vdot: float
    profile: Optional[str]
    riegel_exponent: Optional[float]
    confidence: str
    clip_paces: Dict[str, float] = field(default_factory=dict)
    race_times: Dict[str, float] = field(default_factory=dict)


# ── Time parsing / formatting ────────────────────────────────────────


def parse_time_to_seconds(text: str) -> Optional[float]:
    """Parse a race time into seconds.

    Accepts ``"58.5"`` (seconds), ``"1:58.5"`` (min:sec), ``"14:30"``
    (min:sec) and ``"1:02:30"`` (h:min:sec). Returns ``None`` if the string
    is empty or malformed.

    Args:
        text: User-entered time string.

    Returns:
        Time in seconds, or ``None`` if it cannot be parsed.
    """
    text = (text or "").strip()
    if not text:
        return None
    try:
        parts = [float(p) for p in text.split(":")]
    except ValueError:
        return None
    if any(p < 0 for p in parts):
        return None
    if len(parts) == 1:
        return parts[0]
    if len(parts) == 2:
        return parts[0] * 60 + parts[1]
    if len(parts) == 3:
        return parts[0] * 3600 + parts[1] * 60 + parts[2]
    return None


def format_clock(seconds: float) -> str:
    """Format a race time as ``m:ss.s`` (or ``s.s`` under a minute)."""
    if seconds < 60:
        return f"{seconds:.1f}s"
    minutes = int(seconds // 60)
    rem = seconds - minutes * 60
    return f"{minutes}:{rem:04.1f}"


def format_pace_per_km(seconds_per_km: float) -> str:
    """Format a pace as ``m:ss`` per km."""
    minutes = int(seconds_per_km // 60)
    rem = int(round(seconds_per_km - minutes * 60))
    if rem == 60:
        minutes, rem = minutes + 1, 0
    return f"{minutes}:{rem:02d}/km"


# ── VDOT core ────────────────────────────────────────────────────────


def _vo2_for_velocity(v_m_per_min: float) -> float:
    """Oxygen cost (ml/kg/min) of running at ``v`` m/min."""
    return _VO2_C + _VO2_B * v_m_per_min + _VO2_A * v_m_per_min**2


def _velocity_for_vo2(vo2: float) -> float:
    """Invert :func:`_vo2_for_velocity` — positive root of the quadratic."""
    a, b, c = _VO2_A, _VO2_B, _VO2_C - vo2
    disc = b * b - 4 * a * c
    return (-b + math.sqrt(disc)) / (2 * a)


def _pct_max_for_duration(t_min: float) -> float:
    """Fraction of VO2max sustainable for a race lasting ``t`` minutes."""
    return (
        _PCT_BASE
        + _PCT_K1 * math.exp(_PCT_E1 * t_min)
        + _PCT_K2 * math.exp(_PCT_E2 * t_min)
    )


def vdot_from_performance(distance_m: float, time_s: float) -> float:
    """Compute VDOT from a single race performance.

    Args:
        distance_m: Race distance in metres.
        time_s: Race time in seconds.

    Returns:
        VDOT (ml/kg/min, intensity-adjusted).

    Raises:
        ValueError: if distance or time is non-positive.
    """
    if distance_m <= 0 or time_s <= 0:
        raise ValueError("distance_m and time_s must be positive")
    t_min = time_s / 60.0
    v = distance_m / t_min
    return _vo2_for_velocity(v) / _pct_max_for_duration(t_min)


def predict_time(vdot: float, distance_m: float) -> float:
    """Predict the race time (s) at ``distance_m`` for a given VDOT.

    Solves ``VO2(v) = VDOT · pct(distance/v)`` by fixed-point iteration
    (``pct`` depends on the unknown duration, so velocity and time are
    co-determined).

    Args:
        vdot: VDOT value.
        distance_m: Target race distance in metres.

    Returns:
        Predicted race time in seconds.
    """
    t_min = distance_m / 350.0  # seed at ~350 m/min
    for _ in range(50):
        target_vo2 = vdot * _pct_max_for_duration(t_min)
        v = _velocity_for_vo2(target_vo2)
        new_t = distance_m / v
        if abs(new_t - t_min) < 1e-7:
            t_min = new_t
            break
        t_min = new_t
    return t_min * 60.0


def threshold_velocity(vdot: float) -> float:
    """Lactate-threshold velocity (m/min) for a given VDOT."""
    return _velocity_for_vo2(vdot * PCT_THRESHOLD)


def riegel_exponent(d1: float, t1: float, d2: float, t2: float) -> float:
    """Fit the Riegel fatigue exponent through two performances."""
    return math.log(t2 / t1) / math.log(d2 / d1)


def classify_profile(exponent: float) -> str:
    """Map a Riegel exponent to a coarse speed/endurance label."""
    if exponent <= _PROFILE_ENDURANCE_MAX:
        return "Endurance-leaning"
    if exponent >= _PROFILE_SPEED_MIN:
        return "Speed-leaning"
    return "Balanced"


def _pace_per_km(distance_m: float, time_s: float) -> float:
    """Seconds per km for a race of ``distance_m`` covered in ``time_s``."""
    return time_s / (distance_m / 1000.0)


# ── Top-level prediction ─────────────────────────────────────────────


def predict_paces(pbs: List[Tuple[float, float]]) -> Optional[PacePrediction]:
    """Predict clip paces + profile from one or two PBs.

    Args:
        pbs: List of ``(distance_m, time_s)`` tuples (1 or 2 entries). Entries
            with non-positive values are ignored.

    Returns:
        A :class:`PacePrediction`, or ``None`` if no usable PB was supplied.
    """
    clean = [(d, t) for d, t in pbs if d > 0 and t > 0]
    if not clean:
        return None
    clean.sort(key=lambda dt: dt[0])  # shortest distance first
    vdots = [vdot_from_performance(d, t) for d, t in clean]
    max_dist = max(d for d, _ in clean)

    if len(clean) >= 2:
        (d1, t1), (d2, t2) = clean[0], clean[-1]
        exponent = riegel_exponent(d1, t1, d2, t2)
        profile: Optional[str] = classify_profile(exponent)
        # Personalised Riegel through both points for the race equivalents.
        race_800 = t1 * (800.0 / d1) ** exponent
        race_1500 = t1 * (1500.0 / d1) ** exponent
        # Distance-weighted VDOT for the aerobic threshold (lean on the longer,
        # more aerobic performance).
        w = sum(d for d, _ in clean)
        vdot_thr = sum(v * d for v, (d, _) in zip(vdots, clean)) / w
        confidence = "High" if max_dist >= _LONG_EVENT_M else "Medium"
    else:
        exponent = None
        profile = None
        vdot = vdots[0]
        race_800 = predict_time(vdot, 800.0)
        race_1500 = predict_time(vdot, 1500.0)
        vdot_thr = vdot
        confidence = "Low"

    thr_v = threshold_velocity(vdot_thr)  # m/min
    thr_pace_per_km = 60000.0 / thr_v  # seconds per km

    return PacePrediction(
        vdot=max(vdots),
        profile=profile,
        riegel_exponent=exponent,
        confidence=confidence,
        clip_paces={
            "Threshold": thr_pace_per_km,
            "1500m": _pace_per_km(1500.0, race_1500),
            "800m": _pace_per_km(800.0, race_800),
        },
        race_times={"800m": race_800, "1500m": race_1500},
    )
