"""Elite reference ranges for the Results page's "Elite Range Comparison".

The bands compared against on the Results page. Their provenance -- 16
primary sources, the synthesis behind every value, the citability
constraints and the known gaps -- is recorded in the project's research
notes, which also carry the resulting tables and the per-tier confidence
notes.

These are third-party literature values presented as modelled estimates,
never as a target, a norm or a diagnosis.

How a band is built (rebuilt 2026-09-06)
----------------------------------------

This module used to hold hand-built lookup tables copied from the sources,
and their *widths* were whatever each source happened to print. That made the
narrowest band in the whole table (male 1500m duty factor, `0.245-0.250`,
0.005 wide) rest on a standard ERROR over n=8 digitised finalists -- the
precision of a cohort average, not the spread between athletes -- so it was
narrower than +/- 1 SD about a single mean while two elite men at a matched
speed sit 0.047 apart. Bands are now *derived* rather than transcribed, under
one stated rule:

1. **Centre** on the continuous speed model of research doc sections 6.1-6.3,
   evaluated at the clip's own measured speed where one is available.
2. **Half-width** is `2 x SD`, where SD is the between-athlete spread at a
   fixed speed -- about 95% of genuine elites inside, so "outside range"
   means genuinely unusual rather than merely non-average.
3. **SD is the median of every matched-speed estimate the source set
   supports** for that metric (see `_SD_EVIDENCE`). The median is doing real
   work: it is what lets the ambiguous S2 contact-time `+/- 22` be included
   as one estimate among four rather than argued about, since reading it as a
   standard error implies a 62 ms between-athlete SD, six to ten times every
   other elite estimate.
4. **Speed basis.** A measured speed inside the model's validated 12-28 km/h
   range gives a point band. Otherwise the declared tier's speed span is
   used and the band spans `[ref(v_lo) - 2SD, ref(v_hi) + 2SD]`.
5. **Floor.** No band narrower than `_MIN_RELATIVE_WIDTH`; a modelled
   (extrapolated) cell no narrower than the widest measured band for that
   same metric.

**S2's 0.93 contact-time adjustment is undone here**, because Strideo is
exactly the raw video-based detector the research doc's section 2 warns the
adjustment reads about 7% low against. The adjustment is redistributive, so
step time is preserved and only the contact/flight split moves. Two
independent checks back this rather than one:

- Our own 45-clip cohort means 0.261 duty factor at 1500m pace against the
  unadjusted reference 0.263, where the published 0.245 put only 3 of 15
  clips in band against 10/15 and 9/15 for the other two tiers.
- It repairs the section 6.3 regime join. The two regimes come from
  different cohorts and meet at 22-24 km/h; undoing the adjustment cuts the
  contact-time discontinuity there from about 9% to about 2%, and the duty
  factor discontinuity from 6.5% to 1.1%. Neither cohort was consulted about
  the other, so agreeing better after the undo is evidence the adjustment
  was the thing in the way.

Design decisions carried forward, and where the rebuild revisits them:

- **Tier lookup is now the fallback, not the primary path.** The original
  design chose discrete tier lookup over the continuous model on simplicity
  and extrapolation-clamping grounds. That is revisited here with evidence
  it did not have: every metric varies with actual speed while the band was
  selected by *declared* tier, and our cohort's pace error is entirely
  one-directional (0 of 45 clips undershoot, +3% to +34% over). Since
  duty factor falls with speed, overshooting flattered the runner. The
  clamping concern was real and is handled by falling back to the tier span
  outside 12-28 km/h rather than by avoiding the model.
- **Vertical oscillation is shown at every tier**, flagged `modeled` above
  22 km/h: no whole-cycle VO measurement exists above that speed in any
  population, so those bands are extrapolated and must say so.
- **Duty factor is included** despite not being one of Strideo's original
  four headline metrics -- the research doc's finding #8: cadence, GCT and
  stride length did not significantly separate elite from recreational
  runners in two of its core studies, while duty factor did.
- **No band for sex outside Male/Female, or an unrecognized pace tier.**
  Returns `None`; callers must degrade by omitting the section, not by
  substituting a default.
"""

from dataclasses import dataclass
from statistics import median
from typing import Dict, List, Optional, Tuple

# --------------------------------------------------------------------------
# The speed model (research doc sections 6.1-6.3)
# --------------------------------------------------------------------------

# S2's Table 3 footnote: "Contact and flight times have been adjusted by a
# factor of 0.93". Redistributive, so step time is preserved when undone.
_S2_ADJUSTMENT = 0.93

# Model validity, section 6.3's extrapolation limits. Outside this, fall back
# to the declared tier rather than extrapolating a regression past its cohort.
_MODEL_MIN_KMH = 12.0
_MODEL_MAX_KMH = 28.0

# Section 6.3 joins the two cohorts with a linear blend over this window.
_BLEND_LO_KMH = 22.0
_BLEND_HI_KMH = 24.0


def _regime_a(v_kmh: float) -> Dict[str, float]:
    """Sub-race pace, 12-22 km/h (S3, elite male). `d = v - 14`, v in km/h."""
    d = v_kmh - 14.0
    contact_ms = 230.0 - 6.0 * d
    flight_ms = 147.0 + 2.0 * d
    step_length_m = 1.45 + 0.07 * d
    return {
        "cadence_spm": 60.0 * (2.66 + 0.04 * d),
        "gct_ms": contact_ms,
        "flight_ms": flight_ms,
        "stride_length_m": 2.0 * step_length_m,
        "duty_factor": 0.304 - 0.005 * d,
    }


def _regime_b(v_kmh: float) -> Dict[str, float]:
    """Race pace, 23-28 km/h (S2, elite male), with the 0.93 undo applied.

    S2 publishes adjusted contact and flight times whose sum reproduces the
    reported step frequency, so the adjustment moved time between the two
    rather than scaling both. Undoing it therefore holds step time fixed and
    re-splits it: contact rises to what a raw video detector would return,
    flight takes the remainder, and duty factor follows from the pair rather
    than from S2's own (non-significant) duty-factor regression.
    """
    v_ms = v_kmh / 3.6
    u = v_ms - 7.21
    contact_pub = 147.0 - 23.0 * u
    flight_pub = 153.0 - 18.0 * u
    step_time_ms = contact_pub + flight_pub

    contact_ms = contact_pub / _S2_ADJUSTMENT
    flight_ms = step_time_ms - contact_ms
    return {
        "cadence_spm": 60.0 * v_ms / 2.16,
        "gct_ms": contact_ms,
        "flight_ms": flight_ms,
        # Step length is speed-invariant in S2 (slope 0.01 +/- 0.09, n.s.).
        "stride_length_m": 2.0 * 2.16,
        "duty_factor": contact_ms / (2.0 * step_time_ms),
    }


# Whole-cycle VO as a fraction of standing height, section 7.3. Anchored on
# V1's measurement at 12 km/h, re-levelled flat below 15 km/h per V2's
# digitised curve, extrapolated (and so `modeled`) above 22.
_VO_K_BY_KMH: List[Tuple[float, float]] = [
    (12.0, 0.050),
    (15.0, 0.050),
    (16.0, 0.049),
    (18.0, 0.047),
    (20.0, 0.046),
    (22.0, 0.044),
    (24.0, 0.043),
    (26.0, 0.038),
    (28.0, 0.032),
]

# Above this speed no whole-cycle VO measurement exists in any population.
_VO_MEASURED_MAX_KMH = 22.0

# Default statures when the runner did not give one, matching what the
# original tables were quoted at.
_DEFAULT_HEIGHT_CM = {"Male": 180.0, "Female": 170.0}


def _vo_k(v_kmh: float) -> float:
    """Linear interpolation over the section 7.3 height-fraction table."""
    pts = _VO_K_BY_KMH
    if v_kmh <= pts[0][0]:
        return pts[0][1]
    if v_kmh >= pts[-1][0]:
        return pts[-1][1]
    for (v0, k0), (v1, k1) in zip(pts, pts[1:]):
        if v0 <= v_kmh <= v1:
            return k0 + (k1 - k0) * (v_kmh - v0) / (v1 - v0)
    return pts[-1][1]


# Section 9's reconciliation: one speed-based model for the shape, with
# offsets on the intercept.
#
# **`gct_ms` was added only because the 0.93 undo made it
# derivable.** Section 9 refuses a contact-time sex offset in its own text --
# "the true GCT sex difference could be anywhere from -3% to -10%. Don't build
# on it" -- and names the reason: S1's women are unadjusted while S2's men are
# 0.93-adjusted, so the only comparison available is confounded by exactly the
# adjustment this module undoes. With the men's values undone the two
# speed-matched comparisons agree at -10.0% and -9.6%, landing
# on section 9's own stated lower bound. Same evidential standard as the
# flight-time 0.87 offset that section already ships, which rests on the same
# two comparisons.
#
# Duty factor still takes NO sex adjustment, and that is a deliberate keep
# rather than an oversight: the same two comparisons give -2.6% and +1.8%,
# inconsistent in sign, matching section 9's "S1 0.26-0.27 against S2 0.245 --
# overlapping". Note this leaves the female model not internally closed, since
# a 10% shorter contact against an 8% shorter stride time algebraically implies
# a duty factor about 3% below the male one. That is a property of section 9's
# design, which fits each offset to its own direct comparison rather than
# deriving one metric from the others, and it predates this rebuild (female cadence was
# already offset while duty factor was not). Following the measurements over
# the algebra is the right call while the measurements disagree with it.
_SEX_MULTIPLIERS = {
    "cadence_spm": 1.08,
    "stride_length_m": 0.90,
    "gct_ms": 0.90,
}
_FEMALE_VO_OFFSET_CM = -0.6


def male_reference(v_kmh: float) -> Dict[str, float]:
    """The elite male point reference at `v_kmh`, blended across regimes."""
    if v_kmh <= _BLEND_LO_KMH:
        return _regime_a(v_kmh)
    if v_kmh >= _BLEND_HI_KMH:
        return _regime_b(v_kmh)
    a, b = _regime_a(v_kmh), _regime_b(v_kmh)
    w = (v_kmh - _BLEND_LO_KMH) / (_BLEND_HI_KMH - _BLEND_LO_KMH)
    return {k: (1.0 - w) * a[k] + w * b[k] for k in a}


def reference_at(v_kmh: float, sex: str, height_cm: float) -> Dict[str, float]:
    """Point reference for one sex at one speed, including oscillation."""
    ref = male_reference(v_kmh)
    if sex == "Female":
        for key, mult in _SEX_MULTIPLIERS.items():
            ref[key] = ref[key] * mult
    vo_cm = _vo_k(v_kmh) * height_cm
    if sex == "Female":
        vo_cm += _FEMALE_VO_OFFSET_CM
    ref["oscillation_cm"] = vo_cm
    return ref


# --------------------------------------------------------------------------
# Between-athlete spread
# --------------------------------------------------------------------------

# Every matched-speed between-athlete SD estimate the source set supports,
# per metric. The rule takes the MEDIAN of each list, which is what makes the
# choice mechanical rather than a judgement call -- see the module docstring.
#
#   S2 publishes mean +/- SE over n=8 digitised finalists, so SE * sqrt(8).
#   S1 publishes mean +/- SD directly, per lap.
#   S5 gives individual athletes at one matched speed, so their sample SD.
_SD_EVIDENCE: Dict[str, List[Tuple[float, str]]] = {
    "cadence_spm": [
        (5.09, "S2 mean +/- SE 3.34 +/- 0.03 Hz, n=8 -> SD 0.085 Hz"),
        (7.80, "S1 lap 1 SD 0.13 Hz"),
        (10.20, "S1 lap 2 SD 0.17 Hz"),
        (18.23, "S5 three elite men at ~30 km/h: 242 / 229 / 206 spm"),
    ],
    "gct_ms": [
        (6.00, "S1 lap 1 SD"),
        (11.00, "S1 lap 2 SD"),
        (13.01, "S5 three elite men at ~30 km/h: 112 / 138 / 126 ms"),
        (62.23, "S2 +/- 22 read as SE over n=8; implausible, kept as one vote"),
    ],
    "stride_length_m": [
        (0.1131, "S2 step +/- 0.02 m SE, n=8 -> SD 0.057 m, doubled to stride"),
        (0.1400, "S1 lap 2 step SD 0.07 m, doubled"),
        (0.1800, "S1 lap 1 step SD 0.09 m, doubled"),
        (0.3022, "S5 three elite men at ~30 km/h: 2.13 / 2.31 / 2.43 m step"),
    ],
    "duty_factor": [
        (0.00566, "S2 mean +/- SE 0.245 +/- 0.002, n=8"),
        (0.01000, "S1 lap 1 SD"),
        (0.01000, "S1 lap 2 SD"),
        (0.02380, "S5 three elite men at ~30 km/h: 0.233 / 0.263 / 0.216"),
    ],
}

# Oscillation is the one metric whose spread scales with the runner, so it is
# a fraction of stature rather than an absolute. V1's SD is 0.008 of height
# (n=97); V2's per-speed SDs of 1.0-1.9 cm on trained males bracket the same
# 1.44 cm that fraction gives at 1.80 m, so two independent cohorts agree.
_VO_SD_HEIGHT_FRACTION = 0.008

_SD: Dict[str, float] = {k: median(v for v, _ in ev) for k, ev in _SD_EVIDENCE.items()}

# About 95% of genuine elites inside the band, so "outside range" carries
# meaning against a binary in/above/below verdict.
_SD_MULTIPLE = 2.0

# The floor, in relative width (band width / band midpoint). 0.101 is the
# median relative width of the original measured bands.
_MIN_RELATIVE_WIDTH = 0.101


def sd_for(metric_key: str, height_cm: float) -> float:
    """Between-athlete SD at a fixed speed, for one metric."""
    if metric_key == "oscillation_cm":
        return _VO_SD_HEIGHT_FRACTION * height_cm
    return _SD[metric_key]


# --------------------------------------------------------------------------
# Bands
# --------------------------------------------------------------------------


@dataclass(frozen=True)
class EliteRange:
    """One metric's elite reference band for a given runner and speed."""

    lo: float
    hi: float
    unit: str
    modeled: bool = False  # True: extrapolated, not directly measured
    basis: str = "tier"  # "speed" when built at the clip's measured speed


# metric key matches Strideo's own `metrics` dict keys, so callers can look
# up `m[key]` directly against the returned range.
METRICS = [
    ("cadence_spm", "Cadence"),
    ("gct_ms", "Ground Contact Time"),
    ("stride_length_m", "Stride Length"),
    ("oscillation_cm", "Vertical Oscillation"),
    ("duty_factor", "Duty Factor"),
]

_UNITS = {
    "cadence_spm": "spm",
    "gct_ms": "ms",
    "stride_length_m": "m",
    "oscillation_cm": "cm",
    "duty_factor": "",
}

# Section 4's tier -> speed bridge, sex-specific because a woman's 800 m race
# pace is roughly a man's 1500 m pace. Used for the fallback path, and for the
# published tables; the measured-speed path makes it moot, which section 9
# calls "the larger effect by far".
TIER_SPEED_KMH: Dict[str, Dict[str, Tuple[float, float]]] = {
    "threshold": {"Male": (17.0, 20.0), "Female": (15.5, 18.0)},
    "1500m": {"Male": (23.5, 26.5), "Female": (21.5, 23.5)},
    "800m": {"Male": (26.0, 28.5), "Female": (23.0, 25.5)},
}

TIER_LABELS = {"threshold": "threshold", "1500m": "1500 m", "800m": "800 m"}


def _round_for(metric_key: str, value: float) -> float:
    if metric_key == "duty_factor":
        return round(value, 3)
    if metric_key in ("stride_length_m", "oscillation_cm"):
        return round(value, 2)
    return round(value, 1)


def _build_band(
    metric_key: str,
    v_lo: float,
    v_hi: float,
    sex: str,
    height_cm: float,
    basis: str,
) -> EliteRange:
    """`[ref(v_lo) - k*SD, ref(v_hi) + k*SD]`, then floored."""
    lo_ref = reference_at(v_lo, sex, height_cm)[metric_key]
    hi_ref = reference_at(v_hi, sex, height_cm)[metric_key]
    # Duty factor and GCT fall with speed, so the tier's slow edge is the
    # band's high edge. Order by value rather than by speed.
    lo_ref, hi_ref = min(lo_ref, hi_ref), max(lo_ref, hi_ref)

    half = _SD_MULTIPLE * sd_for(metric_key, height_cm)
    lo, hi = lo_ref - half, hi_ref + half

    mid = (lo + hi) / 2.0
    if mid > 0 and (hi - lo) / mid < _MIN_RELATIVE_WIDTH:
        half_floor = mid * _MIN_RELATIVE_WIDTH / 2.0
        lo, hi = mid - half_floor, mid + half_floor

    return EliteRange(
        lo=_round_for(metric_key, lo),
        hi=_round_for(metric_key, hi),
        unit=_UNITS[metric_key],
        modeled=(metric_key == "oscillation_cm" and v_hi > _VO_MEASURED_MAX_KMH),
        basis=basis,
    )


def _tier_table(pace_level: str, sex: str, height_cm: float) -> Dict[str, EliteRange]:
    v_lo, v_hi = TIER_SPEED_KMH[pace_level][sex]
    return {
        key: _build_band(key, v_lo, v_hi, sex, height_cm, "tier") for key, _ in METRICS
    }


# The published tables, derived from the model over each tier's speed span at
# the default statures. Computed rather than transcribed so that this module,
# its documentation and the research it is drawn from cannot drift apart --
# the drift defect that documentation was written against.
_TIERS: Dict[str, Dict[str, Dict[str, EliteRange]]] = {
    tier: {
        sex: _tier_table(tier, sex, _DEFAULT_HEIGHT_CM[sex])
        for sex in ("Male", "Female")
    }
    for tier in TIER_SPEED_KMH
}


def get_elite_ranges(
    pace_level: Optional[str],
    sex: Optional[str],
    velocity_kmh: Optional[float] = None,
    height_cm: Optional[float] = None,
) -> Optional[Dict[str, EliteRange]]:
    """Return this clip's elite reference bands, or None if not available.

    When `velocity_kmh` is inside the model's validated 12-28 km/h range, the
    bands are built at that speed and each returned range carries
    `basis="speed"`. Otherwise they come from the declared tier's speed span
    with `basis="tier"`, which is wider because it must cover every speed a
    runner declaring that tier might actually have run.

    Returns None (rather than a default band) when `sex` isn't "Male"/"Female"
    or `pace_level` isn't one of the three tiers Strideo collects at Upload --
    there is no validated reference for those cases, and showing one anyway
    would misrepresent the source material.
    """
    if pace_level not in TIER_SPEED_KMH or sex not in ("Male", "Female"):
        return None

    stature = height_cm if height_cm else _DEFAULT_HEIGHT_CM[sex]

    if velocity_kmh is not None and _MODEL_MIN_KMH <= velocity_kmh <= _MODEL_MAX_KMH:
        return {
            key: _build_band(key, velocity_kmh, velocity_kmh, sex, stature, "speed")
            for key, _ in METRICS
        }

    if height_cm:
        return _tier_table(pace_level, sex, stature)
    return _TIERS[pace_level][sex]
