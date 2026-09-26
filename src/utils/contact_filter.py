"""GT-free per-contact filters: presence window + within-clip duration outlier.

One implementation serves two callers: the engine, whose cadence takes intervals
only between contacts this filter keeps, and the validation scoring, which
scores accuracy on the same set. Keeping a single copy is deliberate; two copies
of an earlier gate drifted apart and flattered a published figure. Accessors
read both ``contacts.csv`` rows (dicts) and :class:`GroundContact` objects.

**Experiments that override a constant or helper must patch this module.** A
caller that imports a name has its own binding, so rebinding it there does not
reach the code that reads it.
"""

from __future__ import annotations

import statistics
from typing import Any, Dict, List, Optional, Sequence, Tuple

import numpy as np

# -- The two GT-free per-contact filters ---------------------------------------
#
# These replace an entry/mid/exit thirds visibility gate. It had been retired
# from the production path but not from the scoring path, and its apparent
# accuracy benefit turned out to be luck in where the thirds' boundaries fell
# rather than signal, so the figure it produced was flattered (5.81 ms gated,
# 6.73 ms ungated).
#
# Both filters are GT-free by construction, which is what lets the engine and
# the scoring share one implementation instead of drifting apart the way the
# production and scoring gates did.

#: Floor on the duration scale, as a fraction of the clip's own median GCT.
#:
#: **Measured from the cohort, which is an experiment rather than a fit.** S1 is
#: a balanced replicated design -- 5 runners x 3 target speeds x 3 repeats, 9
#: clips per runner, 15 per condition -- so the median across its 45 clips of
#: (clip MAD / clip median) is a real estimate of the typical within-clip
#: spread, not a threshold tuned against the number this file then reports.
#: Those are different estimands: this calibrates *dispersion*, the reported
#: outcome is *error against Kinovea GT*. Estimating a dispersion parameter
#: from data and using it to bound outliers is what a control chart does.
#:
#: Measured value 2.835% (IQR 1.44-4.60%). It beats the clip's own MAD on 22
#: of 45 clips, which is the point of it: a clip carries only 6-9 contacts,
#: far too few for a stable MAD (Godin et al. put contact-time variability at
#: ~350 steps for reliability), so the floor supplies the stability the clip
#: cannot.
#:
#: Expressed as a fraction of the clip's median rather than as a frame period,
#: which makes the filter **fps-independent** -- deliberately, since a fixed
#: *time* constant silently changes meaning when the frame rate does.
DURATION_MAD_FLOOR_FRACTION = 0.02835

#: Ceiling on the duration scale, also as a fraction of the clip's own median.
#:
#: **The floor stops a clip under-estimating its spread; this stops a clip
#: over-estimating it, which is the more dangerous direction.** A clip that
#: contains defects has an inflated MAD, which raises the very scale meant to
#: catch them -- self-masking. Measured instance: ``S1_02_800m_1`` holds two
#: truncated contacts, its MAD reaches 7.74% of its median (the cohort maximum,
#: 2.7x the anchor), and its own bar rises to 60 ms so neither is caught.
#:
#: Set to the **90th percentile of the cohort's own MAD/median distribution**
#: (P50 2.835%, P75 4.550%, P90 5.902%, P95 6.092%, max 7.744%), so it
#: overrides only the top decile -- the clips whose 6-9-contact MAD estimate is
#: likelier defect-inflated than real -- and leaves genuine adaptation intact
#: everywhere else. **Not load-bearing:** P75, P90 and P95 give identical
#: results on S1, because the gap between a defect-inflated MAD and a genuinely
#: variable clip is wide.
#:
#: Why a cap rather than dropping the clip term entirely: a cohort-only scale
#: scores better (4.185 vs 4.280 ms) but makes a **false drop** --
#: ``S1_03_threshold_2``'s contact 8, which the engine measured to +3.3 ms
#: against Kinovea. The failure directions are asymmetric. A missed defect
#: leaves the reported figure worse than reality; a false drop makes it
#: *better* than reality, which is the mechanism that made 5.81 ms flattered in
#: the first place. Capped scoring drops contacts averaging 86.5 ms of GT error
#: against cohort-only's 63.7 ms, i.e. it stays inside clearly-defective
#: territory.
DURATION_MAD_CAP_FRACTION = 0.05902

#: The standard-normal MAD, so ``dev / (_MAD_TO_SIGMA_INV * scale)`` puts a
#: deviation in sigma-equivalent units. Kept so ``DURATION_MAD_K`` below can be
#: read as "sigmas", which is what makes 3.5 a citable cut-off rather than a
#: number of this project's own devising.
_MAD_TO_SIGMA_INV = 1.4826

#: Cut-off in sigma-equivalent units, applied to the scale above.
#:
#: 3.5 is Iglewicz & Hoaglin's published modified z-score cut-off. **What went
#: wrong the first time was the scale, not this constant** -- paired with a
#: one-frame floor it produced an effective bar of 86.5 ms, ~13x the cohort's
#: own measured spread. With the scale anchored to the cohort's actual
#: dispersion it corresponds to ~15% of a clip's median, which is where the
#: measurement genuinely degrades.
#:
#: **Placed against measured accuracy, not against a gap in the deviation
#: distribution.** Binning every GT-matched in-window contact by scale units
#: and reading its error against Kinovea: 0-2 units 7.4 ms mean error (n=281,
#: the engine's baseline), 2-3 12.0, 3-4 14.4, 4-5 **37.5**, 8-13 56.6, 13+
#: 191.5. Degradation sets in around 2-3 units (z ~2.0-2.7) and is
#: unambiguous by 4 (z ~2.7). 3.5 sigma sits just above that onset. An
#: earlier revision used 15 raw MAD units on the grounds that it sat in a wide
#: gap in the *deviation* distribution; that was the wrong distribution to
#: look at -- a gap says nothing about whether the contacts below it are well
#: measured, and there it left four contacts with 14-57 ms errors in the
#: scored set.
#:
#: GT is used to *validate* this placement, never at runtime: the filter reads
#: only durations and the clip's own median, so it stays deployable on an
#: unannotated clip.
DURATION_MAD_K = 3.5

#: Cut-off for the clip's **first and last** in-window contact.
#:
#: **Why a separate constant rather than a lower uniform one, which is the
#: obvious alternative and was measured.** The two populations interleave. A
#: well-measured *interior* contact sits at z=3.41 (``S1_05_800m_3`` 3/8, GT
#: error 7.1 ms) while the two terminal defects this exists to catch sit at
#: 3.34 and 3.24 (``S1_05_threshold_1`` 9/9, 34.1 ms; ``S1_02_800m_3`` 1/7,
#: 37.2 ms). So **no uniform cut-off reaches those defects without dropping the
#: good contact first** -- measured, not argued: flat 3.0 and flat 3.2 both
#: score 3.941 ms with one false drop, where this scores 3.991 with none.
#: Position is the discriminating variable, and of the five defects anywhere
#: near the threshold, all five are terminal.
#:
#: **The mechanism was established before this constant, not fitted to it.**
#: ``compute_presence_quality``'s window opens when a hip and an ankle first
#: become trackable, not when the step began, so a contact already underway at
#: onset has its IC found late and its duration truncated -- and the mirror
#: case at the exit. Series-terminal attrition was found on 18 of 18 clips
#: examined, and first contacts carry -4.47 ms GT bias against -3.67 mid-clip
#: and +0.23 last.
#:
#: **Anchoring, stated with its limit.** The value sits on a plateau: 2.90-3.20
#: give identical drops, catches and MAE (11 / 5 true / 0 false / 3.991 ms), so
#: it was not selected by its result. The band is bounded by the lowest
#: catchable defect (z=3.24) above and the highest well-measured terminal
#: contact (z=2.89) below -- **three contacts define it**, which is thin, and
#: the exact value inside the band therefore carries much less evidence than
#: the split itself does. Re-measure when a second cohort widens the data.
DURATION_TERMINAL_K = 3.0


# -- field accessors: contacts.csv rows (dicts) and GroundContact alike -------


def _get(c: Any, name: str) -> Any:
    return c.get(name) if isinstance(c, dict) else getattr(c, name, None)


def _ic(c: Any) -> float:
    """Initial contact frame: refined ``contact_frame``, else the fallback.

    For a CSV row or GT dict the fallback is ``f_ic`` (GT); for a
    :class:`GroundContact` it is the coarse ``frame``, as ``calculate_cadence``
    has always used.
    """
    if isinstance(c, dict):
        return float(c.get("contact_frame") or c.get("f_ic") or 0)
    return float(c.contact_frame if c.contact_frame is not None else c.frame)


def _to_f(c: Any) -> float:
    if isinstance(c, dict):
        return float(c.get("toe_off_frame") or c.get("f_to") or 0)
    return float(c.toe_off_frame or 0)


def _duration_scale_ms(durations: List[float], median: float) -> float:
    """The clip's duration scale: its own MAD, floored by the cohort anchor.

    Two estimators, and the floor is not a safety hack but the more reliable of
    the pair on most clips. A clip carries 6-9 contacts, which cannot estimate a
    stable MAD -- and when its durations are near-identical the MAD collapses to
    ~0 and every deviation becomes infinite. The floor supplies what the clip
    cannot, measured over the whole balanced 45-clip design.

    Bounded on both sides, and the two bounds fix opposite failures. The floor
    stops a clip that cannot estimate its own spread (near-identical durations
    drive the MAD to ~0 and every deviation to infinity). The cap stops a clip
    whose MAD is inflated *by the very defects the filter should catch* from
    raising its own bar out of their reach -- self-masking, measured on
    ``S1_02_800m_1``. Between the two bounds a genuinely variable clip is still
    judged against its own spread rather than forced to the cohort's.
    """
    mad = statistics.median([abs(d - median) for d in durations])
    return min(
        max(mad, DURATION_MAD_FLOOR_FRACTION * median),
        DURATION_MAD_CAP_FRACTION * median,
    )


def _in_window(c: Any, window: Optional[Tuple[float, float]]) -> bool:
    """Whether *both* ends of a contact lie inside the runner's presence window.

    Both ends, deliberately. Gating on IC alone misses a contact that starts
    in shot and ends after the runner has left, as on ``S1_04_800m_3`` (IC
    221.3 inside, TO 232.0 outside a window ending at 226).

    A ``None`` window -- no landmarks available, or no frame in the clip
    carrying both a hip and an ankle -- means no gate: a missing input must
    never silently empty the contact set.
    """
    if window is None:
        return True
    lo, hi = window
    return bool(lo <= _ic(c) <= hi and lo <= _to_f(c) <= hi)


def _duration_outlier_flags(
    contacts: Sequence[Any], k: float = DURATION_MAD_K
) -> List[bool]:
    """Per-contact "implausible duration" flags, in input order.

    Outlying is defined against **this clip's own** contact-duration
    distribution, never an absolute physiological bound -- an absolute bound
    misfires across paces and runners.

    Median and MAD rather than mean and SD because the thing being detected --
    a merged contact spanning two strides, e.g. the 336 ms contact on
    ``S1_04_800m_3`` that starts at the previous stride's toe-off -- is exactly
    the kind of value that inflates an SD enough to hide itself inside it.

    Fewer than three contacts yields all-``False``: a median of two is not a
    robust centre, and a clip that thin has no within-clip distribution to
    compare against.

    **The cut-off is position-aware: the first and last contact of
    the clip are judged at ``DURATION_TERMINAL_K``, the rest at ``k``.** The
    bar stays a multiple of the clip's *own* spread either way -- position
    changes the multiplier, never the millisecond bound -- so a terminal
    contact in a genuinely variable clip keeps a wide bar.

    ``k`` scales *both* cut-offs, preserving their measured ratio, rather than
    moving the interior one alone. Two reasons, and the second is a real bug
    the obvious implementation has. The pair was placed together, so a caller
    overriding one and silently inheriting the other gets a combination nobody
    measured. And ``k = inf`` is how a comparison *disables* this filter to
    measure an ungated baseline -- a fixed terminal constant would keep
    filtering the terminal contacts and silently corrupt that baseline.
    """
    if len(contacts) < 3:
        return [False] * len(contacts)
    durations = [float(_get(c, "gct_ms") or 0.0) for c in contacts]
    med = statistics.median(durations)
    scale = _MAD_TO_SIGMA_INV * _duration_scale_ms(durations, med)
    terminal_k = k * (DURATION_TERMINAL_K / DURATION_MAD_K)
    last = len(durations) - 1
    return [
        abs(d - med) / scale > (terminal_k if i in (0, last) else k)
        for i, d in enumerate(durations)
    ]


def deployable_filter(
    contacts: Sequence[Any],
    window: Optional[Tuple[float, float]] = None,
    k: float = DURATION_MAD_K,
) -> Dict:
    """The GT-free contact set a real deployment could build for one clip.

    Applies the two filters to the refined contacts: presence window first,
    then within-clip duration outlier. **That order is load-bearing** -- a
    contact outside the presence window is not a measurement of this runner at
    all, so it must not be allowed to inform the clip's own duration median.

    Nothing here reads ground truth, which is what lets the engine and the
    validation scoring share this one implementation.

    Args:
        contacts: every contact of the clip, as ``contacts.csv`` rows or
            :class:`GroundContact` objects.
        window: ``(start, end)`` from ``compute_presence_quality``, or ``None``
            for no window gate (see :func:`_in_window`).
        k: scale-unit cut-off for the duration filter.

    ``fps`` is deliberately absent: the duration scale is a fraction of the
    clip's own median, not a frame period, so nothing here depends on the frame
    rate, and a parameter that reads as input but reaches nothing would
    mislead the caller.

    Returns:
        Dict with ``refined``, ``kept``, ``dropped_window`` and
        ``dropped_duration``; the last three partition ``refined``.
    """
    refined = sorted(
        (c for c in contacts if _get(c, "detection_method") == "refined"), key=_ic
    )
    in_win = [c for c in refined if _in_window(c, window)]
    dropped_window = [c for c in refined if not _in_window(c, window)]
    flags = _duration_outlier_flags(in_win, k)
    return {
        "refined": refined,
        "kept": [c for c, bad in zip(in_win, flags) if not bad],
        "dropped_window": dropped_window,
        "dropped_duration": [c for c, bad in zip(in_win, flags) if bad],
    }


def presence_window(landmarks: np.ndarray) -> Optional[Tuple[float, float]]:
    """``(start, end)`` of the runner's presence window, or ``None``.

    ``None`` when no frame carries both a hip and an ankle, which
    :func:`_in_window` reads as "no gate".
    """
    # Function-local import: src.utils.quality imports src.utils.metrics,
    # which imports this module. A module-level import would be circular.
    from src.utils.quality import compute_presence_quality

    presence = compute_presence_quality(landmarks)
    lo, hi = presence.get("window_start"), presence.get("window_end")
    return None if lo is None or hi is None else (float(lo), float(hi))
