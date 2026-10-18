"""Moving a deterministic kinematics engine result onto a Results-page entry.

Two routes produce one: the "Run in-depth analysis" button and the
automatic run ``app.pipeline_runner.EngineJob`` starts on every
upload. Both go through ``apply_full_analysis`` so they cannot drift apart.
Lives outside ``app/pages/results.py`` so it can be imported and tested
without executing the page script.
"""

import time
from dataclasses import asdict

from src.utils.vo_correction import correct_engine_metrics_for_display


def _preprocessing_from_full(full_analysis, preprocessing_fast, direction):
    """The ``preprocessing`` diagnostics for a clip once the deterministic
    engine has run.

    Most fields are recomputed from the engine's own ``pose_quality`` and
    should overwrite the fast path's, since the two extractions really do
    differ. ``torso_scale`` is the exception and is **carried forward**: it
    comes out of ``preprocess_for_nn``'s hip-centred normalisation, which is
    StrideoNet's input pipeline and something the engine never runs, so there
    is no engine-side value to supply. It used to be hardcoded ``0.0`` here,
    and because the page resolves ``preprocessing_full or preprocessing_fast``
    the placeholder won outright: every clip the engine had been run on showed
    "Torso scale 0.0000" while every other clip showed the real number. Torso
    length is a property of the clip's landmarks rather than of which
    estimator ran, so the fast path's value is the right one to keep.

    Lives at module level, rather than inline where it is used, so it can be
    tested: the call site sits inside a fragment callback that only runs after
    a multi-minute RTMPose pass, which is why the placeholder shipped
    unnoticed.
    """
    pq = full_analysis.pose_quality
    presence = pq.get("presence_quality") or {}
    return {
        "nn_input_shape": None,
        "mean_visibility": pq["mean_hip_visibility"],
        "detection_rate": pq["detection_rate"] * 100.0,
        "torso_scale": (preprocessing_fast or {}).get("torso_scale", 0.0),
        "direction": direction,
        # Frame statistics for the Video Information section (same
        # fields as the fast path's own).
        "total_frames": pq.get("total_frames"),
        "detected_frames": pq.get("detected_frames"),
        "frames_with_fill": pq.get("frames_with_fill"),
        "presence_window_start": presence.get("window_start"),
        "presence_window_end": presence.get("window_end"),
    }


def apply_full_analysis(r: dict, full_analysis) -> bool:
    """Write an engine ``ClipAnalysis`` onto result entry ``r``.

    An excluded clip (presence window undefined, or tracking lost mid-transit
    for longer than gap-fill repairs) only sets ``full_analysis_excluded``:
    its numbers would not be trustworthy, so they are withheld rather than
    displayed, and ``metrics_full``/``analysis_full`` stay unset.

    Returns:
        True when metrics were written, False for an excluded clip.
    """
    if full_analysis.reliability.excluded:
        r["full_analysis_excluded"] = True
        return False

    from src.preprocessing.nn_preprocessing import detect_running_direction

    # The engine measures oscillation 54.5% low against annotated reference
    # on 15 of 15 annotated clips, and the fit that corrects it had only ever
    # been applied to StrideoNet's training labels -- so the two estimators
    # were showing this metric on scales ~2x apart. The correction is applied
    # HERE, at display, and never in the pipeline: a corrected stored value
    # would have the validation report scoring the fit against the clips it
    # was fitted on. See src/utils/vo_correction.py.
    r["metrics_full"] = correct_engine_metrics_for_display(
        asdict(full_analysis.metrics)
    )
    r["preprocessing_full"] = _preprocessing_from_full(
        full_analysis,
        r.get("preprocessing_fast"),
        detect_running_direction(full_analysis.selected_landmarks),
    )
    r["strike_pattern_full"] = full_analysis.clip_strike_pattern
    r["analysis_full"] = full_analysis
    return True


#: Seconds after which a still-running automatic engine stops holding the
#: Results page in its pending state and the button comes back. A run takes
#: ~40 s; this only matters for a call that hangs. A job that finishes later
#: is still absorbed, unless the button got there first.
ENGINE_PENDING_MAX_S = 180.0

#: How often the engine section re-checks a pending run.
ENGINE_POLL_S = 2.0


def engine_pending(r: dict) -> bool:
    """True while the clip's automatic engine run is still going, and young
    enough to keep waiting for."""
    job = r.get("engine_job")
    return (
        job is not None
        and not job.finished
        and time.monotonic() - job.started < ENGINE_PENDING_MAX_S
    )


def absorb_engine_job(r: dict) -> None:
    """Move a finished automatic engine run (``r["engine_job"]``) onto ``r``.

    Runs on the script thread, at the top of the Results page. ``"done"`` and
    ``"excluded"`` land exactly as the button's result would; ``"skipped"``
    leaves the entry as it was, so the button is the route. The job is
    dropped once absorbed, so it is applied once, and never over a result the
    button already produced.
    """
    job = r.get("engine_job")
    if job is None or not job.finished:
        return
    r["engine_job"] = None
    if job.status in ("done", "excluded") and r.get("metrics_full") is None:
        apply_full_analysis(r, job.analysis)
