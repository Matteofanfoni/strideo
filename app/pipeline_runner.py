# app/pipeline_runner.py
"""Shared clip-analysis entry points for the Streamlit app (F4).

Two independent result sources, both starting from the same ingest step:

- ``run_fast_path`` — BlazePose only, through the shipped FiLM CNN (ONNX,
  The app's default result.
- ``run_full_analysis`` — the classical v1.20 pipeline (RTMPose +
  contact-detection), run only when the user explicitly asks for it
  ("Full analysis").

Previously both lived inline in ``app/pages/upload.py``; factored out so
``app/pages/results.py`` can also trigger ``run_full_analysis`` per clip
without duplicating the ~90-line RTMPose+progress-bar block.
"""

from dataclasses import asdict
from pathlib import Path
from typing import Optional, Tuple

import numpy as np

from src.preprocessing.frame_rate import cap_frames, ensure_cfr
from src.preprocessing.nn_preprocessing import normalize_pace, preprocess_for_nn
from src.preprocessing.pipeline import ClipAnalysis, ProgressCallback
from src.utils.metrics import (
    BiomechanicalMetrics,
    calculate_duty_factor,
    calculate_flight_time,
    calculate_vertical_oscillation_ratio,
    calculate_velocity,
)

# Ingest guardrail: the seeded pass (classical path) and the multi-pass
# BlazePose extraction (both paths) buffer full-resolution frames
# (~25 MB/frame at 4K), so peak RAM scales with frame count. 300 frames ≈ 5 s
# at 60 fps — generous for the 1-3 s clips the protocol targets. Canonical
# home for this constant; app/pages/upload.py imports it rather than
# carrying its own copy.
MAX_PIPELINE_FRAMES = 300

# The shipped artifact. Not in git (see models/*.onnx in .gitignore) —
# present on any machine that ran scripts/export_final_model_to_onnx.py.
# F6 (deploy) is the item that gets this artifact onto the deployed target;
# this module only assumes it exists on disk once there.
DEFAULT_ONNX_PATH = Path(__file__).parent.parent / "models" / "strideo_final.onnx"


class OnnxModelMissingError(RuntimeError):
    """Raised when the shipped ONNX model artifact isn't on disk."""

    def __init__(self, onnx_path: Path):
        super().__init__(
            f"Fast-path model not found at {onnx_path}. Run "
            "`scripts/export_final_model_to_onnx.py` to produce it."
        )


def _diagnostics_from_extraction(extraction) -> Tuple[dict, np.ndarray]:
    """Build the same ``preprocessing`` diagnostics shape both paths already
    show in the app (nn_input_shape/torso_scale/direction/mean_visibility/
    detection_rate) — mirrors the diagnostics block ``upload.py`` builds
    today from ``preprocess_for_nn``'s metadata."""
    nn_input, _status, nn_meta = preprocess_for_nn(
        extraction.selected_landmarks,
        extraction.selected_visibilities,
        frame_width=extraction.res_w,
    )
    torso_scale = float(nn_meta.get("normalization", {}).get("torso_length_px", 0.0))
    direction = nn_meta.get("transforms", {}).get("original_direction")
    pq = extraction.pose_quality
    presence = pq.get("presence_quality") or {}
    return {
        "nn_input_shape": tuple(nn_input.shape),
        "mean_visibility": pq["mean_hip_visibility"],
        "detection_rate": pq["detection_rate"] * 100.0,
        "torso_scale": torso_scale,
        "direction": direction,
        # Frame statistics for the Video Information section.
        "total_frames": pq.get("total_frames"),
        "detected_frames": pq.get("detected_frames"),
        "frames_with_fill": pq.get("frames_with_fill"),
        "presence_window_start": presence.get("window_start"),
        "presence_window_end": presence.get("window_end"),
    }, nn_input


def run_fast_path(
    video_path: str,
    runner_height_cm: float,
    shoe_type: Optional[str],
    shoe_sole_cm: float,
    pace_level: str,
    onnx_model_path: Path = DEFAULT_ONNX_PATH,
    progress: Optional[ProgressCallback] = None,
    bidirectional: bool = True,
    seeded_pass: bool = False,
    pose_model_path: Optional[Path] = None,
) -> dict:
    """BlazePose-only path: extraction (steps 0-8) -> the shipped FiLM CNN
    (ONNX) -> a full ``BiomechanicalMetrics``-shaped dict, so
    ``app/pages/results.py`` can render it exactly like a classical result.

    Bucket-3 fields with no NN equivalent (cadence_std/gct_std/
    stride_length_std, n_contacts/n_refined_contacts — all per-contact
    quantities the model has no equivalent of) are set to NaN/0; results.py
    already renders those as "N/A" (``fmt()``) or "0 total" without needing
    a code change. Fields the UI never reads at all (step_length_m,
    stride_leg_ratio, flight_ratio) are also left NaN — no one has checked a
    formula for these in fast mode.

    ``bidirectional`` stays 3-pass-era default (``True``); ``seeded_pass``
    defaults to ``False`` (``two_pass``, adopted
    2026-08-28) — 35.8% faster with accuracy cost within measured
    replication noise. ``pose_model_path`` still defaults to the shipped
    app's own model.

    Returns:
        {"metrics": dict (BiomechanicalMetrics fields), "preprocessing":
         dict (same diagnostics shape upload.py has always built), "overlay":
         dict (cfr_path/fps/skip_rate/frame_width/selected_landmarks/
         selected_visibilities - everything results.py's annotated-video
         button needs to redraw the skeleton + metrics HUD later,
         without re-running extraction)}
    """
    from src.inference.onnx_infer import OnnxStrideoInference
    from src.preprocessing.pipeline import extract_blazepose_landmarks_and_calibration

    if not Path(onnx_model_path).exists():
        raise OnnxModelMissingError(Path(onnx_model_path))

    cfr_path, _ = ensure_cfr(video_path)
    proc_path, _ = cap_frames(cfr_path, MAX_PIPELINE_FRAMES)

    extraction = extract_blazepose_landmarks_and_calibration(
        proc_path,
        runner_height_cm,
        shoe_sole_cm=shoe_sole_cm,
        shoe_type=shoe_type,
        bidirectional=bidirectional,
        seeded_pass=seeded_pass,
        prune_spurious_contacts=True,  # also the source of the velocity estimate
        pose_model_path=pose_model_path,
        progress=progress,
    )

    preprocessing, nn_input = _diagnostics_from_extraction(extraction)

    if progress is not None:
        progress("Running neural network…", 0.90)
    model = OnnxStrideoInference(onnx_model_path)
    pace = normalize_pace(pace_level)
    prediction = model.predict(
        nn_input.astype(np.float32),
        pace,
        velocity_ms=extraction.estimated_velocity_ms,
    )
    bio = prediction["biomechanics"]
    # Real answer to "does inference actually window the clip": yes -
    # OnnxStrideoInference.predict slides window_size=60/stride=30 frames
    # across the whole nn_input sequence and averages the denormalised
    # per-window predictions (unweighted, confidence_weighted=False
    # default). Surfaced here so the app can show it rather than assert it.
    preprocessing["n_windows"] = prediction.get("n_windows")

    cadence_spm = bio["cadence_spm"]
    gct_ms = bio["gct_ms"]
    oscillation_cm = bio["oscillation_cm"]
    stride_length_m = bio.get("stride_length_m", float("nan"))

    flight_time_ms = calculate_flight_time(cadence_spm, gct_ms)
    duty_factor = calculate_duty_factor(gct_ms, flight_time_ms)
    vertical_oscillation_ratio = calculate_vertical_oscillation_ratio(
        oscillation_cm, stride_length_m
    )
    _, velocity_kmh, pace_per_km = calculate_velocity(stride_length_m, cadence_spm)
    leg_length_cm = extraction.calibration.leg_length_cm
    oscillation_leg_ratio = (
        oscillation_cm / leg_length_cm if leg_length_cm else float("nan")
    )
    # calculate_velocity()'s own velocity_ms is a reconstruction from
    # stride_length_m + cadence_spm (algebraically ~equal to the measured
    # value by construction, since stride_length_m came from _stride_post_hoc
    # on this same estimate) — the measured hip-x estimate is authoritative.
    measured_velocity_ms = (
        extraction.estimated_velocity_ms
        if extraction.estimated_velocity_ms is not None
        else float("nan")
    )

    metrics = BiomechanicalMetrics(
        cadence_spm=cadence_spm,
        cadence_std=float("nan"),  # bucket 3: no per-contact variance
        gct_ms=gct_ms,
        gct_std=float("nan"),
        flight_time_ms=flight_time_ms,
        duty_factor=duty_factor,
        stride_length_m=stride_length_m,
        stride_length_std=float("nan"),
        step_length_m=float("nan"),  # never rendered; not computed
        oscillation_cm=oscillation_cm,
        leg_length_cm=leg_length_cm,
        stride_leg_ratio=float("nan"),  # never rendered; not computed
        oscillation_leg_ratio=oscillation_leg_ratio,
        vertical_oscillation_ratio=vertical_oscillation_ratio,
        flight_ratio=float("nan"),  # never rendered; not computed
        velocity_ms=measured_velocity_ms,
        velocity_kmh=velocity_kmh,
        pace_per_km=pace_per_km,
        n_contacts=0,  # bucket 3: no contact detection in the fast path
        n_refined_contacts=0,
        calibration_confidence=extraction.calibration.confidence,
    )

    # Stops at 0.90, not 1.0: the caller still has the annotated-video encode
    # to do (~3 s against this function's ~27 s of pose passes), and owns the
    # last tenth of the bar so it can walk it frame by frame rather than
    # jumping 95 -> 100 around a silent step.
    if progress is not None:
        progress("Metrics ready", 0.90)
    overlay = {
        "cfr_path": extraction.cfr_path,
        "fps": extraction.fps,
        "skip_rate": extraction.skip_rate,
        "frame_width": extraction.res_w,
        "selected_landmarks": extraction.selected_landmarks,
        "selected_visibilities": extraction.selected_visibilities,
    }
    return {
        "metrics": asdict(metrics),
        "preprocessing": preprocessing,
        "overlay": overlay,
    }


def run_full_analysis(
    video_path: str,
    runner_height_cm: float,
    shoe_type: Optional[str],
    shoe_sole_cm: float,
    progress: Optional[ProgressCallback] = None,
) -> ClipAnalysis:
    """The classical v1.20 pipeline (RTMPose + contact detection), moved out
    of ``upload.py`` verbatim so both pages can trigger it without
    duplicating this block.

    Took ``pace_level``/``footwear_category`` once, purely to
    forward them to ``run_clip_pipeline``'s strike-pattern priors; a later pass
    deleted those priors, so both arguments were inert and are gone.
    ``run_fast_path`` above still takes a live ``pace_level`` -- that is the
    NN's FiLM conditioning input and is unrelated."""
    from src.preprocessing.pipeline import run_clip_pipeline
    from src.preprocessing.rtmpose_extractor import extract_rtmpose_landmarks

    cfr_path, _ = ensure_cfr(video_path)
    proc_path, _ = cap_frames(cfr_path, MAX_PIPELINE_FRAMES)

    def _rtm_progress(done, total, _p=progress):
        if _p is not None:
            frac = (done / total) if total else 0.0
            _p(f"RTMPose pose estimation… frame {done}/{total}", 0.05 + frac * 0.40)

    rtm = extract_rtmpose_landmarks(proc_path, device="cpu", progress=_rtm_progress)

    def _pipe_progress(stage, frac, _p=progress):
        if _p is not None:
            _p(stage, 0.45 + frac * 0.55)

    return run_clip_pipeline(
        proc_path,
        runner_height_cm=runner_height_cm,
        shoe_type=shoe_type,
        shoe_sole_cm=shoe_sole_cm,
        rtm_landmarks=rtm.keypoints,
        rtm_scores=rtm.scores,
        progress=_pipe_progress,
    )
