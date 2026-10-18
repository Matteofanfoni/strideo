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

import logging
from dataclasses import asdict
from pathlib import Path
from typing import Optional, Tuple

import numpy as np

from src.preprocessing.frame_rate import (
    cap_frames,
    converted_frame_count,
    ensure_cfr,
    frame_count,
    needs_cfr_conversion,
)
from src.preprocessing.nn_preprocessing import normalize_pace, preprocess_for_nn
from src.preprocessing.pipeline import ClipAnalysis, ProgressCallback
from src.utils.metrics import (
    BiomechanicalMetrics,
    calculate_duty_factor,
    calculate_flight_time,
    calculate_vertical_oscillation_ratio,
    calculate_velocity,
)

logger = logging.getLogger(__name__)

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


# Seconds to re-encode one 4K frame over seconds for BlazePose to process
# one, on the live CPU Space: ~16 s to cap a clip to 300 frames against ~31 s
# of BlazePose passes on those 300 (one fast-path run, 2026-10-17). Sizes
# the "Preparing video" band against the pose band that follows it.
ENCODE_TO_POSE_COST = 16.0 / 31.0


def _prepare_video(
    video_path: str, progress: Optional[ProgressCallback], pose_band: float
) -> Tuple[str, float]:
    """Convert to 60 fps CFR and cap to ``MAX_PIPELINE_FRAMES``, filling the
    bar from 0 as each ffmpeg encode proceeds.

    Either encode can be skipped (an already-CFR clip, a short one), so the
    band is sized per clip before encoding starts: the frames each encode
    will write, weighted by ``ENCODE_TO_POSE_COST``, against the frames
    BlazePose will then process, inside the first ``pose_band`` of the bar.
    A short CFR clip gets no band at all.

    Returns:
        (path of the prepared clip, fraction of the bar preparation used).
    """
    convert = needs_cfr_conversion(video_path)
    n = converted_frame_count(video_path) if convert else frame_count(video_path)
    convert_frames = n if convert else 0
    cap_n = MAX_PIPELINE_FRAMES if n > MAX_PIPELINE_FRAMES else 0
    encode = (convert_frames + cap_n) * ENCODE_TO_POSE_COST
    pose = min(n, MAX_PIPELINE_FRAMES)
    prep_end = pose_band * encode / (encode + pose) if encode > 0 else 0.0
    convert_end = (
        prep_end * convert_frames / (convert_frames + cap_n) if encode > 0 else 0.0
    )

    def _band(lo: float, hi: float):
        if progress is None:
            return None
        return lambda f: progress("Preparing video…", lo + f * (hi - lo))

    if progress is not None:
        progress("Preparing video…", 0.0)
    cfr_path, _ = ensure_cfr(video_path, progress=_band(0.0, convert_end))
    proc_path, _ = cap_frames(
        cfr_path, MAX_PIPELINE_FRAMES, progress=_band(convert_end, prep_end)
    )
    return proc_path, prep_end


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
         without re-running extraction), "extraction": the
         ``BlazePoseExtraction`` itself, which ``run_full_analysis`` reuses
         rather than running BlazePose a second time}
    """
    from src.inference.onnx_infer import OnnxStrideoInference
    from src.preprocessing.pipeline import extract_blazepose_landmarks_and_calibration

    if not Path(onnx_model_path).exists():
        raise OnnxModelMissingError(Path(onnx_model_path))

    # Extraction reports 0-0.90 and the NN and annotation own the rest, so
    # preparation takes the front of extraction's share and extraction is
    # squeezed into what is left of it.
    proc_path, prep_end = _prepare_video(video_path, progress, 0.90)

    def _extract_progress(stage, frac, _p=progress):
        if _p is not None:
            _p(stage, prep_end + frac * (0.90 - prep_end) / 0.90)

    extraction = extract_blazepose_landmarks_and_calibration(
        proc_path,
        runner_height_cm,
        shoe_sole_cm=shoe_sole_cm,
        shoe_type=shoe_type,
        bidirectional=bidirectional,
        seeded_pass=seeded_pass,
        prune_spurious_contacts=True,  # also the source of the velocity estimate
        pose_model_path=pose_model_path,
        progress=_extract_progress,
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
        "extraction": extraction,
    }


def run_full_analysis(
    video_path: str,
    runner_height_cm: float,
    shoe_type: Optional[str],
    shoe_sole_cm: float,
    progress: Optional[ProgressCallback] = None,
    extraction=None,
) -> ClipAnalysis:
    """The classical v1.20 pipeline (RTMPose + contact detection), moved out
    of ``upload.py`` verbatim so both pages can trigger it without
    duplicating this block.

    ``extraction`` is the fast path's ``BlazePoseExtraction`` for this clip
    (``run_fast_path``'s "extraction"): same capped CFR, same arguments, so
    it is reused instead of running BlazePose again, and its ``cfr_path``
    (the prepared, capped clip) is used instead of preparing the upload a
    second time. None, as after a re-import from a saved results file,
    prepares and extracts here.

    Took ``pace_level``/``footwear_category`` once, purely to
    forward them to ``run_clip_pipeline``'s strike-pattern priors; a later pass
    deleted those priors, so both arguments were inert and are gone.
    ``run_fast_path`` above still takes a live ``pace_level`` -- that is the
    NN's FiLM conditioning input and is unrelated."""
    from src.preprocessing.pipeline import (
        extract_blazepose_landmarks_and_calibration,
        run_clip_pipeline,
    )
    from src.preprocessing.rtmpose_extractor import (
        extract_rtmpose_landmarks_windowed,
        load_rtmpose,
        rtmpose_window,
    )
    from src.preprocessing.rtmpose_remote import (
        RemoteRTMPoseError,
        extract_rtmpose_remote,
        remote_enabled,
    )

    if extraction is not None:
        proc_path = extraction.cfr_path
    else:
        # BlazePose's band starts at 0.02, so preparation fills the bar to
        # there at most; RTMPose dwarfs it on this path.
        proc_path, _ = _prepare_video(video_path, progress, 0.02)

    # RTMPose goes to the Modal GPU when enabled, else, or when that
    # fails, runs here on CPU. The local model is only built when needed:
    # building it is wasted time when Modal answers.
    remote = remote_enabled()
    rtm_model = None
    if remote:
        device = "cuda"
    else:
        if progress is not None:
            progress("Loading RTMPose model…", 0.01)
        rtm_model, device = load_rtmpose()

    # RTMPose's share of the bar depends on which device it actually runs
    # on - a single fixed split can't be right for both, since GPU doesn't
    # just make RTMPose faster, it changes which of the two phases
    # dominates. Measured on real clips both ways: CPU-tier 259.6s/284.6s
    # (91.2%) and 270.9s/298.3s (90.8%) of total wall-clock; GPU-tier
    # (once CUDA actually initialized) 21s/52s (~40%) on one clip -
    # the rest of this function (contact detection, the classical engine)
    # barely benefits from GPU at all, so it goes from a rounding error at
    # CPU tier to the majority of the wall-clock at GPU tier. The previous
    # single CPU-calibrated split (0.92/0.08) made the bar look badly
    # front-loaded once GPU tier flipped which phase is slower. ``device`` is
    # the one the built session runs on, not what onnxruntime-gpu lists; on
    # the Modal path it is assumed "cuda" until a fallback says otherwise.
    # BlazePose comes first, so RTMPose can be windowed on its
    # presence window. The phases keep their measured widths: RTMPose
    # (rtm_end - 0.02), and the rest, ~90% of which is BlazePose (its two
    # passes are ~90% of extraction), ahead of RTMPose instead of after it.
    # Reusing the fast path's extraction drops BlazePose's band entirely, and
    # what is left after RTMPose is only the engine, the other ~10% of the
    # rest, so the bar is re-divided between those two: RTMPose then ends
    # at ~0.99 on CPU and ~0.87 on GPU, not at rtm_end.
    def _split(dev: str) -> tuple:
        rtm_end = 0.40 if dev == "cuda" else 0.92
        rtm_w = rtm_end - 0.02
        if extraction is not None:
            engine_w = 0.1 * (1.0 - rtm_end)
            return 0.02, 0.02 + 0.98 * rtm_w / (rtm_w + engine_w)
        blaze_end = 0.02 + 0.9 * (1.0 - rtm_end)
        return blaze_end, blaze_end + rtm_w

    blaze_end, rtm_stop = _split(device)

    def _blaze_progress(stage, frac, _p=progress):
        if _p is not None:
            _p(stage, 0.02 + min(frac, 1.0) * (blaze_end - 0.02))

    if extraction is None:
        extraction = extract_blazepose_landmarks_and_calibration(
            proc_path,
            runner_height_cm,
            shoe_sole_cm=shoe_sole_cm,
            shoe_type=shoe_type,
            progress=_blaze_progress,
        )
    window = rtmpose_window(extraction.selected_landmarks)

    # RTMPose's band runs from ``bar["start"]`` to ``rtm_stop``. A fallback
    # restarts it from wherever the Modal attempt left the bar, so the bar
    # never moves backwards.
    bar = {"start": blaze_end, "pos": blaze_end}

    def _rtm_progress(done, total, _p=progress):
        if _p is not None:
            frac = (done / total) if total else 0.0
            bar["pos"] = bar["start"] + frac * (rtm_stop - bar["start"])
            _p(f"RTMPose pose estimation… frame {done}/{total}", bar["pos"])

    n_frames = len(extraction.selected_landmarks)
    rtm = None
    if remote:
        if progress is not None:
            progress("Starting the GPU…", bar["pos"])
        try:
            rtm, _ = extract_rtmpose_remote(
                proc_path,
                n_frames,
                window,
                extraction.skip_rate,
                progress=_rtm_progress,
            )
        except RemoteRTMPoseError as exc:
            logger.warning("Modal RTMPose failed (%s); running it on CPU", exc)
    if rtm is None:
        if rtm_model is None:
            if progress is not None:
                progress("Loading RTMPose model…", bar["pos"])
            rtm_model, device = load_rtmpose()
            _, rtm_stop = _split(device)
            rtm_stop = max(rtm_stop, bar["pos"])
            bar["start"] = bar["pos"]
        rtm = extract_rtmpose_landmarks_windowed(
            proc_path,
            n_frames,
            window,
            extraction.skip_rate,
            device=device,
            progress=_rtm_progress,
            model=rtm_model,
        )

    def _pipe_progress(stage, frac, _p=progress):
        if _p is not None:
            _p(stage, rtm_stop + frac * (1.0 - rtm_stop))

    return run_clip_pipeline(
        proc_path,
        runner_height_cm=runner_height_cm,
        shoe_type=shoe_type,
        shoe_sole_cm=shoe_sole_cm,
        rtm_landmarks=rtm.keypoints,
        rtm_scores=rtm.scores,
        progress=_pipe_progress,
        extraction=extraction,
        rtm_window=window,
    )
