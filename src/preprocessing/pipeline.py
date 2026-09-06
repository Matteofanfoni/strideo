from dataclasses import dataclass
from pathlib import Path
from typing import Callable, Dict, List, Optional

import numpy as np

from src.preprocessing.calibration import SpatialCalibration
from src.preprocessing.frame_rate import ensure_cfr
from src.preprocessing.video_io import load_video
from src.preprocessing.normalizer import normalize_training_video
from src.preprocessing.pose_estimator import extract_pose_landmarks
from src.preprocessing.nn_preprocessing import normalize_pace, preprocess_for_nn
from src.utils.metrics import extract_all_metrics
from src.utils.metadata import (
    CaptureInfo,
    FilesPaths,
    FootwearInfo,
    MetadataStore,
    RunnerInfo,
    VideoMetadata,
)


def preprocess_training_video(
    video_path: str,
    metadata: Optional[VideoMetadata] = None,
    metadata_store: Optional[MetadataStore] = None,
    runner_height_cm: Optional[float] = None,
) -> dict:
    """
    Complete preprocessing for a single training video.

    Metadata can be provided in three ways (checked in order):
      1. Explicit ``metadata`` object
      2. Lookup by video_id from a ``metadata_store``
      3. Manual ``runner_height_cm`` fallback

    Args:
        video_path: Path to the raw video file.
        metadata: Pre-loaded VideoMetadata for this video.
        metadata_store: MetadataStore to look up metadata by video_id.
        runner_height_cm: Manual fallback for runner height if no
            metadata is available.

    Returns:
        dict ready for saving as .npz
    """
    # Resolve metadata
    video_meta = _resolve_metadata(
        video_path, metadata, metadata_store, runner_height_cm
    )

    # 0. Ensure video is CFR (convert VFR → CFR if needed)
    cfr_path, was_converted = ensure_cfr(video_path)
    if was_converted:
        print(f"VFR → CFR conversion: {video_path} → {cfr_path}")
    else:
        print("Video is already CFR, skipping conversion.")

    # 1. Load video
    frames, fps, meta = load_video(cfr_path, target_fps=60)

    # 2. Normalize to exactly 120 frames
    norm_result = normalize_training_video(frames)
    normalized_frames = norm_result.frames

    print(f"Video normalization: {norm_result.status.value}")
    print(f"  Original: {norm_result.original_count} frames")
    print(f"  Used: frames {norm_result.start_frame}–{norm_result.end_frame}")
    if norm_result.pad_start > 0 or norm_result.pad_end > 0:
        print(f"  Padded: +{norm_result.pad_start} start, +{norm_result.pad_end} end")

    # 3. Run MediaPipe on normalized frames
    pose_result = extract_pose_landmarks(list(normalized_frames), fps)
    landmarks = pose_result.landmarks
    visibilities = pose_result.visibilities

    print(f"Pose estimation: {pose_result.quality_metrics}")

    # 4. Compute biomechanical metrics
    metrics = extract_all_metrics(
        landmarks,
        visibilities,
        fps,
        video_meta.runner.height_cm,
        resolution_height=meta["height"],
    )

    # 5. Preprocess for NN
    nn_input, status, nn_meta = preprocess_for_nn(
        landmarks, visibilities, frame_width=meta["width"]
    )

    # 6. Normalize pace for FiLM conditioning
    pace_normalized = normalize_pace(video_meta.capture.pace_level)

    return {
        "nn_input": nn_input,
        "status": status,
        "pace": pace_normalized,
        "metrics": metrics,
        "fps": fps,
        "normalization": norm_result.status.value,
        "video_metadata": meta,
        "nn_metadata": nn_meta,
        "runner_height_cm": video_meta.runner.height_cm,
        "pace_level": video_meta.capture.pace_level,
    }


def _resolve_metadata(
    video_path: str,
    metadata: Optional[VideoMetadata],
    metadata_store: Optional[MetadataStore],
    runner_height_cm: Optional[float],
) -> VideoMetadata:
    """
    Resolve metadata from the available sources.

    Priority: explicit metadata > store lookup > manual fallback.
    """
    from pathlib import Path

    if metadata is not None:
        return metadata

    if metadata_store is not None:
        video_id = Path(video_path).stem.removesuffix("_cfr")
        return metadata_store.get(video_id)

    if runner_height_cm is not None:
        # Minimal fallback — construct a stub metadata object
        return VideoMetadata(
            video_id=Path(video_path).stem,
            files=FilesPaths(raw_vfr=video_path),
            runner=RunnerInfo(
                id="unknown",
                height_cm=runner_height_cm,
                stratum="A",
                specialist_event="unknown",
            ),
            footwear=FootwearInfo(category="unknown", model="unknown"),
            capture=CaptureInfo(
                camera_distance_m=12.0,
                pace_level="unknown",
            ),
        )

    raise ValueError(
        "No metadata provided. Supply one of: metadata, metadata_store, "
        "or runner_height_cm."
    )


# ---------------------------------------------------------------------------
# v1.20 inference orchestration (shared by the CLI validator and the web app)
# ---------------------------------------------------------------------------

ProgressCallback = Callable[[str, float], None]


@dataclass
class ClipAnalysis:
    """Result of the v1.20 pipeline on a single clip.

    Mirrors the objects ``scripts/run_prevalidation_single.py`` computes so the
    Day-4 parity check (app output == CLI ``result_full.json``) is a direct
    comparison.
    """

    metrics: object  # BiomechanicalMetrics
    calibration: object  # SpatialCalibration
    contacts: list  # List[GroundContact]
    warnings: List[str]
    pose_quality: Dict
    reliability: object  # ReliabilityAssessment
    fps: float
    n_frames: int
    clip_strike_pattern: Optional[str]
    velocity_ms_estimated: Optional[float]
    cfr_path: str
    # Final (post-filter/gap-fill) landmark stream + frame width, so callers
    # can run nn_preprocessing for display diagnostics without re-extracting.
    selected_landmarks: np.ndarray
    selected_visibilities: np.ndarray
    frame_width: int
    # Source→pose frame stride: pose frame ``t`` is source frame ``t*skip_rate``
    # of ``cfr_path``. Lets overlay viewers (the app scrubber) align frames to
    # landmarks without re-deriving it. 1 when the CFR is already at target fps.
    skip_rate: int = 1


def _buffer_video_frames(video_path: str, skip_rate: int) -> List[np.ndarray]:
    """Buffer frames at ``skip_rate`` into RAM as RGB arrays (seeded-pass input).

    Replicates the helper in ``scripts/run_prevalidation_single.py`` so the
    seeded pass sees identical frames.
    """
    import cv2

    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        raise ValueError(f"Cannot open video for frame buffering: {video_path}")
    frames_rgb: List[np.ndarray] = []
    frame_idx = 0
    while True:
        ok, frame = cap.read()
        if not ok:
            break
        if frame_idx % skip_rate == 0:
            frames_rgb.append(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB))
        frame_idx += 1
    cap.release()
    return frames_rgb


@dataclass
class BlazePoseExtraction:
    """BlazePose-only prefix of ``run_clip_pipeline`` (its steps 0-8, plus the
    pose-quality block and the final spatial calibration): everything the
    pipeline computes before it ever touches ``rtm_landmarks``/``rtm_scores``.
    Extracted so a BlazePose-only caller (the app's fast inference path)
    can reuse it directly instead of duplicating this extraction logic — per
    the measured finding that the NN's input tensor is bit-for-bit
    identical with or without RTMPose.
    """

    cfr_path: str
    fps: float
    res_h: int
    res_w: int
    skip_rate: int
    selected_landmarks: np.ndarray
    selected_visibilities: np.ndarray
    pose_quality: Dict
    body_height_px: float
    calibration: SpatialCalibration  # final (post gap-fill) landmark
    # stream, same fallback ladder as the bootstrap calibration.
    estimated_velocity_ms: Optional[float]
    leg_length_m: Optional[float]


def extract_blazepose_landmarks_and_calibration(
    video_path: str,
    runner_height_cm: float,
    *,
    shoe_sole_cm: float = 2.5,
    shoe_type: Optional[str] = None,
    bidirectional: bool = True,
    # Two_pass is the unified default. Dropping the seeded third pass
    # is bit-identical for contacts, strike, GCT and cadence (RTMPose-driven,
    # and since the windowing change the strike vote runs over the RTM
    # coarse list), moves stride/step/velocity/calibration by <=0.75%,
    # and moves vertical
    # oscillation -- which is recovered by re-deriving the VO correction
    # under this extraction (LOO 14.5% vs 3-pass's 14.6%). One BlazePose
    # pass now serves both StrideoNet and the engine; previously the app
    # extracted the same clip twice under different configurations.
    seeded_pass: bool = False,
    motion_filter: bool = True,
    gap_fill: bool = True,
    interp_method: str = "pchip",
    gap_max_frames: int = 4,
    gap_min_anchor_vis: float = 0.5,
    motion_filter_min_ms: float = 0.5,
    prune_spurious_contacts: bool = True,
    pose_model_path: Optional[Path] = None,
    progress: Optional[ProgressCallback] = None,
) -> BlazePoseExtraction:
    """Run the BlazePose-only prefix of the v1.20 pipeline: VFR->CFR through
    gap-fill, pose-quality metrics, the final spatial calibration, and
    (optionally) the hip-x velocity estimate. A pure lift out of
    ``run_clip_pipeline`` — no behaviour change; ``run_clip_pipeline`` itself
    now calls this as its own first step, and a pinning test confirms its
    output is unchanged by the extraction.

    Args:
        video_path: Path to the source clip (CFR conversion handled here).
        runner_height_cm: Runner standing height (cm), for spatial calibration.
        shoe_sole_cm / shoe_type: Footwear sole offset / SHOE_TYPES key.
        pose_model_path: MediaPipe pose-landmarker model bundle for all
            three BlazePose passes. None (default) uses the production
            model (MODEL_HEAVY) — added for speed-option measurement,
            not otherwise exposed to callers.
        progress: Optional callback(stage_label, fraction_0_to_1) for UI.

    Returns:
        BlazePoseExtraction with everything a BlazePose-only or a
        RTMPose-augmented caller needs to continue.
    """
    from src.preprocessing.pose_estimator import (
        MODEL_HEAVY,
        LandmarkIndex,
        extract_pose_landmarks_seeded,
        extract_pose_landmarks_streaming,
        extract_pose_landmarks_streaming_reverse,
    )

    resolved_model_path = pose_model_path or MODEL_HEAVY
    from src.preprocessing.calibration import create_spatial_calibration
    from src.utils.quality import (
        compute_detected_mean_visibility,
        compute_presence_quality,
    )
    from src.preprocessing.motion_filter import (
        combine_bidirectional_pose,
        combine_three_pass_pose,
        filter_landmarks_by_motion,
    )
    from src.preprocessing.landmarks_cleanup import fill_gaps
    from src.preprocessing.footage_extractor import detect_near_side
    from src.utils.pace_estimator import estimate_velocity_from_hipx

    def _emit(stage: str, frac: float) -> None:
        if progress is not None:
            progress(stage, frac)

    def _band(stage: str, lo: float, hi: float):
        """Per-frame callback that walks the bar across [lo, hi].

        The two pose passes are ~90% of this function's wall-clock (measured
        on the app's fast path: ~13.5 s each against ~3 s for everything
        else), so they get the bar's whole 0.02-0.88 span between them and
        move through it continuously. Everything after them is numpy-fast
        and only gets a point emit near the end.
        """
        if progress is None:
            return None

        def _cb(frac: float, _s=stage, _lo=lo, _hi=hi) -> None:
            progress(_s, _lo + (_hi - _lo) * min(1.0, max(0.0, frac)))

        return _cb

    # Split the pose budget: forward owns it all when there is no reverse pass.
    fwd_hi = 0.45 if bidirectional else 0.88

    # 0. VFR -> CFR
    _emit("Preparing video", 0.02)
    cfr_path, _ = ensure_cfr(video_path)

    # 1. Forward pose pass (v1.0 config: VIDEO + num_poses=1 + det_conf=0.5)
    _emit("Pose estimation (forward)", 0.02)
    pose_result, video_meta = extract_pose_landmarks_streaming(
        cfr_path,
        num_poses=1,
        running_mode="video",
        min_pose_detection_confidence=0.5,
        model_path=resolved_model_path,
        frame_progress=_band("Pose estimation (forward)", 0.02, fwd_hi),
    )
    fps = video_meta["effective_fps"]
    res_h = video_meta["height"]
    res_w = video_meta["width"]
    skip_rate = video_meta["skip_rate"]

    # 2. Reverse pose pass (v1.6)
    pose_result_rev = None
    if bidirectional:
        _emit("Pose estimation (reverse)", 0.45)
        pose_result_rev, _ = extract_pose_landmarks_streaming_reverse(
            cfr_path,
            model_path=resolved_model_path,
            frame_progress=_band("Pose estimation (reverse)", 0.45, 0.88),
        )

    # 3. Bootstrap calibration + body height (px) for downstream gates
    # In a headless Docker environment MediaPipe's EGL context may degrade,
    # producing near-zero visibility scores. Retry progressively: strict →
    # permissive → geometry-only (ignore visibility entirely).
    _cal_args = (
        pose_result.landmarks,
        pose_result.visibilities,
        runner_height_cm,
        shoe_sole_cm,
        shoe_type,
    )
    try:
        bootstrap_cal = create_spatial_calibration(*_cal_args)
    except ValueError:
        try:
            bootstrap_cal = create_spatial_calibration(
                *_cal_args, min_visibility=0.35, min_samples=5
            )
        except ValueError:
            bootstrap_cal = create_spatial_calibration(
                *_cal_args, min_visibility=0.0, min_samples=3
            )
    body_height_cm = float(runner_height_cm) + float(shoe_sole_cm)
    body_height_px = body_height_cm * float(bootstrap_cal.pixels_per_cm)

    # 4. Bidirectional combine (anatomy ratifier)
    if bidirectional and pose_result_rev is not None:
        _emit("Combining passes", 0.885)
        combine_result = combine_bidirectional_pose(
            pose_result.landmarks,
            pose_result.visibilities,
            pose_result.world_landmarks,
            pose_result_rev.landmarks,
            pose_result_rev.visibilities,
            pose_result_rev.world_landmarks,
            body_height_px=body_height_px,
        )
        combined_landmarks = combine_result.landmarks
        combined_visibilities = combine_result.visibilities
        combined_world = combine_result.world_landmarks
    else:
        combined_landmarks = pose_result.landmarks
        combined_visibilities = pose_result.visibilities
        combined_world = pose_result.world_landmarks

    # 5. Seeded pass (v1.11) — requires bidirectional
    if seeded_pass and bidirectional and pose_result_rev is not None:
        _emit("Seeded refinement", 0.89)
        T_clip = len(combined_visibilities)
        near_side = detect_near_side(combined_visibilities)
        near_hip_idx = (
            LandmarkIndex.LEFT_HIP if near_side == "left" else LandmarkIndex.RIGHT_HIP
        )
        presence = combined_visibilities[:, near_hip_idx]

        lo, hi = int(0.2 * T_clip), int(0.8 * T_clip)
        if lo >= hi:
            F_anchor = T_clip // 2
        else:
            present_win = presence[lo:hi] >= 0.3
            if present_win.any():
                first_p = int(np.argmax(present_win))
                last_p = int(len(present_win) - 1 - np.argmax(present_win[::-1]))
                F_anchor = lo + (first_p + last_p) // 2
            else:
                F_anchor = lo + int(np.argmax(presence[lo:hi]))

        quality = combined_visibilities[:, near_hip_idx]
        if quality[F_anchor] < 0.3:
            present_full = presence >= 0.3
            best_start, best_len = 0, 0
            cur_start, cur_len = 0, 0
            for _i, _v in enumerate(present_full):
                if _v:
                    if cur_len == 0:
                        cur_start = _i
                    cur_len += 1
                    if cur_len > best_len:
                        best_len, best_start = cur_len, cur_start
                else:
                    cur_len = 0
            if best_len > 0:
                F_anchor = best_start + best_len // 2
                refine_w = 5
                r_lo = max(0, F_anchor - refine_w)
                r_hi = min(T_clip, F_anchor + refine_w + 1)
                F_anchor = r_lo + int(np.argmax(quality[r_lo:r_hi]))

        frames_rgb = _buffer_video_frames(cfr_path, skip_rate)
        result_seed_bwd, result_seed_fwd = extract_pose_landmarks_seeded(
            frames_rgb, fps, F_anchor, model_path=resolved_model_path
        )
        del frames_rgb
        combine3 = combine_three_pass_pose(
            pose_result.landmarks,
            pose_result.visibilities,
            pose_result.world_landmarks,
            pose_result_rev.landmarks,
            pose_result_rev.visibilities,
            pose_result_rev.world_landmarks,
            result_seed_bwd.landmarks,
            result_seed_bwd.visibilities,
            result_seed_bwd.world_landmarks,
            result_seed_fwd.landmarks,
            result_seed_fwd.visibilities,
            result_seed_fwd.world_landmarks,
            anchor_frame=F_anchor,
            body_height_px=body_height_px,
        )
        combined_landmarks = combine3.landmarks
        combined_visibilities = combine3.visibilities
        combined_world = combine3.world_landmarks

    # 6. Motion filter (v1.2)
    if motion_filter:
        _emit("Motion filtering", 0.893)
        mf = filter_landmarks_by_motion(
            combined_landmarks,
            combined_visibilities,
            combined_world,
            fps=fps,
            pixels_per_cm=bootstrap_cal.pixels_per_cm,
            min_velocity_ms=motion_filter_min_ms,
            body_height_cm=None,
            apply_anatomy_gate=False,
        )
        selected_landmarks = mf.landmarks
        selected_visibilities = mf.visibilities
        selected_world = mf.world_landmarks
        filter_info = {
            "method": "motion_filter_v1.2",
            "threshold_m_s": mf.threshold_m_s,
        }
    else:
        selected_landmarks = combined_landmarks
        selected_visibilities = combined_visibilities
        selected_world = combined_world
        filter_info = {"method": "none"}

    # 7. Gap-fill (v1.7)
    frames_with_fill = 0
    if gap_fill:
        _emit("Gap-filling", 0.896)
        cleanup = fill_gaps(
            selected_landmarks,
            selected_visibilities,
            selected_world,
            max_gap_frames=gap_max_frames,
            min_anchor_visibility=gap_min_anchor_vis,
            method=interp_method,
        )
        selected_landmarks = cleanup.landmarks
        selected_visibilities = cleanup.visibilities
        selected_world = cleanup.world_landmarks
        frames_with_fill = cleanup.n_frames_with_fill

    # Quality metrics on the filtered tensor (mirrors the CLI).
    hip_idx = [LandmarkIndex.LEFT_HIP, LandmarkIndex.RIGHT_HIP]
    detected_mask = ~np.all(np.isnan(selected_landmarks), axis=(1, 2))
    T_total = len(selected_landmarks)
    # Mean_*_visibility is computed over *detected* frames only -
    # selected_visibilities is never NaN on an undetected frame (it is
    # pre-filled 0.0), so a plain nanmean silently counted every undetected
    # frame as "visibility exactly 0" instead of excluding it. See
    # compute_detected_mean_visibility's own docstring for the measured size.
    _mean_vis = compute_detected_mean_visibility(
        selected_landmarks, selected_visibilities
    )
    pose_quality = {
        "total_frames": int(T_total),
        "detected_frames": int(detected_mask.sum()),
        "detection_rate": (float(detected_mask.sum() / T_total) if T_total else 0.0),
        "low_confidence_frames": int(
            np.sum(np.mean(selected_visibilities[:, hip_idx], axis=1) < 0.5)
        ),
        "mean_hip_visibility": _mean_vis["mean_hip_visibility"],
        "mean_ankle_visibility": _mean_vis["mean_ankle_visibility"],
        "filter": filter_info,
        "frames_with_fill": int(frames_with_fill),
        "presence_quality": compute_presence_quality(selected_landmarks),
    }

    # 8. Velocity estimate for spurious-contact prune (v1.9)
    estimated_velocity_ms: Optional[float] = None
    leg_length_m_for_prune: Optional[float] = None
    if prune_spurious_contacts:
        leg_length_m_for_prune = float(bootstrap_cal.leg_length_cm) / 100.0
        try:
            estimated_velocity_ms = estimate_velocity_from_hipx(
                selected_landmarks,
                selected_visibilities,
                pixels_per_m=float(bootstrap_cal.pixels_per_cm) * 100.0,
                fps=fps,
            )
        except ValueError:
            estimated_velocity_ms = None

    # Final (post-filter/gap-fill) spatial calibration - same fallback ladder
    # as the bootstrap above. Order-independent from step 8 (both depend only
    # on selected_landmarks/visibilities and the runner/shoe args), so moving
    # it here from its original position (originally computed just before
    # contact detection) does not change run_clip_pipeline's output.
    _fin_cal_args = (
        selected_landmarks,
        selected_visibilities,
        runner_height_cm,
        shoe_sole_cm,
        shoe_type,
    )
    try:
        calibration = create_spatial_calibration(*_fin_cal_args)
    except ValueError:
        try:
            calibration = create_spatial_calibration(
                *_fin_cal_args, min_visibility=0.35, min_samples=5
            )
        except ValueError:
            calibration = create_spatial_calibration(
                *_fin_cal_args, min_visibility=0.0, min_samples=3
            )

    return BlazePoseExtraction(
        cfr_path=cfr_path,
        fps=fps,
        res_h=res_h,
        res_w=res_w,
        skip_rate=skip_rate,
        selected_landmarks=selected_landmarks,
        selected_visibilities=selected_visibilities,
        pose_quality=pose_quality,
        body_height_px=body_height_px,
        calibration=calibration,
        estimated_velocity_ms=estimated_velocity_ms,
        leg_length_m=leg_length_m_for_prune,
    )


def run_clip_pipeline(
    video_path: str,
    runner_height_cm: float,
    *,
    shoe_sole_cm: float = 2.5,
    shoe_type: Optional[str] = None,
    rtm_landmarks: Optional[np.ndarray] = None,
    rtm_scores: Optional[np.ndarray] = None,
    # v1.20 flags (defaults mirror the canonical CLI invocation in CLAUDE.md)
    bidirectional: bool = True,
    # Two_pass is the unified default. Dropping the seeded third pass
    # is bit-identical for contacts, strike, GCT and cadence (RTMPose-driven,
    # and since the windowing change the strike vote runs over the RTM
    # coarse list), moves stride/step/velocity/calibration by <=0.75%,
    # and moves vertical
    # oscillation -- which is recovered by re-deriving the VO correction
    # under this extraction (LOO 14.5% vs 3-pass's 14.6%). One BlazePose
    # pass now serves both StrideoNet and the engine; previously the app
    # extracted the same clip twice under different configurations.
    seeded_pass: bool = False,
    gap_fill: bool = True,
    motion_filter: bool = True,
    strike_aware_toeoff: bool = True,
    prune_spurious_contacts: bool = True,
    gate_landmark_bounds: bool = True,
    interp_method: str = "pchip",
    gap_max_frames: int = 4,
    gap_min_anchor_vis: float = 0.5,
    motion_filter_min_ms: float = 0.5,
    tau_strike_frac: float = 0.01,
    delta_lift_frac: float = 0.02,
    delta_lift_frac_forefoot: float = 0.01,
    ankle_horiz_vel_gate_frac: float = 0.0,
    cadence_band_frac: float = 0.30,
    clip_strike_pattern_override: str = "auto",
    strike_from_refined_ic: bool = True,
    demote_weak_strike_vote: bool = False,
    unknown_as_forefoot: bool = False,
    progress: Optional[ProgressCallback] = None,
) -> ClipAnalysis:
    """Run the v1.20 preprocessing + metrics pipeline on one clip.

    This is a faithful transcription of ``main()`` in
    ``scripts/run_prevalidation_single.py`` for the canonical v1.20 invocation
    (``--bidirectional --gap-fill --strike-aware-toeoff
    --prune-spurious-contacts --seeded-pass --rtmpose-landmarks
    --gate-landmark-bounds``, the last added at v1.24), factored so
    both the CLI validator and the Streamlit app can share it.

    The orchestration deliberately does NOT run RTMPose itself: pass
    ``rtm_landmarks`` / ``rtm_scores`` (the CLI loads them from a cached npz;
    the app produces them live with ``rtmpose_extractor``). They must be
    frame-aligned to BlazePose (``T_rtm == T_blaze``).

    Args:
        video_path: Path to the source clip (CFR conversion handled here).
        runner_height_cm: Runner standing height (cm), for spatial calibration.
        shoe_sole_cm / shoe_type: Footwear sole offset / SHOE_TYPES key.
            **Removed: ``pace_level`` / ``footwear_category``.**
            They existed only to feed the speed/shoe population priors in
            ``detect_clip_strike_pattern``, which has since deleted them
            (46.7% and 40.0%, both below chance; the shoe prior a constant
            on our cohort), leaving them inert. Not to be reinstated -- an
            argument that reads as conditioning input but reaches nothing is
            the same defect removed one level down, and re-promoting the
            pace prior is separately ruled out. **Do not confuse
            this with ``app.pipeline_runner.run_fast_path``'s own
            ``pace_level``, which is live** -- that one is the NN's FiLM
            conditioning input, via ``normalize_pace``.
        rtm_landmarks / rtm_scores: Live or cached RTMPose-x output, or None.
        gate_landmark_bounds: Drop contacts whose hip/
            ankle landmarks are out of frame at their own frame (entry/
            exit fabrication). Default True as of the v1.24 baseline
            (validation re-run: stride/velocity certify, cadence raw reading
            softens to caveats, second cohort 14/15→13/15).
        progress: Optional callback(stage_label, fraction_0_to_1) for UI.

    Returns:
        ClipAnalysis with metrics, calibration, contacts, quality, fps.
    """
    from src.preprocessing.ground_contact import (
        detect_ground_contacts,
    )
    from src.utils.quality import assess_clip_reliability, validate_metrics

    def _emit(stage: str, frac: float) -> None:
        if progress is not None:
            progress(stage, frac)

    if strike_aware_toeoff:
        toe_off_method = "per_strike_pattern"
    else:
        toe_off_method = "ankle_y_lift"

    extraction = extract_blazepose_landmarks_and_calibration(
        video_path,
        runner_height_cm,
        shoe_sole_cm=shoe_sole_cm,
        shoe_type=shoe_type,
        bidirectional=bidirectional,
        seeded_pass=seeded_pass,
        motion_filter=motion_filter,
        gap_fill=gap_fill,
        interp_method=interp_method,
        gap_max_frames=gap_max_frames,
        gap_min_anchor_vis=gap_min_anchor_vis,
        motion_filter_min_ms=motion_filter_min_ms,
        prune_spurious_contacts=prune_spurious_contacts,
        progress=progress,
    )
    cfr_path = extraction.cfr_path
    fps = extraction.fps
    res_h = extraction.res_h
    res_w = extraction.res_w
    skip_rate = extraction.skip_rate
    selected_landmarks = extraction.selected_landmarks
    selected_visibilities = extraction.selected_visibilities
    pose_quality = extraction.pose_quality
    body_height_px = extraction.body_height_px
    calibration = extraction.calibration
    estimated_velocity_ms = extraction.estimated_velocity_ms
    leg_length_m_for_prune = extraction.leg_length_m

    # Validate RTMPose frame alignment if supplied.
    if rtm_landmarks is not None and rtm_landmarks.shape[0] != len(selected_landmarks):
        raise ValueError(
            f"RTMPose T={rtm_landmarks.shape[0]} != BlazePose T="
            f"{len(selected_landmarks)} — frame extraction misaligned."
        )

    # 9. Clip-level strike pattern (v1.14; moved the vote itself into
    # detect_ground_contacts, so only a user-declared override is resolved
    # here). Leaving it None asks step 10 to vote over its own coarse list,
    # windowed to the runner's presence window -- see the note there.
    clip_strike_pattern: Optional[str] = None
    if toe_off_method == "per_strike_pattern":
        if clip_strike_pattern_override in ("forefoot", "heel"):
            clip_strike_pattern = clip_strike_pattern_override
    auto_strike = toe_off_method == "per_strike_pattern" and clip_strike_pattern is None

    # 10. Metrics (contacts + biomechanics), mirroring compute_metrics_on_window
    # Extraction now ends near 0.90 of its own scale (the two pose passes own
    # the span before it), so this has to sit above that, not at the old 0.85.
    _emit("Extracting metrics", 0.93)
    prune_active = prune_spurious_contacts and estimated_velocity_ms is not None
    contacts, _contact_summary = detect_ground_contacts(
        selected_landmarks,
        selected_visibilities,
        fps,
        resolution_height=res_h,
        toe_off_method=toe_off_method,
        body_height_px=body_height_px,
        tau_strike_frac=tau_strike_frac,
        delta_lift_frac=delta_lift_frac,
        delta_lift_frac_forefoot=delta_lift_frac_forefoot,
        ankle_horiz_vel_gate_frac=ankle_horiz_vel_gate_frac,
        clip_strike_pattern=clip_strike_pattern,
        prune_spurious=prune_active,
        velocity_ms=estimated_velocity_ms,
        leg_length_m=leg_length_m_for_prune,
        cadence_band_frac=cadence_band_frac,
        gate_landmark_bounds=gate_landmark_bounds,
        resolution_width=res_w,
        rtm_landmarks=rtm_landmarks,
        rtm_scores=rtm_scores,
        auto_clip_strike_pattern=auto_strike,
        strike_from_refined_ic=strike_from_refined_ic,
        demote_weak_strike_vote=demote_weak_strike_vote,
        unknown_as_forefoot=unknown_as_forefoot,
    )
    # Resolve the vote once: hand the decided pattern to extract_all_metrics
    # rather than letting it re-vote on its own coarse pass. It then hands
    # it the contacts themselves, so the ClipAnalysis.contacts returned below
    # and the metrics beside them are guaranteed to be the same detection
    # run -- forwarding parameters one by one only ever covered the ones
    # somebody remembered to forward.
    if auto_strike:
        clip_strike_pattern = _contact_summary.get("clip_strike_pattern", {}).get(
            "decision"
        )

    metrics = extract_all_metrics(
        selected_landmarks,
        selected_visibilities,
        fps,
        runner_height_cm,
        shoe_sole_cm=shoe_sole_cm,
        shoe_type=shoe_type,
        resolution_height=res_h,
        toe_off_method=toe_off_method,
        body_height_px=body_height_px,
        tau_strike_frac=tau_strike_frac,
        delta_lift_frac=delta_lift_frac,
        delta_lift_frac_forefoot=delta_lift_frac_forefoot,
        ankle_horiz_vel_gate_frac=ankle_horiz_vel_gate_frac,
        clip_strike_pattern=clip_strike_pattern,
        prune_spurious=prune_active,
        velocity_ms=estimated_velocity_ms,
        leg_length_m=leg_length_m_for_prune,
        cadence_band_frac=cadence_band_frac,
        gate_landmark_bounds=gate_landmark_bounds,
        resolution_width=res_w,
        rtm_landmarks=rtm_landmarks,
        rtm_scores=rtm_scores,
        unknown_as_forefoot=unknown_as_forefoot,
        contacts=contacts,
    )
    _, warnings = validate_metrics(metrics)
    reliability = assess_clip_reliability(pose_quality, metrics)

    _emit("Complete", 1.0)
    return ClipAnalysis(
        metrics=metrics,
        calibration=calibration,
        contacts=contacts,
        warnings=warnings,
        pose_quality=pose_quality,
        reliability=reliability,
        fps=fps,
        n_frames=int(len(selected_landmarks)),
        clip_strike_pattern=clip_strike_pattern,
        velocity_ms_estimated=estimated_velocity_ms,
        cfr_path=cfr_path,
        selected_landmarks=selected_landmarks,
        selected_visibilities=selected_visibilities,
        frame_width=int(res_w),
        skip_rate=int(skip_rate),
    )
