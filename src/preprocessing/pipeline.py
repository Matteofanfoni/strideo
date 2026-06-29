from dataclasses import dataclass
from typing import Callable, Dict, List, Optional

import numpy as np

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


def run_clip_pipeline(
    video_path: str,
    runner_height_cm: float,
    *,
    shoe_sole_cm: float = 2.5,
    shoe_type: Optional[str] = None,
    pace_level: str = "unknown",
    footwear_category: Optional[str] = None,
    rtm_landmarks: Optional[np.ndarray] = None,
    rtm_scores: Optional[np.ndarray] = None,
    # v1.20 flags (defaults mirror the canonical CLI invocation in CLAUDE.md)
    bidirectional: bool = True,
    seeded_pass: bool = True,
    gap_fill: bool = True,
    motion_filter: bool = True,
    strike_aware_toeoff: bool = True,
    prune_spurious_contacts: bool = True,
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
    progress: Optional[ProgressCallback] = None,
) -> ClipAnalysis:
    """Run the v1.20 preprocessing + metrics pipeline on one clip.

    This is a faithful transcription of ``main()`` in
    ``scripts/run_prevalidation_single.py`` for the canonical v1.20 invocation
    (``--bidirectional --gap-fill --strike-aware-toeoff
    --prune-spurious-contacts --seeded-pass --rtmpose-landmarks``), factored so
    both the CLI validator and the Streamlit app can share it.

    The orchestration deliberately does NOT run RTMPose itself: pass
    ``rtm_landmarks`` / ``rtm_scores`` (the CLI loads them from a cached npz;
    the app produces them live with ``rtmpose_extractor``). They must be
    frame-aligned to BlazePose (``T_rtm == T_blaze``).

    Args:
        video_path: Path to the source clip (CFR conversion handled here).
        runner_height_cm: Runner standing height (cm), for spatial calibration.
        shoe_sole_cm / shoe_type: Footwear sole offset / SHOE_TYPES key.
        pace_level / footwear_category: Hints for clip strike-pattern auto-detect.
        rtm_landmarks / rtm_scores: Live or cached RTMPose-x output, or None.
        progress: Optional callback(stage_label, fraction_0_to_1) for UI.

    Returns:
        ClipAnalysis with metrics, calibration, contacts, quality, fps.
    """
    from src.preprocessing.pose_estimator import (
        LandmarkIndex,
        extract_pose_landmarks_seeded,
        extract_pose_landmarks_streaming,
        extract_pose_landmarks_streaming_reverse,
    )
    from src.preprocessing.calibration import create_spatial_calibration
    from src.preprocessing.motion_filter import (
        combine_bidirectional_pose,
        combine_three_pass_pose,
        filter_landmarks_by_motion,
    )
    from src.preprocessing.landmarks_cleanup import fill_gaps
    from src.preprocessing.footage_extractor import detect_near_side
    from src.preprocessing.ground_contact import (
        detect_clip_strike_pattern,
        detect_contacts_coarse,
        detect_ground_contacts,
    )
    from src.utils.pace_estimator import estimate_velocity_from_hipx
    from src.utils.quality import validate_metrics

    def _emit(stage: str, frac: float) -> None:
        if progress is not None:
            progress(stage, frac)

    if strike_aware_toeoff:
        toe_off_method = "per_strike_pattern"
    else:
        toe_off_method = "ankle_y_lift"

    # 0. VFR -> CFR
    _emit("Preparing video", 0.02)
    cfr_path, _ = ensure_cfr(video_path)

    # 1. Forward pose pass — det_conf=0.2 matches the seeded pass and maximises
    # recall in headless CPU environments (e.g. HF Spaces) where the decoder
    # produces slightly different pixel values, pushing marginal detections
    # below the original 0.5 threshold.
    _emit("Pose estimation (forward)", 0.05)
    pose_result, video_meta = extract_pose_landmarks_streaming(
        cfr_path,
        num_poses=1,
        running_mode="video",
        min_pose_detection_confidence=0.2,
    )
    fps = video_meta["effective_fps"]
    res_h = video_meta["height"]
    res_w = video_meta["width"]
    skip_rate = video_meta["skip_rate"]

    # 2. Reverse pose pass (v1.6)
    pose_result_rev = None
    if bidirectional:
        _emit("Pose estimation (reverse)", 0.20)
        pose_result_rev, _ = extract_pose_landmarks_streaming_reverse(cfr_path)

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
        _emit("Combining passes", 0.35)
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
        _emit("Seeded refinement", 0.45)
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
            frames_rgb, fps, F_anchor
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
        _emit("Motion filtering", 0.60)
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
    if gap_fill:
        _emit("Gap-filling", 0.68)
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

    # Quality metrics on the filtered tensor (mirrors the CLI).
    hip_idx = [LandmarkIndex.LEFT_HIP, LandmarkIndex.RIGHT_HIP]
    ankle_idx = [LandmarkIndex.LEFT_ANKLE, LandmarkIndex.RIGHT_ANKLE]
    detected_mask = ~np.all(np.isnan(selected_landmarks), axis=(1, 2))
    T_total = len(selected_landmarks)
    pose_quality = {
        "total_frames": int(T_total),
        "detected_frames": int(detected_mask.sum()),
        "detection_rate": (float(detected_mask.sum() / T_total) if T_total else 0.0),
        "low_confidence_frames": int(
            np.sum(np.mean(selected_visibilities[:, hip_idx], axis=1) < 0.5)
        ),
        "mean_hip_visibility": float(np.nanmean(selected_visibilities[:, hip_idx])),
        "mean_ankle_visibility": float(np.nanmean(selected_visibilities[:, ankle_idx])),
        "filter": filter_info,
    }

    # Validate RTMPose frame alignment if supplied.
    if rtm_landmarks is not None and rtm_landmarks.shape[0] != len(selected_landmarks):
        raise ValueError(
            f"RTMPose T={rtm_landmarks.shape[0]} != BlazePose T="
            f"{len(selected_landmarks)} — frame extraction misaligned."
        )

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

    # 9. Clip-level strike pattern (v1.14)
    clip_strike_pattern: Optional[str] = None
    if toe_off_method == "per_strike_pattern":
        if clip_strike_pattern_override in ("forefoot", "heel"):
            clip_strike_pattern = clip_strike_pattern_override
        else:
            _coarse_contacts, _ = detect_contacts_coarse(
                selected_landmarks, selected_visibilities, fps, res_h
            )
            _pat = detect_clip_strike_pattern(
                pace_level,
                footwear_category,
                _coarse_contacts,
                selected_landmarks,
                selected_visibilities,
                body_height_px or 0.4 * res_h,
                rtm_landmarks=rtm_landmarks,
                rtm_scores=rtm_scores,
            )
            clip_strike_pattern = _pat["decision"]

    # 10. Metrics (contacts + biomechanics), mirroring compute_metrics_on_window
    _emit("Extracting metrics", 0.85)
    prune_active = prune_spurious_contacts and estimated_velocity_ms is not None
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
    contacts, _ = detect_ground_contacts(
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
        rtm_landmarks=rtm_landmarks,
        rtm_scores=rtm_scores,
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
        rtm_landmarks=rtm_landmarks,
        rtm_scores=rtm_scores,
    )
    _, warnings = validate_metrics(metrics)

    _emit("Complete", 1.0)
    return ClipAnalysis(
        metrics=metrics,
        calibration=calibration,
        contacts=contacts,
        warnings=warnings,
        pose_quality=pose_quality,
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
