from dataclasses import dataclass
from enum import IntEnum
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import cv2
import mediapipe as mp
import numpy as np

# Default model paths (relative to project root)
MODEL_DIR = Path(__file__).resolve().parents[2] / "models"
MODEL_HEAVY = MODEL_DIR / "pose_landmarker_heavy.task"
MODEL_FULL = MODEL_DIR / "pose_landmarker_full.task"


def create_pose_estimator(
    model_path: Path = MODEL_HEAVY,
    num_poses: int = 1,
    running_mode: str = "video",
    min_pose_detection_confidence: float = 0.5,
    min_pose_presence_confidence: float = 0.5,
    min_tracking_confidence: float = 0.5,
) -> mp.tasks.vision.PoseLandmarker:
    """
    Create MediaPipe PoseLandmarker with optimal settings for
    running analysis.

    Args:
        model_path: Path to the .task model bundle.
            MODEL_HEAVY (default) for highest accuracy,
            MODEL_FULL for faster processing.
        num_poses: Maximum number of pose candidates returned per frame.
            1 (default) preserves v1.0 behaviour. Higher values allow a
            downstream selector (see src/preprocessing/pose_selection.py)
            to pick the runner among background ghost detections.
        running_mode: "video" (default) uses VIDEO mode with frame-to-frame
            tracking. "image" uses IMAGE mode (per-frame detection, no
            tracking) — slower but immune to tracker lock-in on background
            ghosts. Required when num_poses > 1 is expected to actually
            return multiple candidates: VIDEO mode's tracker only follows
            the person(s) found at initial detection.
        min_pose_detection_confidence: Detector acceptance threshold.
            Lower values (e.g. 0.2) surface weaker detections — useful
            when the target (runner) may compete with stronger background
            priors (e.g. fence).
        min_pose_presence_confidence: Landmark presence threshold.
        min_tracking_confidence: Tracking confidence threshold (ignored
            in IMAGE mode).

    Returns:
        Configured PoseLandmarker object
    """
    mode_map = {
        "video": mp.tasks.vision.RunningMode.VIDEO,
        "image": mp.tasks.vision.RunningMode.IMAGE,
    }
    if running_mode not in mode_map:
        raise ValueError(
            f"Unknown running_mode={running_mode!r}; expected 'video' or 'image'"
        )

    options = mp.tasks.vision.PoseLandmarkerOptions(
        base_options=mp.tasks.BaseOptions(
            model_asset_path=str(model_path),
            delegate=mp.tasks.BaseOptions.Delegate.CPU,
        ),
        running_mode=mode_map[running_mode],
        num_poses=num_poses,
        min_pose_detection_confidence=min_pose_detection_confidence,
        min_pose_presence_confidence=min_pose_presence_confidence,
        min_tracking_confidence=min_tracking_confidence,
        output_segmentation_masks=False,
    )

    return mp.tasks.vision.PoseLandmarker.create_from_options(options)


class LandmarkIndex(IntEnum):
    """MediaPipe Pose landmark indices."""

    NOSE = 0
    LEFT_EYE_INNER = 1
    LEFT_EYE = 2
    LEFT_EYE_OUTER = 3
    RIGHT_EYE_INNER = 4
    RIGHT_EYE = 5
    RIGHT_EYE_OUTER = 6
    LEFT_EAR = 7
    RIGHT_EAR = 8
    MOUTH_LEFT = 9
    MOUTH_RIGHT = 10
    LEFT_SHOULDER = 11
    RIGHT_SHOULDER = 12
    LEFT_ELBOW = 13
    RIGHT_ELBOW = 14
    LEFT_WRIST = 15
    RIGHT_WRIST = 16
    LEFT_PINKY = 17
    RIGHT_PINKY = 18
    LEFT_INDEX = 19
    RIGHT_INDEX = 20
    LEFT_THUMB = 21
    RIGHT_THUMB = 22
    LEFT_HIP = 23
    RIGHT_HIP = 24
    LEFT_KNEE = 25
    RIGHT_KNEE = 26
    LEFT_ANKLE = 27
    RIGHT_ANKLE = 28
    LEFT_HEEL = 29
    RIGHT_HEEL = 30
    LEFT_FOOT_INDEX = 31
    RIGHT_FOOT_INDEX = 32


@dataclass
class PoseExtractionResult:
    """Container for pose extraction results.

    Single-pose fields (`landmarks`, `visibilities`, `world_landmarks`) carry
    candidate 0 — the top-scoring pose MediaPipe returns for that frame.
    Callers wanting backward compatibility with the v1.0 pipeline ignore the
    multi-* fields and use these three arrays.

    Multi-pose fields are populated when the estimator was created with
    `num_poses > 1`. They carry all candidates for every frame and are used
    by `src/preprocessing/pose_selection.py` to pick the true runner when
    background structures get misdetected as people.
    """

    landmarks: np.ndarray  # (T, 33, 2) pixel coordinates, candidate 0
    visibilities: np.ndarray  # (T, 33) visibility scores [0-1], candidate 0
    world_landmarks: np.ndarray  # (T, 33, 3) metric coordinates, candidate 0
    quality_metrics: Dict  # Detection statistics

    # Multi-candidate tensors (populated when num_poses > 1)
    multi_landmarks: Optional[np.ndarray] = None  # (T, N, 33, 2)
    multi_visibilities: Optional[np.ndarray] = None  # (T, N, 33)
    multi_world_landmarks: Optional[np.ndarray] = None  # (T, N, 33, 3)
    n_poses: int = 1  # Max candidates the estimator was configured for


def extract_pose_landmarks(
    frames: List[np.ndarray],
    fps: float,
    pose_estimator: mp.tasks.vision.PoseLandmarker = None,
    num_poses: int = 1,
) -> PoseExtractionResult:
    """
    Extract pose landmarks from video frames.

    Args:
        frames: List of RGB frame arrays
        fps: Video frame rate (needed for timestamp computation)
        pose_estimator: PoseLandmarker object (created if None)
        num_poses: Max candidates per frame. 1 = v1.0 behaviour; >1 also
            populates the multi-* arrays on the result.

    Returns:
        PoseExtractionResult with landmarks, visibilities,
        and quality metrics
    """
    owns_estimator = pose_estimator is None
    if owns_estimator:
        pose_estimator = create_pose_estimator(num_poses=num_poses)

    height, width = frames[0].shape[:2]
    T = len(frames)

    landmarks = np.full((T, 33, 2), np.nan)
    visibilities = np.zeros((T, 33))
    world_landmarks = np.full((T, 33, 3), np.nan)

    multi_landmarks = np.full((T, num_poses, 33, 2), np.nan) if num_poses > 1 else None
    multi_visibilities = np.zeros((T, num_poses, 33)) if num_poses > 1 else None
    multi_world_landmarks = (
        np.full((T, num_poses, 33, 3), np.nan) if num_poses > 1 else None
    )

    detection_count = 0
    low_confidence_frames = 0
    total_candidates = 0

    for t, frame in enumerate(frames):
        mp_image = mp.Image(image_format=mp.ImageFormat.SRGB, data=frame)
        timestamp_ms = int(t * (1000 / fps))

        result = pose_estimator.detect_for_video(mp_image, timestamp_ms)

        if result.pose_landmarks:
            detection_count += 1
            total_candidates += len(result.pose_landmarks)

            # Candidate 0 → backward-compatible single-pose fields
            for i, lm in enumerate(result.pose_landmarks[0]):
                landmarks[t, i, 0] = lm.x * width
                landmarks[t, i, 1] = lm.y * height
                visibilities[t, i] = lm.visibility

            hip_vis = np.mean(
                [
                    visibilities[t, LandmarkIndex.LEFT_HIP],
                    visibilities[t, LandmarkIndex.RIGHT_HIP],
                ]
            )
            if hip_vis < 0.5:
                low_confidence_frames += 1

            # All candidates → multi-pose fields
            if multi_landmarks is not None and multi_visibilities is not None:
                for k, cand in enumerate(result.pose_landmarks[:num_poses]):
                    for i, lm in enumerate(cand):
                        multi_landmarks[t, k, i, 0] = lm.x * width
                        multi_landmarks[t, k, i, 1] = lm.y * height
                        multi_visibilities[t, k, i] = lm.visibility

        if result.pose_world_landmarks:
            for i, lm in enumerate(result.pose_world_landmarks[0]):
                world_landmarks[t, i] = [lm.x, lm.y, lm.z]
            if multi_world_landmarks is not None:
                for k, cand in enumerate(result.pose_world_landmarks[:num_poses]):
                    for i, lm in enumerate(cand):
                        multi_world_landmarks[t, k, i] = [lm.x, lm.y, lm.z]

    if owns_estimator:
        pose_estimator.close()

    hip_indices = [
        LandmarkIndex.LEFT_HIP,
        LandmarkIndex.RIGHT_HIP,
    ]
    ankle_indices = [
        LandmarkIndex.LEFT_ANKLE,
        LandmarkIndex.RIGHT_ANKLE,
    ]
    quality_metrics = {
        "total_frames": T,
        "detected_frames": detection_count,
        "detection_rate": detection_count / T if T > 0 else 0,
        "low_confidence_frames": low_confidence_frames,
        "mean_hip_visibility": np.nanmean(visibilities[:, hip_indices]),
        "mean_ankle_visibility": np.nanmean(visibilities[:, ankle_indices]),
        "mean_n_candidates": (
            total_candidates / detection_count if detection_count > 0 else 0
        ),
    }
    print(
        f"[pose] detection_rate={quality_metrics['detection_rate']:.2%}"
        f" ({detection_count}/{T} frames)",
        flush=True,
    )

    return PoseExtractionResult(
        landmarks=landmarks,
        visibilities=visibilities,
        world_landmarks=world_landmarks,
        quality_metrics=quality_metrics,
        multi_landmarks=multi_landmarks,
        multi_visibilities=multi_visibilities,
        multi_world_landmarks=multi_world_landmarks,
        n_poses=num_poses,
    )


def extract_pose_landmarks_streaming(
    video_path: str,
    target_fps: int = 60,
    pose_estimator: mp.tasks.vision.PoseLandmarker = None,
    progress_every: int = 60,
    num_poses: int = 1,
    running_mode: str = "video",
    min_pose_detection_confidence: float = 0.5,
    start_frame: int = 0,
    end_frame: Optional[int] = None,
) -> Tuple[PoseExtractionResult, Dict]:
    """
    Extract pose landmarks directly from a video file without
    materialising all frames in memory.

    Use when the source video is too large to fit in RAM (e.g. 4K CFR
    clips). Functionally equivalent to ``load_video`` + ``extract_pose_landmarks``
    but keeps at most one frame alive at a time.

    Args:
        video_path: Path to the CFR video file.
        target_fps: Target frame rate; source is downsampled when higher.
        pose_estimator: Pre-built PoseLandmarker; created if None.
        progress_every: Print a progress line every N frames (0 = silent).
        num_poses: Max candidates per frame. 1 = v1.0 behaviour; >1 also
            populates the multi-* arrays on the result.
        running_mode: "video" (VIDEO + tracking) or "image" (per-frame
            detection, no tracking). See ``create_pose_estimator``.
        min_pose_detection_confidence: Lowering from 0.5 surfaces weaker
            detections so a small/distant runner can appear as a candidate
            alongside background priors.
        start_frame: Source-video frame index to start processing from
            (inclusive). Used by motion gating (M10) to skip idle leading
            frames so the VIDEO-mode tracker's first detection happens on
            a frame where the runner is already moving.
        end_frame: Source-video frame index to stop at (inclusive). None
            means process until the video ends.

    Returns:
        result: PoseExtractionResult with (T, 33, *) landmark/visibility arrays.
        video_meta: Same dict structure as video_io.load_video metadata,
            with added ``start_frame`` / ``end_frame`` fields.

    Raises:
        ValueError: If the video cannot be opened or is shorter than 1.5 s.
    """
    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        raise ValueError(f"Cannot open video: {video_path}")

    source_fps = cap.get(cv2.CAP_PROP_FPS)
    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    duration = total_frames / source_fps if source_fps > 0 else 0.0

    if duration < 1.5:
        cap.release()
        raise ValueError(f"Video too short: {duration:.2f}s (absolute minimum 1.5s)")

    skip_rate = max(1, int(source_fps / target_fps))
    effective_fps = source_fps / skip_rate

    if start_frame > 0:
        cap.set(cv2.CAP_PROP_POS_FRAMES, start_frame)
    resolved_end = (
        total_frames - 1 if end_frame is None else min(end_frame, total_frames - 1)
    )
    window_frames = max(0, resolved_end - start_frame + 1)
    expected_extracted = (window_frames + skip_rate - 1) // skip_rate

    owns_estimator = pose_estimator is None
    if owns_estimator:
        pose_estimator = create_pose_estimator(
            num_poses=num_poses,
            running_mode=running_mode,
            min_pose_detection_confidence=min_pose_detection_confidence,
        )
    is_image_mode = running_mode == "image"

    landmarks = np.full((expected_extracted, 33, 2), np.nan)
    visibilities = np.zeros((expected_extracted, 33))
    world_landmarks = np.full((expected_extracted, 33, 3), np.nan)

    multi_landmarks = (
        np.full((expected_extracted, num_poses, 33, 2), np.nan)
        if num_poses > 1
        else None
    )
    multi_visibilities = (
        np.zeros((expected_extracted, num_poses, 33)) if num_poses > 1 else None
    )
    multi_world_landmarks = (
        np.full((expected_extracted, num_poses, 33, 3), np.nan)
        if num_poses > 1
        else None
    )

    detection_count = 0
    low_confidence_frames = 0
    total_candidates = 0
    t = 0
    frame_idx = 0
    source_frame_idx = start_frame

    while True:
        ret, frame = cap.read()
        if not ret:
            break
        if source_frame_idx > resolved_end:
            break

        if frame_idx % skip_rate == 0:
            if t >= expected_extracted:
                break
            frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            mp_image = mp.Image(image_format=mp.ImageFormat.SRGB, data=frame_rgb)
            if is_image_mode:
                result = pose_estimator.detect(mp_image)
            else:
                timestamp_ms = int(t * (1000 / effective_fps))
                result = pose_estimator.detect_for_video(mp_image, timestamp_ms)

            if result.pose_landmarks:
                detection_count += 1
                total_candidates += len(result.pose_landmarks)

                for i, lm in enumerate(result.pose_landmarks[0]):
                    landmarks[t, i, 0] = lm.x * width
                    landmarks[t, i, 1] = lm.y * height
                    visibilities[t, i] = lm.visibility

                hip_vis = np.mean(
                    [
                        visibilities[t, LandmarkIndex.LEFT_HIP],
                        visibilities[t, LandmarkIndex.RIGHT_HIP],
                    ]
                )
                if hip_vis < 0.5:
                    low_confidence_frames += 1

                if multi_landmarks is not None and multi_visibilities is not None:
                    for k, cand in enumerate(result.pose_landmarks[:num_poses]):
                        for i, lm in enumerate(cand):
                            multi_landmarks[t, k, i, 0] = lm.x * width
                            multi_landmarks[t, k, i, 1] = lm.y * height
                            multi_visibilities[t, k, i] = lm.visibility

            if result.pose_world_landmarks:
                for i, lm in enumerate(result.pose_world_landmarks[0]):
                    world_landmarks[t, i] = [lm.x, lm.y, lm.z]
                if multi_world_landmarks is not None:
                    for k, cand in enumerate(result.pose_world_landmarks[:num_poses]):
                        for i, lm in enumerate(cand):
                            multi_world_landmarks[t, k, i] = [
                                lm.x,
                                lm.y,
                                lm.z,
                            ]

            t += 1
            if progress_every and t % progress_every == 0:
                print(f"  pose streaming: {t}/{expected_extracted} frames")

        frame_idx += 1
        source_frame_idx += 1

    cap.release()
    if owns_estimator:
        pose_estimator.close()

    T = t
    landmarks = landmarks[:T]
    visibilities = visibilities[:T]
    world_landmarks = world_landmarks[:T]
    if (
        multi_landmarks is not None
        and multi_visibilities is not None
        and multi_world_landmarks is not None
    ):
        multi_landmarks = multi_landmarks[:T]
        multi_visibilities = multi_visibilities[:T]
        multi_world_landmarks = multi_world_landmarks[:T]

    hip_indices = [LandmarkIndex.LEFT_HIP, LandmarkIndex.RIGHT_HIP]
    ankle_indices = [LandmarkIndex.LEFT_ANKLE, LandmarkIndex.RIGHT_ANKLE]
    quality_metrics = {
        "total_frames": T,
        "detected_frames": detection_count,
        "detection_rate": detection_count / T if T > 0 else 0,
        "low_confidence_frames": low_confidence_frames,
        "mean_hip_visibility": float(np.nanmean(visibilities[:, hip_indices])),
        "mean_ankle_visibility": float(np.nanmean(visibilities[:, ankle_indices])),
        "mean_n_candidates": (
            total_candidates / detection_count if detection_count > 0 else 0
        ),
    }
    print(
        f"[pose] detection_rate={quality_metrics['detection_rate']:.2%}"
        f" ({detection_count}/{T} frames)",
        flush=True,
    )

    video_meta = {
        "source_path": video_path,
        "source_fps": source_fps,
        "effective_fps": effective_fps,
        "width": width,
        "height": height,
        "total_frames": total_frames,
        "extracted_frames": T,
        "duration_seconds": duration,
        "skip_rate": skip_rate,
        "start_frame": int(start_frame),
        "end_frame": int(resolved_end),
    }

    result_obj = PoseExtractionResult(
        landmarks=landmarks,
        visibilities=visibilities,
        world_landmarks=world_landmarks,
        quality_metrics=quality_metrics,
        multi_landmarks=multi_landmarks,
        multi_visibilities=multi_visibilities,
        multi_world_landmarks=multi_world_landmarks,
        n_poses=num_poses,
    )
    return result_obj, video_meta


def extract_pose_landmarks_streaming_reverse(
    video_path: str,
    target_fps: int = 60,
    pose_estimator: mp.tasks.vision.PoseLandmarker = None,
    progress_every: int = 60,
) -> Tuple[PoseExtractionResult, Dict]:
    """
    Extract pose landmarks from a video file with frames fed in reverse
    order. Companion to ``extract_pose_landmarks_streaming`` — used by the
    bidirectional pipeline (v1.6) to recover the pre-lock-in window the
    forward pass misses.

    The tracker is unaware of real-world time direction; it only sees
    inter-frame motion. Feeding frames in reverse therefore makes the
    detector enter the clip from the post-runner-exit period (empty FOV,
    same ghost-lock failure mode as forward), then snap to the runner
    around the same mid-clip moment, and track *backwards* through the
    original frames the forward pass missed.

    Implementation:
      1. Buffer every frame into RAM at the target_fps grid (so the
         forward and reverse passes share an identical temporal index).
         Memory: ~25 MB / 4K frame × ~330 frames ≈ 8 GB peak.
      2. Feed frames in reverse to MediaPipe with monotonically-increasing
         timestamps (0, 1/fps, 2/fps, …) — the API requires monotonic
         timestamps regardless of the wall-clock direction of motion.
      3. Reverse the resulting arrays along the time axis before
         returning, so callers see forward-time indexing identical to
         ``extract_pose_landmarks_streaming``.

    Args:
        video_path: Path to the CFR video file.
        target_fps: Target frame rate; source is downsampled when higher.
        pose_estimator: Pre-built PoseLandmarker; created if None. The
            estimator is fed the entire reverse stream — do not reuse a
            forward-pass estimator instance (its tracker state and
            timestamp cursor would be poisoned).
        progress_every: Print a progress line every N frames (0 = silent).

    Returns:
        result: PoseExtractionResult in source-frame order (index 0 =
            first source frame).
        video_meta: Same shape as ``extract_pose_landmarks_streaming``'s
            metadata dict, with ``pass_direction = "reverse"``.

    Raises:
        ValueError: If the video cannot be opened or is shorter than 1.5 s.
    """
    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        raise ValueError(f"Cannot open video: {video_path}")

    source_fps = cap.get(cv2.CAP_PROP_FPS)
    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    duration = total_frames / source_fps if source_fps > 0 else 0.0

    if duration < 1.5:
        cap.release()
        raise ValueError(f"Video too short: {duration:.2f}s (absolute minimum 1.5s)")

    skip_rate = max(1, int(source_fps / target_fps))
    effective_fps = source_fps / skip_rate

    frames_rgb: List[np.ndarray] = []
    frame_idx = 0
    while True:
        ret, frame = cap.read()
        if not ret:
            break
        if frame_idx % skip_rate == 0:
            frames_rgb.append(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB))
        frame_idx += 1
    cap.release()

    T = len(frames_rgb)
    if T == 0:
        raise ValueError(f"No frames extracted from {video_path}")

    owns_estimator = pose_estimator is None
    if owns_estimator:
        pose_estimator = create_pose_estimator(
            num_poses=1,
            running_mode="video",
            min_pose_detection_confidence=0.5,
        )

    # Reverse-feed buffers: index t in feed order = source-frame index (T-1-t).
    landmarks_rev = np.full((T, 33, 2), np.nan)
    visibilities_rev = np.zeros((T, 33))
    world_landmarks_rev = np.full((T, 33, 3), np.nan)

    detection_count = 0
    low_confidence_frames = 0
    total_candidates = 0

    for t in range(T):
        frame_rgb = frames_rgb[T - 1 - t]
        mp_image = mp.Image(image_format=mp.ImageFormat.SRGB, data=frame_rgb)
        timestamp_ms = int(t * (1000 / effective_fps))
        result = pose_estimator.detect_for_video(mp_image, timestamp_ms)

        if result.pose_landmarks:
            detection_count += 1
            total_candidates += len(result.pose_landmarks)

            for i, lm in enumerate(result.pose_landmarks[0]):
                landmarks_rev[t, i, 0] = lm.x * width
                landmarks_rev[t, i, 1] = lm.y * height
                visibilities_rev[t, i] = lm.visibility

            hip_vis = np.mean(
                [
                    visibilities_rev[t, LandmarkIndex.LEFT_HIP],
                    visibilities_rev[t, LandmarkIndex.RIGHT_HIP],
                ]
            )
            if hip_vis < 0.5:
                low_confidence_frames += 1

        if result.pose_world_landmarks:
            for i, lm in enumerate(result.pose_world_landmarks[0]):
                world_landmarks_rev[t, i] = [lm.x, lm.y, lm.z]

        if progress_every and (t + 1) % progress_every == 0:
            print(f"  pose streaming (reverse): {t + 1}/{T} frames")

    if owns_estimator:
        pose_estimator.close()
    del frames_rgb

    landmarks = landmarks_rev[::-1].copy()
    visibilities = visibilities_rev[::-1].copy()
    world_landmarks = world_landmarks_rev[::-1].copy()

    hip_indices = [LandmarkIndex.LEFT_HIP, LandmarkIndex.RIGHT_HIP]
    ankle_indices = [LandmarkIndex.LEFT_ANKLE, LandmarkIndex.RIGHT_ANKLE]
    quality_metrics = {
        "total_frames": T,
        "detected_frames": detection_count,
        "detection_rate": detection_count / T if T > 0 else 0,
        "low_confidence_frames": low_confidence_frames,
        "mean_hip_visibility": float(np.nanmean(visibilities[:, hip_indices])),
        "mean_ankle_visibility": float(np.nanmean(visibilities[:, ankle_indices])),
        "mean_n_candidates": (
            total_candidates / detection_count if detection_count > 0 else 0
        ),
        "pass_direction": "reverse",
    }

    video_meta = {
        "source_path": video_path,
        "source_fps": source_fps,
        "effective_fps": effective_fps,
        "width": width,
        "height": height,
        "total_frames": total_frames,
        "extracted_frames": T,
        "duration_seconds": duration,
        "skip_rate": skip_rate,
        "pass_direction": "reverse",
    }

    result_obj = PoseExtractionResult(
        landmarks=landmarks,
        visibilities=visibilities,
        world_landmarks=world_landmarks,
        quality_metrics=quality_metrics,
        n_poses=1,
    )
    return result_obj, video_meta


def extract_pose_landmarks_seeded(
    frames: List[np.ndarray],
    fps: float,
    anchor_frame: int,
) -> Tuple[PoseExtractionResult, PoseExtractionResult]:
    """Run seeded-backward (anchor→0) and seeded-forward (anchor→T-1) passes.

    Both sub-passes start from a well-detected mid-clip anchor frame using
    the same VIDEO-mode timestamp trick as the reverse pass: frames are fed
    with monotonically-increasing synthetic timestamps into a fresh estimator.
    Starting from the anchor gives the tracker a warm start at the point of
    highest detection quality, recovering entry/exit zones that cold-start
    passes miss.

    Args:
        frames: RGB frame arrays in source-frame order, pre-extracted at
            the target fps grid (same list the reverse pass would consume).
        fps: Effective frame rate of the frames list.
        anchor_frame: Index of the anchor frame in ``frames``.

    Returns:
        result_bwd: Source-frame-order result. Frames [0, anchor_frame]
            populated; frames (anchor_frame, T-1] are NaN.
        result_fwd: Source-frame-order result. Frames [anchor_frame, T-1]
            populated; frames [0, anchor_frame) are NaN.
    """
    T = len(frames)
    if T == 0:
        raise ValueError("frames list is empty")
    anchor_frame = int(np.clip(anchor_frame, 0, T - 1))
    height, width = frames[0].shape[:2]

    hip_indices = [LandmarkIndex.LEFT_HIP, LandmarkIndex.RIGHT_HIP]
    ankle_indices = [LandmarkIndex.LEFT_ANKLE, LandmarkIndex.RIGHT_ANKLE]

    # --- seeded-backward: anchor_frame → anchor_frame-1 → ... → 0 ---
    # Lower detection threshold vs fwd/rev (0.2 vs 0.5): the anchor frame is
    # mid-gait at race pace, where the cold-start detector fires marginally.
    # The tracker maintains tracking after initial lock-on.
    n_bwd = anchor_frame + 1
    estimator_bwd = create_pose_estimator(
        num_poses=1,
        running_mode="video",
        min_pose_detection_confidence=0.2,
    )
    lm_bwd_feed = np.full((n_bwd, 33, 2), np.nan)
    vis_bwd_feed = np.zeros((n_bwd, 33))
    world_bwd_feed = np.full((n_bwd, 33, 3), np.nan)

    for t in range(n_bwd):
        frame_rgb = frames[anchor_frame - t]
        mp_image = mp.Image(image_format=mp.ImageFormat.SRGB, data=frame_rgb)
        timestamp_ms = int(t * (1000.0 / fps))
        result = estimator_bwd.detect_for_video(mp_image, timestamp_ms)
        if result.pose_landmarks:
            for i, lm in enumerate(result.pose_landmarks[0]):
                lm_bwd_feed[t, i, 0] = lm.x * width
                lm_bwd_feed[t, i, 1] = lm.y * height
                vis_bwd_feed[t, i] = lm.visibility
        if result.pose_world_landmarks:
            for i, lm in enumerate(result.pose_world_landmarks[0]):
                world_bwd_feed[t, i] = [lm.x, lm.y, lm.z]
    estimator_bwd.close()

    # Flip feed order → source order: feed index t = source frame anchor_frame - t
    lm_bwd_src = lm_bwd_feed[::-1].copy()
    vis_bwd_src = vis_bwd_feed[::-1].copy()
    world_bwd_src = world_bwd_feed[::-1].copy()

    lm_bwd = np.full((T, 33, 2), np.nan)
    vis_bwd = np.zeros((T, 33))
    world_bwd = np.full((T, 33, 3), np.nan)
    lm_bwd[:n_bwd] = lm_bwd_src
    vis_bwd[:n_bwd] = vis_bwd_src
    world_bwd[:n_bwd] = world_bwd_src

    # --- seeded-forward: anchor_frame → anchor_frame+1 → ... → T-1 ---
    n_fwd = T - anchor_frame
    estimator_fwd = create_pose_estimator(
        num_poses=1,
        running_mode="video",
        min_pose_detection_confidence=0.2,
    )
    lm_fwd_feed = np.full((n_fwd, 33, 2), np.nan)
    vis_fwd_feed = np.zeros((n_fwd, 33))
    world_fwd_feed = np.full((n_fwd, 33, 3), np.nan)

    for t in range(n_fwd):
        frame_rgb = frames[anchor_frame + t]
        mp_image = mp.Image(image_format=mp.ImageFormat.SRGB, data=frame_rgb)
        timestamp_ms = int(t * (1000.0 / fps))
        result = estimator_fwd.detect_for_video(mp_image, timestamp_ms)
        if result.pose_landmarks:
            for i, lm in enumerate(result.pose_landmarks[0]):
                lm_fwd_feed[t, i, 0] = lm.x * width
                lm_fwd_feed[t, i, 1] = lm.y * height
                vis_fwd_feed[t, i] = lm.visibility
        if result.pose_world_landmarks:
            for i, lm in enumerate(result.pose_world_landmarks[0]):
                world_fwd_feed[t, i] = [lm.x, lm.y, lm.z]
    estimator_fwd.close()

    # Already in source order: lm_fwd_feed[0] = source frame anchor_frame
    lm_fwd = np.full((T, 33, 2), np.nan)
    vis_fwd = np.zeros((T, 33))
    world_fwd = np.full((T, 33, 3), np.nan)
    lm_fwd[anchor_frame:] = lm_fwd_feed
    vis_fwd[anchor_frame:] = vis_fwd_feed
    world_fwd[anchor_frame:] = world_fwd_feed

    det_bwd = int(np.sum(~np.all(np.isnan(lm_bwd[:n_bwd]), axis=(1, 2))))
    qm_bwd = {
        "total_frames": T,
        "detected_frames": det_bwd,
        "detection_rate": det_bwd / T if T > 0 else 0.0,
        "low_confidence_frames": 0,
        "mean_hip_visibility": float(np.nanmean(vis_bwd[:, hip_indices])),
        "mean_ankle_visibility": float(np.nanmean(vis_bwd[:, ankle_indices])),
        "pass_direction": "seed_bwd",
    }
    det_fwd_count = int(np.sum(~np.all(np.isnan(lm_fwd[anchor_frame:]), axis=(1, 2))))
    qm_fwd = {
        "total_frames": T,
        "detected_frames": det_fwd_count,
        "detection_rate": det_fwd_count / T if T > 0 else 0.0,
        "low_confidence_frames": 0,
        "mean_hip_visibility": float(np.nanmean(vis_fwd[:, hip_indices])),
        "mean_ankle_visibility": float(np.nanmean(vis_fwd[:, ankle_indices])),
        "pass_direction": "seed_fwd",
    }

    result_bwd = PoseExtractionResult(
        landmarks=lm_bwd,
        visibilities=vis_bwd,
        world_landmarks=world_bwd,
        quality_metrics=qm_bwd,
        n_poses=1,
    )
    result_fwd = PoseExtractionResult(
        landmarks=lm_fwd,
        visibilities=vis_fwd,
        world_landmarks=world_fwd,
        quality_metrics=qm_fwd,
        n_poses=1,
    )
    return result_bwd, result_fwd
