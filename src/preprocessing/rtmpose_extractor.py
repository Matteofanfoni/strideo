"""In-process RTMPose-x landmark extraction (live, no pre-cached npz).

The pre-validation workflow extracts RTMPose landmarks once per clip with
``scripts/rtmpose/extract_landmarks.py`` and caches them to an .npz that the
pipeline later loads with ``--rtmpose-landmarks``. The live web app cannot do
that: a visitor uploads a brand-new clip, so RTMPose must run on the fly.

This module factors the core extraction into a reusable function that returns
the same ``(keypoints, scores)`` arrays the cached path produces, frame-aligned
to BlazePose via the shared ``skip_rate`` so the downstream
``T_rtm == T_blaze`` invariant holds.

RTMPose runs through rtmlib's ONNX backend; on the CPU host this is plain
``onnxruntime`` (no CUDA payload).
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Callable, List, Optional

import cv2
import numpy as np

# COCO-WholeBody body keypoint indices (0-16), used to pick the runner among
# any background detections by highest mean body-keypoint confidence.
_BODY_KP_INDICES: List[int] = list(range(17))

ProgressCallback = Callable[[int, int], None]


@dataclass
class RTMPoseResult:
    """Frame-aligned RTMPose-x output for one clip.

    keypoints: (T, 133, 2) pixel xy in COCO-WholeBody order.
    scores:    (T, 133) per-keypoint confidence.
    fps:       effective fps after skip_rate downsampling.
    """

    keypoints: np.ndarray
    scores: np.ndarray
    fps: float


def _select_runner(keypoints_all: np.ndarray, scores_all: np.ndarray) -> int:
    """Index of the person most likely to be the runner.

    Mirrors scripts/rtmpose/extract_landmarks.py: highest mean confidence
    across the 17 body keypoints (the most fully visible / largest person).
    """
    if keypoints_all.shape[0] == 1:
        return 0
    body_scores = scores_all[:, _BODY_KP_INDICES].mean(axis=1)
    return int(np.argmax(body_scores))


def extract_rtmpose_landmarks(
    video_path: str,
    *,
    target_fps: int = 60,
    start_frame: int = 0,
    end_frame: Optional[int] = None,
    device: str = "cpu",
    mode: str = "performance",
    progress: Optional[ProgressCallback] = None,
) -> RTMPoseResult:
    """Run RTMPose-x over a clip and return frame-aligned landmarks.

    Frame selection mirrors ``extract_pose_landmarks_streaming``:
    ``skip_rate = max(1, int(source_fps / target_fps))`` and every
    ``frame_idx % skip_rate == 0`` frame in ``[start_frame, end_frame]`` is
    processed. With the same ``target_fps`` this yields ``T`` identical to the
    BlazePose pass, satisfying the downstream length invariant.

    Args:
        video_path: Path to the CFR video.
        target_fps: Target fps; source is downsampled when higher.
        start_frame: First source-frame index to process (inclusive).
        end_frame: Last source-frame index (inclusive); None = end of video.
        device: rtmlib device, "cpu" (default) or "cuda".
        mode: rtmlib model tier — "performance" (RTMPose-x, the v1.20 model),
            "balanced", or "lightweight".
        progress: Optional callback(done, total) invoked per processed frame.

    Returns:
        RTMPoseResult with (T, 133, 2) keypoints, (T, 133) scores, effective fps.

    Raises:
        ValueError: If the video cannot be opened.
    """
    # Imported lazily so importing this module never pulls in rtmlib/onnxruntime
    # unless live extraction is actually requested.
    from rtmlib import Wholebody

    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        raise ValueError(f"Cannot open video: {video_path}")

    source_fps = cap.get(cv2.CAP_PROP_FPS)
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    skip_rate = max(1, int(source_fps / target_fps)) if source_fps > 0 else 1
    effective_fps = source_fps / skip_rate if source_fps > 0 else float(target_fps)

    resolved_end = (
        total_frames - 1 if end_frame is None else min(end_frame, total_frames - 1)
    )
    window_frames = max(0, resolved_end - start_frame + 1)
    expected = (window_frames + skip_rate - 1) // skip_rate

    wholebody = Wholebody(
        mode=mode,
        to_openpose=False,
        backend="onnxruntime",
        device=device,
    )

    keypoints = np.zeros((expected, 133, 2), dtype=np.float32)
    scores = np.zeros((expected, 133), dtype=np.float32)

    if start_frame > 0:
        cap.set(cv2.CAP_PROP_POS_FRAMES, start_frame)

    frame_idx = start_frame
    t = 0
    try:
        while t < expected:
            ok, frame = cap.read()
            if not ok or frame_idx > resolved_end:
                break
            if (frame_idx - start_frame) % skip_rate == 0:
                kps, kp_scores = wholebody(frame)  # (N, 133, 2), (N, 133)
                if kps is not None and len(kps) > 0:
                    idx = _select_runner(kps, kp_scores)
                    keypoints[t] = kps[idx]
                    scores[t] = kp_scores[idx]
                # else: leave zeros (no detection this frame)
                t += 1
                if progress is not None:
                    progress(t, expected)
            frame_idx += 1
    finally:
        cap.release()

    # Trim if the video ended early (fewer real frames than expected).
    if t < expected:
        keypoints = keypoints[:t]
        scores = scores[:t]

    return RTMPoseResult(keypoints=keypoints, scores=scores, fps=effective_fps)
