"""In-process RTMPose-x landmark extraction (live, no pre-cached npz).

The pre-validation workflow extracts RTMPose landmarks once per clip with
``scripts/rtmpose/extract_landmarks.py`` and caches them to an .npz that the
pipeline later loads with ``--rtmpose-landmarks``. The live web app cannot do
that: a visitor uploads a brand-new clip, so RTMPose must run on the fly.

This module factors the core extraction into a reusable function that returns
the same ``(keypoints, scores)`` arrays the cached path produces, frame-aligned
to BlazePose via the shared ``skip_rate`` so the downstream
``T_rtm == T_blaze`` invariant holds.

RTMPose runs through rtmlib's ONNX backend. The app image installs
``onnxruntime-gpu`` on every tier, so ``load_rtmpose`` decides the device
from the session it builds rather than from what is installed.
"""

from __future__ import annotations

import logging
import math
from dataclasses import dataclass
from typing import Any, Callable, List, Optional, Tuple

import cv2
import numpy as np

logger = logging.getLogger(__name__)

# COCO-WholeBody body keypoint indices (0-16), used to pick the runner among
# any background detections by highest mean body-keypoint confidence.
_BODY_KP_INDICES: List[int] = list(range(17))

ProgressCallback = Callable[[int, int], None]

#: Frames added each side of BlazePose's presence window before RTMPose is
#: run on it. Replaying all 50 GT clips with RTMPose masked outside the
#: window, 20 frames was the smallest padding that left every contact and
#: metric unchanged against the full-clip run; 30 keeps a 10-frame margin.
RTM_WINDOW_PAD_FRAMES = 30


def rtmpose_window(
    landmarks: np.ndarray, pad: int = RTM_WINDOW_PAD_FRAMES
) -> Optional[Tuple[int, int]]:
    """Inclusive pose-frame range RTMPose needs to run on, or None for all.

    BlazePose's presence window (``contact_filter.presence_window``, the
    window ``deployable_filter`` gates contacts on) widened by ``pad`` frames
    each side and clamped to the clip. ``landmarks`` is BlazePose's final
    ``selected_landmarks``, which exist before RTMPose runs. None when no
    frame has both a hip and an ankle, so the caller runs the whole clip.
    """
    from src.utils.contact_filter import presence_window

    pres = presence_window(landmarks)
    if pres is None:
        return None
    last = len(landmarks) - 1
    return (
        max(0, int(math.floor(pres[0])) - pad),
        min(last, int(math.ceil(pres[1])) + pad),
    )


def mask_to_window(
    keypoints: np.ndarray, scores: np.ndarray, window: Optional[Tuple[int, int]]
) -> Tuple[np.ndarray, np.ndarray]:
    """Copies of full-clip RTMPose arrays with every frame outside ``window``
    zeroed, i.e. what a windowed extraction returns: RTMPose is stateless per
    frame and writes zeros for a frame it does not detect on. Lets a cached
    full-clip extraction stand in for a windowed live one."""
    kp, sc = keypoints.copy(), scores.copy()
    if window is not None:
        outside = np.ones(len(kp), dtype=bool)
        outside[window[0] : window[1] + 1] = False
        kp[outside] = 0
        sc[outside] = 0
    return kp, sc


def load_rtmpose(mode: str = "performance") -> Tuple[Any, str]:
    """Build rtmlib's Wholebody on CUDA when it actually initialises, else on
    CPU, and return it with the device it really runs on.

    Lets the live app pick GPU on GPU-tier hardware and CPU on CPU-tier
    hardware from the same image. The device is read from the providers of
    the ONNX Runtime session rtmlib built, not from
    ``ort.get_available_providers()``: that lists what the installed
    ``onnxruntime-gpu`` was compiled with, so it names CUDA on a machine
    with no GPU, and the session then falls back to CPU on its own. Trusting
    the list put the CPU Space on the GPU progress split.

    Returns:
        (model, device): the ``Wholebody`` to pass to
        ``extract_rtmpose_landmarks(model=...)``, and "cuda" or "cpu".
    """
    import onnxruntime as ort
    from rtmlib import Wholebody

    def _build(device: str) -> Any:
        return Wholebody(
            mode=mode, to_openpose=False, backend="onnxruntime", device=device
        )

    model = None
    if "CUDAExecutionProvider" in ort.get_available_providers():
        try:
            model = _build("cuda")
        except Exception as exc:  # a failed CUDA session should not end the run
            logger.warning("RTMPose CUDA session failed (%s); using CPU", exc)
    if model is None:
        model = _build("cpu")
    providers = model.pose_model.session.get_providers()
    device = "cuda" if "CUDAExecutionProvider" in providers else "cpu"
    logger.info("RTMPose device: %s (session providers %s)", device, providers)
    return model, device


def detect_device() -> str:
    """The device RTMPose really runs on here, "cuda" or "cpu".

    Builds the model to find out (see ``load_rtmpose``), so a caller that
    then extracts should call ``load_rtmpose`` and pass its model instead of
    loading it twice.
    """
    return load_rtmpose()[1]


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
    model: Any = None,
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
        model: A ``Wholebody`` from ``load_rtmpose``, used instead of building
            one; ``device`` and ``mode`` are then ignored.

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
    # Fast sources only; a slower source is passed through at its own rate, and
    # that is safe only because ensure_cfr normalises a constant-but-slow clip
    # to target_fps upstream.
    skip_rate = max(1, int(source_fps / target_fps)) if source_fps > 0 else 1
    effective_fps = source_fps / skip_rate if source_fps > 0 else float(target_fps)

    resolved_end = (
        total_frames - 1 if end_frame is None else min(end_frame, total_frames - 1)
    )
    window_frames = max(0, resolved_end - start_frame + 1)
    expected = (window_frames + skip_rate - 1) // skip_rate

    wholebody = model
    if wholebody is None:
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


def extract_rtmpose_landmarks_windowed(
    video_path: str,
    n_frames: int,
    window: Optional[Tuple[int, int]],
    skip_rate: int,
    *,
    device: str = "cpu",
    mode: str = "performance",
    progress: Optional[ProgressCallback] = None,
    model: Any = None,
) -> RTMPoseResult:
    """RTMPose on pose frames ``window`` only, returned at full clip length.

    ``window`` is in pose frames (BlazePose's index, see ``rtmpose_window``);
    pose frame ``t`` is source frame ``t * skip_rate``, so the source range
    starts on the same ``skip_rate`` grid BlazePose sampled. Frames outside
    the window are zeros, as for a frame with no detection, so the result is
    ``n_frames`` long and ``T_rtm == T_blaze`` still holds. A None window
    runs the whole clip.
    """
    if window is None:
        return extract_rtmpose_landmarks(
            video_path, device=device, mode=mode, progress=progress, model=model
        )
    lo, hi = window
    part = extract_rtmpose_landmarks(
        video_path,
        start_frame=lo * skip_rate,
        end_frame=hi * skip_rate,
        device=device,
        mode=mode,
        progress=progress,
        model=model,
    )
    keypoints = np.zeros((n_frames, 133, 2), dtype=np.float32)
    scores = np.zeros((n_frames, 133), dtype=np.float32)
    n = min(len(part.keypoints), n_frames - lo)
    keypoints[lo : lo + n] = part.keypoints[:n]
    scores[lo : lo + n] = part.scores[:n]
    return RTMPoseResult(keypoints=keypoints, scores=scores, fps=part.fps)
