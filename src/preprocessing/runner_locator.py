"""Find where the runner is in a long clip, cheaply, before any pose pass.

The app keeps only ``MAX_PIPELINE_FRAMES`` (300) frames of an upload, a RAM
guard. Keeping the *first* 300 cuts the run short when the runner crosses
late. To place the 300 frames on the runner instead, the runner has to be
found without the full pose pass the cap exists to bound; the frame-
difference motion gate was measured unfit for that (camera motion dominates
it).

``locate_runner_window`` is the one the app uses: every
``LOCATE_EVERY``-th frame, decoded at ``DECODE_WIDTH``, goes to YOLOX-m,
RTMPose's own person detector, and the window moves only when the frames
with a person run past the cap. Measured on the 19 annotated clips over 300
frames: all 162 ground-truth contacts kept, against 153 for the first 300
frames, and no clip worse.

``locate_runner`` is the first attempt, kept as its record with no caller:
it decodes only the clip's keyframes and runs BlazePose's image
mode on each, and was measured blind, since a 60 fps conversion carries a
keyframe only every 250 frames and BlazePose cannot see a small side-on
runner in a single frame.
"""

from __future__ import annotations

import subprocess
from dataclasses import dataclass, field
from pathlib import Path
from typing import List, Optional, Tuple

import numpy as np

#: Width keyframes are decoded at; BlazePose's detector works on a 224 px
#: square crop, so 4K buys nothing for a yes/no person check.
DECODE_WIDTH = 1280

#: Frames between YOLOX samples. Every 30 kept as many GT contacts in the
#: survey; 15 gives a briefly visible runner a few more chances to be seen.
LOCATE_EVERY = 15


@dataclass
class RunnerWindow:
    """Where to keep ``length`` frames of an ``n_frames`` clip."""

    start: int
    length: int
    n_frames: int
    keyframes: List[int] = field(default_factory=list)
    detected: List[int] = field(default_factory=list)

    @property
    def end(self) -> int:
        """Last kept frame, inclusive."""
        return self.start + self.length - 1


def keyframe_indices(video_path: str) -> List[int]:
    """Display-order frame indices of the clip's keyframes, from packet
    timestamps (decode order is not display order with B-frames)."""
    cmd = [
        "ffprobe",
        "-v",
        "error",
        "-select_streams",
        "v:0",
        "-show_entries",
        "packet=pts,flags",
        "-of",
        "csv=p=0",
        video_path,
    ]
    out = subprocess.run(cmd, capture_output=True, text=True, check=True).stdout
    pts, key = [], set()
    for line in out.split():
        parts = line.strip().split(",")
        if len(parts) < 2 or not parts[0].lstrip("-").isdigit():
            continue
        p = int(parts[0])
        pts.append(p)
        if "K" in parts[1]:
            key.add(p)
    order = sorted(pts)
    rank = {p: i for i, p in enumerate(order)}
    return sorted(rank[p] for p in key)


def _video_size(video_path: str) -> Tuple[int, int]:
    import cv2

    cap = cv2.VideoCapture(video_path)
    w = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    h = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    cap.release()
    return w, h


def decode_keyframes(video_path: str, width: int = DECODE_WIDTH) -> List[np.ndarray]:
    """Every keyframe, in display order, as RGB arrays ``width`` wide."""
    w, h = _video_size(video_path)
    out_w = width
    out_h = int(round(h * width / w / 2.0)) * 2
    cmd = [
        "ffmpeg",
        "-v",
        "error",
        "-skip_frame",
        "nokey",
        "-i",
        video_path,
        "-fps_mode",
        "passthrough",
        "-vf",
        f"scale={out_w}:{out_h}",
        "-pix_fmt",
        "rgb24",
        "-f",
        "rawvideo",
        "-",
    ]
    raw = subprocess.run(cmd, capture_output=True, check=True).stdout
    size = out_w * out_h * 3
    return [
        np.frombuffer(raw[i : i + size], dtype=np.uint8).reshape(out_h, out_w, 3)
        for i in range(0, len(raw) - size + 1, size)
    ]


def _detects_person(frames: List[np.ndarray], model_path: Optional[Path]) -> List[bool]:
    import mediapipe as mp

    from src.preprocessing.pose_estimator import MODEL_HEAVY, create_pose_estimator

    est = create_pose_estimator(
        model_path=model_path or MODEL_HEAVY, num_poses=1, running_mode="image"
    )
    try:
        hits = []
        for frame in frames:
            img = mp.Image(
                image_format=mp.ImageFormat.SRGB, data=np.ascontiguousarray(frame)
            )
            hits.append(bool(est.detect(img).pose_landmarks))
        return hits
    finally:
        est.close()


def locate_runner(
    video_path: str,
    n_frames: int,
    length: int,
    model_path: Optional[Path] = None,
) -> RunnerWindow:
    """Where to keep ``length`` frames so that they hold the runner.

    The window is centred on the span between the first and last keyframes
    with a person detection, then clamped to the clip. With no detection at
    all it falls back to the first ``length`` frames, today's behaviour.
    """
    keys = keyframe_indices(video_path)
    frames = decode_keyframes(video_path)
    n = min(len(keys), len(frames))
    hits = _detects_person(frames[:n], model_path)
    detected = [k for k, hit in zip(keys[:n], hits) if hit]
    if not detected or n_frames <= length:
        return RunnerWindow(0, min(length, n_frames), n_frames, keys, detected)
    centre = (detected[0] + detected[-1]) / 2.0
    start = int(round(centre - length / 2.0))
    start = max(0, min(start, n_frames - length))
    return RunnerWindow(start, length, n_frames, keys, detected)


def decode_every(
    video_path: str, every: int, width: int = DECODE_WIDTH
) -> List[np.ndarray]:
    """Frames 0, ``every``, 2 * ``every``, ... as BGR arrays ``width`` wide."""
    w, h = _video_size(video_path)
    out_h = int(round(h * width / w / 2.0)) * 2
    cmd = [
        "ffmpeg",
        "-v",
        "error",
        "-i",
        video_path,
        "-vf",
        f"select='not(mod(n\\,{every}))',scale={width}:{out_h}",
        "-fps_mode",
        "passthrough",
        "-pix_fmt",
        "bgr24",
        "-f",
        "rawvideo",
        "-",
    ]
    raw = subprocess.run(cmd, capture_output=True, check=True).stdout
    size = width * out_h * 3
    return [
        np.frombuffer(raw[i : i + size], dtype=np.uint8).reshape(out_h, width, 3)
        for i in range(0, len(raw) - size + 1, size)
    ]


#: Upper bound on YOLOX's onnxruntime threads: the live Space's vCPUs.
MAX_DETECT_THREADS = 8


def _detect_threads() -> int:
    """CPUs this process may run on, capped at ``MAX_DETECT_THREADS``.

    onnxruntime sizes its default thread pool from the host's cores, not the
    process's: on 8 CPUs of a larger host its threads thrash, and YOLOX took
    877 ms a frame against 120 ms with 8 threads (pinned on the Legion;
    ~25 s per upload live). A container's CPU quota does not show in the
    affinity mask, hence the cap as well.
    """
    import os

    try:
        n = len(os.sched_getaffinity(0))  # type: ignore[attr-defined]
    except AttributeError:
        n = os.cpu_count() or 1
    return max(1, min(n, MAX_DETECT_THREADS))


def _person_detector():
    """YOLOX-m alone, the detector of RTMPose-x's ``Wholebody`` (same model
    file, already cached in the app image), without loading RTMPose, its
    session rebuilt with an explicit thread count (``_detect_threads``)."""
    import onnxruntime as ort
    from rtmlib import YOLOX, Wholebody
    from rtmlib.tools.file import download_checkpoint

    mode = Wholebody.MODE["performance"]
    path = download_checkpoint(mode["det"])
    det = YOLOX(path, model_input_size=mode["det_input_size"], backend="onnxruntime")
    opts = ort.SessionOptions()
    opts.intra_op_num_threads = _detect_threads()
    opts.inter_op_num_threads = 1
    det.session = ort.InferenceSession(
        path, sess_options=opts, providers=["CPUExecutionProvider"]
    )
    return det


def locate_runner_window(
    video_path: str, n_frames: int, length: int, every: int = LOCATE_EVERY
) -> RunnerWindow:
    """Where to keep ``length`` frames so that they hold the runner.

    The span runs from the first to the last sampled frame with a person,
    widened by one sampling step each side. The window stays at frame 0 when
    the span ends inside it, so a clip that fits today keeps today's frames;
    otherwise it is centred on the span and clamped to the clip. No person
    at all also keeps frame 0.
    """
    if n_frames <= length:
        return RunnerWindow(0, n_frames, n_frames)
    det = _person_detector()
    detected = []
    for i, frame in enumerate(decode_every(video_path, every)):
        boxes = det(frame)
        if boxes is not None and len(boxes):
            detected.append(i * every)
    if not detected or detected[-1] + every < length:
        return RunnerWindow(0, length, n_frames, detected=detected)
    lo = max(0, detected[0] - every)
    hi = min(n_frames - 1, detected[-1] + every)
    start = int(round((lo + hi) / 2.0 - length / 2.0))
    start = max(0, min(start, n_frames - length))
    return RunnerWindow(start, length, n_frames, detected=detected)


def keyframe_start(window: RunnerWindow) -> int:
    """The window's start snapped back to the nearest keyframe at or before
    it, where a stream copy can begin exactly."""
    earlier = [k for k in window.keyframes if k <= window.start]
    return earlier[-1] if earlier else 0
