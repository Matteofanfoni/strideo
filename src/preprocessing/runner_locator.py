"""Find where the runner is in a long clip, cheaply, before any pose pass.

The app keeps only ``MAX_PIPELINE_FRAMES`` (300) frames of an upload, a RAM
guard. Keeping the *first* 300 cuts the run short when the runner crosses
late. To place the 300 frames on the runner instead, the runner has to be
found without the full pose pass the cap exists to bound; the frame-
difference motion gate was measured unfit for that (camera motion dominates
it).

This decodes only the clip's keyframes (ffmpeg ``-skip_frame nokey``, about
one a second on phone footage), downscaled, and runs BlazePose's image mode
on each. The keyframes with a detection bracket the crossing, and the
window is centred on them. Keyframes are also where a stream copy can
start a cut exactly, so ``keyframe_start`` offers the window's start
snapped back to one.
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


def keyframe_start(window: RunnerWindow) -> int:
    """The window's start snapped back to the nearest keyframe at or before
    it, where a stream copy can begin exactly."""
    earlier = [k for k in window.keyframes if k <= window.start]
    return earlier[-1] if earlier else 0
