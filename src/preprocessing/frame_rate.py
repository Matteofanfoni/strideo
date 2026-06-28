"""VFR detection and CFR conversion for smartphone video.

Smartphones (including iPhones) record in Variable Frame Rate (VFR),
where actual inter-frame intervals fluctuate. This module detects VFR
and converts to Constant Frame Rate (CFR) using FFmpeg, ensuring
frame indices are consistent across Kinovea and OpenCV.

Reference: docs/technical/01_data_collection_protocol.md §11.2
"""

import json
import subprocess
from pathlib import Path
from typing import Optional, Tuple

import cv2


def is_cfr(video_path: str, tolerance: float = 0.01) -> bool:
    """
    Check whether a video is Constant Frame Rate.

    Compares r_frame_rate (container-level) and avg_frame_rate (stream-level).
    If they match within tolerance, the file is considered CFR.

    Args:
        video_path: Path to the video file.
        tolerance: Maximum relative difference between r_frame_rate and
            avg_frame_rate to still be considered CFR.

    Returns:
        True if the video is CFR, False if VFR.

    Raises:
        FileNotFoundError: If video file does not exist.
        RuntimeError: If ffprobe fails.
    """
    path = Path(video_path)
    if not path.exists():
        raise FileNotFoundError(f"Video not found: {video_path}")

    cmd = [
        "ffprobe",
        "-v",
        "error",
        "-select_streams",
        "v:0",
        "-show_entries",
        "stream=r_frame_rate,avg_frame_rate",
        "-of",
        "json",
        str(path),
    ]

    result = subprocess.run(cmd, capture_output=True, text=True)
    if result.returncode != 0:
        raise RuntimeError(f"ffprobe failed: {result.stderr.strip()}")

    data = json.loads(result.stdout)
    stream = data["streams"][0]

    r_fps = _parse_fraction(stream["r_frame_rate"])
    avg_fps = _parse_fraction(stream["avg_frame_rate"])

    if r_fps == 0:
        return False

    relative_diff = abs(r_fps - avg_fps) / r_fps
    return relative_diff <= tolerance


def convert_to_cfr(
    input_path: str,
    output_path: str | None = None,
    target_fps: int = 60,
    crf: int = 18,
) -> str:
    """
    Convert a video from VFR to CFR using FFmpeg.

    Uses H.264 encoding with a visually near-lossless CRF value,
    as specified in the data collection protocol.

    Args:
        input_path: Path to the input (VFR) video file.
        output_path: Path for the output CFR file. If None, appends
            '_cfr' suffix to the input filename.
        target_fps: Target constant frame rate (default 60).
        crf: Constant Rate Factor for quality (default 18, near-lossless).

    Returns:
        Path to the converted CFR video file.

    Raises:
        FileNotFoundError: If input video does not exist.
        RuntimeError: If FFmpeg conversion fails.
    """
    path = Path(input_path)
    if not path.exists():
        raise FileNotFoundError(f"Video not found: {input_path}")

    if output_path is None:
        output_path = str(path.with_stem(path.stem + "_cfr"))

    out = Path(output_path)

    cmd = [
        "ffmpeg",
        "-i",
        str(path),
        "-r",
        str(target_fps),
        "-c:v",
        "libx264",
        "-preset",
        "fast",
        "-crf",
        str(crf),
        "-y",
        str(out),
    ]

    result = subprocess.run(cmd, capture_output=True, text=True)
    if result.returncode != 0:
        raise RuntimeError(f"FFmpeg conversion failed: {result.stderr.strip()}")

    # Verify output is truly CFR (protocol §11.2 verification step)
    _verify_cfr(str(out), target_fps)

    return str(out)


def ensure_cfr(
    video_path: str,
    target_fps: int = 60,
    crf: int = 18,
) -> Tuple[str, bool]:
    """
    Ensure a video is CFR, converting only if necessary.

    This is the main entry point for the pipeline. It checks whether the
    video is already CFR and skips conversion if so.

    Args:
        video_path: Path to the video file.
        target_fps: Target constant frame rate (default 60).
        crf: Constant Rate Factor for quality (default 18).

    Returns:
        Tuple of (path_to_cfr_video, was_converted).
        If the video was already CFR, returns the original path.
    """
    if is_cfr(video_path):
        return video_path, False

    output_path = convert_to_cfr(video_path, target_fps=target_fps, crf=crf)
    return output_path, True


def cap_frames(
    video_path: str,
    max_frames: int,
    output_path: Optional[str] = None,
    crf: int = 18,
) -> Tuple[str, bool]:
    """Trim a video to at most ``max_frames`` frames (keeping the first ones).

    The web app's pipeline buffers full-resolution frames for the seeded pass
    (~25 MB/frame at 4K), so peak RAM scales linearly with frame count. This is
    the ingest guardrail that bounds RAM (and per-clip runtime) on the CPU host:
    a clip longer than ``max_frames`` is re-encoded down to its first
    ``max_frames`` frames before any pose estimator touches it. Because the trim
    happens once, up front, every downstream stage (RTMPose + all BlazePose
    passes) sees the same shortened clip and the ``T_rtm == T_blaze`` invariant
    is preserved automatically.

    Capping by *frames* (not seconds) is deliberate: RAM is a function of frame
    count and resolution, independent of fps.

    Args:
        video_path: Path to the (ideally already-CFR) input video.
        max_frames: Maximum number of frames to keep.
        output_path: Output path; defaults to a ``_cap{N}`` suffix on the input.
        crf: x264 quality for the re-encode (default 18, near-lossless).

    Returns:
        Tuple of (path_to_use, was_capped). If the clip already has
        ``<= max_frames`` frames, returns the original path and ``False``.

    Raises:
        FileNotFoundError: If the input video does not exist.
        RuntimeError: If the FFmpeg trim fails.
    """
    path = Path(video_path)
    if not path.exists():
        raise FileNotFoundError(f"Video not found: {video_path}")

    cap = cv2.VideoCapture(str(path))
    n_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    cap.release()

    if n_frames <= max_frames:
        return str(path), False

    if output_path is None:
        output_path = str(path.with_stem(path.stem + f"_cap{max_frames}"))

    cmd = [
        "ffmpeg",
        "-i",
        str(path),
        "-frames:v",
        str(max_frames),
        "-c:v",
        "libx264",
        "-preset",
        "fast",
        "-crf",
        str(crf),
        "-an",  # drop audio; metrics are video-only
        "-y",
        str(output_path),
    ]
    result = subprocess.run(cmd, capture_output=True, text=True)
    if result.returncode != 0:
        raise RuntimeError(f"FFmpeg frame-cap failed: {result.stderr.strip()}")

    return str(output_path), True


def _verify_cfr(video_path: str, expected_fps: int) -> None:
    """
    Verify that a converted video is truly CFR at the expected frame rate.

    Checks that both r_frame_rate and avg_frame_rate equal expected_fps/1,
    as specified in the data collection protocol §11.2.

    Args:
        video_path: Path to the converted video file.
        expected_fps: Expected constant frame rate.

    Raises:
        RuntimeError: If ffprobe fails or the video is not CFR at the
            expected frame rate.
    """
    cmd = [
        "ffprobe",
        "-v",
        "error",
        "-select_streams",
        "v:0",
        "-show_entries",
        "stream=r_frame_rate,avg_frame_rate",
        "-of",
        "json",
        str(video_path),
    ]

    result = subprocess.run(cmd, capture_output=True, text=True)
    if result.returncode != 0:
        raise RuntimeError(f"ffprobe verification failed: {result.stderr.strip()}")

    data = json.loads(result.stdout)
    stream = data["streams"][0]

    r_fps = _parse_fraction(stream["r_frame_rate"])
    avg_fps = _parse_fraction(stream["avg_frame_rate"])

    if round(r_fps) != expected_fps or round(avg_fps) != expected_fps:
        raise RuntimeError(
            f"CFR verification failed: r_frame_rate={stream['r_frame_rate']}, "
            f"avg_frame_rate={stream['avg_frame_rate']} "
            f"(expected {expected_fps}/1 for both)"
        )


def _parse_fraction(fraction_str: str) -> float:
    """Parse an FFmpeg fraction string like '60/1' to a float."""
    parts = fraction_str.split("/")
    if len(parts) == 2:
        num, den = int(parts[0]), int(parts[1])
        return num / den if den != 0 else 0.0
    return float(fraction_str)


def probe_video_metadata(video_path: str) -> dict:
    """Read shooting metadata (creation time, resolution, fps) via ffprobe.

    Best-effort: returns whatever ffprobe exposes and never raises for missing
    tags or a probe failure — callers fall back to a user-supplied value. The
    ``creation_time`` tag is written by most smartphones (iPhone included) and is
    read from the container format first, then the video stream.

    Args:
        video_path: Path to the video file.

    Returns:
        Dict with keys ``creation_time`` (str | None, ISO-8601 as ffprobe
        reports it), ``width`` (int | None), ``height`` (int | None), and
        ``fps`` (float | None).
    """
    empty: dict = {
        "creation_time": None,
        "width": None,
        "height": None,
        "fps": None,
    }
    if not Path(video_path).exists():
        return empty

    cmd = [
        "ffprobe",
        "-v",
        "error",
        "-select_streams",
        "v:0",
        "-show_entries",
        "stream=width,height,avg_frame_rate:stream_tags=creation_time"
        ":format_tags=creation_time",
        "-of",
        "json",
        str(video_path),
    ]
    try:
        result = subprocess.run(cmd, capture_output=True, text=True)
        if result.returncode != 0:
            return empty
        data = json.loads(result.stdout)
    except (subprocess.SubprocessError, json.JSONDecodeError, OSError):
        return empty

    streams = data.get("streams") or [{}]
    stream = streams[0]
    fmt_tags = (data.get("format") or {}).get("tags") or {}
    stream_tags = stream.get("tags") or {}

    fps_raw = stream.get("avg_frame_rate")
    try:
        fps = _parse_fraction(fps_raw) if fps_raw else None
    except (ValueError, ZeroDivisionError):
        fps = None

    return {
        "creation_time": fmt_tags.get("creation_time")
        or stream_tags.get("creation_time"),
        "width": stream.get("width"),
        "height": stream.get("height"),
        "fps": fps if fps else None,
    }
