"""StrideoNet annotated-video encoding for the Results page.

Bakes per-frame overlays (skeleton, contact markers, GCT labels) in Python
onto a display-resized copy of the capped CFR clip, using the SAME helpers as
the QA video (``src/preprocessing/visualisation.py``), then pipes raw frames
to ffmpeg for H.264 encoding. Pure/widget-free (``encode_strideonet_video``,
) - called eagerly from ``upload.py``'s analyze loop, not on demand from
a page.

The interactive frame-by-frame ground-contact scrubber this module used to
also provide (``render_gc_scrubber``) was removed (2026-08-27, user:
"I don't see the value of it as it is") - the same per-contact IC/TO data it
overlaid on the clip image is now shown as a table instead (
``app/pages/results.py::_render_contact_tracker``).
"""

from __future__ import annotations

import os
import subprocess
import sys
import tempfile
from pathlib import Path
from types import SimpleNamespace
from typing import Callable, List, Optional, Tuple

import cv2
import numpy as np
import streamlit as st

# Project root on sys.path so ``src.`` resolves when imported from a page.
_PROJECT_ROOT = Path(__file__).resolve().parents[2]
if str(_PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(_PROJECT_ROOT))

from src.preprocessing.visualisation import (  # noqa: E402
    draw_contact_markers,
    draw_metrics_hud,
    draw_skeleton,
    scale_landmarks,
)

VIDEO_MAX_H = 720


@st.cache_data(show_spinner="Loading frames…", max_entries=3)
def _load_frames(
    cfr_path: str,
    skip_rate: int,
    max_h: int,
    cache_token: str,
    _progress: Optional[Callable[[float], None]] = None,
) -> List[np.ndarray]:
    """Decode the clip once into display-resized BGR frames (cached by path).

    ``cache_token`` (mtime+size) invalidates the cache if the same path is
    reused for different content within a session. Frame ``t`` of the returned
    list aligns with pose frame ``t`` because we keep every ``skip_rate``-th
    source frame, matching the pipeline's sampling.

    ``_progress`` (0-1 per decoded frame) exists because this decode, not the
    encode loop after it, is the bulk of the annotated-video step: measured
    6.5 s of 8.0 s on a 289-frame 4K clip, which is exactly the stretch the
    caller's progress bar used to sit still through. Underscore-prefixed so
    ``st.cache_data`` keeps it out of the cache key.
    """
    cap = cv2.VideoCapture(cfr_path)
    frames: List[np.ndarray] = []
    if not cap.isOpened():
        return frames
    expected = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    expected = (expected + skip_rate - 1) // skip_rate if expected > 0 else 0
    idx = 0
    while True:
        ok, frame = cap.read()
        if not ok:
            break
        if idx % skip_rate == 0:
            h, w = frame.shape[:2]
            if h > max_h:
                ratio = max_h / float(h)
                frame = cv2.resize(
                    frame,
                    (max(1, round(w * ratio)), max_h),
                    interpolation=cv2.INTER_AREA,
                )
            frames.append(frame)
            if _progress is not None and expected:
                _progress(min(1.0, len(frames) / expected))
        idx += 1
    cap.release()
    return frames


@st.cache_data(show_spinner=False, max_entries=4)
def _encode_annotated_video(
    cfr_path: str,
    skip_rate: int,
    fps: float,
    show_skel: bool,
    show_marks: bool,
    show_labels: bool,
    cache_token: str,
    _analysis,
    hud_lines: Optional[Tuple[str, ...]] = None,
    hud_font_scale: float = 1.0,
    _progress: Optional[Callable[[float], None]] = None,
) -> bytes:
    """Bake overlays into an H.264 MP4 and return its bytes (cached per config).

    Overlays are drawn per frame in Python, then raw BGR is piped to ffmpeg
    libx264 (NOT ``cv2.VideoWriter('avc1')``, which is absent in
    ``opencv-python-headless`` on headless Linux). ``yuv420p`` + ``faststart``
    make it stream in the browser. ``_analysis`` is underscore-prefixed so
    Streamlit does not try to hash it - the cache key is the clip + toggles.
    ``hud_lines``, when given, burns a StrideoNet-style metrics HUD box into
    every frame - a tuple, not a list, so it stays hashable for the
    cache key. ``hud_font_scale`` shrinks the whole HUD box (text + line
    height) so a longer ``hud_lines`` doesn't grow to cover the frame.
    ``_progress``, if given, is called with a 0-1 fraction per drawn frame so
    the caller's progress bar keeps moving through the encode; it is
    underscore-prefixed so ``st.cache_data`` leaves it out of the cache key
    (a cache hit skips the loop, and therefore the callback, entirely).
    """
    # Decode is ~70% of this function's wall-clock, the draw+pipe loop the
    # rest, so the caller's fraction is split in that proportion rather than
    # handing the whole span to the loop and stalling through the decode.
    decode_weight = 0.7
    frames = _load_frames(
        cfr_path,
        skip_rate,
        VIDEO_MAX_H,
        cache_token,
        _progress=(
            None
            if _progress is None
            else (lambda f: _progress(decode_weight * f))  # noqa: E731
        ),
    )
    total = min(len(frames), int(len(_analysis.selected_landmarks)))
    if total == 0:
        raise RuntimeError("No frames available to encode.")

    # libx264 + yuv420p needs even dimensions; crop the odd edge if any.
    h0, w0 = frames[0].shape[:2]
    width, height = w0 - (w0 % 2), h0 - (h0 % 2)

    out_path = tempfile.NamedTemporaryFile(suffix=".mp4", delete=False).name
    cmd = [
        "ffmpeg",
        "-y",
        "-f",
        "rawvideo",
        "-pix_fmt",
        "bgr24",
        "-s",
        f"{width}x{height}",
        "-r",
        f"{fps:.4f}",
        "-i",
        "-",
        "-an",
        "-c:v",
        "libx264",
        "-preset",
        "fast",
        "-pix_fmt",
        "yuv420p",
        "-movflags",
        "+faststart",
        out_path,
    ]
    proc = subprocess.Popen(
        cmd, stdin=subprocess.PIPE, stdout=subprocess.DEVNULL, stderr=subprocess.PIPE
    )
    try:
        assert proc.stdin is not None
        for t in range(total):
            frame = frames[t][:height, :width].copy()
            _draw_overlays(frame, _analysis, t, show_skel, show_marks, show_labels)
            if hud_lines:
                draw_metrics_hud(frame, list(hud_lines), font_scale=hud_font_scale)
            proc.stdin.write(frame.tobytes())
            if _progress is not None:
                _progress(decode_weight + (1.0 - decode_weight) * ((t + 1) / total))
        proc.stdin.close()
        err = proc.stderr.read() if proc.stderr else b""
        if proc.wait() != 0:
            raise RuntimeError(
                "ffmpeg encode failed: " + err.decode("utf-8", "replace")[-500:]
            )
        return Path(out_path).read_bytes()
    finally:
        try:
            os.unlink(out_path)
        except OSError:
            pass


def _cache_token(path: str) -> str:
    """Cheap content fingerprint (mtime + size) to key the frame cache."""
    try:
        stat = os.stat(path)
        return f"{int(stat.st_mtime)}-{stat.st_size}"
    except OSError:
        return path


def _draw_overlays(
    frame: np.ndarray,
    analysis,
    t: int,
    show_skel: bool,
    show_marks: bool,
    show_labels: bool,
    label_scale: float = 0.5,
) -> np.ndarray:
    """Draw the selected overlays onto ``frame`` (in place) for pose frame ``t``.

    Landmarks are stored in source pixels; they are scaled to ``frame``'s width
    so this works at any display/encode resolution. Returns the scaled
    landmarks (used by the zoom crop).
    """
    ratio = frame.shape[1] / float(analysis.frame_width)
    lm = scale_landmarks(analysis.selected_landmarks[t], ratio)
    if show_skel:
        draw_skeleton(frame, lm, analysis.selected_visibilities[t])
    if show_marks:
        draw_contact_markers(
            frame,
            t,
            lm,
            analysis.contacts,
            label_scale=label_scale,
            draw_labels=show_labels,
        )
    return lm


def encode_strideonet_video(
    overlay: dict,
    metrics: dict,
    preprocessing: Optional[dict] = None,
    progress: Optional[Callable[[float], None]] = None,
) -> bytes:
    """Encode a StrideoNet-only annotated video: BlazePose skeleton +
    StrideoNet's own metrics HUD, no contact markers (contacts are a
    Deterministic Kinematics Engine concept, StrideoNet doesn't detect them).

    Pure and widget-free by design (made eager per user feedback
    2026-08-27) - called from ``upload.py``'s analyze loop so the video ships
    already-generated alongside the rest of a clip's results, rather than
    behind an on-demand button on the Results page. Raises on failure
    (missing clip file, ffmpeg error); the caller decides how to surface
    that, since "no video" shouldn't fail the whole analysis.

    HUD line set matches ``scripts/render_strideonet_hero_overlay.py``'s
    Home-page hero HUD (user feedback: this video's HUD was missing fields
    the Home page's own overlay shows) - Velocity/Flight+Duty/VO ratio/Cal/
    Detect added to the original Cadence/GCT/Stride/Osc set, at a smaller
    ``hud_font_scale`` so the taller box doesn't grow to cover the frame.

    Args:
        overlay: the ``overlay`` dict from ``pipeline_runner.run_fast_path``
            (cfr_path/fps/skip_rate/frame_width/selected_landmarks/
            selected_visibilities).
        metrics: the fast-path ``metrics`` dict, for the burned-in HUD text.
        preprocessing: the fast-path ``preprocessing`` dict, for detection
            rate. Optional - HUD drops that one line if not given (e.g. a
            re-imported result with no cached preprocessing dict).
        progress: Optional callback(fraction_0_to_1) fired per encoded frame,
            so ``upload.py`` can walk its progress bar through this step
            instead of parking it while the encode runs.
    """
    cfr_path = overlay["cfr_path"]
    if not os.path.exists(cfr_path):
        raise FileNotFoundError(
            f"Processed clip no longer available for the annotated video: {cfr_path}"
        )
    hud_lines = [
        "StrideoNet result",
        f"Cadence  : {metrics['cadence_spm']:.1f} spm",
        f"GCT      : {metrics['gct_ms']:.1f} ms",
        f"Stride   : {metrics['stride_length_m']:.2f} m",
        f"Osc      : {metrics['oscillation_cm']:.2f} cm "
        f"({metrics['oscillation_leg_ratio']:.3f}x leg)",
        f"Velocity : {metrics['velocity_ms']:.2f} m/s "
        f"({metrics['pace_per_km']}/km)",
        f"Flight   : {metrics['flight_time_ms']:.1f} ms   "
        f"Duty: {metrics['duty_factor']:.2f}",
        f"VO ratio : {metrics['vertical_oscillation_ratio']:.1f} %",
        f"Cal      : conf={metrics['calibration_confidence']:.2f}",
    ]
    if preprocessing is not None and preprocessing.get("detection_rate") is not None:
        hud_lines.append(f"Detect   : {preprocessing['detection_rate']:.1f} %")
    _analysis = SimpleNamespace(
        selected_landmarks=overlay["selected_landmarks"],
        selected_visibilities=overlay["selected_visibilities"],
        frame_width=overlay["frame_width"],
    )
    return _encode_annotated_video(
        cfr_path,
        int(overlay["skip_rate"]),
        float(overlay["fps"]),
        True,  # show_skel - the pose overlay
        False,  # show_marks - no contact dots (StrideoNet has none)
        False,  # show_labels
        _cache_token(cfr_path),
        _analysis,
        tuple(hud_lines),
        0.72,  # hud_font_scale - ~2x the line count of the original 5-line HUD
        _progress=progress,
    )
