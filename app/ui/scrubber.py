"""Interactive ground-contact frame scrubber for the Results page.

Draws per-frame overlays (skeleton, contact markers, GCT labels) in Python on a
display-resized copy of the capped CFR clip, using the SAME helpers as the QA
video (``src/preprocessing/visualisation.py``). Scrubbing and toggling never
re-run the analysis pipeline: the ``ClipAnalysis`` lives in session memory and
the decoded frames are cached by clip path.

Memory note: frames are cached at ``SCRUB_MAX_H`` (900p ≈ 1600px wide, so the
display stays crisp at the ~1440px content width) - roughly 4.3 MB/frame, i.e.
~1.3 GB for a 300-frame clip. ``max_entries`` bounds the number of clips held at
once; on a memory-tight multi-user host, drop ``SCRUB_MAX_H`` to 720 (~0.8 GB).
"""

from __future__ import annotations

import os
import subprocess
import sys
import tempfile
from pathlib import Path
from typing import List

import cv2
import numpy as np
import streamlit as st

# Project root on sys.path so ``src.`` resolves when imported from a page.
_PROJECT_ROOT = Path(__file__).resolve().parents[2]
if str(_PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(_PROJECT_ROOT))

from src.preprocessing.visualisation import (  # noqa: E402
    draw_contact_markers,
    draw_skeleton,
    scale_landmarks,
)

SCRUB_MAX_H = 900
VIDEO_MAX_H = 720


@st.cache_data(show_spinner="Loading frames…", max_entries=3)
def _load_frames(
    cfr_path: str, skip_rate: int, max_h: int, cache_token: str
) -> List[np.ndarray]:
    """Decode the clip once into display-resized BGR frames (cached by path).

    ``cache_token`` (mtime+size) invalidates the cache if the same path is
    reused for different content within a session. Frame ``t`` of the returned
    list aligns with pose frame ``t`` because we keep every ``skip_rate``-th
    source frame, matching the pipeline's sampling.
    """
    cap = cv2.VideoCapture(cfr_path)
    frames: List[np.ndarray] = []
    if not cap.isOpened():
        return frames
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
) -> bytes:
    """Bake overlays into an H.264 MP4 and return its bytes (cached per config).

    Overlays are drawn per frame in Python, then raw BGR is piped to ffmpeg
    libx264 (NOT ``cv2.VideoWriter('avc1')``, which is absent in
    ``opencv-python-headless`` on headless Linux). ``yuv420p`` + ``faststart``
    make it stream in the browser. ``_analysis`` is underscore-prefixed so
    Streamlit does not try to hash it - the cache key is the clip + toggles.
    """
    frames = _load_frames(cfr_path, skip_rate, VIDEO_MAX_H, cache_token)
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
            proc.stdin.write(frame.tobytes())
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


def _step(state_key: str, delta: int, lo: int, hi: int) -> None:
    """Button callback: nudge the frame index, clamped to ``[lo, hi]``."""
    st.session_state[state_key] = max(
        lo, min(hi, st.session_state.get(state_key, 0) + delta)
    )


def _jump(state_key: str, frame_idx: int) -> None:
    """Button callback: jump to a specific frame."""
    st.session_state[state_key] = frame_idx


def render_gc_scrubber(analysis, key_prefix: str) -> None:
    """Render the ground-contact verifier as a single-frame scrubber.

    Prev/next buttons and a frame slider step through the clip; the overlays
    (skeleton, IC/TO markers, GCT labels) are always drawn. A full annotated
    clip can be generated on demand. Nothing here re-runs the analysis.

    Args:
        analysis: The ``ClipAnalysis`` from ``run_clip_pipeline`` (carries
            ``cfr_path``, ``selected_landmarks``, ``selected_visibilities``,
            ``contacts``, ``frame_width``, ``skip_rate``).
        key_prefix: Unique per-clip prefix for Streamlit widget keys.
    """
    cfr_path = getattr(analysis, "cfr_path", None)
    if not cfr_path or not os.path.exists(cfr_path):
        st.info(
            "The processed clip is no longer available for review (the "
            "per-session files were cleared). Re-run the analysis to verify."
        )
        return

    frames = _load_frames(
        cfr_path, int(analysis.skip_rate), SCRUB_MAX_H, _cache_token(cfr_path)
    )
    total = min(len(frames), int(len(analysis.selected_landmarks)))
    if total == 0:
        st.info("Frame data is unavailable for this clip.")
        return

    # Frame index lives in session_state so prev/next and the slider share it.
    fkey = f"{key_prefix}_frame"
    if fkey not in st.session_state:
        st.session_state[fkey] = 0
    if st.session_state[fkey] > total - 1:
        st.session_state[fkey] = total - 1

    # One numbered button per detected contact - jumps to the IC frame.
    contact_frames = [
        (i + 1, min(int(c.contact_frame) + 1, total - 1))
        for i, c in enumerate(analysis.contacts)
        if c.contact_frame is not None
    ]
    if contact_frames:
        # Fixed-width narrow columns + large trailing spacer = left-aligned row
        n = len(contact_frames)
        spacer = max(n * 3, 10)
        btn_cols = st.columns([1] * n + [spacer], gap="small")
        for col, (num, fi) in zip(btn_cols, contact_frames):
            col.button(
                str(num),
                key=f"{key_prefix}_contact_{num}",
                on_click=_jump,
                args=(fkey, fi),
                use_container_width=True,
            )

    t = st.slider("Frame", 0, total - 1, key=fkey) if total > 1 else 0

    # Overlays are always on (skeleton + IC/TO markers + GCT labels).
    frame = frames[t].copy()
    _draw_overlays(frame, analysis, t, True, True, True)
    st.image(
        cv2.cvtColor(frame, cv2.COLOR_BGR2RGB),
        use_container_width=True,
        caption=f"Frame {t} / {total - 1}",
    )

    # Optional full annotated clip (overlays baked in) - on demand (heavy).
    want_key = f"{key_prefix}_wantvid"
    if st.button(
        "Generate annotated video",
        key=f"{key_prefix}_genvid",
        use_container_width=True,
    ):
        st.session_state[want_key] = True

    if st.session_state.get(want_key):
        with st.spinner("Encoding H.264 video…"):
            try:
                video_bytes = _encode_annotated_video(
                    cfr_path,
                    int(analysis.skip_rate),
                    float(analysis.fps),
                    True,
                    True,
                    True,
                    _cache_token(cfr_path),
                    analysis,
                )
                st.video(video_bytes)
                st.download_button(
                    "Download video (MP4)",
                    data=video_bytes,
                    file_name=f"{key_prefix}_annotated.mp4",
                    mime="video/mp4",
                    key=f"{key_prefix}_dlvid",
                )
            except Exception as exc:  # noqa: BLE001 - surface encode failures
                st.error(f"Could not generate the video: {exc}")
