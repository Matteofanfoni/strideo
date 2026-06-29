"""Container startup tasks — runs once when the Streamlit process starts.

Imported at the top of app.py so it executes before any page is served:
  1. Removes per-session upload dirs older than 4 hours (disk hygiene).
  2. Downloads pose_landmarker_heavy.task if not present (MediaPipe needs
     this file on disk; it is not pip-installable or committed to the repo).
  3. Warms the RTMPose-x ONNX model so the first user upload does not stall
     waiting for a ~300 MB download mid-request.

All three steps are best-effort: failures are logged to stderr but never
crash the app.
"""

from __future__ import annotations

import shutil
import sys
import tempfile
import time
import urllib.request
from pathlib import Path

_MEDIAPIPE_URL = (
    "https://storage.googleapis.com/mediapipe-models/"
    "pose_landmarker/pose_landmarker_heavy/float16/latest/"
    "pose_landmarker_heavy.task"
)

_MODELS_DIR = Path(__file__).parent.parent / "models"
_TASK_PATH = _MODELS_DIR / "pose_landmarker_heavy.task"


def _clean_stale_sessions(max_age_hours: float = 4.0) -> None:
    """Remove per-session upload dirs older than max_age_hours.

    Each browser session writes to tempdir/strideo_sessions/<uuid12>/. On HF
    Spaces container restarts clear /tmp entirely, but within a long-running
    container these dirs accumulate. We sweep on startup to bound disk use.
    Sessions younger than max_age_hours are left alone so the results scrubber
    keeps working for active users.
    """
    sessions_root = Path(tempfile.gettempdir()) / "strideo_sessions"
    if not sessions_root.exists():
        return
    cutoff = time.time() - max_age_hours * 3600
    for session_dir in sessions_root.iterdir():
        try:
            if session_dir.is_dir() and session_dir.stat().st_mtime < cutoff:
                shutil.rmtree(session_dir, ignore_errors=True)
        except OSError:
            pass


def _download_mediapipe() -> None:
    if _TASK_PATH.exists():
        size_mb = _TASK_PATH.stat().st_size / 1_048_576
        print(
            f"[preload] MediaPipe task file ready: {_TASK_PATH} ({size_mb:.1f} MB)",
            flush=True,
        )
        return
    print(f"[preload] downloading MediaPipe weights → {_TASK_PATH}", flush=True)
    try:
        _MODELS_DIR.mkdir(parents=True, exist_ok=True)
        urllib.request.urlretrieve(_MEDIAPIPE_URL, _TASK_PATH)
        print("[preload] MediaPipe weights ready.", flush=True)
    except Exception as exc:
        print(
            f"[preload] WARNING: MediaPipe download failed: {exc}",
            file=sys.stderr,
            flush=True,
        )


def _warm_rtmpose() -> None:
    print("[preload] warming RTMPose-x ONNX model…", flush=True)
    try:
        from rtmlib import Wholebody  # triggers ONNX download on first use

        Wholebody(
            mode="performance",
            to_openpose=False,
            backend="onnxruntime",
            device="cpu",
        )
        print("[preload] RTMPose-x ready.", flush=True)
    except Exception as exc:
        print(
            f"[preload] WARNING: RTMPose warmup failed: {exc}",
            file=sys.stderr,
            flush=True,
        )


_clean_stale_sessions()
_download_mediapipe()
_warm_rtmpose()
