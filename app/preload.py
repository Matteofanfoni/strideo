"""Container startup tasks — runs once when the Streamlit process starts.

Imported at the top of app.py so it executes before any page is served:
  1. Downloads pose_landmarker_heavy.task if not present (MediaPipe needs this
     file on disk; it is not pip-installable and is not committed to the repo).
  2. Warms the RTMPose-x ONNX model so the first user upload does not stall
     waiting for a ~300 MB download mid-request.

Both steps are best-effort: failures are logged to stderr but never crash the
app — a missing weight file will surface as a pipeline error on the first
upload, which is more informative than a startup crash.
"""

from __future__ import annotations

import sys
import urllib.request
from pathlib import Path

_MEDIAPIPE_URL = (
    "https://storage.googleapis.com/mediapipe-models/"
    "pose_landmarker/pose_landmarker_heavy/float16/latest/"
    "pose_landmarker_heavy.task"
)

_MODELS_DIR = Path(__file__).parent.parent / "models"
_TASK_PATH = _MODELS_DIR / "pose_landmarker_heavy.task"


def _download_mediapipe() -> None:
    if _TASK_PATH.exists():
        return
    print(f"[preload] downloading MediaPipe weights → {_TASK_PATH}", flush=True)
    try:
        _MODELS_DIR.mkdir(parents=True, exist_ok=True)
        urllib.request.urlretrieve(_MEDIAPIPE_URL, _TASK_PATH)
        print("[preload] MediaPipe weights ready.", flush=True)
    except Exception as exc:
        print(f"[preload] WARNING: MediaPipe download failed: {exc}", file=sys.stderr, flush=True)


def _warm_rtmpose() -> None:
    print("[preload] warming RTMPose-x ONNX model…", flush=True)
    try:
        from rtmlib import Wholebody  # triggers ONNX download on first import

        Wholebody(mode="performance", to_openpose=False, backend="onnxruntime", device="cpu")
        print("[preload] RTMPose-x ready.", flush=True)
    except Exception as exc:
        print(f"[preload] WARNING: RTMPose warmup failed: {exc}", file=sys.stderr, flush=True)


_download_mediapipe()
_warm_rtmpose()
