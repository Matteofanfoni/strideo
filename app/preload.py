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


def _log_env_versions() -> None:
    """Print key inference-library versions so the deployed environment can be
    compared against the validated local one (BlazePose detection rate is
    sensitive to the MediaPipe wheel build, which differs per-version and
    per-platform)."""
    try:
        import cv2
        import mediapipe
        import numpy
        import onnxruntime

        print(
            "[preload] env versions: "
            f"mediapipe={mediapipe.__version__} "
            f"numpy={numpy.__version__} "
            f"opencv={cv2.__version__} "
            f"onnxruntime={onnxruntime.__version__}",
            flush=True,
        )
    except Exception as exc:  # noqa: BLE001
        print(f"[preload] WARNING: version probe failed: {exc}", flush=True)


def _clean_stale_sessions() -> None:
    """Remove idle per-session upload dirs at container start.

    The same rule as every page run's sweep (``utils.session_files``): a
    folder untouched for an hour is deleted. That per-run sweep is what keeps
    the privacy policy true on a container that runs for weeks; this one only
    clears what a restart left behind before the first visitor arrives.
    """
    from utils.session_files import sweep_idle_sessions

    sweep_idle_sessions(force=True)


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


_log_env_versions()
_clean_stale_sessions()
_download_mediapipe()
_warm_rtmpose()
