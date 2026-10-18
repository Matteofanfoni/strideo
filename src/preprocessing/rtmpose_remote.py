"""Client for RTMPose-x on a Modal GPU (the ``strideo-rtmpose`` Modal app).

``extract_rtmpose_remote`` sends the prepared clip to the deployed Modal class
and returns the same ``RTMPoseResult`` the local
``extract_rtmpose_landmarks_windowed`` would. Every failure, from a missing
token to a stalled call or a wrong-shaped answer, raises
``RemoteRTMPoseError``, so the caller has one thing to catch before falling
back to local CPU RTMPose.

The call is opt-in: ``remote_enabled`` reads ``STRIDEO_RTMPOSE_REMOTE``, so the
code can ship before the privacy policy names Modal.
"""

from __future__ import annotations

import os
import queue
import threading
from pathlib import Path
from typing import Any, Callable, Optional, Tuple

import numpy as np

from src.preprocessing.rtmpose_extractor import RTMPoseResult

APP_NAME = "strideo-rtmpose"
CLS_NAME = "RTMPoseGPU"

#: Seconds to wait for the first event: upload plus a cold container start.
FIRST_EVENT_TIMEOUT_S = 60.0
#: Seconds allowed between later events before the call counts as stalled.
STALL_TIMEOUT_S = 30.0

ProgressCallback = Callable[[int, int], None]


class RemoteRTMPoseError(RuntimeError):
    """The Modal call could not produce a usable result."""


def remote_enabled() -> bool:
    """True when the app should try Modal first (``STRIDEO_RTMPOSE_REMOTE=modal``)."""
    return os.environ.get("STRIDEO_RTMPOSE_REMOTE", "").strip().lower() == "modal"


def _array(payload: dict, key: str) -> np.ndarray:
    shape = tuple(payload[f"{key}_shape"])
    return np.frombuffer(payload[key], dtype=np.float32).reshape(shape).copy()


def extract_rtmpose_remote(
    video_path: str,
    n_frames: int,
    window: Optional[Tuple[int, int]],
    skip_rate: int,
    *,
    progress: Optional[ProgressCallback] = None,
    first_event_timeout_s: float = FIRST_EVENT_TIMEOUT_S,
    stall_timeout_s: float = STALL_TIMEOUT_S,
    cls: Any = None,
) -> Tuple[RTMPoseResult, str]:
    """RTMPose-x for ``video_path`` on the Modal GPU.

    Args:
        video_path: The prepared (CFR, capped) clip, as for the local call.
        n_frames: BlazePose's frame count; the result has this length.
        window: Pose-frame window from ``rtmpose_window``, or None.
        skip_rate: BlazePose's ``skip_rate``.
        progress: Optional callback(done, total), as for the local call.
        first_event_timeout_s: Deadline for the first event.
        stall_timeout_s: Deadline between later events.
        cls: The Modal class to call, instead of looking up the deployed
            one; for an ephemeral ``app.run()`` and for tests.

    Returns:
        (result, device): the ``RTMPoseResult`` and the device the remote
        session actually ran on.

    Raises:
        RemoteRTMPoseError: On any failure; the caller falls back to local.
    """
    try:
        if cls is None:
            import modal

            cls = modal.Cls.from_name(APP_NAME, CLS_NAME)
        video = Path(video_path).read_bytes()
    except Exception as exc:
        raise RemoteRTMPoseError(f"cannot start Modal call: {exc}") from exc

    events: "queue.Queue[tuple]" = queue.Queue()

    def _consume() -> None:
        try:
            for event in cls().extract.remote_gen(
                video, n_frames, list(window) if window else None, skip_rate
            ):
                events.put(event)
        except Exception as exc:
            events.put(("error", f"{type(exc).__name__}: {exc}"))
        events.put(("end",))

    # Daemon: on a timeout the thread is abandoned, not joined; the remote
    # call is bounded by the Modal function's own timeout.
    threading.Thread(target=_consume, daemon=True).start()

    timeout = first_event_timeout_s
    while True:
        try:
            event = events.get(timeout=timeout)
        except queue.Empty:
            raise RemoteRTMPoseError(f"no answer from Modal within {timeout:.0f}s")
        timeout = stall_timeout_s
        kind = event[0]
        if kind == "progress":
            if progress is not None:
                progress(event[1], event[2])
        elif kind == "result":
            payload = event[1]
            try:
                keypoints = _array(payload, "keypoints")
                scores = _array(payload, "scores")
            except Exception as exc:
                raise RemoteRTMPoseError(f"unreadable result: {exc}") from exc
            if keypoints.shape != (n_frames, 133, 2) or scores.shape != (
                n_frames,
                133,
            ):
                raise RemoteRTMPoseError(
                    f"wrong shape {keypoints.shape}/{scores.shape} "
                    f"for {n_frames} frames"
                )
            result = RTMPoseResult(
                keypoints=keypoints, scores=scores, fps=float(payload["fps"])
            )
            return result, str(payload["device"])
        elif kind == "error":
            raise RemoteRTMPoseError(f"Modal call failed: {event[1]}")
        else:  # "end" with no result
            raise RemoteRTMPoseError("Modal call ended without a result")
