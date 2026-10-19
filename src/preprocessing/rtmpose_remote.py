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

import logging
import os
import queue
import threading
import time
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
#: How often ``on_wait`` is called while waiting for the first event.
WAIT_TICK_S = 0.5

ProgressCallback = Callable[[int, int], None]

logger = logging.getLogger(__name__)


class RemoteRTMPoseError(RuntimeError):
    """The Modal call could not produce a usable result."""


#: Containers one RTMPose pass is split across. RTMPose has no state from
#: frame to frame, so each part is the same as that stretch of a whole-clip
#: run; two halve the pass, which is then shorter than the fast path.
RTMPOSE_PARTS = 2

#: Seconds each warm-up holds its container, so that the parts' warm-ups
#: each start a container instead of queueing on the first one to answer.
WARM_HOLD_S = 2.0


def warm_remote(cls: Any = None, parts: int = RTMPOSE_PARTS) -> threading.Event:
    """Start ``parts`` Modal GPU containers ahead of the real call, without
    blocking.

    The containers' start-up (~8-12 s, model load included) then overlaps
    whatever the caller does next, such as preparing the video. Wait on the
    returned event before the real call: a container takes one call at a
    time, so a call sent while one is still starting could start yet another
    container instead of reusing it.

    Never raises. A failure (no token, no ``warm`` method on an older
    deploy, Modal down) is logged and still counts as answered, so the real
    call just pays the cold start it would have paid anyway.

    Args:
        cls: The Modal class, instead of looking up the deployed one; for
            tests.
        parts: How many containers to wake, one per part of the real call.

    Returns:
        An event set once every warm-up has answered or failed.
    """
    done = threading.Event()
    remaining = [parts]
    lock = threading.Lock()

    def _run() -> None:
        try:
            target = cls
            if target is None:
                import modal

                target = modal.Cls.from_name(APP_NAME, CLS_NAME)
            target().warm.remote(WARM_HOLD_S)
        except Exception as exc:  # noqa: BLE001 - a warm-up is best effort
            logger.info("Modal warm-up failed: %s", exc)
        finally:
            with lock:
                remaining[0] -= 1
                if remaining[0] <= 0:
                    done.set()

    for _ in range(parts):
        threading.Thread(target=_run, daemon=True).start()
    return done


def _split(lo: int, hi: int, parts: int) -> list:
    """``[lo, hi]`` (inclusive) as ``parts`` contiguous, near-equal ranges."""
    n = hi - lo + 1
    parts = max(1, min(parts, n))
    edges = [lo + (n * i) // parts for i in range(parts + 1)]
    return [(edges[i], edges[i + 1] - 1) for i in range(parts)]


def _pose_frame_count(video_path: str, skip_rate: int) -> int:
    import cv2

    cap = cv2.VideoCapture(video_path)
    n = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    cap.release()
    if n <= 0:
        raise RemoteRTMPoseError(f"cannot read the frame count of {video_path}")
    return (n + skip_rate - 1) // skip_rate


def extract_rtmpose_remote_parallel(
    video_path: str,
    n_frames: Optional[int],
    window: Optional[Tuple[int, int]],
    skip_rate: int,
    *,
    parts: int = RTMPOSE_PARTS,
    progress: Optional[ProgressCallback] = None,
    on_wait: Optional[Callable[[float], None]] = None,
    cls: Any = None,
    poll_s: float = 0.25,
) -> Tuple[RTMPoseResult, str]:
    """``extract_rtmpose_remote`` with the frames split across ``parts``
    containers that run at the same time.

    Each part is one ``extract_rtmpose_remote`` call on a sub-window, which
    returns a full-length result with zeros outside its sub-window; the
    parts are stitched by range. RTMPose detects and selects the runner per
    frame with nothing carried between frames, so the result equals one
    call on the whole range.

    ``progress`` and ``on_wait`` are called on the caller's thread (a
    Streamlit bar cannot be updated from another): the parts' threads only
    record how far they have got.

    Args:
        video_path, window, skip_rate: As for ``extract_rtmpose_remote``.
        n_frames: BlazePose's frame count, or None for the whole clip, whose
            pose-frame count is then read from the file.
        parts: How many containers to split across.
        progress: Optional callback(done, total) over all parts together.
        on_wait: Optional callback(seconds_elapsed) until any part's first
            frame arrives.
        cls: The Modal class; for tests.
        poll_s: How often the caller's thread reports.

    Returns:
        (result, device), as for ``extract_rtmpose_remote``.

    Raises:
        RemoteRTMPoseError: If any part fails.
    """
    if n_frames is None:
        if window is not None:
            raise ValueError("a window needs BlazePose's frame count")
        n_frames = _pose_frame_count(video_path, skip_rate)
    lo, hi = window if window is not None else (0, n_frames - 1)
    ranges = _split(lo, hi, parts)

    done = [0] * len(ranges)
    totals = [b - a + 1 for a, b in ranges]
    results: list = [None] * len(ranges)
    errors: list = []
    finished = [threading.Event() for _ in ranges]

    def _run(i: int, sub: Tuple[int, int]) -> None:
        def _p(d: int, _t: int) -> None:
            done[i] = d

        try:
            results[i] = extract_rtmpose_remote(
                video_path, n_frames, sub, skip_rate, progress=_p, cls=cls
            )
        except Exception as exc:  # noqa: BLE001 - re-raised on the caller
            errors.append(exc)
        finally:
            finished[i].set()

    for i, sub in enumerate(ranges):
        threading.Thread(target=_run, args=(i, sub), daemon=True).start()

    started = time.monotonic()
    while not all(f.is_set() for f in finished):
        time.sleep(poll_s)
        if errors:
            break
        if any(done):
            if progress is not None:
                progress(sum(done), sum(totals))
        elif on_wait is not None:
            on_wait(time.monotonic() - started)
    if errors:
        exc = errors[0]
        if isinstance(exc, RemoteRTMPoseError):
            raise exc
        raise RemoteRTMPoseError(f"{type(exc).__name__}: {exc}") from exc

    keypoints = np.zeros((n_frames, 133, 2), dtype=np.float32)
    scores = np.zeros((n_frames, 133), dtype=np.float32)
    for (a, b), (res, _dev) in zip(ranges, results):
        keypoints[a : b + 1] = res.keypoints[a : b + 1]
        scores[a : b + 1] = res.scores[a : b + 1]
    if progress is not None:
        progress(sum(totals), sum(totals))
    first, device = results[0]
    return RTMPoseResult(keypoints=keypoints, scores=scores, fps=first.fps), device


def remote_enabled() -> bool:
    """True when the app should try Modal first (``STRIDEO_RTMPOSE_REMOTE=modal``)."""
    return os.environ.get("STRIDEO_RTMPOSE_REMOTE", "").strip().lower() == "modal"


def _array(payload: dict, key: str) -> np.ndarray:
    shape = tuple(payload[f"{key}_shape"])
    return np.frombuffer(payload[key], dtype=np.float32).reshape(shape).copy()


def extract_rtmpose_remote(
    video_path: str,
    n_frames: Optional[int],
    window: Optional[Tuple[int, int]],
    skip_rate: int,
    *,
    progress: Optional[ProgressCallback] = None,
    first_event_timeout_s: float = FIRST_EVENT_TIMEOUT_S,
    stall_timeout_s: float = STALL_TIMEOUT_S,
    on_wait: Optional[Callable[[float], None]] = None,
    cls: Any = None,
) -> Tuple[RTMPoseResult, str]:
    """RTMPose-x for ``video_path`` on the Modal GPU.

    Args:
        video_path: The prepared (CFR, capped) clip, as for the local call.
        n_frames: BlazePose's frame count; the result has this length. None
            when the call starts before BlazePose has run (the app's
            automatic engine): the whole clip is extracted, its length is not
            checked here, and the caller compares it with BlazePose's.
            Needs ``window`` None, since a window is in BlazePose's frames.
        window: Pose-frame window from ``rtmpose_window``, or None.
        skip_rate: BlazePose's ``skip_rate``.
        progress: Optional callback(done, total), as for the local call.
        first_event_timeout_s: Deadline for the first event.
        stall_timeout_s: Deadline between later events.
        on_wait: Optional callback(seconds_elapsed), called about every
            ``WAIT_TICK_S`` until the first event arrives.
        cls: The Modal class to call, instead of looking up the deployed
            one; for an ephemeral ``app.run()`` and for tests.

    Returns:
        (result, device): the ``RTMPoseResult`` and the device the remote
        session actually ran on.

    Raises:
        RemoteRTMPoseError: On any failure; the caller falls back to local.
    """
    if n_frames is None and window is not None:
        raise ValueError("a window needs BlazePose's frame count")
    try:
        if cls is None:
            import modal

            cls = modal.Cls.from_name(APP_NAME, CLS_NAME)
        video = Path(video_path).read_bytes()
    except Exception as exc:
        raise RemoteRTMPoseError(f"cannot start Modal call: {exc}") from exc

    events: "queue.Queue[tuple]" = queue.Queue()

    def _consume() -> None:
        # With no window the remote side reads neither the frame count nor
        # the skip rate, so 0 stands in for an unknown count.
        try:
            for event in cls().extract.remote_gen(
                video, n_frames or 0, list(window) if window else None, skip_rate
            ):
                events.put(event)
        except Exception as exc:
            events.put(("error", f"{type(exc).__name__}: {exc}"))
        events.put(("end",))

    # Daemon: on a timeout the thread is abandoned, not joined; the remote
    # call is bounded by the Modal function's own timeout.
    threading.Thread(target=_consume, daemon=True).start()

    # Before the first event (upload plus container start, ~13 s live) wake
    # every WAIT_TICK_S so the caller's bar can keep moving.
    started = time.monotonic()
    first = True
    while True:
        if first:
            elapsed = time.monotonic() - started
            if elapsed >= first_event_timeout_s:
                raise RemoteRTMPoseError(
                    f"no answer from Modal within {first_event_timeout_s:.0f}s"
                )
            try:
                event = events.get(
                    timeout=min(WAIT_TICK_S, first_event_timeout_s - elapsed)
                )
            except queue.Empty:
                if on_wait is not None:
                    on_wait(time.monotonic() - started)
                continue
            first = False
        else:
            try:
                event = events.get(timeout=stall_timeout_s)
            except queue.Empty:
                raise RemoteRTMPoseError(
                    f"no answer from Modal within {stall_timeout_s:.0f}s"
                )
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
            n = len(keypoints) if n_frames is None else n_frames
            if keypoints.shape != (n, 133, 2) or scores.shape != (n, 133):
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
