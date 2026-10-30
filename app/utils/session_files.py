"""Per-visitor upload folders, and their deletion once the visitor is idle.

Each browser session keeps its uploads (and the prepared clips derived from
them) in ``<tempdir>/strideo_sessions/<session_id>/``, because the Results
page's video scrubber reads them back during the visit. The privacy policy
promises they are deleted within an hour of the visitor's last activity.

Until 2026-10-18 the only sweep ran once, at container start
(``preload._clean_stale_sessions``, 4 hours), and since the Space is set
never to sleep a container could run for weeks: uploads stayed on disk long
after the policy said they were gone. From then every page run marked the
current visitor's folder as active and swept idle ones.

That still needed traffic: a folder outlived the hour until someone else
opened a page, which on a quiet Space could be days. So the
first page run in a process also starts a background sweeper, and the idle
limit plus the sweep interval add up to exactly the hour the policy states.
"""

from __future__ import annotations

import os
import shutil
import tempfile
import threading
import time
from pathlib import Path
from typing import Optional

#: The hour the privacy policy promises, from last activity to deletion.
PROMISED_MAX_S = 3600
#: How often the background sweeper runs; the sweep is a directory listing.
SWEEP_INTERVAL_S = 60
#: A folder untouched for this long belongs to a visitor who has left. Set so
#: that a folder going idle just after a sweep is still gone within the hour.
SESSION_MAX_IDLE_S = PROMISED_MAX_S - SWEEP_INTERVAL_S

_SWEEPER_NAME = "strideo-session-sweeper"
_sweeper_lock = threading.Lock()
_stop_sweeper = threading.Event()

_last_sweep = 0.0


def sessions_root() -> Path:
    return Path(tempfile.gettempdir()) / "strideo_sessions"


def session_dir(session_id: str) -> Path:
    """The visitor's upload folder, created if needed and marked active."""
    path = sessions_root() / session_id
    path.mkdir(parents=True, exist_ok=True)
    os.utime(path)
    return path


def mark_active(session_id: Optional[str]) -> None:
    """Refresh the visitor's folder, if it exists, so the sweep keeps it."""
    if not session_id:
        return
    path = sessions_root() / session_id
    try:
        if path.is_dir():
            os.utime(path)
    except OSError:
        pass


def sweep_idle_sessions(
    max_idle_s: float = SESSION_MAX_IDLE_S,
    *,
    now: Optional[float] = None,
    force: bool = False,
) -> int:
    """Delete session folders idle for more than ``max_idle_s``.

    Throttled to once per ``SWEEP_INTERVAL_S`` per process unless ``force``.
    Returns how many folders were deleted.
    """
    global _last_sweep
    now = time.time() if now is None else now
    if not force and now - _last_sweep < SWEEP_INTERVAL_S:
        return 0
    _last_sweep = now
    root = sessions_root()
    if not root.exists():
        return 0
    removed = 0
    for path in root.iterdir():
        try:
            if path.is_dir() and now - path.stat().st_mtime > max_idle_s:
                shutil.rmtree(path, ignore_errors=True)
                removed += 1
        except OSError:
            pass
    return removed


def _sweep_until_stopped(interval_s: float) -> None:
    while not _stop_sweeper.wait(interval_s):
        try:
            sweep_idle_sessions(force=True)
        except Exception:  # noqa: BLE001 - a failed sweep must not end the loop
            pass


def start_sweeper(interval_s: float = SWEEP_INTERVAL_S) -> threading.Thread:
    """Start the background sweeper once per process; return the running one.

    Found by thread name rather than a module flag, so a module re-import
    (Streamlit's dev reloader) does not start a second one.
    """
    with _sweeper_lock:
        for thread in threading.enumerate():
            if thread.name == _SWEEPER_NAME and thread.is_alive():
                return thread
        _stop_sweeper.clear()
        thread = threading.Thread(
            target=_sweep_until_stopped,
            args=(interval_s,),
            name=_SWEEPER_NAME,
            daemon=True,
        )
        thread.start()
        return thread


def stop_sweeper(timeout_s: float = 5.0) -> None:
    """Stop the background sweeper (tests only; the app never stops it)."""
    _stop_sweeper.set()
    for thread in threading.enumerate():
        if thread.name == _SWEEPER_NAME:
            thread.join(timeout_s)


def on_page_run(session_id: Optional[str]) -> None:
    """Called on every page run: keep this visitor's folder, drop idle ones.

    Also makes sure the background sweeper is running, which is what keeps
    the hour when nobody opens a page.
    """
    mark_active(session_id)
    start_sweeper()
    sweep_idle_sessions()
