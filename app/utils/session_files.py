"""Per-visitor upload folders, and their deletion once the visitor is idle.

Each browser session keeps its uploads (and the prepared clips derived from
them) in ``<tempdir>/strideo_sessions/<session_id>/``, because the Results
page's video scrubber reads them back during the visit. The privacy policy
promises they are deleted within an hour of the visitor's last activity.

Until 2026-10-18 the only sweep ran once, at container start
(``preload._clean_stale_sessions``, 4 hours), and since the Space is set
never to sleep a container could run for weeks: uploads stayed on disk long
after the policy said they were gone. Now every page run marks the current
visitor's folder as active, and sweeps idle ones at most every
``SWEEP_INTERVAL_S``.
"""

from __future__ import annotations

import os
import shutil
import tempfile
import time
from pathlib import Path
from typing import Optional

#: A folder untouched for this long belongs to a visitor who has left.
SESSION_MAX_IDLE_S = 3600
#: The sweep is a directory listing; this bounds how often a page run pays it.
SWEEP_INTERVAL_S = 300

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


def on_page_run(session_id: Optional[str]) -> None:
    """Called on every page run: keep this visitor's folder, drop idle ones."""
    mark_active(session_id)
    sweep_idle_sessions()
