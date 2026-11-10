"""An anonymous count of completed analyses, by campaign link tag.

Marketing links carry ``?src=<tt|rd>-<angle>-<n>`` (``strideo.org`` forwards
its query string into the Space's iframe). The tag is remembered for the
session on whichever page the visitor lands, because a page switch drops the
query string, and every clip that gets a result records one event.

An event is an empty file in a Modal Volume at
``/events/<YYYY-MM-DD>/<tag>/<random hex>``: the date, the tag, and an id that
exists only so two events never share a path. No IP, no session, nothing from
the video or the runner. A Volume rather than a Modal Dict because a Dict's
entries expire after 7 days without a read or write.

Off unless ``STRIDEO_USAGE_COUNT=1``, so local runs and tests write nothing.
The write runs on a daemon thread and swallows every error: counting must
never slow or break an analysis. ``scripts/count_analyses.py`` reads the
count.

**Source frame rates, a separate record behind its own flag.** Each
analysed clip can also write an empty file at
``/sources/<YYYY-MM-DD>/<rate class>/<random hex>``: the date and a coarse
class of the upload's frame rate (``60``, ``30``, ``120``, ``vfr-30``,
``other``, ...), which decides whether ``ensure_cfr`` re-encodes it. It
carries no tag, so it cannot be joined to a campaign, and nothing else from
the video. This *is* something from the video, which the privacy policy's
description of the count ruled out until the policy named it
(2026-11-10). It is off unless ``STRIDEO_SOURCE_LOG=1`` on the Space.
"""

import datetime
import io
import logging
import os
import re
import threading
import uuid
from typing import Optional

logger = logging.getLogger(__name__)

VOLUME_NAME = "strideo-usage"
ENV_FLAG = "STRIDEO_USAGE_COUNT"
SOURCE_ENV_FLAG = "STRIDEO_SOURCE_LOG"
SESSION_KEY = "campaign_tag"

#: Nominal rates a source is matched to; anything else is ``other``.
RATE_CLASSES = (24, 25, 30, 50, 60, 120, 240)
#: Same relative tolerance as ``ensure_cfr``'s constancy and rate checks.
RATE_TOLERANCE = 0.01

#: The documented link format only. Anything else is stored as ``other``, so
#: a hand-edited URL cannot write free text (a name, say) into the count.
_TAG_RE = re.compile(r"^(tt|rd)-[a-z0-9]{1,24}(-[a-z0-9]{1,24}){0,3}-[0-9]{1,3}$")


def clean_tag(raw: Optional[str]) -> str:
    """``raw`` if it matches the link format, ``none`` if absent, else
    ``other``. Idempotent, so an already-cleaned tag passes through."""
    if not raw:
        return "none"
    tag = str(raw).strip().lower()
    if tag in ("none", "other") or _TAG_RE.match(tag):
        return tag
    return "other"


def remember_campaign_tag() -> None:
    """Store the landing page's ``src`` tag for the session, once."""
    import streamlit as st

    if SESSION_KEY not in st.session_state:
        st.session_state[SESSION_KEY] = clean_tag(st.query_params.get("src"))


def event_path(tag: str, now: datetime.datetime, uid: str) -> str:
    """The Volume path for one event."""
    return f"/events/{now:%Y-%m-%d}/{tag}/{uid}"


def record_completed_analysis(tag: Optional[str]) -> None:
    """Record one completed analysis under ``tag``, in the background."""
    if os.environ.get(ENV_FLAG, "").strip() != "1":
        return
    path = event_path(
        clean_tag(tag), datetime.datetime.now(datetime.timezone.utc), uuid.uuid4().hex
    )
    threading.Thread(target=_write_event, args=(path,), daemon=True).start()


def rate_class(r_fps: float, avg_fps: float) -> str:
    """A coarse class for a source's frame rate: the nominal rate it matches
    (``"30"``), prefixed ``vfr-`` when the stream is variable by
    ``ensure_cfr``'s own test, or ``"other"``/``"vfr-other"``; ``"unknown"``
    when the probe read nothing."""
    if r_fps <= 0 or avg_fps <= 0:
        return "unknown"
    constant = abs(r_fps - avg_fps) / r_fps <= RATE_TOLERANCE
    nominal = next(
        (str(c) for c in RATE_CLASSES if abs(avg_fps - c) / c <= RATE_TOLERANCE),
        None,
    )
    if nominal is None and not constant:
        # A variable stream's average drifts; its nominal rate is the header's.
        nominal = next(
            (str(c) for c in RATE_CLASSES if abs(r_fps - c) / c <= RATE_TOLERANCE),
            None,
        )
    name = nominal or "other"
    return name if constant else f"vfr-{name}"


def source_path(rate: str, now: datetime.datetime, uid: str) -> str:
    """The Volume path for one source-frame-rate record."""
    return f"/sources/{now:%Y-%m-%d}/{rate}/{uid}"


def record_source_rate(video_path: str) -> None:
    """Record ``video_path``'s frame-rate class, in the background, when
    ``STRIDEO_SOURCE_LOG=1``."""
    if os.environ.get(SOURCE_ENV_FLAG, "").strip() != "1":
        return
    threading.Thread(target=_probe_and_write, args=(video_path,), daemon=True).start()


def _probe_and_write(video_path: str) -> None:
    try:
        from src.preprocessing.frame_rate import _probe_rates

        rate = rate_class(*_probe_rates(video_path))
    except Exception:  # noqa: BLE001 - recording must never raise
        rate = "unknown"
    now = datetime.datetime.now(datetime.timezone.utc)
    _write_event(source_path(rate, now, uuid.uuid4().hex))


def _write_event(path: str) -> None:
    try:
        import modal

        vol = modal.Volume.from_name(VOLUME_NAME, create_if_missing=True)
        with vol.batch_upload() as batch:
            batch.put_file(io.BytesIO(b""), path)
    except Exception as exc:  # noqa: BLE001 - counting must never raise
        # WARNING, not INFO: the Space prints no INFO lines.
        logger.warning("usage count not recorded: %s", type(exc).__name__)
