"""Shared per-frame overlay drawing helpers for pose/contact visualisation.

Single source of truth for the skeleton, ground-contact markers, and outlined
text drawn on top of pipeline frames. Consumed by both:

* ``scripts/run_prevalidation_single.py`` — QA annotated-video rendering.
* ``app/`` — the interactive frame scrubber and optional ffmpeg-encoded video.

All functions operate on a single BGR image (the OpenCV convention) and one
frame's worth of landmarks/visibilities, so they work identically for a single
scrubbed frame and for a streaming video pass.

Coordinate convention
----------------------
Landmarks are expected in the pixel coordinate space of ``img``. When drawing
on a resized (e.g. 720p) frame, scale the landmarks first with
:func:`scale_landmarks` — the radius/thickness of glyphs auto-scale from
``img.shape[1]`` so they stay legible at any resolution.
"""

from __future__ import annotations

from typing import TYPE_CHECKING, Dict, List, Tuple

import cv2
import numpy as np

if TYPE_CHECKING:  # avoid importing mediapipe-heavy modules at runtime
    from src.preprocessing.ground_contact import GroundContact

# Ankle landmark indices in the 33-point MediaPipe Pose topology.
LEFT_ANKLE_IDX = 27
RIGHT_ANKLE_IDX = 28

# 33-landmark MediaPipe Pose connection topology (face + body + hands + feet).
# Hard-coded because mp.solutions is not available in the tasks-only build
# used by src/preprocessing/pose_estimator.py.
POSE_CONNECTIONS: Tuple[Tuple[int, int], ...] = (
    # Face
    (0, 1),
    (1, 2),
    (2, 3),
    (3, 7),
    (0, 4),
    (4, 5),
    (5, 6),
    (6, 8),
    (9, 10),
    # Torso
    (11, 12),
    (11, 23),
    (12, 24),
    (23, 24),
    # Left arm
    (11, 13),
    (13, 15),
    (15, 17),
    (15, 19),
    (15, 21),
    (17, 19),
    # Right arm
    (12, 14),
    (14, 16),
    (16, 18),
    (16, 20),
    (16, 22),
    (18, 20),
    # Left leg
    (23, 25),
    (25, 27),
    (27, 29),
    (27, 31),
    (29, 31),
    # Right leg
    (24, 26),
    (26, 28),
    (28, 30),
    (28, 32),
    (30, 32),
)

CONTACT_COLORS: Dict[str, Tuple[int, int, int]] = {
    "red": (0, 0, 255),
    "white": (255, 255, 255),
    "yellow": (0, 255, 255),
    "green": (0, 200, 0),
}


def vis_colour(v: float) -> Tuple[int, int, int]:
    """BGR colour keyed to a visibility score (green/yellow/red)."""
    if v >= 0.7:
        return (0, 220, 0)  # green
    if v >= 0.4:
        return (0, 215, 230)  # yellow
    return (0, 0, 230)  # red


def scale_landmarks(landmarks: np.ndarray, ratio: float) -> np.ndarray:
    """Return a copy of ``landmarks`` (..., 2) scaled by ``ratio``.

    Use when drawing source-pixel landmarks onto a resized display frame:
    ``ratio = display_width / source_width``. NaNs are preserved.
    """
    return landmarks * ratio


def put_text(
    img: np.ndarray,
    text: str,
    org: Tuple[int, int],
    scale: float = 0.8,
    colour: Tuple[int, int, int] = (255, 255, 255),
    thickness: int = 2,
) -> None:
    """Draw outlined text (black halo + coloured fill) for legibility on video."""
    cv2.putText(
        img,
        text,
        org,
        cv2.FONT_HERSHEY_SIMPLEX,
        scale,
        (0, 0, 0),
        thickness + 2,
        cv2.LINE_AA,
    )
    cv2.putText(
        img,
        text,
        org,
        cv2.FONT_HERSHEY_SIMPLEX,
        scale,
        colour,
        thickness,
        cv2.LINE_AA,
    )


def draw_skeleton(
    img: np.ndarray, frame_landmarks: np.ndarray, frame_visibilities: np.ndarray
) -> None:
    """Draw the 33-point skeleton on ``img`` with visibility-keyed colours."""
    if np.all(np.isnan(frame_landmarks)):
        return
    radius = max(3, img.shape[1] // 600)
    thick = max(2, img.shape[1] // 900)
    for i, j in POSE_CONNECTIONS:
        p_i = frame_landmarks[i]
        p_j = frame_landmarks[j]
        if np.any(np.isnan(p_i)) or np.any(np.isnan(p_j)):
            continue
        v = min(float(frame_visibilities[i]), float(frame_visibilities[j]))
        if v < 0.2:
            continue
        cv2.line(
            img,
            (int(p_i[0]), int(p_i[1])),
            (int(p_j[0]), int(p_j[1])),
            vis_colour(v),
            thick,
            cv2.LINE_AA,
        )
    for i in range(33):
        p = frame_landmarks[i]
        v = float(frame_visibilities[i])
        if np.any(np.isnan(p)) or v < 0.2:
            continue
        cv2.circle(img, (int(p[0]), int(p[1])), radius, vis_colour(v), -1, cv2.LINE_AA)


def draw_contact_markers(
    img: np.ndarray,
    frame_idx: int,
    frame_landmarks: np.ndarray,
    contacts: "List[GroundContact]",
    contact_color: str = "red",
    label_scale: float = 0.8,
    draw_labels: bool = True,
) -> None:
    """Mark active ground contacts at the ankle for the given frame.

    A contact is active when ``contact_frame <= frame_idx <= toe_off_frame``.
    Each active contact draws a small dot at the relevant ankle plus (when
    ``draw_labels``) a label with its progressive index, strike pattern, IC/TO
    frames, leg, GCT, and confidence. The scrubber toggles dots and labels
    independently by calling with ``draw_labels=False``.
    """
    ankle_l = frame_landmarks[LEFT_ANKLE_IDX]
    ankle_r = frame_landmarks[RIGHT_ANKLE_IDX]
    radius = max(8, img.shape[1] // 200)
    bgr = CONTACT_COLORS.get(contact_color, (0, 0, 255))
    # Sort by coarse frame so progressive index is stable across the clip.
    sorted_contacts = sorted(contacts, key=lambda c: c.frame)
    for prog, c in enumerate(sorted_contacts, start=1):
        if c.contact_frame is None or c.toe_off_frame is None:
            continue
        if not (c.contact_frame <= frame_idx <= c.toe_off_frame):
            continue
        ankle = ankle_l if c.leg == "L" else ankle_r
        if np.any(np.isnan(ankle)):
            continue
        centre = (int(ankle[0]), int(ankle[1]))
        dot_r = max(4, img.shape[1] // 640)  # ~6 px at 4K — small dot
        cv2.circle(img, centre, dot_r, bgr, -1, cv2.LINE_AA)
        if not draw_labels:
            continue
        # Label: progressive index, refined IC + TO frames (sub-frame
        # floats rounded to nearest int), side, GCT, confidence.
        ic = round(c.contact_frame) if c.contact_frame is not None else c.frame
        to = round(c.toe_off_frame) if c.toe_off_frame is not None else "?"
        sp = (c.strike_pattern or "?")[0].upper()  # F / H / U / ?
        if c.gct_ms is not None:
            label = (
                f"#{prog}[{sp}] IC{ic} TO{to} {c.leg}  "
                f"{c.gct_ms:.0f}ms  {c.confidence:.2f}"
            )
        else:
            label = f"#{prog}[{sp}] IC{ic} TO{to} {c.leg}"
        font = cv2.FONT_HERSHEY_SIMPLEX
        thickness = 2
        (tw, _), _ = cv2.getTextSize(label, font, label_scale, thickness)
        margin = radius + 6
        x = centre[0] + margin
        if x + tw > img.shape[1]:
            x = centre[0] - margin - tw
        put_text(
            img,
            label,
            (max(0, x), centre[1] - 4),
            scale=label_scale,
            colour=bgr,
        )
