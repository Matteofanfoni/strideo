"""Local (position-in-frame) pole calibration lookup.

Shared by the Kinovea GT builder and (eventually) the production pipeline to
correct for S1's confirmed camera yaw: one clip-wide px/m ratio (either the
pooled Kinovea pole average or the anthropometric self-calibration) silently
over/under-estimates measurements taken near the frame edges, where perspective
makes pixels-per-metre non-uniform across the visible track.

See ~/.claude/plans/idempotent-foraging-hippo.md for the full design and the
critical-correctness note on the correction's sign (this module only exposes
local ratios; applying the sign-sensitive correction itself is a downstream
concern, e.g. `SpatialCalibration.px_to_m`).
"""

from dataclasses import dataclass
from typing import Dict, Optional, Sequence

import numpy as np

POLE_LETTERS = ("A", "B", "C", "D")
PAIR_SPAN_LABELS = ("A-B", "B-C", "C-D")


@dataclass
class PoleSegment:
    """One pole-pair span's own local px/m ratio, with optional pixel-x anchors."""

    label: str
    measured_px: float
    ground_truth_m: float
    x_start_px: Optional[float] = None
    x_end_px: Optional[float] = None

    @property
    def px_per_m(self) -> float:
        """This span's own measured_px / ground_truth_m ratio."""
        return self.measured_px / self.ground_truth_m

    @property
    def midpoint_px(self) -> Optional[float]:
        """Pixel-x midpoint of this span, if both anchors are known."""
        if self.x_start_px is None or self.x_end_px is None:
            return None
        return (self.x_start_px + self.x_end_px) / 2.0


@dataclass
class PoleCurve:
    """Piecewise local px/m scale across the visible A-B/B-C/C-D pole spans."""

    segments: Dict[str, PoleSegment]

    def local_px_per_m(self, label: str) -> Optional[float]:
        """Direct lookup of a pole-pair span's own px/m ratio (Place 1's need)."""
        segment = self.segments.get(label)
        return segment.px_per_m if segment is not None else None

    @property
    def reference_px_per_m(self) -> float:
        """Pooled sum(px)/sum(m) across all available pair-spans.

        Used only as the "no correction" baseline ratio (Place 2's use case),
        not as a measurement itself.
        """
        total_px = sum(segment.measured_px for segment in self.segments.values())
        total_m = sum(segment.ground_truth_m for segment in self.segments.values())
        return total_px / total_m

    def px_per_m_at_x(self, x_px: float) -> Optional[float]:
        """Piecewise-linear interpolation of local px/m at a pixel-x position.

        Returns None if fewer than two segments carry pixel-x anchors
        (`pole_x_px` was never filled in, so continuous lookup isn't possible —
        callers should fall back to `local_px_per_m` categorical lookup instead).
        """
        anchors = [
            (s.midpoint_px, s.px_per_m)
            for s in self.segments.values()
            if s.midpoint_px is not None
        ]
        if len(anchors) < 2:
            return None
        anchors.sort(key=lambda pair: pair[0])  # type: ignore[arg-type,return-value]
        xp = [pair[0] for pair in anchors]
        fp = [pair[1] for pair in anchors]
        return float(np.interp(x_px, xp, fp))


def build_pole_curve(
    calibration_rows: Sequence[Dict[str, object]], resolution: str
) -> Optional[PoleCurve]:
    """Build a PoleCurve from kinovea_calibration.csv-shaped rows.

    Reads the three pair-span rows (A-B/B-C/C-D) directly at `resolution`.
    If any A/B/C/D pole-height row for that resolution carries a `pole_x_px`
    reading, derives absolute pole x-positions via cumulative sums of the
    pair-span pixel distances, anchored at that pole — enabling
    `PoleCurve.px_per_m_at_x`. Without any `pole_x_px`, the returned curve
    still supports `local_px_per_m` (categorical lookup; all Place 1 needs).

    Args:
        calibration_rows: Rows shaped like `kinovea_calibration.csv` (e.g. from
            `csv.DictReader`) — each a dict with `label`, `resolution`,
            `measured_px`, `ground_truth_m`, and optionally `pole_x_px`.
        resolution: Which resolution's rows to use (`"4K"` or `"1080p"`).

    Returns:
        A PoleCurve, or None if no usable pair-span rows exist at `resolution`.
    """
    span_segments: Dict[str, PoleSegment] = {}
    for row in calibration_rows:
        if row.get("resolution") != resolution:
            continue
        label = row.get("label")
        if label not in PAIR_SPAN_LABELS:
            continue
        try:
            measured_px = float(row["measured_px"])  # type: ignore[arg-type]
            ground_truth_m = float(row["ground_truth_m"])  # type: ignore[arg-type]
        except (KeyError, TypeError, ValueError):
            continue
        if ground_truth_m == 0:
            continue
        span_segments[str(label)] = PoleSegment(
            label=str(label), measured_px=measured_px, ground_truth_m=ground_truth_m
        )

    if not span_segments:
        return None

    pole_x_px: Dict[str, float] = {}
    for row in calibration_rows:
        if row.get("resolution") != resolution:
            continue
        label = row.get("label")
        if label not in POLE_LETTERS:
            continue
        raw_x = row.get("pole_x_px")
        if raw_x in (None, ""):
            continue
        try:
            pole_x_px[label] = float(raw_x)  # type: ignore[index,arg-type]
        except (TypeError, ValueError):
            continue

    anchor_letter = next(
        (letter for letter in POLE_LETTERS if letter in pole_x_px), None
    )

    if anchor_letter is not None:
        offsets: Dict[str, float] = {"A": 0.0}
        cumulative = 0.0
        for start, end in zip(POLE_LETTERS, POLE_LETTERS[1:]):
            span_label = f"{start}-{end}"
            segment = span_segments.get(span_label)
            if segment is None:
                break  # cumulative sum can't cross a missing span
            cumulative += segment.measured_px
            offsets[end] = cumulative

        if anchor_letter in offsets:
            shift = pole_x_px[anchor_letter] - offsets[anchor_letter]
            positions = {letter: offset + shift for letter, offset in offsets.items()}

            for span_label, segment in span_segments.items():
                start, end = span_label.split("-")
                segment.x_start_px = positions.get(start)
                segment.x_end_px = positions.get(end)

    return PoleCurve(segments=span_segments)
