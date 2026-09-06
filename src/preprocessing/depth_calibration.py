"""Camera-geometry depth calibration for the Kinovea pole-anchored speed gate.

Sibling to :mod:`src.preprocessing.pole_calibration`. Where ``pole_calibration``
answers *"what is the local px/m scale at this image column?"* (a
position-in-frame correction for camera yaw), this module answers the
orthogonal question *"how far from the calibrated pole line was the runner
actually running?"* — the depth/parallax term.

Promoted verbatim (logic unchanged) out of
the one-time correction-factor generator, so that it and the
diagnosis script share a single implementation instead of duplicating it.
Everything here is pure: callers hand in already-loaded CSV rows and landmark
arrays; nothing in this module opens ``data/results/`` or ``ground_truth.json``.

**The estimator.** For a ground point imaged at column ``x`` and row ``y``::

    k(x, y) = (h_pole(x) / H_POLE) * (h_cam / (y - cy))
            = Z_runner / Z_poleline

``h_pole(x)`` is a 1.4 m pole's *pixel height* interpolated across image column
``x`` from the four measured poles in ``kinovea_calibration.csv``. ``k`` is the
ratio of perpendicular camera distances, which for two parallel lines and a
pencil of rays from one point is exactly the factor by which the gate's true
ground distance scales — so a depth-corrected GT quantity is simply ``GT * k``,
identically for velocity and for stride (they share the same gate distance),
while cadence and GCT (pure frame timing) are untouched.

**Hard constraint, still binding.** Only camera-geometry
signals may enter: pole pixel measurements, camera height/distance, and
landmark *positions*. Never body-segment or anthropometric quantities
(``pixels_per_cm``, ``segments_px``) — those are what the pipeline's own
``create_spatial_calibration`` uses, so correcting GT with them would make any
later pipeline-vs-GT comparison tautological. Nothing in this module reads one.

**Coordinate systems — a known trap.** ``kinovea_calibration.csv``'s
``pole_x_px`` and ``pole_base_y_px`` are Kinovea hover-and-read values, which
are **centre origin** (per the annotation guide
line 40), and ``pole_base_y_px`` is additionally **y-up**. BlazePose's
``landmarks_px`` are top-left origin, y-down. Feeding one into the other
without converting is the exact bug
``validate_anthropometric_scale.py::to_kinovea_centered_x`` was written to fix
on 2026-08-04 and that was later reintroduced: 57.5% of stance-foot samples
fell outside the *raw* pole-column range and were silently clamped flat to
pole D's pixel height instead of being interpolated. Use
:func:`kinovea_x_to_image` / :func:`kinovea_y_to_image` — or
:func:`load_pole_rows`, which applies both — anywhere pole CSV coordinates meet
landmark pixel coordinates.
"""

from __future__ import annotations

import csv
from pathlib import Path
from typing import Any, Dict, List, Optional, Sequence, Tuple

import numpy as np

POLE_LETTERS = ("A", "B", "C", "D")

#: Frame size per calibration resolution. Kinovea's hover-and-read columns and
#: rows are recorded relative to the *image centre* at whatever resolution the
#: annotator was working in, so the offset is resolution-dependent.
RESOLUTION_FRAME: Dict[str, Tuple[int, int]] = {
    "4K": (3840, 2160),
    "1080p": (1920, 1080),
}

FRAME_W, FRAME_H = RESOLUTION_FRAME["4K"]

#: BlazePose landmark indices, left/right ordered.
ANKLE_IDX = [27, 28]
HEEL_IDX = [29, 30]
FOOT_INDEX_IDX = [31, 32]


# ------------------------------------------------------ coordinate systems --


def kinovea_x_to_image(v: float, width: int) -> float:
    """Kinovea centre-origin column -> top-left-origin image column."""
    return width / 2.0 + v


def kinovea_y_to_image(v: float, height: int) -> float:
    """Kinovea centre-origin, **y-up** row -> top-left-origin image row.

    The y-up direction is not assumed: ``c43_pole_parallax_depth_correction.py``
    's ``coordinate_convention()`` tests all four sign/offset combinations
    against three independent physical checks and this is the only admissible
    one.
    """
    return height / 2.0 - v


def read_calibration_rows(path: Path) -> List[Dict[str, Any]]:
    """Every row of a ``kinovea_calibration.csv``-shaped file, as dicts.

    The one CSV-loading helper shared by this module's callers, so the
    diagnosis script and the correction-factor generator read the same file
    the same way.
    """
    with path.open(newline="", encoding="utf-8") as f:
        return [dict(r) for r in csv.DictReader(f)]


def load_pole_rows(
    calibration_rows: Sequence[Dict[str, Any]], resolution: str
) -> Dict[str, Dict[str, float]]:
    """Raw A/B/C/D pole-height rows for one resolution, plus image coords.

    Each returned record carries ``measured_px`` and, where the CSV has them,
    both the raw (centre-origin) and converted (top-left-origin image) forms of
    the pole's column and base row — ``pole_x_raw``/``pole_x_img`` and
    ``pole_base_y_raw``/``pole_base_y_img``. Only the ``_img`` values are safe
    to compare against BlazePose ``landmarks_px``.
    """
    w, h = RESOLUTION_FRAME[resolution]
    out: Dict[str, Dict[str, float]] = {}
    for r in calibration_rows:
        lab = str(r.get("label") or "")
        if r.get("resolution") != resolution or lab not in POLE_LETTERS:
            continue
        rec: Dict[str, float] = {"measured_px": float(r["measured_px"])}
        if r.get("pole_x_px"):
            rec["pole_x_raw"] = float(r["pole_x_px"])
            rec["pole_x_img"] = kinovea_x_to_image(rec["pole_x_raw"], w)
        if r.get("pole_base_y_px"):
            rec["pole_base_y_raw"] = float(r["pole_base_y_px"])
            rec["pole_base_y_img"] = kinovea_y_to_image(rec["pole_base_y_raw"], h)
        out[lab] = rec
    return out


# ------------------------------------------------------------- the probe --


class DepthProbe:
    """k(x, y) = Z_runner / Z_poleline, from pole pixel measurements alone.

    Two horizon models, both camera-geometry-only:

    ``constant``  -- one scalar ``cy`` for the whole frame (the ``cy`` argument
      to ``__call__``). This is the Step 1b headline model.
    ``column``    -- ``cy(x)`` interpolated across image column from each
      pole's own base row: ``cy(x) = ybase(x) - pole_scale(x) * cam_h``. Zero
      residual at every pole by construction, so it absorbs camera roll and
      any residual trend in the four per-pole ``cy`` estimates. Reported as a
      robustness check, not the headline.

    ``pole_x``/``pole_base_y`` must already be in **image** (top-left origin)
    coordinates -- see :func:`load_pole_rows`.
    """

    def __init__(
        self,
        pole_x: Dict[str, float],
        pole_h: Dict[str, float],
        pole_height_m: float,
        cam_height_m: float,
        pole_base_y: Dict[str, float] | None = None,
    ) -> None:
        labels = sorted(pole_x, key=lambda k: pole_x[k])
        self.labels = labels
        self.xp = np.array([pole_x[k] for k in labels], dtype=float)
        self.hp = np.array([pole_h[k] for k in labels], dtype=float)
        self.yb = (
            np.array([pole_base_y[k] for k in labels], dtype=float)
            if pole_base_y and all(k in pole_base_y for k in labels)
            else None
        )
        self.pole_height_m = pole_height_m
        self.cam_height_m = cam_height_m

    def pole_scale(self, x: np.ndarray) -> np.ndarray:
        """Vertical px/m of the pole line at image column ``x``."""
        return np.asarray(np.interp(x, self.xp, self.hp) / self.pole_height_m)

    def cy_column(self, x: np.ndarray) -> np.ndarray:
        """Per-column horizon row implied by the pole bases (``column`` model)."""
        if self.yb is None:
            raise ValueError("pole base rows not available")
        return np.asarray(
            np.interp(x, self.xp, self.yb) - self.pole_scale(x) * self.cam_height_m
        )

    def __call__(self, x: np.ndarray, y: np.ndarray, cy: float) -> np.ndarray:
        return np.asarray(
            self.pole_scale(x) * (self.cam_height_m / (y - cy)), dtype=float
        )

    def call_column(self, x: np.ndarray, y: np.ndarray) -> np.ndarray:
        return np.asarray(
            self.pole_scale(x) * (self.cam_height_m / (y - self.cy_column(x))),
            dtype=float,
        )


def probe_k(
    probe: DepthProbe, x: np.ndarray, y: np.ndarray, cy: float, cy_model: str
) -> np.ndarray:
    """Evaluate the probe under the selected horizon model."""
    if cy_model == "column":
        return probe.call_column(x, y)
    return probe(x, y, cy)


# ------------------------------------------------------- landmark probes --


def stance_foot_pixels(
    landmarks: np.ndarray,
    contact_frames: Sequence[float],
    frame_w: int = FRAME_W,
    frame_h: int = FRAME_H,
) -> Tuple[np.ndarray, np.ndarray, int]:
    """Image column/row of the stance ankle at each contact (the depth probe).

    The stance ankle is the lower (larger y) of the two at the contact frame.
    Uses no segment length, so it is independent of anthropometry. Contacts
    whose ankle falls outside the frame are dropped (they cannot carry a valid
    ground projection) and counted in the third return value.
    """
    xs: List[float] = []
    ys: List[float] = []
    n_out = 0
    for f in contact_frames:
        i = int(round(f))
        if i < 0 or i >= len(landmarks):
            continue
        pair = landmarks[i, ANKLE_IDX, :]
        if np.all(np.isnan(pair[:, 1])):
            continue
        j = int(np.nanargmax(pair[:, 1]))
        x, y = float(pair[j, 0]), float(pair[j, 1])
        if np.isnan(x) or np.isnan(y):
            continue
        if not (0.0 <= x <= frame_w and 0.0 <= y <= frame_h):
            n_out += 1
            continue
        xs.append(x)
        ys.append(y)
    return np.array(xs), np.array(ys), n_out


def stance_foot_pixels_lowest(
    landmarks: np.ndarray,
    contact_frames: Sequence[float],
    frame_w: int = FRAME_W,
    frame_h: int = FRAME_H,
) -> Tuple[np.ndarray, np.ndarray, int]:
    """Same as :func:`stance_foot_pixels` but takes the LOWEST foot landmark.

    Robustness variant for the strike-pattern confound, and the one
    validated as least-confounded. The ankle landmark sits higher above the
    ground for a forefoot contact (heel off the floor) than for a heel contact,
    which shows up as a spurious depth difference. Taking the lowest of
    {ankle, heel, foot_index} on the stance side tracks the true ground-contact
    point far more closely regardless of strike, and still uses only landmark
    *positions* -- no segment lengths, so the anthropometry ban is untouched.
    """
    xs: List[float] = []
    ys: List[float] = []
    n_out = 0
    for f in contact_frames:
        i = int(round(f))
        if i < 0 or i >= len(landmarks):
            continue
        pair = landmarks[i, ANKLE_IDX, :]
        if np.all(np.isnan(pair[:, 1])):
            continue
        side = int(np.nanargmax(pair[:, 1]))  # 0 = left, 1 = right
        cand = landmarks[i, [ANKLE_IDX[side], HEEL_IDX[side], FOOT_INDEX_IDX[side]], :]
        if np.all(np.isnan(cand[:, 1])):
            continue
        j = int(np.nanargmax(cand[:, 1]))
        x, y = float(cand[j, 0]), float(cand[j, 1])
        if np.isnan(x) or np.isnan(y):
            continue
        if not (0.0 <= x <= frame_w and 0.0 <= y <= frame_h):
            n_out += 1
            continue
        xs.append(x)
        ys.append(y)
    return np.array(xs), np.array(ys), n_out


# ------------------------------------------------- cy and the common-mode --


def resolve_cy(
    calibration_rows: Sequence[Dict[str, Any]],
    cam_h: float,
    pole_height_m: float,
) -> Dict[str, Any]:
    """Solve ``cy = y_base - pole_scale * cam_h`` at each pole.

    A pole's base is a ground-plane point *on the pole line*, so the probe's own
    definition forces ``k = 1`` there. Inverting gives one ``cy`` estimate per
    pole per resolution. 1080p estimates are reported rescaled to 4K pixels.
    Needs ``M8``'s ``pole_base_y_px`` column; poles without it are skipped.
    """
    per: List[Dict[str, Any]] = []
    for res in ("4K", "1080p"):
        poles = load_pole_rows(calibration_rows, res)
        scale_to_4k = FRAME_W / RESOLUTION_FRAME[res][0]
        for lab in POLE_LETTERS:
            rec = poles.get(lab, {})
            if "pole_base_y_img" not in rec:
                continue
            s = rec["measured_px"] / pole_height_m
            cy = rec["pole_base_y_img"] - s * cam_h
            per.append(
                {
                    "pole": lab,
                    "resolution": res,
                    "pole_scale_px_per_m": float(s),
                    "pole_base_row_px": float(rec["pole_base_y_img"]),
                    "image_column_px": float(rec["pole_x_img"]),
                    "cy_px": float(cy),
                    "cy_px_4k_equivalent": float(cy * scale_to_4k),
                }
            )
    cy4k = np.array([p["cy_px_4k_equivalent"] for p in per])
    by_pole: Dict[str, List[float]] = {}
    for p in per:
        by_pole.setdefault(p["pole"], []).append(p["cy_px_4k_equivalent"])
    cross_res = {k: float(abs(v[0] - v[1])) for k, v in by_pole.items() if len(v) == 2}
    pole_mean = {k: float(np.mean(v)) for k, v in by_pole.items()}

    # camera roll: cy trends linearly with image column across the four poles
    cols = np.array(
        [
            float(
                np.mean(
                    [
                        p["image_column_px"]
                        * (FRAME_W / RESOLUTION_FRAME[p["resolution"]][0])
                        for p in per
                        if p["pole"] == lab
                    ]
                )
            )
            for lab in POLE_LETTERS
        ]
    )
    means = np.array([pole_mean[lab] for lab in POLE_LETTERS])
    roll_slope, roll_int = np.polyfit(cols, means, 1)
    roll_resid = means - (roll_slope * cols + roll_int)

    return {
        "estimates": per,
        "n_estimates": len(per),
        "cy_px_4k_median": float(np.median(cy4k)),
        "cy_px_4k_mean": float(np.mean(cy4k)),
        "cy_px_4k_sd": float(np.std(cy4k, ddof=1)),
        "cy_px_4k_range": [float(cy4k.min()), float(cy4k.max())],
        "frame_centre_row_px": FRAME_H / 2.0,
        "offset_from_frame_centre_px": float(np.median(cy4k) - FRAME_H / 2.0),
        "cross_resolution_abs_diff_px_4k": cross_res,
        "max_cross_resolution_abs_diff_px_4k": float(max(cross_res.values())),
        "per_pole_mean_px_4k": pole_mean,
        "cross_pole_spread_px_4k": float(
            max(pole_mean.values()) - min(pole_mean.values())
        ),
        "camera_roll_fit": {
            "slope_px_per_px": float(roll_slope),
            "intercept_px": float(roll_int),
            "implied_roll_deg": float(np.degrees(np.arctan(roll_slope))),
            "max_abs_residual_px": float(np.abs(roll_resid).max()),
            "note": (
                "The four per-pole cy estimates trend monotonically with image "
                "column. A single tilted horizon line (camera roll) explains that "
                "trend to well under a pixel, which is why the 'column' cy model "
                "exists as a robustness check on the constant one."
            ),
        },
        "finding": (
            "cy is now measured, not assumed. Cross-resolution agreement is the "
            "strong check: the 4K and 1080p reads are independent hover-and-reads "
            "of the same scene (measured_px differs by ~0.3% between them, so they "
            "are not one derived from the other), and after rescaling they agree "
            "to about a pixel at 4K on every pole. Cross-pole spread is larger and "
            "is explained by camera roll, not by read error."
        ),
    }


def resolve_cohort_offset_m(ks: Sequence[float] | np.ndarray, cam_h: float) -> float:
    """The common-mode landmark-height offset ``a``, in metres.

    **Diagnostic, not the production offset.** This *solves* ``a`` from the data
    by asserting the cohort median clip is on the pole line. A later step briefly
    used it that way (one cohort-wide ~1.35 cm offset for all 45 clips) and no
    longer does: ``scripts/analysis/c43_compute_depth_correction_factors.py``
    now looks up each runner's own measured shoe stack height on the side of the
    shoe that runner actually strikes with (``RUNNER_SHOE`` there), which is a
    real measurement rather than a solved anchor and does not assume every
    runner's shoe is equally thick. What this function is still *good* for is
    falsification -- see the note about ~1.5 cm vs ~9.5 cm below -- and that is
    how ``c43_pole_parallax_depth_correction.py`` uses it throughout.

    ``k`` as measured is ``(Z_runner / Z_poleline) * h / (h - a)``, where ``a``
    is the probed landmark's height above the true ground-contact point. Solving
    ``a`` so the *cohort median* clip sits exactly on the pole line -- which is
    what the S1 protocol asserts, poles set on the runners' lane -- gives::

        a = cam_h * (1 - 1 / median(k))

    ``a`` is a pure common-mode multiplier: it moves the cohort anchor and never
    the ranking or the *relative* size of per-clip deviations. With the
    lowest-foot landmark it comes out around 1.5 cm (against ~9.5 cm for the
    ankle landmark), which is itself a falsification check on the geometric
    model -- a physically absurd value would sink it.
    """
    return float(cam_h * (1.0 - 1.0 / float(np.median(np.asarray(ks, dtype=float)))))


def apply_offset(
    ks: Sequence[float] | np.ndarray,
    cam_h: float,
    offset_m: float | Sequence[float] | np.ndarray,
) -> np.ndarray:
    """Divide the landmark-height offset back out of ``k``.

    ``k_true = k * (cam_h - a) / cam_h``, the inverse of the model
    :func:`resolve_cohort_offset_m` documents.

    ``offset_m`` may be a scalar (one ``a`` for every element of ``ks``) or an
    array broadcastable against ``ks`` (one ``a`` per element) -- the latter is
    what the generator uses, passing each clip's own runner's shoe
    stack height rather than a single cohort constant.
    """
    a = np.asarray(offset_m, dtype=float)
    return np.asarray(np.asarray(ks, dtype=float) * (cam_h - a) / cam_h, dtype=float)


def pole_height_m_from_calibration(path: Path) -> Optional[float]:
    """The physical pole height in metres, from a ``calibration.csv``.

    Reads the first row's ``pole_height_cm``; ``None`` if the file has no rows
    or no such column.
    """
    with path.open(encoding="utf-8") as f:
        row = next(csv.DictReader(f), None)
    if not row or not row.get("pole_height_cm"):
        return None
    return float(row["pole_height_cm"]) / 100.0
