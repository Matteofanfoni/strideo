"""DORMANT — v1.1 experiment, superseded by v1.2 motion_filter.

This module implements M7 (composite score) and M9 (velocity filter) from
01c §10.6. Both were tested on PV_800m_1_Victory in April 2026 and both
underperformed the v1.0 baseline because MediaPipe's detector — not its
tracker — is the bottleneck on fast subjects at 4K. IMAGE mode with
num_poses=4 returned only 1.17 candidates/frame on average (detector
threshold too conservative even at 0.2), so there was nothing for a
selector to choose between. See 01c §10.6 table for the detailed failure
numbers.

The v1.2 default pipeline does *not* import this module. It remains in
the tree for future experiments — e.g. if we ever get MediaPipe to emit
multiple candidates reliably, the longest-path DAG selector below is a
reasonable starting point.

Per-frame pose-candidate selection for Phase 2 ghost-lock mitigation.

MediaPipe with num_poses > 1 returns up to N candidate people per frame.
When the scene contains background structures that can be misdetected as
humans (e.g. construction fences, banners), candidate 0 is not always
the runner. These selectors pick the runner by exploiting the one
signal the ghost cannot fake — **horizontal motion**.

- ``select_by_velocity`` (M9): pick the candidate whose hip-x velocity
  sits inside the pace's expected running band.
- ``select_by_composite`` (M7): weighted score over velocity match,
  mean hip visibility, and bounding-box area.

Both selectors expose the same signature and return a ``SelectionResult``
with a single-candidate landmark tensor shaped for the v1.0 downstream
pipeline (metrics, ground-contact detection, annotation rendering).

Configuration is loaded from ``configs/pose_selection.yaml`` so weights
and velocity bands can be tuned without touching code. See
``docs/technical/01c_pre_validation_session_report.md`` §10.6 for the
motivation and the M7/M9 proposal.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import numpy as np
import yaml  # type: ignore[import-untyped]

from src.preprocessing.pose_estimator import LandmarkIndex

DEFAULT_CONFIG_PATH = (
    Path(__file__).resolve().parents[2] / "configs" / "pose_selection.yaml"
)


@dataclass
class SelectionResult:
    """Output of a pose-selection pass.

    ``landmarks`` / ``visibilities`` / ``world_landmarks`` match the v1.0
    single-candidate shape so downstream metrics code is unchanged.
    Frames where no candidate passes selection carry NaN landmarks and
    zero visibilities, which the existing quality metrics already treat
    as a missed detection.
    """

    landmarks: np.ndarray  # (T, 33, 2)
    visibilities: np.ndarray  # (T, 33)
    world_landmarks: Optional[np.ndarray]  # (T, 33, 3) or None
    log: List[Dict[str, Any]] = field(default_factory=list)
    method: str = ""

    @property
    def n_accepted(self) -> int:
        return sum(1 for entry in self.log if entry["accepted"])


def load_config(path: Optional[Path] = None) -> Dict[str, Any]:
    """Load the YAML config; falls back to the package default."""
    cfg_path = Path(path) if path else DEFAULT_CONFIG_PATH
    with cfg_path.open(encoding="utf-8") as fh:
        data: Dict[str, Any] = yaml.safe_load(fh)
        return data


def _mean_hip_visibility(multi_visibilities: np.ndarray) -> np.ndarray:
    """Per-frame per-candidate mean of LEFT_HIP and RIGHT_HIP visibility."""
    vis_l = multi_visibilities[:, :, LandmarkIndex.LEFT_HIP]
    vis_r = multi_visibilities[:, :, LandmarkIndex.RIGHT_HIP]
    out: np.ndarray = np.mean(np.stack([vis_l, vis_r]), axis=0)
    return out


def _hip_x(multi_landmarks: np.ndarray) -> np.ndarray:
    """Per-frame per-candidate mean hip X position (pixels)."""
    hip_l = multi_landmarks[:, :, LandmarkIndex.LEFT_HIP, 0]
    hip_r = multi_landmarks[:, :, LandmarkIndex.RIGHT_HIP, 0]
    with np.errstate(invalid="ignore"):
        out: np.ndarray = np.nanmean(np.stack([hip_l, hip_r]), axis=0)
        return out


def _bbox_area(multi_landmarks: np.ndarray) -> np.ndarray:
    """Per-frame per-candidate bounding-box area from all 33 landmarks."""
    with np.errstate(invalid="ignore"):
        x_min = np.nanmin(multi_landmarks[:, :, :, 0], axis=2)
        x_max = np.nanmax(multi_landmarks[:, :, :, 0], axis=2)
        y_min = np.nanmin(multi_landmarks[:, :, :, 1], axis=2)
        y_max = np.nanmax(multi_landmarks[:, :, :, 1], axis=2)
    width = np.where(np.isnan(x_max - x_min), 0.0, x_max - x_min)
    height = np.where(np.isnan(y_max - y_min), 0.0, y_max - y_min)
    return width * height


def _pace_band_m_s(pace_level: str, config: Dict[str, Any]) -> Tuple[float, float]:
    bands = config["velocity_bands_ms"]
    if pace_level in bands:
        v_min, v_max = bands[pace_level]
    else:
        v_min, v_max = bands["unknown"]
    return float(v_min), float(v_max)


def _empty_selection(
    T: int,
    has_world: bool,
    method: str,
) -> SelectionResult:
    return SelectionResult(
        landmarks=np.full((T, 33, 2), np.nan),
        visibilities=np.zeros((T, 33)),
        world_landmarks=np.full((T, 33, 3), np.nan) if has_world else None,
        log=[],
        method=method,
    )


def _log_entry(
    t: int,
    *,
    accepted: bool,
    chosen_idx: Optional[int],
    reason: str,
    n_eligible: int,
    mean_hip_vis: float,
    velocity_px_per_frame: float,
    bbox_area: float,
    score: Optional[float] = None,
) -> Dict[str, Any]:
    return {
        "frame": int(t),
        "accepted": bool(accepted),
        "chosen_idx": int(chosen_idx) if chosen_idx is not None else -1,
        "reason": reason,
        "n_eligible": int(n_eligible),
        "mean_hip_visibility": float(mean_hip_vis),
        "velocity_px_per_frame": float(velocity_px_per_frame),
        "bbox_area": float(bbox_area),
        "score": float(score) if score is not None else float("nan"),
    }


def _assign(
    landmarks: np.ndarray,
    visibilities: np.ndarray,
    world_landmarks: Optional[np.ndarray],
    t: int,
    best_idx: int,
    multi_landmarks: np.ndarray,
    multi_visibilities: np.ndarray,
    multi_world_landmarks: Optional[np.ndarray],
) -> None:
    landmarks[t] = multi_landmarks[t, best_idx]
    visibilities[t] = multi_visibilities[t, best_idx]
    if world_landmarks is not None and multi_world_landmarks is not None:
        world_landmarks[t] = multi_world_landmarks[t, best_idx]


def _find_longest_track(
    hip_x: np.ndarray,
    mean_hip_vis: np.ndarray,
    v_min_pxf: float,
    v_max_pxf: float,
    min_vis: float,
    max_gap: int,
    edge_cost: Optional[Any] = None,
) -> Tuple[Dict[int, int], np.ndarray]:
    """DAG longest-path through the candidate graph.

    Nodes: (t, c) for each eligible candidate (visible enough, has hip_x).
    Edges: (t, c) -> (t+k, c') for 1 <= k <= max_gap whose implied
    horizontal hip velocity |Δhip_x| / k is inside [v_min_pxf, v_max_pxf].

    Returns the chain (dict frame -> candidate index) and the per-node
    chain length matrix. When ``edge_cost`` is provided, it is evaluated
    on each edge and added to the score; the function then maximises
    (chain_length + cumulative_edge_score) instead of chain length alone.
    Used by select_by_composite.
    """
    T, N = hip_x.shape
    eligible = (mean_hip_vis >= min_vis) & np.isfinite(hip_x)

    dp = np.zeros((T, N), dtype=np.float64)
    parent = np.full((T, N, 2), -1, dtype=np.int32)

    for t in range(T):
        for c in range(N):
            if not eligible[t, c]:
                continue
            dp[t, c] = 1.0
            lo = max(0, t - max_gap)
            for prev_t in range(lo, t):
                for prev_c in range(N):
                    if not eligible[prev_t, prev_c]:
                        continue
                    gap = t - prev_t
                    vel = abs(hip_x[t, c] - hip_x[prev_t, prev_c]) / gap
                    if not (v_min_pxf <= vel <= v_max_pxf):
                        continue
                    gain = 1.0
                    if edge_cost is not None:
                        gain = edge_cost(prev_t, prev_c, t, c, vel)
                    cand = dp[prev_t, prev_c] + gain
                    if cand > dp[t, c]:
                        dp[t, c] = cand
                        parent[t, c] = (prev_t, prev_c)

    if dp.max() <= 1.0:
        # No edges found — single-node best case.
        return {}, dp

    flat = int(np.argmax(dp))
    t_end, c_end = divmod(flat, N)
    chain: Dict[int, int] = {}
    cur_t, cur_c = t_end, c_end
    while cur_t >= 0:
        chain[cur_t] = cur_c
        prev_t, prev_c = parent[cur_t, cur_c]
        if prev_t < 0:
            break
        cur_t, cur_c = int(prev_t), int(prev_c)
    return chain, dp


def select_by_velocity(
    multi_landmarks: np.ndarray,
    multi_visibilities: np.ndarray,
    fps: float,
    pixels_per_cm: float,
    pace_level: str,
    multi_world_landmarks: Optional[np.ndarray] = None,
    config: Optional[Dict[str, Any]] = None,
) -> SelectionResult:
    """M9 — longest candidate chain whose every transition is in-band.

    Algorithm:
      1. For every frame, drop candidates with mean hip visibility below
         ``min_hip_visibility``.
      2. Build a DAG: directed edge from (t, c) to (t+k, c') for
         k ∈ [1, velocity_fallback_max_gap] whose |Δhip_x| / k falls
         inside the pace-level velocity band.
      3. Find the longest path — that is the runner's track.
      4. Emit selected landmarks for frames on the track; NaN elsewhere.

    This treats per-frame candidate identity correctly in IMAGE mode (where
    candidate 0 in frame t is not necessarily the same person as candidate 0
    in frame t+1). It also avoids the greedy "bootstrap on highest
    visibility" trap, which otherwise locks on the ghost in frame 0.
    """
    if config is None:
        config = load_config()

    T, _ = multi_landmarks.shape[:2]
    has_world = multi_world_landmarks is not None
    result = _empty_selection(T, has_world, method="velocity")
    if T == 0:
        return result

    mean_hip_vis = _mean_hip_visibility(multi_visibilities)
    hip_x = _hip_x(multi_landmarks)
    bbox_areas = _bbox_area(multi_landmarks)

    v_min_ms, v_max_ms = _pace_band_m_s(pace_level, config)
    ms_to_pxf = (pixels_per_cm * 100.0) / fps
    v_min_pxf = v_min_ms * ms_to_pxf
    v_max_pxf = v_max_ms * ms_to_pxf
    min_vis = float(config["min_hip_visibility"])
    fallback_gap = int(config["velocity_fallback_max_gap"])

    chain, dp = _find_longest_track(
        hip_x, mean_hip_vis, v_min_pxf, v_max_pxf, min_vis, fallback_gap
    )

    if not chain:
        # No track found — fall back to all-NaN output (worst-case behaviour).
        for t in range(T):
            result.log.append(
                _log_entry(
                    t,
                    accepted=False,
                    chosen_idx=None,
                    reason="no_valid_track",
                    n_eligible=int(np.sum(mean_hip_vis[t] >= min_vis)),
                    mean_hip_vis=float(np.max(mean_hip_vis[t])),
                    velocity_px_per_frame=float("nan"),
                    bbox_area=float(np.max(bbox_areas[t])),
                )
            )
        return result

    # Populate selection + log for every frame.
    prev_t = None
    prev_hip = np.nan
    for t in range(T):
        if t in chain:
            c = chain[t]
            _assign(
                result.landmarks,
                result.visibilities,
                result.world_landmarks,
                t,
                c,
                multi_landmarks,
                multi_visibilities,
                multi_world_landmarks,
            )
            if prev_t is None:
                vel = float("nan")
                reason = "track_start"
            else:
                vel = abs(hip_x[t, c] - prev_hip) / max(1, t - prev_t)
                reason = "on_track"
            result.log.append(
                _log_entry(
                    t,
                    accepted=True,
                    chosen_idx=int(c),
                    reason=reason,
                    n_eligible=int(np.sum(mean_hip_vis[t] >= min_vis)),
                    mean_hip_vis=float(mean_hip_vis[t, c]),
                    velocity_px_per_frame=vel,
                    bbox_area=float(bbox_areas[t, c]),
                )
            )
            prev_t = t
            prev_hip = float(hip_x[t, c])
        else:
            n_elig = int(np.sum(mean_hip_vis[t] >= min_vis))
            max_vis = float(np.max(mean_hip_vis[t])) if n_elig else 0.0
            max_area = float(np.max(bbox_areas[t])) if n_elig else 0.0
            result.log.append(
                _log_entry(
                    t,
                    accepted=False,
                    chosen_idx=None,
                    reason="off_track",
                    n_eligible=n_elig,
                    mean_hip_vis=max_vis,
                    velocity_px_per_frame=float("nan"),
                    bbox_area=max_area,
                )
            )

    return result


def select_by_composite(
    multi_landmarks: np.ndarray,
    multi_visibilities: np.ndarray,
    fps: float,
    pixels_per_cm: float,
    pace_level: str,
    multi_world_landmarks: Optional[np.ndarray] = None,
    config: Optional[Dict[str, Any]] = None,
) -> SelectionResult:
    """M7 — weighted composite score (velocity + visibility + area).

    Uses the same longest-path DAG skeleton as select_by_velocity, but
    edges are scored by the composite (velocity match + destination
    visibility + destination area). Chains with higher average score win
    even when they are slightly shorter.
    """
    if config is None:
        config = load_config()

    T, _ = multi_landmarks.shape[:2]
    has_world = multi_world_landmarks is not None
    result = _empty_selection(T, has_world, method="composite")
    if T == 0:
        return result

    mean_hip_vis = _mean_hip_visibility(multi_visibilities)
    hip_x = _hip_x(multi_landmarks)
    bbox_areas = _bbox_area(multi_landmarks)

    v_min_ms, v_max_ms = _pace_band_m_s(pace_level, config)
    ms_to_pxf = (pixels_per_cm * 100.0) / fps
    v_min_pxf = v_min_ms * ms_to_pxf
    v_max_pxf = v_max_ms * ms_to_pxf
    v_mid_pxf = (v_min_pxf + v_max_pxf) / 2.0
    band_half = max(1e-6, (v_max_pxf - v_min_pxf) / 2.0)

    min_vis = float(config["min_hip_visibility"])
    fallback_gap = int(config["velocity_fallback_max_gap"])
    weights = config["composite_weights"]
    w_v = float(weights["velocity"])
    w_vis = float(weights["visibility"])
    w_a = float(weights["area"])

    # Normalise area against the global max so the score stays in [0, 1].
    max_area = float(np.nanmax(bbox_areas)) if bbox_areas.size else 1.0
    max_area = max(max_area, 1e-6)

    def edge_cost(prev_t: int, prev_c: int, t: int, c: int, vel: float) -> float:
        v_score = float(np.clip(1.0 - abs(vel - v_mid_pxf) / band_half, 0.0, 1.0))
        vis_score = float(mean_hip_vis[t, c])
        area_score = float(bbox_areas[t, c] / max_area)
        return w_v * v_score + w_vis * vis_score + w_a * area_score

    chain, dp = _find_longest_track(
        hip_x,
        mean_hip_vis,
        v_min_pxf,
        v_max_pxf,
        min_vis,
        fallback_gap,
        edge_cost=edge_cost,
    )

    if not chain:
        for t in range(T):
            result.log.append(
                _log_entry(
                    t,
                    accepted=False,
                    chosen_idx=None,
                    reason="no_valid_track",
                    n_eligible=int(np.sum(mean_hip_vis[t] >= min_vis)),
                    mean_hip_vis=float(np.max(mean_hip_vis[t])),
                    velocity_px_per_frame=float("nan"),
                    bbox_area=float(np.max(bbox_areas[t])),
                    score=0.0,
                )
            )
        return result

    prev_t = None
    prev_hip = np.nan
    for t in range(T):
        if t in chain:
            c = chain[t]
            _assign(
                result.landmarks,
                result.visibilities,
                result.world_landmarks,
                t,
                c,
                multi_landmarks,
                multi_visibilities,
                multi_world_landmarks,
            )
            if prev_t is None:
                vel = float("nan")
                score = float(dp[t, c])
                reason = "track_start"
            else:
                vel = abs(hip_x[t, c] - prev_hip) / max(1, t - prev_t)
                score = float(dp[t, c] - dp[prev_t, chain[prev_t]])
                reason = "on_track"
            result.log.append(
                _log_entry(
                    t,
                    accepted=True,
                    chosen_idx=int(c),
                    reason=reason,
                    n_eligible=int(np.sum(mean_hip_vis[t] >= min_vis)),
                    mean_hip_vis=float(mean_hip_vis[t, c]),
                    velocity_px_per_frame=vel,
                    bbox_area=float(bbox_areas[t, c]),
                    score=score,
                )
            )
            prev_t = t
            prev_hip = float(hip_x[t, c])
        else:
            n_elig = int(np.sum(mean_hip_vis[t] >= min_vis))
            max_vis = float(np.max(mean_hip_vis[t])) if n_elig else 0.0
            max_a = float(np.max(bbox_areas[t])) if n_elig else 0.0
            result.log.append(
                _log_entry(
                    t,
                    accepted=False,
                    chosen_idx=None,
                    reason="off_track",
                    n_eligible=n_elig,
                    mean_hip_vis=max_vis,
                    velocity_px_per_frame=float("nan"),
                    bbox_area=max_a,
                    score=0.0,
                )
            )

    return result


SELECTORS = {
    "velocity": select_by_velocity,
    "composite": select_by_composite,
}
