import numpy as np
from scipy.ndimage import gaussian_filter1d
from scipy.signal import savgol_filter
from dataclasses import dataclass
from typing import List, Optional, Tuple, Dict

from src.preprocessing.pose_estimator import LandmarkIndex


@dataclass
class GroundContact:
    """
    Ground contact event with coarse and refined timing.

    The two-pass detection provides:
    - Coarse timing (frame): Integer frame from leg compression method
    - Refined timing (contact_frame, toe_off_frame): Sub-frame float from toe method
    """

    # Coarse detection (leg compression method)
    frame: int  # Integer frame of detected contact
    leg: str  # 'L' or 'R'

    # Refined detection (toe velocity method, sub-frame precision)
    contact_frame: Optional[float] = None  # Precise touchdown (sub-frame)
    toe_off_frame: Optional[float] = None  # Precise toe-off (sub-frame)

    # Derived metrics (computed during detection)
    gct_ms: Optional[float] = None  # Ground contact time in ms

    # Quality metadata
    detection_method: str = "coarse"  # 'coarse' or 'refined'
    confidence: float = 0.0  # Detection confidence [0-1]

    # Phase 6c (Fix B3): per-contact strike pattern detected at touchdown.
    # 'heel' / 'forefoot' / 'unknown' / None. None when toe_off_method is not
    # 'per_strike_pattern' (i.e., legacy v1.5–v1.7 paths that don't classify).
    strike_pattern: Optional[str] = None


@dataclass
class ContactDetectionResult:
    """Container for ground contact detection results."""

    contacts: List[GroundContact]  # Detected contacts
    summary: Dict  # Detection statistics


# RTMPose COCO-WholeBody foot keypoint indices (v1.18 Option-B port).
# Used when rtm_landmarks are passed to refine_contacts and its callees;
# all non-foot signals (ankles, hips) remain on BlazePose.
_RTM_FOOT = {
    "L": {"big_toe": 17, "small_toe": 18, "heel": 19},
    "R": {"big_toe": 20, "small_toe": 21, "heel": 22},
}

# RTMPose COCO-body keypoint indices used by the coarse pass (v1.20).
# Standard COCO body keypoints (indices 0–16); always present and high-
# confidence because hips and ankles are large, easily-visible segments.
_RTM_BODY = {
    "L": {"hip": 11, "ankle": 15},
    "R": {"hip": 12, "ankle": 16},
}


def _interpolate_nans(arr: np.ndarray) -> np.ndarray:
    """Interpolate NaN values using linear interpolation."""
    arr = arr.copy()
    nans = np.isnan(arr)
    if nans.all() or not nans.any():
        return arr
    valid = np.where(~nans)[0]
    arr[nans] = np.interp(np.where(nans)[0], valid, arr[valid])
    return arr


def _get_leg_compression(
    landmarks: np.ndarray, hip_idx: int, ankle_idx: int, sigma: float
) -> np.ndarray:
    """
    Extract and smooth leg compression signal.

    Compression = ankle_y - hip_y (positive when leg extended)
    """
    hip_y = _interpolate_nans(landmarks[:, hip_idx, 1].copy())
    ankle_y = _interpolate_nans(landmarks[:, ankle_idx, 1].copy())

    compression = ankle_y - hip_y
    result: np.ndarray = gaussian_filter1d(compression, sigma=sigma)
    return result


def detect_contacts_coarse(
    landmarks: np.ndarray,
    visibilities: np.ndarray,
    fps: float,
    resolution_height: int = 2160,
    min_interval_ms: float = 100.0,
    ankle_horiz_vel_gate_frac: float = 0.0,
    rtm_landmarks: Optional[np.ndarray] = None,
    rtm_scores: Optional[np.ndarray] = None,
) -> Tuple[List[GroundContact], Dict]:
    """
    COARSE PASS: Detect ground contacts using leg compression velocity.

    The leg compression signal (ankle_y - hip_y) is immune to camera shake
    because both landmarks translate together during shake events.

    Args:
        landmarks: (T, 33, 2) pose landmarks in pixels
        visibilities: (T, 33) visibility scores
        fps: Video frame rate
        resolution_height: Video height for adaptive thresholds
        min_interval_ms: Minimum time between same-leg contacts
        ankle_horiz_vel_gate_frac: Horizontal-velocity gate — candidate is
            rejected if the smoothed ankle x-velocity magnitude exceeds this
            fraction of resolution_height (px/frame) at the detection frame.
            A high horizontal velocity indicates the foot is still swinging
            forward and has not yet contacted the ground (heel-strike false
            positive from dorsiflexion pre-contact reversal). Default 0.007
            (~15 px/frame at 4K). Set to 0 to disable.
        rtm_landmarks: (T, 133, 2) RTMPose COCO-WholeBody landmarks. When
            provided, the hip and ankle y-positions for the compression signal
            are taken from RTMPose body indices (_RTM_BODY) instead of
            BlazePose. Scores in [1, 8]; zero means no detection (v1.20).
        rtm_scores: (T, 133) RTMPose detection scores. Required when
            rtm_landmarks is provided.

    Returns:
        contacts: List of GroundContact with coarse timing
        diagnostics: Detection statistics per leg
    """
    T = len(landmarks)
    min_interval_frames = int(min_interval_ms / 1000.0 * fps)
    # Scale threshold by fps: at lower fps the ankle travels more px/frame for
    # the same physical speed, so the threshold must be proportionally larger.
    horiz_vel_threshold_px = (
        ankle_horiz_vel_gate_frac * resolution_height * (60.0 / fps)
    )

    # Landmark indices (BlazePose fallback)
    HIPS = {"L": LandmarkIndex.LEFT_HIP, "R": LandmarkIndex.RIGHT_HIP}
    ANKLES = {"L": LandmarkIndex.LEFT_ANKLE, "R": LandmarkIndex.RIGHT_ANKLE}

    # Adaptive threshold based on resolution
    # At 4K (2160p): ~2.16 px/frame; at 1080p: ~1.08 px/frame
    stance_threshold = resolution_height * 0.001

    all_contacts = []
    diagnostics = {}

    use_rtm = rtm_landmarks is not None and rtm_scores is not None

    for leg in ["L", "R"]:
        if use_rtm:
            assert rtm_landmarks is not None and rtm_scores is not None
            rtm_hip_idx = _RTM_BODY[leg]["hip"]
            rtm_ankle_idx = _RTM_BODY[leg]["ankle"]

            # Build hip/ankle y-signals: NaN-fill score=0 frames, then interp.
            # RTMPose scores are in [1, 8]; 0 indicates no detection.
            h_y = rtm_landmarks[:, rtm_hip_idx, 1].astype(float).copy()
            a_y = rtm_landmarks[:, rtm_ankle_idx, 1].astype(float).copy()
            h_y[rtm_scores[:, rtm_hip_idx] <= 0] = np.nan
            a_y[rtm_scores[:, rtm_ankle_idx] <= 0] = np.nan
            h_y = _interpolate_nans(h_y)
            a_y = _interpolate_nans(a_y)

            # Binary valid-frame mask (score > 0) used for confidence and sigma.
            hip_vis = (rtm_scores[:, rtm_hip_idx] > 0).astype(float)
            ankle_vis = (rtm_scores[:, rtm_ankle_idx] > 0).astype(float)
            mean_hip_vis = float(np.nanmean(hip_vis))
            mean_ankle_vis = float(np.nanmean(ankle_vis))

            sigma = 2.0 if mean_hip_vis > 0.5 else 3.0
            compression = gaussian_filter1d(a_y - h_y, sigma=sigma)
            compression_velocity = np.gradient(compression)

            # Ankle x-velocity for the horizontal gate (RTMPose ankle x).
            a_x = rtm_landmarks[:, rtm_ankle_idx, 0].astype(float).copy()
            a_x[rtm_scores[:, rtm_ankle_idx] <= 0] = np.nan
            ankle_x_vel = np.gradient(
                gaussian_filter1d(_interpolate_nans(a_x), sigma=sigma)
            )
        else:
            hip_idx = HIPS[leg]
            ankle_idx = ANKLES[leg]

            hip_vis = visibilities[:, hip_idx]
            ankle_vis = visibilities[:, ankle_idx]
            mean_hip_vis = np.nanmean(hip_vis)
            mean_ankle_vis = np.nanmean(ankle_vis)

            sigma = 2.0 if mean_hip_vis > 0.5 else 3.0
            compression = _get_leg_compression(landmarks, hip_idx, ankle_idx, sigma)
            compression_velocity = np.gradient(compression)

            ankle_x_vel = np.gradient(
                gaussian_filter1d(
                    _interpolate_nans(landmarks[:, ankle_idx, 0].copy()), sigma=sigma
                )
            )

        diagnostics[leg] = {
            "mean_hip_visibility": float(mean_hip_vis),
            "mean_ankle_visibility": float(mean_ankle_vis),
        }

        # Detect contacts: velocity crosses from above threshold to below
        # (leg transitions from extending to loading)
        leg_contacts: List[GroundContact] = []

        for i in range(1, T - 1):
            if (
                compression_velocity[i - 1] > stance_threshold
                and compression_velocity[i] <= stance_threshold
            ):
                # Horizontal-velocity gate: reject if ankle is still moving
                # fast sideways (foot in forward swing, not yet planted).
                if horiz_vel_threshold_px > 0:
                    if abs(ankle_x_vel[i]) > horiz_vel_threshold_px:
                        continue

                # Check minimum interval from previous same-leg contact
                if (
                    len(leg_contacts) == 0
                    or (i - leg_contacts[-1].frame) >= min_interval_frames
                ):

                    # Confidence based on hip and ankle visibility
                    frame_vis = min(hip_vis[i], ankle_vis[i])
                    confidence = min(1.0, max(0.2, frame_vis / 0.6))

                    leg_contacts.append(
                        GroundContact(
                            frame=i,
                            leg=leg,
                            contact_frame=float(i),  # Will be refined
                            detection_method="coarse",
                            confidence=confidence,
                        )
                    )

        diagnostics[leg]["coarse_contacts"] = len(leg_contacts)
        all_contacts.extend(leg_contacts)

    # Sort all contacts by frame
    all_contacts.sort(key=lambda c: c.frame)

    default_stance_frames = int(0.18 * fps)  # ~180 ms default stance estimate

    def _refresh_coarse_stance(contacts: List[GroundContact]) -> None:
        """Set toe_off_frame and gct_ms based on same-leg neighbour spacing.

        Fix B2 (2026-04-26, Phase 6d): toe-off is bounded by the next
        SAME-LEG contact, not the absolute next contact.  Using the
        absolute next contact caused a cascade: a phantom cross-leg
        contact firing a few frames after a real same-leg contact would
        truncate that contact's estimated stance to a few ms, causing
        Fix A to drop the real contact (the "R91 killed by phantom L95"
        finding on PV_800m_2).  Same-leg bounds are biomechanically
        correct: a contact's stance ends before the NEXT same-leg
        touchdown, which is one full stride (not one step) away.
        """
        n = len(contacts)
        for i, contact in enumerate(contacts):
            # Find the next same-leg contact for the stance bound.
            next_same_leg_frame: Optional[int] = None
            for j in range(i + 1, n):
                if contacts[j].leg == contact.leg:
                    next_same_leg_frame = contacts[j].frame
                    break
            if next_same_leg_frame is not None:
                max_stance = next_same_leg_frame - 2
            else:
                max_stance = T - 1
            contact.toe_off_frame = float(
                min(contact.frame + default_stance_frames, max_stance)
            )
            if contact.contact_frame is not None and contact.toe_off_frame is not None:
                contact.gct_ms = (contact.toe_off_frame - contact.contact_frame) * (
                    1000.0 / fps
                )

    _refresh_coarse_stance(all_contacts)

    # Fix A (v1.3): biomechanical-floor sanity gate on coarse contacts.
    # The same 80-400 ms band already used at refinement time
    # (refine_contacts, 80–400 ms band) is propagated to the coarse set.
    # Spurious flight-phase detections (pose-landmark jitter) inflate
    # contact counts and produce sub-80 ms apparent durations after the
    # next-contact upper bound clips them — see 01c §10.10 / §11.4.3.
    spurious_per_leg: Dict[str, int] = {"L": 0, "R": 0}
    survivors: List[GroundContact] = []
    for contact in all_contacts:
        if contact.gct_ms is None or 80.0 < contact.gct_ms < 400.0:
            survivors.append(contact)
        else:
            spurious_per_leg[contact.leg] += 1
    all_contacts = survivors

    # Recompute stance estimates for survivors — neighbour distances may
    # have grown when squeezing spurious contacts dropped out.
    _refresh_coarse_stance(all_contacts)

    # Refresh per-leg diagnostics so downstream sees post-filter counts.
    for leg in ("L", "R"):
        diagnostics[leg]["coarse_contacts"] = sum(
            1 for c in all_contacts if c.leg == leg
        )
        diagnostics[leg]["spurious_filtered"] = spurious_per_leg[leg]

    return all_contacts, diagnostics


# ---------------------------------------------------------------------------
# Phase 6d — Fix A2 spurious-contact pruning
# ---------------------------------------------------------------------------
#
# Fix A (v1.3) drops contacts whose coarse_gct_ms falls outside 80–400 ms.
# Phantoms that fire close to a real contact still produce a plausible coarse
# gct (bounded by the next real contact's frame) and slip through Fix A.
# Fix A2 catches these using three independent biomechanical signals:
#
#  1. Pace-derived timing gate: adjacent pairs (any leg order) with gap
#     < min_alt_step_frames.
#  2. Same-leg alternation gate: same-leg pairs with gap < min_same_leg_frames
#     (= 2 × min_alt_step) — in normal gait same-leg contacts are separated
#     by a full stride (~33 frames at 800m pace), not a step.
#  3. Foot-altitude adjudication: when a pair is suspect, the candidate with
#     smaller pixel-y (= higher in image = foot in the air) is the phantom.
#
# Visual QA on PV_800m_1's annotated.mp4 confirmed 3 of 12 detected contacts
# were biomechanically impossible (L 138→L 147 = 9 frames, R 121→R 131 = 10,
# R 188→R 196 = 8) — well below the pace-derived minimum of ~16 frames at
# 800m pace.

_TIEBREAKER_PX = 5.0  # if foot-altitude difference < this, drop the second


def _foot_y_at_contact(
    contact: GroundContact,
    landmarks: np.ndarray,
    visibilities: np.ndarray,
    min_vis: float = 0.5,
    rtm_landmarks: Optional[np.ndarray] = None,
    rtm_scores: Optional[np.ndarray] = None,
) -> Optional[float]:
    """Return the pixel-y of the foot at the contact's refined IC frame.

    Uses ``contact_frame`` (refined sub-frame IC) rounded to the nearest
    integer.  This is the frame where the foot is closest to the ground
    for a REAL contact; for a PHANTOM the foot is in the air even at its
    reported IC, so its pixel-y will be smaller (higher in image).

    Using the refined IC rather than the coarse detection frame avoids
    unfair comparisons: coarse frame fires at the compression-velocity
    zero-crossing, which can be 3–5 frames before true touchdown while
    the foot is still descending.  At that point a real contact's foot
    appears higher in the image than it will be at actual touchdown,
    making it look like a phantom when compared against a planted foot
    at a different frame.

    Priority: max(toe_y, heel_y) with visibility gating, fallback to
    ankle_y if both foot landmarks are below threshold, None if even the
    ankle is occluded.

    A TRUE contact has its foot near the ground (maximum pixel-y, since
    pixel-y increases downward).  A PHANTOM contact has its foot in the
    air (smaller pixel-y).  Comparing this value across two suspect
    candidates identifies the phantom as the one with the smaller value.
    """
    ic_frame = (
        int(round(contact.contact_frame))
        if contact.contact_frame is not None
        else int(contact.frame)
    )
    t = min(ic_frame, len(landmarks) - 1)
    is_left = contact.leg == "L"
    ANKLE = LandmarkIndex.LEFT_ANKLE if is_left else LandmarkIndex.RIGHT_ANKLE

    if rtm_landmarks is not None and rtm_scores is not None:
        _rtm_toe = _RTM_FOOT[contact.leg]["big_toe"]
        _rtm_heel = _RTM_FOOT[contact.leg]["heel"]
        candidates = []
        if rtm_scores[t, _rtm_toe] > 0 and not np.isnan(rtm_landmarks[t, _rtm_toe, 1]):
            candidates.append(float(rtm_landmarks[t, _rtm_toe, 1]))
        if rtm_scores[t, _rtm_heel] > 0 and not np.isnan(
            rtm_landmarks[t, _rtm_heel, 1]
        ):
            candidates.append(float(rtm_landmarks[t, _rtm_heel, 1]))
        if candidates:
            return max(candidates)
        ankle_y = landmarks[t, ANKLE, 1]
        if not np.isnan(ankle_y) and visibilities[t, ANKLE] >= min_vis:
            return float(ankle_y)
        return None

    TOE = LandmarkIndex.LEFT_FOOT_INDEX if is_left else LandmarkIndex.RIGHT_FOOT_INDEX
    HEEL = LandmarkIndex.LEFT_HEEL if is_left else LandmarkIndex.RIGHT_HEEL

    toe_y = landmarks[t, TOE, 1]
    heel_y = landmarks[t, HEEL, 1]
    ankle_y = landmarks[t, ANKLE, 1]
    toe_vis = visibilities[t, TOE]
    heel_vis = visibilities[t, HEEL]
    ankle_vis = visibilities[t, ANKLE]

    candidates = []
    if not np.isnan(toe_y) and toe_vis >= min_vis:
        candidates.append(float(toe_y))
    if not np.isnan(heel_y) and heel_vis >= min_vis:
        candidates.append(float(heel_y))
    if candidates:
        return max(candidates)  # largest pixel-y = lowest in image = on ground
    if not np.isnan(ankle_y) and ankle_vis >= min_vis:
        return float(ankle_y)
    return None


def prune_spurious_contacts(
    contacts: List[GroundContact],
    landmarks: np.ndarray,
    visibilities: np.ndarray,
    fps: float,
    velocity_ms: float,
    leg_length_m: float,
    cadence_band_frac: float = 0.30,
    min_vis: float = 0.5,
    rtm_landmarks: Optional[np.ndarray] = None,
    rtm_scores: Optional[np.ndarray] = None,
) -> Tuple[List[GroundContact], Dict]:
    """Sliding-window prune of biomechanically-impossible-close contacts.

    Detects phantom flight-phase contacts (F1 mechanism, 01c §10.18.6 /
    Phase 6d) by comparing every adjacent pair sorted by frame against
    a pace-derived plausible inter-contact band, then adjudicating via
    foot altitude (the one with smaller pixel-y is in the air → phantom).

    Two suspicion gates:
    - **Timing gate** (cross-leg): ``gap < min_alt_step_frames`` — catches
      any pair that is impossibly close regardless of which leg fired.
    - **Same-leg alternation gate**: ``same_leg AND gap < min_same_leg_frames``
      (where ``min_same_leg = 2 × min_alt``) — catches same-leg pairs that
      are under the stride-period minimum even when they might slip the
      alternating-step timing gate.

    Args:
        contacts: contacts from ``detect_contacts_coarse`` after Fix A.
        landmarks: (T, 33, 2) post-pipeline landmark tensor.
        visibilities: (T, 33) visibility scores.
        fps: video frame rate.
        velocity_ms: runner velocity estimate in m/s (from pace_estimator).
        leg_length_m: runner leg length in metres (from calibration).
        cadence_band_frac: ±width of the cadence band (default 0.25 = ±25 %).
        min_vis: minimum visibility for foot-landmark altitude check.

    Returns:
        (survivors, diagnostics). diagnostics: ``n_dropped_total``,
        ``min_alt_step_frames_used``, ``drop_log`` (list of per-dropped-
        contact dicts with frame / leg / prev_foot_y / curr_foot_y /
        reason).
    """
    from src.utils.pace_estimator import min_step_interval_frames

    min_alt = min_step_interval_frames(
        velocity_ms, leg_length_m, fps, cadence_band_frac
    )
    survivors: List[GroundContact] = sorted(contacts, key=lambda c: c.frame)
    drop_log: List[Dict] = []

    i = 1
    while i < len(survivors):
        prev = survivors[i - 1]
        curr = survivors[i]
        # Gap = max(refined_gap, coarse_gap).
        # A pair is only pruned when BOTH gaps are below the threshold —
        # if either indicates the contacts are genuinely far apart, keep
        # them. This prevents IC refinement (which can legitimately move
        # contact_frame later for heel clips) from artificially collapsing
        # the gap between two real contacts below the pruning threshold.
        # Foot-altitude adjudication (below) still uses contact_frame.
        # prev_ic / curr_ic (refined) are also used by the same-leg
        # tiebreaker below to find the best pace-interval match.
        prev_ic = float(
            prev.contact_frame if prev.contact_frame is not None else prev.frame
        )
        curr_ic = float(
            curr.contact_frame if curr.contact_frame is not None else curr.frame
        )
        gap = max(
            curr_ic - prev_ic,
            float(curr.frame) - float(prev.frame),
        )
        same_leg = prev.leg == curr.leg

        # Single timing gate: any adjacent pair (regardless of leg) with gap
        # below the pace-derived minimum step interval is suspect.  The
        # same-leg alternation signal is recorded in the drop_log for audit
        # but does NOT add a separate, looser suspicion gate — that caused
        # over-pruning via cascade in the initial pilot on PV_800m_1 (the
        # same-leg threshold 2×min_alt flagged a 25-frame same-leg pair that
        # was a legitimate contact, triggering a 6-drop cascade from 12
        # contacts down to 6 vs truth 8; the pure timing gate gives 7).
        is_suspect = gap < min_alt

        if not is_suspect:
            i += 1
            continue

        prev_y = _foot_y_at_contact(
            prev,
            landmarks,
            visibilities,
            min_vis,
            rtm_landmarks=rtm_landmarks,
            rtm_scores=rtm_scores,
        )
        curr_y = _foot_y_at_contact(
            curr,
            landmarks,
            visibilities,
            min_vis,
            rtm_landmarks=rtm_landmarks,
            rtm_scores=rtm_scores,
        )

        if prev_y is None and curr_y is None:
            drop_idx = i  # both untrustworthy: keep prev, drop curr
            reason = "both_untrustworthy"
        elif prev_y is None:
            # Cannot adjudicate prev — keep both and let the next pair
            # comparison handle the phantom (e.g. the curr's neighbour
            # will be compared with it next iteration).  Dropping prev
            # blind caused real contacts with occluded foot landmarks
            # (coarse-only detections) to be wrongly eliminated.
            i += 1
            continue
        elif curr_y is None:
            drop_idx = i  # curr untrustworthy → drop curr
            reason = "curr_untrustworthy"
        elif abs(prev_y - curr_y) < _TIEBREAKER_PX:
            # Foot altitudes indistinguishable.  For same-leg pairs, use
            # pace-plausibility from the previous same-leg contact: drop
            # whichever candidate has the less plausible interval from the
            # most recent prior contact on the same leg.
            #
            # Example (PV_800m_2): same-leg L pair at IC=95 (#3) and
            # IC=105 (#4), previous L at IC=75 (#2).  Intervals: 20f vs
            # 30f; expected same-leg stride ≈ 32f → #4 wins; drop #3.
            # The original "drop second" tiebreaker would keep #3 (wrong).
            if same_leg:
                # Expected same-leg period in frames.
                _step_pred = leg_length_m * (1.4 + 0.1 * velocity_ms)
                _cadence_pred = 60.0 * velocity_ms / _step_pred
                expected_sl_frames = 120.0 * fps / _cadence_pred

                # Find the most recent prior contact on the same leg.
                prev_same_ic: Optional[float] = None
                for _j in range(i - 2, -1, -1):
                    if survivors[_j].leg == prev.leg:
                        _cf = survivors[_j].contact_frame
                        prev_same_ic = (
                            float(_cf)
                            if _cf is not None
                            else float(survivors[_j].frame)
                        )
                        break

                if prev_same_ic is not None:
                    err_prev = abs((prev_ic - prev_same_ic) - expected_sl_frames)
                    err_curr = abs((curr_ic - prev_same_ic) - expected_sl_frames)
                    drop_idx = i - 1 if err_prev > err_curr else i
                    reason = "tiebreaker_interval"
                else:
                    drop_idx = i  # no prior same-leg anchor: drop second
                    reason = "tiebreaker"
            else:
                drop_idx = i  # cross-leg: drop second
                reason = "tiebreaker"
        else:
            # Drop the one with smaller pixel-y (higher in image = in the air).
            drop_idx = i - 1 if prev_y < curr_y else i
            reason = "foot_altitude"

        dropped = survivors[drop_idx]
        drop_log.append(
            {
                "frame": int(dropped.frame),
                "leg": dropped.leg,
                "prev_foot_y": prev_y,
                "curr_foot_y": curr_y,
                "gap_frames": int(gap),
                "same_leg": same_leg,
                "reason": reason,
            }
        )
        survivors.pop(drop_idx)
        if drop_idx == i - 1:
            # Dropped prev → curr shifted to i-1; re-check against new prev.
            i = max(1, i - 1)
        # Dropped curr → i now points at the next element; no change needed.

    diagnostics: Dict = {
        "n_dropped_total": len(drop_log),
        "min_alt_step_frames_used": float(min_alt),
        "velocity_ms_used": float(velocity_ms),
        "drop_log": drop_log,
    }
    return survivors, diagnostics


# ---------------------------------------------------------------------------
# Phase 6d fix #3 — pace-grid contact interpolation
# ---------------------------------------------------------------------------
#
# After pruning phantoms, gaps larger than ~1.5× the expected step interval
# indicate missing real contacts.  This happens when:
#   (a) A real contact's coarse detection was killed by Fix A because a
#       phantom next to it truncated its estimated stance (Fix A2 with same-
#       leg bounds reduces this but does not eliminate it entirely).
#   (b) The compression-velocity zero-crossing for a real contact is so
#       shallow or masked that the coarse detector never fires it.
#
# The interpolation pass:
#   1. Finds gaps > `min_hole_factor` × expected_step_interval between
#      adjacent contacts.
#   2. Estimates the expected IC frame for each missing contact using the
#      pace model (cadence_pred from velocity + leg length).
#   3. Searches for a toe-velocity touchdown zero-crossing within ±search_window
#      frames of the estimated IC.  Guards: requires non-NaN toe landmarks
#      with visibility ≥ min_vis.
#   4. If the refinement succeeds, inserts a new contact (detection_method=
#      "interpolated"); otherwise skips the hole.
#   5. Re-runs _refresh_coarse_stance and the full refine pass on the augmented
#      list so the new contacts also get sub-frame TO refinement.
#
# Maximum contacts inserted per hole is capped at `max_per_hole` to avoid
# unreasonably dense interpolation on clips with very large gaps (Recovery_2,
# Steady_1) where landmark quality may be poor — if toe visibility is too
# low the search naturally returns None and the contact is not inserted.


def _find_touchdown_near(
    leg: str,
    estimated_ic: float,
    landmarks: np.ndarray,
    visibilities: np.ndarray,
    fps: float,
    resolution_height: int = 2160,
    search_window: int = 8,
    min_vis: float = 0.4,
    rtm_landmarks: Optional[np.ndarray] = None,
    rtm_scores: Optional[np.ndarray] = None,
) -> Optional[float]:
    """Search for a toe-velocity touchdown crossing near `estimated_ic`.

    Uses the same toe-falling zero-crossing as ``refine_contacts_with_toe``
    but centred on the pace-grid estimated IC rather than on a coarse
    detection frame.  Returns the sub-frame refined IC or None if no
    crossing is found or toe visibility is too low.
    """
    T = len(landmarks)

    if rtm_landmarks is not None and rtm_scores is not None:
        _rtm_toe = _RTM_FOOT[leg]["big_toe"]
        _raw = rtm_landmarks[:, _rtm_toe, 1].astype(float).copy()
        _raw[rtm_scores[:, _rtm_toe] <= 0] = np.nan
        toe_y_raw = _interpolate_nans(_raw)
        toe_vis_ok = rtm_scores[:, _rtm_toe] > 0
    else:
        TOE = (
            LandmarkIndex.LEFT_FOOT_INDEX
            if leg == "L"
            else LandmarkIndex.RIGHT_FOOT_INDEX
        )
        toe_y_raw = _interpolate_nans(landmarks[:, TOE, 1].copy())
        toe_vis_ok = visibilities[:, TOE] >= min_vis

    window = min(5, T if T % 2 == 1 else T - 1)
    if window < 3:
        return None
    toe_smooth = savgol_filter(toe_y_raw, window_length=window, polyorder=2)
    toe_velocity = np.gradient(toe_smooth)

    velocity_threshold = resolution_height * 0.0007
    est = int(round(estimated_ic))
    s_start = max(1, est - search_window)
    s_end = min(T - 1, est + search_window)

    # Require at least a few visible frames in the search window before
    # attempting refinement.
    vis_in_window = toe_vis_ok[s_start : s_end + 1]
    if float(vis_in_window.sum()) < 3:
        return None

    # Search backward from s_end for zero-crossing (toe falling → arrested).
    for i in range(s_end, s_start, -1):
        v_prev = toe_velocity[i - 1]
        v_curr = toe_velocity[i]
        if v_prev > velocity_threshold and v_curr <= velocity_threshold:
            if v_prev != v_curr:
                fraction = (velocity_threshold - v_prev) / (v_curr - v_prev)
                return float((i - 1) + fraction)
            return float(i - 1)
    return None


def interpolate_missing_contacts(
    contacts: List[GroundContact],
    landmarks: np.ndarray,
    visibilities: np.ndarray,
    fps: float,
    velocity_ms: float,
    leg_length_m: float,
    resolution_height: int = 2160,
    cadence_band_frac: float = 0.30,
    min_hole_factor: float = 1.5,
    max_per_hole: int = 3,
    search_window: int = 8,
    min_vis: float = 0.4,
    rtm_landmarks: Optional[np.ndarray] = None,
    rtm_scores: Optional[np.ndarray] = None,
) -> Tuple[List[GroundContact], Dict]:
    """Insert contacts for gaps that exceed `min_hole_factor` × expected step.

    For each gap between adjacent contacts, estimates the expected IC frames
    using the pace model (``velocity_ms`` / ``leg_length_m``) and attempts
    touchdown refinement at each estimate.  Inserts a new ``GroundContact``
    with ``detection_method="interpolated"`` for every hole where the toe-
    velocity signal confirms a real touchdown near the estimated frame.

    Args:
        contacts: post-prune contact list sorted by frame.
        landmarks: (T, 33, 2) post-pipeline landmark tensor.
        visibilities: (T, 33) visibility scores.
        fps: video frame rate.
        velocity_ms: runner velocity estimate (from pace_estimator).
        leg_length_m: runner leg length in metres (from calibration).
        resolution_height: for velocity_threshold scaling.
        cadence_band_frac: half-width of cadence band; sets expected step.
        min_hole_factor: gaps > this × expected_step_interval are treated
            as containing at least one missing contact.
        max_per_hole: maximum contacts inserted per gap.
        search_window: ±frames around estimated IC to search for touchdown.
        min_vis: minimum toe visibility required to attempt refinement.

    Returns:
        (augmented_contacts, diagnostics).
        diagnostics: ``n_inserted``, ``n_holes_found``, ``insert_log``.
    """
    from src.utils.pace_estimator import expected_cadence_band

    if not contacts:
        return contacts, {"n_inserted": 0, "n_holes_found": 0, "insert_log": []}

    cadence_min, cadence_max = expected_cadence_band(
        velocity_ms, leg_length_m, cadence_band_frac
    )
    cadence_pred = (cadence_min + cadence_max) / 2.0
    expected_step_frames = 60.0 * fps / cadence_pred

    sorted_contacts = sorted(contacts, key=lambda c: c.frame)
    insert_log: List[Dict] = []
    n_holes = 0

    i = 0
    while i < len(sorted_contacts) - 1:
        prev = sorted_contacts[i]
        curr = sorted_contacts[i + 1]

        prev_ic = float(
            prev.contact_frame if prev.contact_frame is not None else prev.frame
        )
        curr_ic = float(
            curr.contact_frame if curr.contact_frame is not None else curr.frame
        )
        gap = curr_ic - prev_ic
        n_missing = int(round(gap / expected_step_frames)) - 1

        if n_missing < 1 or gap <= min_hole_factor * expected_step_frames:
            i += 1
            continue

        n_holes += 1
        n_to_insert = min(n_missing, max_per_hole)
        inserted_this_hole = 0

        # Leg alternates starting from the leg AFTER prev.
        next_leg = "R" if prev.leg == "L" else "L"
        for k in range(1, n_to_insert + 1):
            est_ic = prev_ic + k * expected_step_frames
            leg = next_leg if k % 2 == 1 else prev.leg

            refined_ic = _find_touchdown_near(
                leg,
                est_ic,
                landmarks,
                visibilities,
                fps,
                resolution_height=resolution_height,
                search_window=search_window,
                min_vis=min_vis,
                rtm_landmarks=rtm_landmarks,
                rtm_scores=rtm_scores,
            )
            if refined_ic is None:
                insert_log.append(
                    {
                        "estimated_ic": float(est_ic),
                        "leg": leg,
                        "result": "no_crossing_found",
                    }
                )
                continue

            new_contact = GroundContact(
                frame=int(round(refined_ic)),
                leg=leg,
                contact_frame=refined_ic,
                toe_off_frame=refined_ic + 0.18 * fps,  # placeholder; refined later
                detection_method="interpolated",
                confidence=0.5,
            )
            sorted_contacts.append(new_contact)
            sorted_contacts.sort(key=lambda c: c.frame)
            inserted_this_hole += 1
            insert_log.append(
                {
                    "estimated_ic": float(est_ic),
                    "refined_ic": float(refined_ic),
                    "leg": leg,
                    "result": "inserted",
                }
            )

        i += inserted_this_hole + 1  # skip past inserted contacts in this gap

    diagnostics: Dict = {
        "n_inserted": sum(1 for e in insert_log if e["result"] == "inserted"),
        "n_holes_found": n_holes,
        "insert_log": insert_log,
    }
    return sorted_contacts, diagnostics


_FOREFOOT_PACES = {"800m", "1500m"}
_HEEL_PACES = {"steady", "recovery", "easy", "threshold"}
_FOREFOOT_SHOES = {"track_spike", "super_shoe", "carbon_plate"}
_HEEL_SHOES = {"trainer", "racing_flat", "spikes_distance", "neutral"}


def detect_clip_strike_pattern(
    pace_level: str,
    shoe_category: Optional[str],
    contacts: List[GroundContact],
    landmarks: np.ndarray,
    visibilities: np.ndarray,
    body_height_px: float,
    tau_strike_frac: float = 0.01,
    min_vis: float = 0.5,
    rtm_landmarks: Optional[np.ndarray] = None,
    rtm_scores: Optional[np.ndarray] = None,
) -> Dict:
    """Classify the dominant strike pattern for an entire clip.

    Combines three independent signals:
      1. Speed prior  — pace_level maps to forefoot (800m/1500m) or heel
      2. Shoe prior   — shoe_category maps to forefoot (track_spike) or heel
      3. Majority vote — per-contact zero-crossing classification across all
         detected contacts; takes the plurality

    Returns a dict with per-signal votes, counts, and the final 'decision'
    ('forefoot' | 'heel' | 'unknown') plus a 'confidence' rating.
    """
    # --- Signal 1: speed ---
    pace_norm = (pace_level or "").lower().strip()
    if pace_norm in _FOREFOOT_PACES:
        speed_signal = "forefoot"
    elif pace_norm in _HEEL_PACES:
        speed_signal = "heel"
    else:
        speed_signal = "neutral"

    # --- Signal 2: shoe ---
    shoe_norm = (shoe_category or "").lower().strip()
    if shoe_norm in _FOREFOOT_SHOES:
        shoe_signal = "forefoot"
    elif shoe_norm in _HEEL_SHOES:
        shoe_signal = "heel"
    else:
        shoe_signal = "neutral"

    # --- Signal 3: majority vote across contacts ---
    vote_counts: Dict[str, int] = {"forefoot": 0, "heel": 0, "unknown": 0}
    for contact in contacts:
        label = _detect_strike_pattern_for_contact(
            contact,
            landmarks,
            visibilities,
            body_height_px,
            tau_strike_frac=tau_strike_frac,
            min_vis=min_vis,
            rtm_landmarks=rtm_landmarks,
            rtm_scores=rtm_scores,
        )
        vote_counts[label] = vote_counts.get(label, 0) + 1

    vote_total = sum(vote_counts.values())
    classified = vote_counts["forefoot"] + vote_counts["heel"]
    if classified == 0:
        vote_signal = "unknown"
    elif vote_counts["forefoot"] > vote_counts["heel"]:
        vote_signal = "forefoot"
    elif vote_counts["heel"] > vote_counts["forefoot"]:
        vote_signal = "heel"
    else:
        vote_signal = "unknown"

    # --- Combine ---
    signals = [s for s in (speed_signal, shoe_signal, vote_signal) if s != "neutral"]
    forefoot_votes = signals.count("forefoot")
    heel_votes = signals.count("heel")

    if forefoot_votes > heel_votes:
        decision = "forefoot"
    elif heel_votes > forefoot_votes:
        decision = "heel"
    else:
        decision = "unknown"

    n_agree = max(forefoot_votes, heel_votes)
    n_signals = len(signals)
    if n_signals == 0:
        confidence = "none"
    elif n_agree == n_signals:
        confidence = "strong"
    elif n_agree >= 2:
        confidence = "moderate"
    else:
        confidence = "weak"

    return {
        "speed_signal": speed_signal,
        "shoe_signal": shoe_signal,
        "vote_signal": vote_signal,
        "vote_counts": vote_counts,
        "vote_total": vote_total,
        "decision": decision,
        "confidence": confidence,
    }


VALID_TOE_OFF_METHODS = (
    "ankle_y_lift",  # Fix B-B1 (v1.5 default; preserved as v1.7 default)
    "foot_index_position_lift",  # Fix B2c (Phase 6c)
    "per_strike_pattern",  # Fix B3 (Phase 6c) — dispatches B-B1 vs B2c per contact
)


def _detect_strike_pattern_for_contact(
    contact: GroundContact,
    landmarks: np.ndarray,
    visibilities: np.ndarray,
    body_height_px: float,
    tau_strike_frac: float = 0.01,
    min_vis: float = 0.5,
    rtm_landmarks: Optional[np.ndarray] = None,
    rtm_scores: Optional[np.ndarray] = None,
) -> str:
    """Per-contact strike pattern detection at touchdown.

    Two-stage detection (pixel-y increases downward):

    Stage 1 — zero-crossing pre-check over [F_IC-4, F_IC+1]:
      Searches for a frame t where delta[t-1] > 0 (heel-dominant approach)
      and delta[t] < -tau (toe contacts first). This catches forefoot
      contacts where the foot flattens within 1 frame of IC so the signal
      is too brief to survive a window mean — but the crossing itself is
      unambiguous. Both frames must pass the visibility gate.

    Stage 2 — window mean over [F_IC-1, F_IC, F_IC+1]:
      Fallback for contacts that are consistently forefoot/heel throughout
      the window (no crossing needed). Returns 'heel' / 'forefoot' /
      'unknown' using the same tau threshold.

    Returns 'heel' / 'forefoot' / 'unknown'.
    """
    F_IC = int(contact.frame)
    T = len(landmarks)
    threshold = tau_strike_frac * body_height_px

    if rtm_landmarks is not None and rtm_scores is not None:
        _rtm_heel = _RTM_FOOT[contact.leg]["heel"]
        _rtm_toe = _RTM_FOOT[contact.leg]["big_toe"]
        h_y = rtm_landmarks[:, _rtm_heel, 1]
        t_y = rtm_landmarks[:, _rtm_toe, 1]

        def _vis_ok(fr: int) -> bool:
            return bool(rtm_scores[fr, _rtm_heel] > 0 and rtm_scores[fr, _rtm_toe] > 0)

    else:
        HEEL_IDX = (
            LandmarkIndex.LEFT_HEEL if contact.leg == "L" else LandmarkIndex.RIGHT_HEEL
        )
        TOE_IDX = (
            LandmarkIndex.LEFT_FOOT_INDEX
            if contact.leg == "L"
            else LandmarkIndex.RIGHT_FOOT_INDEX
        )
        h_y = landmarks[:, HEEL_IDX, 1]
        t_y = landmarks[:, TOE_IDX, 1]

        def _vis_ok(fr: int) -> bool:
            return bool(
                visibilities[fr, HEEL_IDX] >= min_vis
                and visibilities[fr, TOE_IDX] >= min_vis
            )

    # Stage 1: zero-crossing pre-check.
    s_start = max(1, F_IC - 4)
    s_end = min(T - 1, F_IC + 1)
    for t in range(s_start, s_end + 1):
        prev = t - 1
        if not _vis_ok(prev) or not _vis_ok(t):
            continue
        if np.any(np.isnan([h_y[prev], t_y[prev], h_y[t], t_y[t]])):
            continue
        delta_prev = h_y[prev] - t_y[prev]
        delta_t = h_y[t] - t_y[t]
        if delta_prev > 0 and delta_t < -threshold:
            return "forefoot"

    # Stage 2: window mean fallback.
    window = [t for t in (F_IC - 1, F_IC, F_IC + 1) if 0 <= t < T]
    if not window:
        return "unknown"
    heel_ys = np.array([h_y[t] for t in window])
    toe_ys = np.array([t_y[t] for t in window])
    if (
        np.any(np.isnan(heel_ys))
        or np.any(np.isnan(toe_ys))
        or not all(_vis_ok(t) for t in window)
    ):
        return "unknown"
    delta = float(np.mean(heel_ys) - np.mean(toe_ys))
    if delta > threshold:
        return "heel"
    if delta < -threshold:
        return "forefoot"
    return "unknown"


def _refine_toe_off_ankle_y_lift(
    contact: GroundContact,
    ankle_velocity: np.ndarray,
    lift_threshold: float,
    T: int,
    ic_frame_override: Optional[float] = None,
    fps: Optional[float] = None,
) -> Optional[float]:
    """Fix B-B1 (v1.5 default): toe-off via ankle-y velocity crossing
    lift_threshold (rising). Refactored from the previous inline body of
    refine_contacts_with_toe for re-use under per_strike_pattern dispatch.

    When ``ic_frame_override`` is supplied (Phase C, v1.17), the search
    window is anchored on ``ic_frame_override + default_stance`` rather than
    on the coarse ``contact.toe_off_frame``.  This corrects the systematic
    window shift on heel clips where the coarse TO was computed from the
    coarse IC (C_IC) rather than the refined IC(S).

    Returns the (sub-frame) toe-off frame if the threshold-crossing is
    found within the search window, else falls back to the contact's
    pre-existing toe_off_frame (default coarse stance estimate).
    """
    if ic_frame_override is not None and fps is not None:
        default_stance = int(0.18 * fps)
        coarse_end = int(ic_frame_override) + default_stance
    else:
        coarse_end = (
            int(contact.toe_off_frame) if contact.toe_off_frame else contact.frame + 12
        )
    search_start = max(1, coarse_end - 5)
    search_end = min(T - 1, coarse_end + 5)

    refined_toe_off = contact.toe_off_frame
    for i in range(search_start, search_end):
        v_prev = ankle_velocity[i - 1]
        v_curr = ankle_velocity[i]
        if v_prev > lift_threshold and v_curr <= lift_threshold:
            if v_prev != v_curr:
                fraction = (lift_threshold - v_prev) / (v_curr - v_prev)
                refined_toe_off = (i - 1) + fraction
            else:
                refined_toe_off = float(i - 1)
            break
    return refined_toe_off


def _refine_toe_off_foot_index_velocity(
    contact: GroundContact,
    foot_index_velocity: np.ndarray,
    lift_threshold: float,
    T: int,
    ic_frame_override: Optional[float] = None,
    fps: Optional[float] = None,
) -> Optional[float]:
    """Fix B-B2: toe-off via foot-index (ball of foot) velocity crossing
    lift_threshold (rising). For clear forefoot contacts (spikes, super
    shoes) where foot_index stays grounded until true TO and its velocity
    onset is sharper than ankle velocity (B-B1), which rises early during
    plantar push-off before the foot leaves the ground.

    Identical search logic to _refine_toe_off_ankle_y_lift but applied to
    foot_index_y velocity rather than ankle_y velocity.

    When ic_frame_override is supplied, re-anchors the search window on
    ic_frame_override + default_stance instead of contact.toe_off_frame.
    This corrects the systematic window shift on forefoot clips where
    C_IC fires 1-5 frames after true IC (Phase B, v1.16).
    """
    if ic_frame_override is not None:
        default_stance = int(0.18 * fps) if fps is not None else T // 6
        coarse_end = int(ic_frame_override) + default_stance
    else:
        coarse_end = (
            int(contact.toe_off_frame) if contact.toe_off_frame else contact.frame + 12
        )
    search_start = max(1, coarse_end - 6)
    search_end = min(T - 1, coarse_end + 6)
    # TO cannot precede IC by less than 120 ms; clamp search_start to block
    # mid-stance plantar push-off wobbles that cross lift_threshold_forefoot
    # 3-5 frames before true TO at 800m pace (v1.19 diagnostic finding).
    # 120 ms (7 fr at 60 fps) is the empirical floor that blocks the wobble
    # without affecting contacts with genuine GCT >= 80 ms (sanity gate).
    if ic_frame_override is not None:
        min_gct_frames = int(0.120 * fps) if fps is not None else 7
        search_start = max(search_start, int(ic_frame_override) + min_gct_frames)

    refined_toe_off = contact.toe_off_frame
    for i in range(search_start, search_end):
        v_prev = foot_index_velocity[i - 1]
        v_curr = foot_index_velocity[i]
        if v_prev > lift_threshold and v_curr <= lift_threshold:
            if v_prev != v_curr:
                fraction = (lift_threshold - v_prev) / (v_curr - v_prev)
                refined_toe_off = (i - 1) + fraction
            else:
                refined_toe_off = float(i - 1)
            break
    return refined_toe_off


def _refine_toe_off_foot_index_position(
    contact: GroundContact,
    landmarks: np.ndarray,
    visibilities: np.ndarray,
    body_height_px: float,
    fps: float,
    delta_lift_frac: float = 0.02,
    min_vis: float = 0.5,
    ic_frame_override: Optional[float] = None,
    window_factor: float = 1.5,
    rtm_landmarks: Optional[np.ndarray] = None,
    rtm_scores: Optional[np.ndarray] = None,
) -> Optional[float]:
    """Fix B2c (Phase 6c): toe-off via foot-index pixel-y leaving a
    per-stance ground-line proxy by delta_lift_frac × body_height_px.

    Pixel-y increases downward. During stance the toe (foot-index) is
    pinned to the ground at maximum pixel-y; at toe-off it rises and
    pixel-y decreases. The ground-line proxy is the maximum (= lowest in
    image) of foot_index_y over the stance window. Toe-off is the
    sub-frame transition between the LAST frame where the toe was at
    ground level (toe_y >= ground_line - Δ) and the next valid frame
    (where toe_y < ground_line - Δ).

    The search direction is BACKWARD from stance_end. A forward search
    from F_IC misinterprets the heel-to-toe roll on heel strikers (the
    toe is in the air for ~3 frames after F_IC and only descends to
    ground_line during the roll) as toe-off — see 01c §10.18 / Phase 6c
    Step 4 finding for the diagnostic. Backward search is robust to both
    strike patterns: heel (toe descends to ground at frames F_IC+0..3,
    pinned at frames F_IC+3..11, lifts at F_IC+12) and forefoot (toe
    pinned from F_IC, lifts at F_IC+12); in both cases the LAST pinned
    frame is the one immediately before true toe-off.

    Returns None if foot-index visibility is too low across the stance
    window or no clear lift is detected — caller falls back to Fix B-B1.
    """
    # Anchor the stance window on the caller-supplied refined IC when
    # available (ic_frame_override), then contact_frame, then coarse frame.
    # ic_frame_override is passed by refine_contacts for heel contacts so the
    # B2c window covers the actual stance phase rather than starting from the
    # coarse IC (which fires 6-9 frames early on heel clips).
    if ic_frame_override is not None:
        F_IC = int(ic_frame_override)
    elif contact.contact_frame is not None:
        F_IC = int(contact.contact_frame)
    else:
        F_IC = int(contact.frame)
    T = len(landmarks)
    TOE_IDX = (
        LandmarkIndex.LEFT_FOOT_INDEX
        if contact.leg == "L"
        else LandmarkIndex.RIGHT_FOOT_INDEX
    )

    default_stance_frames = int(0.18 * fps)
    stance_end = min(F_IC + int(window_factor * default_stance_frames), T - 1)
    if stance_end <= F_IC + 2:
        return None  # too short a window to detect lift

    if rtm_landmarks is not None and rtm_scores is not None:
        _rtm_toe = _RTM_FOOT[contact.leg]["big_toe"]
        toe_y_window = rtm_landmarks[F_IC : stance_end + 1, _rtm_toe, 1].astype(float)
        _rtm_sc_win = rtm_scores[F_IC : stance_end + 1, _rtm_toe]
        valid_mask = (~np.isnan(toe_y_window)) & (_rtm_sc_win > 0)
    else:
        toe_y_window = landmarks[F_IC : stance_end + 1, TOE_IDX, 1]
        toe_vis_window = visibilities[F_IC : stance_end + 1, TOE_IDX]
        valid_mask = (~np.isnan(toe_y_window)) & (toe_vis_window >= min_vis)
    if int(valid_mask.sum()) < 3:
        return None  # insufficient visible / finite frames

    valid_toe_y = toe_y_window[valid_mask]
    ground_line = float(np.max(valid_toe_y))
    delta_lift = delta_lift_frac * body_height_px
    target = ground_line - delta_lift

    # Backward search: find the LAST frame where the toe was at ground
    # level (toe_y >= target). Then find the NEXT valid frame after it
    # — the toe-off transition is sub-frame between the two.
    last_pinned_offset: Optional[int] = None
    for offset in range(len(toe_y_window) - 1, -1, -1):
        if not bool(valid_mask[offset]):
            continue
        if float(toe_y_window[offset]) >= target:
            last_pinned_offset = offset
            break

    if last_pinned_offset is None:
        # Toe never observed at/near ground in the search window — should
        # be rare given ground_line = max(toe_y). Conservative fallback.
        return None
    if last_pinned_offset >= len(toe_y_window) - 1:
        # Toe still pinned at end of search window; no clear lift detected
        # within the stance budget. Fall back to Fix B-B1.
        return None

    # Find the next valid frame after the last pinned one.
    next_offset = last_pinned_offset + 1
    while next_offset < len(toe_y_window) and not bool(valid_mask[next_offset]):
        next_offset += 1
    if next_offset >= len(toe_y_window):
        return None

    y_prev = float(toe_y_window[last_pinned_offset])
    y_curr = float(toe_y_window[next_offset])
    if y_prev != y_curr:
        fraction = (target - y_prev) / (y_curr - y_prev)
        return float(F_IC + last_pinned_offset) + fraction
    return float(F_IC + last_pinned_offset)


def refine_contacts(
    contacts: List[GroundContact],
    landmarks: np.ndarray,
    visibilities: np.ndarray,
    fps: float,
    resolution_height: int = 2160,
    savgol_window: int = 5,
    savgol_polyorder: int = 2,
    min_toe_visibility: float = 0.5,
    toe_off_method: str = "ankle_y_lift",
    body_height_px: Optional[float] = None,
    tau_strike_frac: float = 0.01,
    delta_lift_frac: float = 0.02,
    delta_lift_frac_forefoot: float = 0.01,
    clip_strike_pattern: Optional[str] = None,
    rtm_landmarks: Optional[np.ndarray] = None,
    rtm_scores: Optional[np.ndarray] = None,
) -> Tuple[List[GroundContact], Dict]:
    """
    FINE PASS: Refine contact timing using foot landmark velocity signals.

    IC refinement dispatches on ``clip_strike_pattern``:

    - ``"heel"``: heel y-velocity falling zero-crossing in a forward-biased
      window ``[C_IC−2, C_IC+10]``. The coarse IC fires ~6–9 frames early
      on heel clips (dorsiflexion pre-contact reversal); the forward bias
      reaches the true IC while avoiding the toe-velocity signal, which
      only arrests when the toe rolls to ground several frames post-IC.
    - anything else (``"forefoot"``, ``None``): toe (foot index) y-velocity
      falling zero-crossing in the symmetric window ``[C_IC−5, C_IC+5]``.
      Unchanged from v1.5–v1.14.

    Toe-off detection is dispatched on ``toe_off_method``:

    - ``"ankle_y_lift"`` (default, v1.5 / v1.7 byte-identical): ankle-y
      velocity crossing lift_threshold (rising). Fix B-B1.
    - ``"foot_index_position_lift"``: foot-index pixel-y rising above a
      per-stance ground-line proxy by delta_lift_frac × body_height_px.
      Fix B2c (Phase 6c). Targets the heel-strike F3 mid-stance plantar
      flexion case where the ankle rises early during the heel-to-toe
      roll (01c §10.13.2).
    - ``"per_strike_pattern"``: detect strike pattern per contact via
      heel-y vs foot-index-y at touchdown; route heel contacts through
      foot_index_position_lift, route forefoot / unknown / midfoot
      contacts through ankle_y_lift. Fix B3 (Phase 6c). Forefoot
      passes preserved by construction.

    Args:
        contacts: List of coarse GroundContact from leg compression detection
        landmarks: (T, 33, 2) pose landmarks
        visibilities: (T, 33) visibility scores
        fps: Video frame rate
        resolution_height: Video height for adaptive thresholds
        savgol_window: Savitzky-Golay filter window (must be odd)
        savgol_polyorder: Polynomial order for Savitzky-Golay
        min_toe_visibility: Minimum toe visibility to attempt refinement
        toe_off_method: One of VALID_TOE_OFF_METHODS. Default
            ``"ankle_y_lift"`` reproduces v1.5 / v1.7 behaviour byte-
            identical (no strike-pattern detection runs).
        body_height_px: Body height in pixels, used to scale strike-
            pattern threshold and foot-index lift threshold. Required
            when toe_off_method is ``foot_index_position_lift`` or
            ``per_strike_pattern``. If None and a method needing it is
            passed, falls back to ``0.4 × resolution_height`` (~864 px
            at 4K, a reasonable runner-in-frame proxy).
        tau_strike_frac: Strike-pattern detection threshold as a
            fraction of body_height_px. Default 0.01 (1 % body height ≈
            4–5 px at 4K).
        delta_lift_frac: Foot-index toe-off lift threshold for heel
            contacts as a fraction of body_height_px. Default 0.02
            (2 % body height ≈ 9 px at 4K).
        delta_lift_frac_forefoot: Same threshold for forefoot contacts
            (Task 2, v1.14). Default 0.01 (1 %) — smaller because
            foot_index starts lower for a forefoot striker and the
            lift off ground is less pronounced than for heel contacts.
        clip_strike_pattern: Clip-level strike pattern ('forefoot' |
            'heel' | None). Controls IC refinement signal and window
            (heel → heel y-velocity + forward-biased window; else →
            toe y-velocity + symmetric window). Also controls TO
            dispatch under ``per_strike_pattern``: 'forefoot' → B-B2,
            'heel' → B2c.

    Returns:
        contacts: Same list with refined timing where possible
        diagnostics: Refinement statistics per leg
    """
    if toe_off_method not in VALID_TOE_OFF_METHODS:
        raise ValueError(
            f"toe_off_method must be one of {VALID_TOE_OFF_METHODS}, "
            f"got {toe_off_method!r}"
        )

    if toe_off_method != "ankle_y_lift" and body_height_px is None:
        # Sensible runner-in-4K-frame fallback. Documented above.
        body_height_px = 0.4 * resolution_height
    T = len(landmarks)

    # Landmark indices
    TOES = {"L": LandmarkIndex.LEFT_FOOT_INDEX, "R": LandmarkIndex.RIGHT_FOOT_INDEX}
    HEELS = {"L": LandmarkIndex.LEFT_HEEL, "R": LandmarkIndex.RIGHT_HEEL}
    ANKLES = {"L": LandmarkIndex.LEFT_ANKLE, "R": LandmarkIndex.RIGHT_ANKLE}

    # Velocity thresholds (resolution-adaptive)
    # At 4K: ~1.5 px/frame noise floor
    velocity_threshold = resolution_height * 0.0007
    lift_threshold = -velocity_threshold * 1.5  # More negative for lift detection

    diagnostics: Dict[str, Dict[str, object]] = {"L": {}, "R": {}}

    for leg in ["L", "R"]:
        toe_idx = TOES[leg]
        heel_idx = HEELS[leg]
        ankle_idx = ANKLES[leg]

        # Extract and smooth toe / heel Y positions.
        # When RTMPose landmarks are provided, use COCO-WholeBody big_toe and
        # heel indices instead of BlazePose foot_index / heel.  Score = 0
        # frames (no detection) are NaN-filled before interpolation so the
        # SavGol filter operates on a continuous signal.
        if rtm_landmarks is not None and rtm_scores is not None:
            _rtm_big_toe = _RTM_FOOT[leg]["big_toe"]
            _rtm_heel_idx = _RTM_FOOT[leg]["heel"]
            _raw_toe = rtm_landmarks[:, _rtm_big_toe, 1].astype(float).copy()
            _raw_heel = rtm_landmarks[:, _rtm_heel_idx, 1].astype(float).copy()
            _raw_toe[rtm_scores[:, _rtm_big_toe] <= 0] = np.nan
            _raw_heel[rtm_scores[:, _rtm_heel_idx] <= 0] = np.nan
            toe_y = _interpolate_nans(_raw_toe)
            heel_y = _interpolate_nans(_raw_heel)
            toe_vis = (rtm_scores[:, _rtm_big_toe] > 0).astype(float)
        else:
            toe_y = _interpolate_nans(landmarks[:, toe_idx, 1].copy())
            toe_vis = visibilities[:, toe_idx]
            heel_y = _interpolate_nans(landmarks[:, heel_idx, 1].copy())

        # Extract and smooth ankle Y position (toe-off signal — Fix B-B1,
        # v1.5). On heel strikers the toe wobbles via plantar flexion at
        # mid-stance, false-triggering a toe-velocity threshold ~3-4
        # frames early (01c §11.4.4). The ankle does not rise vertically
        # until the foot is leaving the ground, so it is robust to that
        # mid-stance plantar flexion.
        ankle_y = _interpolate_nans(landmarks[:, ankle_idx, 1].copy())

        # Savitzky-Golay filter preserves sharp impact edges
        window = min(savgol_window, T if T % 2 == 1 else T - 1)
        if window < 3:
            diagnostics[leg]["refinement_skipped"] = "video_too_short"
            continue

        toe_smooth = savgol_filter(
            toe_y, window_length=window, polyorder=savgol_polyorder
        )
        toe_velocity = np.gradient(toe_smooth)

        heel_smooth = savgol_filter(
            heel_y, window_length=window, polyorder=savgol_polyorder
        )
        heel_velocity = np.gradient(heel_smooth)

        ankle_smooth = savgol_filter(
            ankle_y, window_length=window, polyorder=savgol_polyorder
        )
        ankle_velocity = np.gradient(ankle_smooth)

        # Hip-calibrated lift threshold for forefoot B-B2 (v1.16).
        #
        # A rigid-leg constraint (constant Euclidean hip-to-toe distance)
        # gives the toe's vertical liftoff velocity as:
        #
        #   V_toe_y  =  V_hip_x  ×  Δx / Δy
        #
        # where Δx = toe_x − hip_x, Δy = toe_y − hip_y at late stance.
        # The ratio Δx/Δy is the tangent of the leg angle from vertical,
        # typically tan(30°–35°) ≈ 0.55–0.70 for elite forefoot runners at
        # TO.  Using k = 0.55 (conservative end) with the clip-mean hip
        # velocity makes the threshold self-calibrating: faster pace →
        # larger |V_hip_x| → deeper threshold → fires later, matching the
        # higher toe liftoff speed at race pace.  The fixed resolution-based
        # lift_threshold (−2.268 px/frame at 4K/60 fps) corresponds to
        # k ≈ 0.075, far too shallow and pace-blind.
        _K_FOREFOOT_LIFTOFF = 0.55
        hip_x_mid = _interpolate_nans(
            (
                landmarks[:, LandmarkIndex.LEFT_HIP, 0].astype(float)
                + landmarks[:, LandmarkIndex.RIGHT_HIP, 0].astype(float)
            )
            / 2
        )
        hip_x_smooth = savgol_filter(
            hip_x_mid, window_length=window, polyorder=savgol_polyorder
        )
        v_hip_x = np.gradient(hip_x_smooth)
        lift_threshold_forefoot = -_K_FOREFOOT_LIFTOFF * float(
            np.abs(np.nanmean(v_hip_x))
        )

        # Hip-calibrated IC threshold for forefoot Phase A (v1.16).
        # At IC the toe is anterior to the hip (Δx < 0), so the geometric
        # ratio Δx/Δy is negative and the rigid-leg formula predicts
        # +13–14 px/frame.  Active braking by the runner reduces the
        # observed V_toe_y to 4–12 px/frame at true IC, giving an
        # empirical k_ic ≈ 0.40 (vs the geometric 0.45).  The fixed
        # velocity_threshold (1.512 px/frame) fires too late because it
        # only catches the very last moment of deceleration.
        _K_FOREFOOT_IC = 0.40
        velocity_threshold_forefoot_ic = _K_FOREFOOT_IC * float(
            np.abs(np.nanmean(v_hip_x))
        )

        diagnostics[leg]["refinement_method"] = toe_off_method

        refined_count = 0
        skipped_low_vis = 0
        failed_sanity = 0
        # Per-strike-pattern dispatch: count routings per leg.
        strike_counts: Dict[str, int] = {"heel": 0, "forefoot": 0, "unknown": 0}

        for contact in contacts:
            if contact.leg != leg:
                continue

            # ─────────────────────────────────────────────────────────────
            # CHECK TOE VISIBILITY DURING STANCE WINDOW
            # ─────────────────────────────────────────────────────────────
            stance_start = contact.frame
            stance_end = (
                int(contact.toe_off_frame)
                if contact.toe_off_frame
                else contact.frame + 12
            )
            stance_end = min(stance_end, T - 1)

            # Mean toe visibility during stance (when it should be visible)
            stance_toe_vis = float(np.mean(toe_vis[stance_start : stance_end + 1]))

            if stance_toe_vis < min_toe_visibility:
                skipped_low_vis += 1
                continue

            # ─────────────────────────────────────────────────────────────
            # REFINE TOUCHDOWN: falling y-velocity zero-crossing.
            # Signal and search direction dispatched on clip_strike_pattern:
            #   heel     → heel y-velocity, backward search [C_IC-2, C_IC+10]
            #              C_IC fires early (dorsiflexion reversal); LAST
            #              crossing = true heel IC. C_IC floor applied.
            #   forefoot → toe  y-velocity, forward  search [C_IC-5, C_IC+5]
            #              C_IC fires AFTER true IC (load delayed); FIRST
            #              crossing = true forefoot IC. No floor.
            # Fallback: contact.contact_frame (= C_IC) if none found.
            # ─────────────────────────────────────────────────────────────
            refined_touchdown: float = (
                float(contact.contact_frame)
                if contact.contact_frame is not None
                else float(contact.frame)
            )

            if clip_strike_pattern == "heel":
                ic_velocity = heel_velocity
                search_start = max(1, contact.frame - 2)
                search_end = min(T - 1, contact.frame + 10)
                for i in range(search_end, search_start, -1):
                    v_prev = ic_velocity[i - 1]
                    v_curr = ic_velocity[i]
                    if v_prev > velocity_threshold and v_curr <= velocity_threshold:
                        if v_prev != v_curr:
                            fraction = (velocity_threshold - v_prev) / (v_curr - v_prev)
                            refined_touchdown = (i - 1) + fraction
                        else:
                            refined_touchdown = float(i - 1)
                        break
                # C_IC floor: heel coarse detector fires ≤ true IC.
                refined_touchdown = max(refined_touchdown, float(contact.frame))
            else:
                ic_velocity = toe_velocity
                search_start = max(1, contact.frame - 5)
                search_end = min(T - 1, contact.frame + 5)
                # Forward search: first crossing = earliest toe arrest = true IC.
                # No C_IC floor — forefoot C_IC fires AFTER true IC.
                # Hip-calibrated threshold (v1.16): see velocity_threshold_forefoot_ic.
                for i in range(search_start + 1, search_end + 1):
                    v_prev = ic_velocity[i - 1]
                    v_curr = ic_velocity[i]
                    if (
                        v_prev > velocity_threshold_forefoot_ic
                        and v_curr <= velocity_threshold_forefoot_ic
                    ):
                        if v_prev != v_curr:
                            fraction = (velocity_threshold_forefoot_ic - v_prev) / (
                                v_curr - v_prev
                            )
                            refined_touchdown = (i - 1) + fraction
                        else:
                            refined_touchdown = float(i - 1)
                        break

            # ─────────────────────────────────────────────────────────────
            # REFINE TOE-OFF: dispatched on toe_off_method.
            # ─────────────────────────────────────────────────────────────
            refined_toe_off: Optional[float] = None
            if toe_off_method == "ankle_y_lift":
                refined_toe_off = _refine_toe_off_ankle_y_lift(
                    contact, ankle_velocity, lift_threshold, T
                )
            elif toe_off_method == "foot_index_position_lift":
                assert body_height_px is not None
                refined_toe_off = _refine_toe_off_foot_index_position(
                    contact,
                    landmarks,
                    visibilities,
                    body_height_px,
                    fps,
                    delta_lift_frac=delta_lift_frac,
                    min_vis=min_toe_visibility,
                    ic_frame_override=refined_touchdown,
                    rtm_landmarks=rtm_landmarks,
                    rtm_scores=rtm_scores,
                )
                if refined_toe_off is None:
                    # Fallback to Fix B-B1 when foot-index visibility too
                    # low or no clear lift detected.
                    refined_toe_off = _refine_toe_off_ankle_y_lift(
                        contact, ankle_velocity, lift_threshold, T
                    )
            elif toe_off_method == "per_strike_pattern":
                assert body_height_px is not None
                # Clip-level override: skip per-contact classification and
                # route all contacts through the clip's dominant path.
                if clip_strike_pattern in ("forefoot", "heel"):
                    contact.strike_pattern = clip_strike_pattern
                else:
                    contact.strike_pattern = _detect_strike_pattern_for_contact(
                        contact,
                        landmarks,
                        visibilities,
                        body_height_px,
                        tau_strike_frac=tau_strike_frac,
                        min_vis=min_toe_visibility,
                        rtm_landmarks=rtm_landmarks,
                        rtm_scores=rtm_scores,
                    )
                strike_counts[contact.strike_pattern] = (
                    strike_counts.get(contact.strike_pattern, 0) + 1
                )
                if contact.strike_pattern == "heel":
                    # Heel strikers — Fix B2c (delta=0.02).
                    # Phase B (v1.17): use 2.0× stance window instead of 1.5×
                    # so the search covers contacts where true TO falls at the
                    # far edge of the old budget (e.g. Steady_2 #5L).
                    refined_toe_off = _refine_toe_off_foot_index_position(
                        contact,
                        landmarks,
                        visibilities,
                        body_height_px,
                        fps,
                        delta_lift_frac=delta_lift_frac,
                        min_vis=min_toe_visibility,
                        ic_frame_override=refined_touchdown,
                        window_factor=2.0,
                        rtm_landmarks=rtm_landmarks,
                        rtm_scores=rtm_scores,
                    )
                    if refined_toe_off is None:
                        # Phase C (v1.17): anchor on refined IC(S), not coarse TO.
                        refined_toe_off = _refine_toe_off_ankle_y_lift(
                            contact,
                            ankle_velocity,
                            lift_threshold,
                            T,
                            ic_frame_override=refined_touchdown,
                            fps=fps,
                        )
                elif contact.strike_pattern == "forefoot":
                    # Forefoot strikers — Fix B-B2 (foot_index velocity),
                    # hip-calibrated threshold (v1.16).
                    refined_toe_off = _refine_toe_off_foot_index_velocity(
                        contact,
                        toe_velocity,
                        lift_threshold_forefoot,
                        T,
                        ic_frame_override=refined_touchdown,
                        fps=fps,
                    )
                    if refined_toe_off is None:
                        refined_toe_off = _refine_toe_off_ankle_y_lift(
                            contact, ankle_velocity, lift_threshold, T
                        )
                else:
                    # unknown → Fix B-B1.
                    refined_toe_off = _refine_toe_off_ankle_y_lift(
                        contact, ankle_velocity, lift_threshold, T
                    )

            # ─────────────────────────────────────────────────────────────
            # VALIDATE AND APPLY REFINEMENT
            # ─────────────────────────────────────────────────────────────
            if refined_toe_off is None or refined_touchdown is None:
                continue
            refined_gct_ms = (refined_toe_off - refined_touchdown) * (1000.0 / fps)

            # Sanity check: GCT should be 80-400ms for running
            if 80 < refined_gct_ms < 400:
                contact.contact_frame = refined_touchdown
                contact.toe_off_frame = refined_toe_off
                contact.gct_ms = refined_gct_ms
                contact.detection_method = "refined"
                contact.confidence = min(1.0, stance_toe_vis / 0.7)
                refined_count += 1
            else:
                failed_sanity += 1

        diagnostics[leg]["refined_contacts"] = refined_count
        diagnostics[leg]["skipped_low_visibility"] = skipped_low_vis
        diagnostics[leg]["failed_sanity_check"] = failed_sanity
        if toe_off_method == "per_strike_pattern":
            diagnostics[leg]["strike_pattern_counts"] = dict(strike_counts)

    return contacts, diagnostics


def detect_ground_contacts(
    landmarks: np.ndarray,
    visibilities: np.ndarray,
    fps: float,
    resolution_height: int = 2160,
    refine: bool = True,
    min_toe_visibility: float = 0.4,
    toe_off_method: str = "ankle_y_lift",
    body_height_px: Optional[float] = None,
    tau_strike_frac: float = 0.01,
    delta_lift_frac: float = 0.02,
    delta_lift_frac_forefoot: float = 0.01,
    ankle_horiz_vel_gate_frac: float = 0.0,
    clip_strike_pattern: Optional[str] = None,
    prune_spurious: bool = False,
    velocity_ms: Optional[float] = None,
    leg_length_m: Optional[float] = None,
    cadence_band_frac: float = 0.25,
    interpolate_missing: bool = False,
    rtm_landmarks: Optional[np.ndarray] = None,
    rtm_scores: Optional[np.ndarray] = None,
) -> Tuple[List[GroundContact], Dict]:
    """
    Two-pass bilateral ground contact detection.

    Pass 1 (Coarse): Leg compression velocity for both legs
                     → Camera-shake immune, robust stance window detection

    Pass 2 (Fine):   Toe velocity refinement for both legs
                     → Sub-frame precision, visibility-gated

    Args:
        landmarks: (T, 33, 2) pose landmarks
        visibilities: (T, 33) visibility scores
        fps: Video frame rate
        resolution_height: Video height in pixels (for adaptive thresholds)
        refine: Whether to apply toe-based refinement
        min_toe_visibility: Minimum toe visibility to attempt refinement
        toe_off_method: Toe-off detection method (Phase 6c). Default
            ``"ankle_y_lift"`` reproduces v1.5 / v1.7 behaviour byte-
            identical. See refine_contacts_with_toe docstring for the
            other options.
        body_height_px: Body height in pixels. Required for
            non-default toe_off_method values; auto-fallback to
            ``0.4 × resolution_height`` if None.
        tau_strike_frac: Strike-pattern detection threshold (Fix B3).
        delta_lift_frac: Foot-index toe-off lift threshold (Fix B2c).

    Returns:
        contacts: List of GroundContact with timing and GCT
        summary: Detection statistics
    """
    # Pass 1: Coarse detection using leg compression.
    contacts, coarse_diag = detect_contacts_coarse(
        landmarks,
        visibilities,
        fps,
        resolution_height,
        ankle_horiz_vel_gate_frac=ankle_horiz_vel_gate_frac,
        rtm_landmarks=rtm_landmarks,
        rtm_scores=rtm_scores,
    )

    if len(contacts) < 2:
        summary = {
            "total_contacts": len(contacts),
            "detection_failed": True,
            "reason": "Insufficient contacts in coarse pass",
        }
        return contacts, summary

    # Pass 2: Refine with toe velocity (if enabled)
    if refine:
        contacts, refine_diag = refine_contacts(
            contacts,
            landmarks,
            visibilities,
            fps,
            resolution_height=resolution_height,
            min_toe_visibility=min_toe_visibility,
            toe_off_method=toe_off_method,
            body_height_px=body_height_px,
            tau_strike_frac=tau_strike_frac,
            delta_lift_frac=delta_lift_frac,
            delta_lift_frac_forefoot=delta_lift_frac_forefoot,
            clip_strike_pattern=clip_strike_pattern,
            rtm_landmarks=rtm_landmarks,
            rtm_scores=rtm_scores,
        )
    else:
        refine_diag = {"L": {}, "R": {}}

    # Pass 3 (Phase 6d): Prune spurious contacts using REFINED timing.
    # Must run after refinement so contact_frame holds the sub-frame
    # refined IC (not the coarse integer), which is what the gap
    # computation and foot_y measurement need to be accurate.
    prune_diag: Optional[Dict] = None
    if prune_spurious:
        if velocity_ms is None or leg_length_m is None:
            raise ValueError(
                "--prune-spurious-contacts requires velocity_ms and "
                "leg_length_m (pass via detect_ground_contacts)."
            )
        contacts, prune_diag = prune_spurious_contacts(
            contacts,
            landmarks,
            visibilities,
            fps,
            velocity_ms=velocity_ms,
            leg_length_m=leg_length_m,
            cadence_band_frac=cadence_band_frac,
            rtm_landmarks=rtm_landmarks,
            rtm_scores=rtm_scores,
        )

    # Compile summary
    summary = {
        "total_contacts": len(contacts),
        "by_leg": {
            "L": {
                "contacts": sum(1 for c in contacts if c.leg == "L"),
                "refined": sum(
                    1
                    for c in contacts
                    if c.leg == "L" and c.detection_method == "refined"
                ),
                **coarse_diag.get("L", {}),
                **refine_diag.get("L", {}),
            },
            "R": {
                "contacts": sum(1 for c in contacts if c.leg == "R"),
                "refined": sum(
                    1
                    for c in contacts
                    if c.leg == "R" and c.detection_method == "refined"
                ),
                **coarse_diag.get("R", {}),
                **refine_diag.get("R", {}),
            },
        },
        "overall_refined_rate": (
            sum(1 for c in contacts if c.detection_method == "refined") / len(contacts)
            if contacts
            else 0
        ),
    }
    if prune_diag is not None:
        summary["prune_spurious"] = prune_diag

    # Pass 4 (Phase 6d fix #3): pace-grid interpolation of missing contacts.
    # Runs after the prune so interpolation only fills genuine holes, not
    # phantom-adjacent gaps.  A second refinement pass updates IC and TO
    # on the newly-inserted contacts.
    interp_diag: Optional[Dict] = None
    if interpolate_missing:
        if velocity_ms is None or leg_length_m is None:
            raise ValueError(
                "--interpolate-missing-contacts requires velocity_ms and "
                "leg_length_m (pass via detect_ground_contacts)."
            )
        contacts, interp_diag = interpolate_missing_contacts(
            contacts,
            landmarks,
            visibilities,
            fps,
            velocity_ms=velocity_ms,
            leg_length_m=leg_length_m,
            resolution_height=resolution_height,
            cadence_band_frac=cadence_band_frac,
            rtm_landmarks=rtm_landmarks,
            rtm_scores=rtm_scores,
        )
        if interp_diag["n_inserted"] > 0 and refine:
            # Second refinement pass to give interpolated contacts accurate
            # sub-frame IC and TO (uses the same toe_off_method as Pass 2).
            contacts, refine_diag2 = refine_contacts(
                contacts,
                landmarks,
                visibilities,
                fps,
                resolution_height=resolution_height,
                min_toe_visibility=min_toe_visibility,
                toe_off_method=toe_off_method,
                body_height_px=body_height_px,
                tau_strike_frac=tau_strike_frac,
                delta_lift_frac=delta_lift_frac,
                delta_lift_frac_forefoot=delta_lift_frac_forefoot,
                clip_strike_pattern=clip_strike_pattern,
                rtm_landmarks=rtm_landmarks,
                rtm_scores=rtm_scores,
            )
        summary["total_contacts"] = len(contacts)
        summary["interpolate_missing"] = interp_diag

    return contacts, summary


def validate_contact_pattern(contacts: List[GroundContact]) -> Dict:
    """
    Validate that detected contacts follow expected alternating pattern.

    Args:
        contacts: List of detected ground contacts

    Returns:
        validation: Dictionary with pattern analysis
    """
    if len(contacts) < 3:
        return {"valid": False, "reason": "Too few contacts detected"}

    # Check for alternating left/right pattern
    alternating_errors = 0
    for i in range(1, len(contacts)):
        if contacts[i].leg == contacts[i - 1].leg:
            alternating_errors += 1

    # Calculate inter-contact intervals (using refined timing if available)
    def get_contact_time(c):
        return c.contact_frame if c.contact_frame is not None else float(c.frame)

    intervals = [
        get_contact_time(contacts[i + 1]) - get_contact_time(contacts[i])
        for i in range(len(contacts) - 1)
    ]
    interval_cv = np.std(intervals) / np.mean(intervals) if intervals else float("inf")

    # Count refined vs coarse
    n_refined = sum(1 for c in contacts if c.detection_method == "refined")

    return {
        "valid": alternating_errors <= len(contacts) * 0.2,  # Allow 20% errors
        "n_contacts": len(contacts),
        "n_refined": n_refined,
        "n_coarse": len(contacts) - n_refined,
        "alternating_errors": alternating_errors,
        "mean_interval_frames": float(np.mean(intervals)) if intervals else 0,
        "interval_cv": float(interval_cv),
    }


def get_stance_phase_frames(
    contact: GroundContact, phase_start: float = 0.1, phase_end: float = 0.5
) -> Tuple[int, int]:
    """
    Get frame range for a specific portion of stance phase.

    Uses refined timing if available, falls back to coarse.

    Stance phase is divided as:
    - 0.0-0.2: Initial contact / loading
    - 0.2-0.5: Mid-stance (leg most vertical)
    - 0.5-0.8: Terminal stance
    - 0.8-1.0: Pre-swing / toe-off

    Args:
        contact: GroundContact event
        phase_start: Start of phase window (0-1)
        phase_end: End of phase window (0-1)

    Returns:
        start_frame, end_frame: Frame indices for the phase window
    """
    # Use refined timing if available
    contact_time = (
        contact.contact_frame
        if contact.contact_frame is not None
        else float(contact.frame)
    )
    toe_off_time = (
        contact.toe_off_frame
        if contact.toe_off_frame is not None
        else contact_time + 12
    )

    stance_duration = toe_off_time - contact_time

    start_frame = int(contact_time + stance_duration * phase_start)
    end_frame = int(contact_time + stance_duration * phase_end)

    return start_frame, end_frame
