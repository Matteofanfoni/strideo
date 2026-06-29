# app/pages/upload.py
import streamlit as st
import sys
import tempfile
import uuid
import numpy as np
from pathlib import Path
from dataclasses import asdict

project_root = Path(__file__).parent.parent.parent
app_dir = Path(__file__).parent.parent
for p in [str(project_root), str(app_dir)]:
    if p not in sys.path:
        sys.path.insert(0, p)

from ui.styles import (  # noqa: E402
    inject_shared_styles,
    inject_page_css,
    render_navbar,
)
from src.utils.pace_predictor import (  # noqa: E402
    format_clock,
    format_pace_per_km,
    parse_time_to_seconds,
    predict_paces,
)

st.set_page_config(page_title="Upload | strideo.it", page_icon="🏃", layout="wide")

# Ingest guardrail: the seeded pass buffers full-resolution frames
# (~25 MB/frame at 4K), so peak RAM scales with frame count. Cap long uploads to
# bound RAM (and runtime) on the hosted CPU tier. 300 frames ≈ 5 s at 60 fps -
# generous for the 1-3 s clips the protocol targets.
MAX_PIPELINE_FRAMES = 300

inject_shared_styles()

inject_page_css("""
/* File uploader dropzone */
[data-testid="stFileUploader"] section {
  background: linear-gradient(135deg, var(--card) 0%, var(--card-2) 100%) !important;
  border: 2px dashed var(--brand) !important;
  border-radius: var(--radius-lg) !important;
  padding: 26px !important;
  transition: all 0.3s ease !important;
}
[data-testid="stFileUploader"] section:hover {
  border-color: var(--accent) !important;
}
[data-testid="stFileUploaderDropzoneInstructions"] { color: var(--text-muted) !important; }
[data-testid="stFileUploaderDropzoneInstructions"] span { color: var(--text) !important; }

/* Per-clip card (st.container border) */
[data-testid="stVerticalBlockBorderWrapper"] {
  border-radius: var(--radius-lg) !important;
  border-color: #B8C4D8 !important;
}

video {
  border-radius: var(--radius-md) !important;
  border: 1px solid var(--border) !important;
  box-shadow: 0 4px 20px rgba(16,24,40,0.10) !important;
}

.video-preview-label {
  font-size: 0.88rem;
  font-weight: 600;
  color: var(--text-muted);
  text-align: center;
  margin: 0;
  padding: 4px 0 8px;
}

/* Remove button */
.stButton > button:not([kind="primary"]):not(:disabled) {
  background: transparent !important;
  color: var(--text-muted) !important;
  border: 1px solid var(--border) !important;
  border-radius: var(--radius-sm) !important;
  font-size: 0.85rem !important;
  font-weight: 600 !important;
  transition: all 0.3s ease !important;
}
.stButton > button:not([kind="primary"]):not(:disabled):hover {
  background: rgba(244, 67, 54, 0.12) !important;
  color: #F44336 !important;
  border-color: rgba(244, 67, 54, 0.3) !important;
}

.helper-text {
  text-align: center;
  color: var(--text-subtle);
  font-size: 0.95rem;
  margin-top: 12px;
}

.pace-label {
  font-size: 0.82rem;
  font-weight: 600;
  color: var(--text-subtle);
  text-align: center;
  margin: 8px 0 2px;
  letter-spacing: 0.3px;
}

/* Keep number inputs as simple fields - hide the +/- steppers */
[data-testid="stNumberInput"] button { display: none !important; }
/* Hide the cloud glyph inside the upload dropzone (page is kept icon-free) */
[data-testid="stFileUploader"] section svg { display: none !important; }

/* Article typography, matching the recording-guide page */
[data-testid="stMarkdownContainer"] h1 {
  font-size: 2.6rem; font-weight: 800; letter-spacing: -1px;
  color: var(--text); margin: 8px 0 2px;
}
[data-testid="stMarkdownContainer"] p:not([class]) {
  font-size: 1.06rem; line-height: 1.8; color: var(--text-muted);
}
[data-testid="stMarkdownContainer"] strong { color: var(--text); font-weight: 700; }
.section-title {
  color: var(--text);
  font-size: 1.05rem;
  font-weight: 700;
  margin: 24px 0 10px;
  letter-spacing: -0.2px;
}

/* Left-accented box for PB-derived pace targets - matches homepage blockquote */
.prediction-box {
  background: rgba(13,148,136,0.06);
  border-left: 3px solid var(--brand);
  border-radius: 0 8px 8px 0;
  padding: 12px 18px;
  margin: 18px 0 14px;
}
.prediction-box p { margin: 2px 0 !important; }

/* Normalize st.caption to body rhythm */
[data-testid="stCaptionContainer"] p {
  font-size: 0.88rem !important;
  color: var(--text-muted) !important;
  line-height: 1.5 !important;
}
/* Supporting lines under the lead share one size (clean type scale) */
.up-note { font-size: 0.95rem; line-height: 1.6; margin: 8px 0; color: var(--text); }
.up-note.muted { color: var(--text-muted); }
[data-testid="stPageLink"] a p { font-size: 0.95rem !important; }

""")

render_navbar("Upload")

st.markdown("# Upload Videos")

st.markdown(
    "Upload up to three clips of the same runner, one at each effort: "
    "threshold, 1500 m, and 800 m race pace. "
    "Each clip should capture the runner crossing the frame, about 2 to 5 seconds. "
    f"Longer clips are trimmed to the first ~{MAX_PIPELINE_FRAMES // 60} seconds.\n\n"
    "Your clips are analysed on the spot and are never stored or shared."
)

st.page_link(
    "pages/recording_guide.py",
    label="New here? Read the recording guide →",
)

# ─────────────────────────────────────────────────────────────
# Runner info (profile + height for spatial calibration)
# ─────────────────────────────────────────────────────────────

EVENTS = [
    "800 m",
    "1500 m",
    "Mile",
    "3000 m",
    "3000 m steeplechase",
    "5000 m",
    "400 m",
    "10000 m",
    "Other",
]

# Track distances (m) for PB-based pace inference. "Other" has no fixed
# distance, so it cannot seed a prediction.
EVENT_DISTANCES = {
    "800 m": 800,
    "1500 m": 1500,
    "Mile": 1609,
    "3000 m": 3000,
    "3000 m steeplechase": 3000,
    "5000 m": 5000,
    "400 m": 400,
    "10000 m": 10000,
}

st.markdown('<p class="section-title">Runner Information</p>', unsafe_allow_html=True)

r1c1, r1c2, r1c3, r1c4 = st.columns(4)
with r1c1:
    runner_name = st.text_input("Name", placeholder="Optional")
with r1c2:
    runner_age = st.number_input("Age", min_value=10, max_value=100, value=None, step=1)
with r1c3:
    runner_sex = st.selectbox(
        "Sex", ["Male", "Female", "Other"], index=None, placeholder="Select…"
    )
with r1c4:
    runner_height = st.number_input(
        "Height (cm)", min_value=140, max_value=220, value=None, step=1
    )

r2c1, r2c2, r2c3, r2c4 = st.columns(4)
with r2c1:
    primary_event = st.selectbox(
        "Primary event", EVENTS, index=None, placeholder="Select event"
    )
with r2c2:
    primary_pb = st.text_input(
        "Primary PB", placeholder="e.g. 1:58.5", key="pb_primary"
    )
with r2c3:
    secondary_event = st.selectbox(
        "Secondary event", EVENTS, index=None, placeholder="Optional"
    )
with r2c4:
    secondary_pb = st.text_input(
        "Secondary PB", placeholder="e.g. 4:05.2", key="pb_secondary"
    )

# PB → pace inference. Each entered (event, PB) with a known distance and a
# parseable time seeds the predictor; two PBs personalise the curve (profile +
# better target paces). These are filming suggestions only - the measured hip
# velocity reported after analysis stays authoritative.
_pb_pairs = []
for _event, _pb_text in [(primary_event, primary_pb), (secondary_event, secondary_pb)]:
    _dist = EVENT_DISTANCES.get(_event or "")
    _secs = parse_time_to_seconds(_pb_text)
    if _dist and _secs:
        _pb_pairs.append((float(_dist), _secs))

prediction = predict_paces(_pb_pairs) if _pb_pairs else None

if prediction is not None:
    _bits = [f"VDOT ≈ {prediction.vdot:.0f}"]
    if prediction.profile:
        _bits.append(prediction.profile)
    _bits.append(f"confidence: {prediction.confidence}")
    _meta_str = " · ".join(_bits)
    _th = format_pace_per_km(prediction.clip_paces["Threshold"])
    _t15 = format_clock(prediction.race_times["1500m"])
    _p15 = format_pace_per_km(prediction.clip_paces["1500m"])
    _t8 = format_clock(prediction.race_times["800m"])
    _p8 = format_pace_per_km(prediction.clip_paces["800m"])
    st.markdown(
        '<div class="prediction-box">'
        f'<p class="up-note"><strong>Target paces</strong> - {_meta_str}</p>'
        f'<p class="up-note muted">'
        f"Threshold: {_th} &nbsp;&middot;&nbsp; "
        f"1500 m: {_t15} ({_p15}) &nbsp;&middot;&nbsp; "
        f"800 m: {_t8} ({_p8})"
        f"</p></div>",
        unsafe_allow_html=True,
    )
    st.markdown(
        '<p class="up-note muted">Paces are extrapolated from your PB(s) based on the '
        '<a href="https://vdoto2.com" target="_blank">VDOT equivalency system</a> '
        "(Jack Daniels); the threshold estimate is approximate for middle-distance runners. "
        "The pace measured from your video is what is reported - these are only filming "
        "targets for the three paces.</p>",
        unsafe_allow_html=True,
    )

# Profile metadata - height drives spatial calibration; the rest is collected
# for the per-runner record / future report and stored alongside each result.
runner_info = {
    "name": (runner_name or "").strip(),
    "age": int(runner_age) if runner_age is not None else None,
    "sex": runner_sex,
    "primary_event": primary_event,
    "primary_pb": (primary_pb or "").strip(),
    "secondary_event": secondary_event,
    "secondary_pb": (secondary_pb or "").strip(),
    "height_cm": float(runner_height) if runner_height is not None else None,
    "predicted_paces": (
        {
            "vdot": prediction.vdot,
            "profile": prediction.profile,
            "confidence": prediction.confidence,
            "clip_paces_s_per_km": prediction.clip_paces,
            "race_times_s": prediction.race_times,
        }
        if prediction is not None
        else None
    ),
}
st.session_state["runner_info"] = runner_info
st.session_state["predicted_clip_paces"] = (
    prediction.clip_paces if prediction is not None else None
)

# ─────────────────────────────────────────────────────────────
# Upload Cards - session-state driven
# ─────────────────────────────────────────────────────────────

# Per-session temp directory: concurrent visitors on the shared HF container
# must not clobber each other's uploads. Each browser session gets an isolated
# directory under the system temp dir; clips are processed transiently.
if "session_id" not in st.session_state:
    st.session_state["session_id"] = uuid.uuid4().hex[:12]
uploads_dir = (
    Path(tempfile.gettempdir()) / "strideo_sessions" / st.session_state["session_id"]
)
uploads_dir.mkdir(parents=True, exist_ok=True)

MAX_CLIPS = 3
VALID_EXTS = {".mp4", ".mov", ".m4v", ".avi", ".webm"}

# Pace label (shown) → pace_level string understood by the strike classifier
# (src/preprocessing/ground_contact.py) and the NN normaliser. "threshold"
# biases toward a heel prior; "1500m"/"800m" toward forefoot.
PACE_LABELS = ["Threshold", "1500m", "800m"]
PACE_LEVEL_MAP = {"Threshold": "threshold", "1500m": "1500m", "800m": "800m"}

# Shoe type (shown) → (SHOE_TYPES key | None, footwear_category for the strike
# prior). The key drives calibration sole thickness (SHOE_TYPES[key].sole_cm);
# the footwear_category maps onto the classifier's forefoot/heel shoe sets.
SHOE_LABELS = [
    "Trainer",
    "Track spike",
    "Super shoe",
    "Racing flat",
    "Barefoot",
    "Other / unknown",
]
SHOE_TYPE_MAP = {
    "Track spike": ("track_spike", "track_spike"),
    "Trainer": ("training_shoe", "trainer"),
    "Super shoe": ("super_shoe", "super_shoe"),
    "Racing flat": ("racing_flat", "racing_flat"),
    # Barefoot / unknown → no shoe prior. (The classifier treats the literal
    # string "neutral" as a HEEL signal, so pass None for a true no-prior.)
    "Barefoot": ("barefoot", None),
    "Other / unknown": (None, None),
}


def _validate_clip(path: str):
    """Inspect a saved clip; return (ok, error_message, warnings)."""
    warns: list = []
    try:
        import cv2

        cap = cv2.VideoCapture(str(path))
        fps = cap.get(cv2.CAP_PROP_FPS)
        n_frames = cap.get(cv2.CAP_PROP_FRAME_COUNT)
        w = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        h = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
        cap.release()
    except Exception:
        return True, None, warns  # don't block ingest if OpenCV can't read here
    duration = n_frames / fps if fps > 0 else 0.0
    if duration and duration < 1.0:
        return False, f"Too short ({duration:.1f}s) - minimum is 1 second.", warns
    if n_frames > MAX_PIPELINE_FRAMES:
        warns.append(
            f"{duration:.1f}s ({int(n_frames)} frames) - trimmed to the first "
            f"{MAX_PIPELINE_FRAMES} frames (~{MAX_PIPELINE_FRAMES // 60}s @ 60fps)."
        )
    if 0 < fps < 30:
        warns.append(f"Low frame rate ({fps:.0f} fps); 60 fps recommended for GCT.")
    if h > w:
        warns.append("Portrait orientation - landscape is recommended.")
    return True, None, warns


st.divider()

uploaded_files = st.file_uploader(
    "Upload clips",
    type=sorted(e.lstrip(".") for e in VALID_EXTS),
    accept_multiple_files=True,
    key="clip_uploader",
    label_visibility="collapsed",
)

if uploaded_files and len(uploaded_files) > MAX_CLIPS:
    st.warning(
        f"Up to {MAX_CLIPS} clips are analysed at a time - using the first "
        f"{MAX_CLIPS}."
    )
    uploaded_files = uploaded_files[:MAX_CLIPS]

# Default pace per slot (first→Threshold, second→1500m, third→800m).
upload_info = []
for _i, uf in enumerate(uploaded_files or []):
    fid = getattr(uf, "file_id", None) or uf.name
    ext = Path(uf.name).suffix.lower()
    size_mb = uf.size / (1024 * 1024)

    # Persist to the per-session temp dir once; probe metadata once.
    save_path = uploads_dir / f"{_i + 1:02d}_{uf.name}"
    if not save_path.exists():
        save_path.write_bytes(uf.getbuffer())
    meta_key = f"meta::{fid}"
    if meta_key not in st.session_state:
        try:
            from src.preprocessing.frame_rate import probe_video_metadata

            st.session_state[meta_key] = probe_video_metadata(str(save_path))
        except Exception:
            st.session_state[meta_key] = {}
    meta = st.session_state.get(meta_key) or {}

    with st.container(border=True):
        st.markdown(
            f'<p class="video-preview-label">Clip {_i + 1} · {uf.name}</p>',
            unsafe_allow_html=True,
        )
        vid_col, ctrl_col = st.columns([1.4, 1], vertical_alignment="top")
        with vid_col:
            st.video(str(save_path))
        with ctrl_col:
            if ext not in VALID_EXTS:
                st.error(f"Unsupported format: {ext}.")
                continue
            if size_mb > 500:
                st.error(f"File too large ({size_mb:.0f} MB). Max 500 MB.")
                continue
            ok, err, warns = _validate_clip(str(save_path))
            if not ok:
                st.error(err)
                continue

            pace = st.selectbox(
                "Pace",
                PACE_LABELS,
                index=None,
                placeholder="Select pace",
                key=f"pace_{fid}",
            )
            _pred_paces = st.session_state.get("predicted_clip_paces")
            if _pred_paces and pace in _pred_paces:
                st.caption(
                    f"Suggested {pace} pace ≈ {format_pace_per_km(_pred_paces[pace])}"
                )
            shoe = st.selectbox(
                "Shoe type",
                SHOE_LABELS,
                index=None,
                placeholder="Select shoe type",
                key=f"shoe_{fid}",
            )
            _raw_dt = meta.get("creation_time") or ""
            try:
                from datetime import datetime as _dt

                _date_str = (
                    _dt.fromisoformat(_raw_dt.replace("Z", "+00:00")).strftime(
                        "%Y-%m-%d"
                    )
                    if _raw_dt
                    else ""
                )
            except (ValueError, AttributeError):
                _date_str = _raw_dt
            shoot_dt = st.text_input(
                "Shooting date",
                value=_date_str,
                key=f"dt_{fid}",
                placeholder="e.g. 2026-06-22",
            )
            for _w in warns:
                st.caption(_w)

    shoe_key, footwear_category = (
        SHOE_TYPE_MAP[shoe] if shoe is not None else (None, None)
    )
    upload_info.append(
        {
            "path": str(save_path),
            "name": uf.name,
            "pace": pace,
            "pace_level": PACE_LEVEL_MAP[pace] if pace is not None else None,
            "shoe_label": shoe,
            "shoe_type": shoe_key,
            "footwear_category": footwear_category,
            "shooting_datetime": (shoot_dt or "").strip(),
            "source_meta": meta,
        }
    )

num_uploaded = len(upload_info)
clips_ready = num_uploaded > 0 and all(
    r["pace"] is not None and r["shoe_label"] is not None for r in upload_info
)

# ─────────────────────────────────────────────────────────────
# Analyze Button
# ─────────────────────────────────────────────────────────────

st.markdown("<br>", unsafe_allow_html=True)

if num_uploaded == 0:
    st.button(
        "Please upload at least 1 video to continue",
        disabled=True,
        use_container_width=True,
    )
    analyze_clicked = False
elif not clips_ready:
    st.button(
        "Select pace and shoe type for each clip to continue",
        disabled=True,
        use_container_width=True,
    )
    analyze_clicked = False
else:
    analyze_clicked = st.button(
        f"Analyze Running Form ({num_uploaded}/3 videos)",
        type="primary",
        use_container_width=True,
    )

if num_uploaded > 0 and not analyze_clicked:
    label = (
        "All videos uploaded - ready to analyze."
        if num_uploaded == 3
        else f"{num_uploaded}/3 videos uploaded - you can analyze now or add more."
    )
    st.markdown(f'<p class="helper-text">{label}</p>', unsafe_allow_html=True)

# ─────────────────────────────────────────────────────────────
# Pipeline execution
# ─────────────────────────────────────────────────────────────

if analyze_clicked and num_uploaded > 0:
    # v1.20 pipeline path - the same orchestration as
    # scripts/run_prevalidation_single.py (shared via src.preprocessing.pipeline),
    # with RTMPose-x run live on CPU for each fresh upload.
    from src.preprocessing.calibration import SHOE_TYPES
    from src.preprocessing.frame_rate import cap_frames, ensure_cfr
    from src.preprocessing.pipeline import run_clip_pipeline
    from src.preprocessing.rtmpose_extractor import extract_rtmpose_landmarks
    from src.preprocessing.nn_preprocessing import preprocess_for_nn

    st.markdown(
        '<div class="prediction-box"><p class="up-note">Analysis runs on CPU: roughly 8-12 minutes per clip '
        "on Hugging Face Spaces (RTMPose-x pose estimation is the slow step). "
        "Please keep this tab open.</p></div>",
        unsafe_allow_html=True,
    )

    all_results = []

    for vid_idx, vid in enumerate(upload_info):
        video_path = vid["path"]
        pace_label = vid["pace"]
        pace_level = vid["pace_level"]
        video_name = vid["name"]
        shoe_type = vid["shoe_type"]
        footwear_category = vid["footwear_category"]
        shoe_label = vid["shoe_label"]
        shoe_sole_cm = SHOE_TYPES[shoe_type].sole_cm if shoe_type else 2.5
        shooting_datetime = vid["shooting_datetime"]
        source_meta = vid["source_meta"]

        run_num = vid_idx + 1
        st.markdown(
            f'<hr style="border:none;border-top:2px solid var(--border);margin:28px 0 20px;">'
            f'<p style="font-weight:700;font-size:1rem;color:var(--text);margin:0 0 2px;">'
            f"Run {run_num}</p>"
            f'<p style="font-size:0.82rem;color:var(--text-muted);margin:0 0 10px;">'
            f"{video_name} &nbsp;·&nbsp; {pace_label} &nbsp;·&nbsp; {shoe_label}</p>",
            unsafe_allow_html=True,
        )

        progress = st.progress(0, text="Preparing video…")
        metrics_dict = None
        metrics_error = None
        analysis = None

        try:
            # 0. VFR→CFR, then cap to MAX_PIPELINE_FRAMES. The capped CFR file is
            #    fed to BOTH RTMPose and the pipeline so their frame counts stay
            #    aligned (T_rtm == T_blaze) and peak RAM is bounded.
            cfr_path, _ = ensure_cfr(video_path)
            proc_path, was_capped = cap_frames(cfr_path, MAX_PIPELINE_FRAMES)
            if was_capped:
                st.caption(
                    f"Clip trimmed to the first {MAX_PIPELINE_FRAMES} frames "
                    f"(~{MAX_PIPELINE_FRAMES // 60}s at 60 fps) to fit the hosted "
                    "CPU/memory limits."
                )

            # 1. RTMPose-x live (CPU) - mapped to 5-45 % of the bar.
            def _rtm_progress(done, total, _p=progress):
                frac = (done / total) if total else 0.0
                _p.progress(
                    5 + int(frac * 40),
                    text=f"RTMPose pose estimation… frame {done}/{total}",
                )

            rtm = extract_rtmpose_landmarks(
                proc_path, device="cpu", progress=_rtm_progress
            )

            # 2. Full v1.20 pipeline - mapped to 45-100 % of the bar.
            def _pipe_progress(stage, frac, _p=progress):
                _p.progress(45 + int(frac * 55), text=stage)

            analysis = run_clip_pipeline(
                proc_path,
                runner_height_cm=(
                    float(runner_height) if runner_height is not None else 175.0
                ),
                shoe_type=shoe_type,
                shoe_sole_cm=shoe_sole_cm,
                pace_level=pace_level,
                footwear_category=footwear_category,
                rtm_landmarks=rtm.keypoints,
                rtm_scores=rtm.scores,
                progress=_pipe_progress,
            )
            metrics_dict = asdict(analysis.metrics)
        except Exception as e:  # noqa: BLE001 - surface any pipeline failure
            metrics_error = str(e)

        progress.progress(100, text="Complete!")

        # NN-preprocessing diagnostics (display only - no inference in the MVP).
        nn_shape = None
        torso_scale = None
        direction = None
        mean_vis = 0.0
        detection_rate = 0.0
        has_nans = True
        if analysis is not None:
            mean_vis = float(analysis.pose_quality["mean_hip_visibility"])
            detection_rate = float(analysis.pose_quality["detection_rate"]) * 100.0
            try:
                nn_input, _nn_status, nn_meta = preprocess_for_nn(
                    analysis.selected_landmarks,
                    analysis.selected_visibilities,
                    frame_width=analysis.frame_width,
                )
                nn_shape = tuple(nn_input.shape)
                torso_scale = float(
                    nn_meta.get("normalization", {}).get("torso_length_px", 0.0)
                )
                direction = nn_meta.get("transforms", {}).get("direction")
                has_nans = bool(np.isnan(nn_input).any())
            except Exception:
                pass  # diagnostics are best-effort; metrics are the deliverable

        # Result cards - a responsive grid (reflows cleanly on mobile), with
        # strike pattern and detection rate as separate cards.
        if analysis is not None:
            m = analysis.metrics
            strike = (analysis.clip_strike_pattern or "n/a").capitalize()
            cards = [
                ("Cadence", f"{m.cadence_spm:.0f} spm"),
                ("GCT", f"{m.gct_ms:.0f} ms"),
                ("Stride", f"{m.stride_length_m:.2f} m"),
                ("Strike", strike),
                ("Detection Rate", f"{detection_rate:.1f}%"),
            ]
            grid = '<div class="result-grid">'
            for _label, _value in cards:
                grid += (
                    '<div class="result-card">'
                    f'<p class="result-label">{_label}</p>'
                    f'<p class="result-value">{_value}</p></div>'
                )
            grid += "</div>"
            st.markdown(grid, unsafe_allow_html=True)

        if metrics_error:
            st.error(f"Analysis failed: {metrics_error}")

        # Resolution/fps for the Results header strip: prefer the source-file
        # probe; fall back to the analysed clip's values.
        res_w = source_meta.get("width") or (
            analysis.frame_width if analysis is not None else None
        )
        res_h = source_meta.get("height")
        clip_fps = source_meta.get("fps") or (
            analysis.fps if analysis is not None else None
        )

        result_entry = {
            "video_name": video_name,
            "pace_label": pace_label,
            "pace_level": pace_level,
            "shoe_label": shoe_label,
            "shoe_type": shoe_type,
            "footwear_category": footwear_category,
            "shooting_datetime": shooting_datetime,
            "runner_info": st.session_state.get("runner_info"),
            "resolution": ({"width": res_w, "height": res_h} if res_w else None),
            "fps": clip_fps,
            "strike_pattern": (
                analysis.clip_strike_pattern if analysis is not None else None
            ),
            "metrics": metrics_dict,
            "metrics_error": metrics_error,
            "preprocessing": {
                "nn_input_shape": nn_shape,
                "mean_visibility": mean_vis,
                "detection_rate": detection_rate,
                "torso_scale": torso_scale or 0.0,
                "direction": direction,
            },
            # Full ClipAnalysis object (landmarks, contacts, cfr_path, …) kept in
            # session memory so the Results frame scrubber can draw overlays
            # without re-running the pipeline. Not JSON-serialised; session-only.
            "analysis": analysis,
            "success": metrics_dict is not None and not has_nans,
        }
        all_results.append(result_entry)

    # Final summary
    st.markdown("---")
    successes = [r for r in all_results if r["metrics"] is not None]
    if successes:
        st.session_state["analysis_results"] = all_results
        if len(successes) < len(all_results):
            n_failed = len(all_results) - len(successes)
            clip_word = "clip" if n_failed == 1 else "clips"
            st.warning(f"{n_failed} {clip_word} couldn't be analyzed - see above.")
    else:
        st.error("No clips were analyzed successfully. Check the errors above.")

# Persistent "View Results" button. It lives OUTSIDE the analyze block so it is
# re-rendered on the rerun that the click triggers (a button inside the
# analyze-only block would not exist on that rerun, so its click would be lost).
# st.switch_page keeps the session, so the cached analysis survives the hop.
if st.session_state.get("analysis_results"):
    st.markdown("<br>", unsafe_allow_html=True)
    if st.button("View Full Results", type="primary", key="goto_results_persistent"):
        st.switch_page("pages/results.py")
