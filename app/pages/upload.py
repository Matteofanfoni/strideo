# app/pages/upload.py
import streamlit as st
import sys
import tempfile
import uuid
from pathlib import Path

project_root = Path(__file__).parent.parent.parent
app_dir = Path(__file__).parent.parent
for p in [str(project_root), str(app_dir)]:
    if p not in sys.path:
        sys.path.insert(0, p)

from ui.styles import (  # noqa: E402
    inject_shared_styles,
    inject_page_css,
    render_navbar,
    render_footer,
    scroll_to_top,
)
from ui.scrubber import encode_strideonet_video  # noqa: E402
from src.utils.pace_predictor import (  # noqa: E402
    format_clock,
    format_pace_per_km,
    parse_time_to_seconds,
    predict_paces,
)
from pipeline_runner import MAX_PIPELINE_FRAMES  # noqa: E402

st.set_page_config(
    page_title="Upload | strideo.it",
    page_icon=str(app_dir / "assets" / "favicon.png"),
    layout="wide",
)

inject_shared_styles()

inject_page_css("""
/* ── Page header ── */
.up-title{
  font-family: 'Space Grotesk', ui-sans-serif, system-ui, sans-serif;
  font-size: clamp(2.1rem, 3.5vw, 3.1rem) !important;
  font-weight: 700;
  letter-spacing: -0.035em;
  line-height: 1.04 !important;
  color: var(--text);
  margin: 16px 0 0 !important;
  padding: 0 !important;
}
.up-lead{
  font-size: 1.08rem;
  line-height: 1.68;
  color: var(--text-muted);
  margin: 20px 0 0 !important;
}
.up-lead strong{ color: var(--text); font-weight: 600; }
/* 250x48px matches "Upload clips"/"Analyse a clip" - this button never had
   use_container_width set, so its container was shrink-wrapping to the
   button's own content width regardless of the shared width:100% button CSS
   (which only fills a container, it can't grow one). Now stretched via the
   column + use_container_width pair, then capped to the same size. */
.st-key-goto_results_persistent{ max-width: 250px; }
/* Font-size/line-height match .section-h2 (home page's "How it's validated"
   heading) exactly - same visual weight for a heading of this rank. */
.up-h2{
  font-family: 'Space Grotesk', ui-sans-serif, system-ui, sans-serif;
  font-size: 2.15rem !important;
  font-weight: 700;
  letter-spacing: -0.025em;
  line-height: 1.14 !important;
  color: var(--text);
  margin: 16px 0 0 !important;
  padding: 0 !important;
}
/* A real gap between this heading and the upload button below (they read as
   one cramped block otherwise). No TOP margin: since L1 the heading sits
   inside the clips band, whose own padding already provides that space, and
   the two stacked to 136px of dead height before the words "Add up to three
   clips" (measured 2026-08-31, user: too much scrolling). */
.clips-head{ margin: 0 0 22px; }

/* ── Cards ──
   Both the runner-information panel and each per-clip panel are
   st.container(border=True). Streamlit 1.54 puts that border directly on the
   container's own stVerticalBlock (there is no stVerticalBlockBorderWrapper any
   more, and no attribute marking a block as bordered), so each card is matched
   by the heading it opens with - a direct-child element container, which cannot
   match an outer block by accident. The legacy wrapper selector is kept so the
   cards still get the right surface on an older Streamlit. */
[data-testid="stVerticalBlock"]:has(> [data-testid="stElementContainer"] .card-h),
[data-testid="stVerticalBlock"]:has(> [data-testid="stElementContainer"] .video-preview-label),
[data-testid="stVerticalBlockBorderWrapper"]{
  background: var(--card) !important;
  /* var(--border)'s default 1px renders as a 0.667px sub-pixel line on this
     browser's device pixel ratio (confirmed via getComputedStyle) - crisp on
     some edges, near-invisible on others. A wider, fully-opaque border reads
     as a solid, even line on every side regardless of that rounding. */
  border: 1.5px solid rgba(217, 222, 229, 0.95) !important;
  border-radius: var(--radius-lg) !important;
  padding: 22px 24px !important;
}
/* .card-h marks an st.container card (and is what the rule above matches on);
   .aside-h is the identical heading for the HTML-only aside cards, kept under a
   separate name so those cards do not get wrapped in a card surface as well. */
.card-h, .aside-h{
  font-family: 'Space Grotesk', ui-sans-serif, system-ui, sans-serif;
  font-size: 1rem !important;
  font-weight: 600;
  color: var(--text);
  margin: 0 0 14px !important;
  padding: 0 !important;
}

/* ── File uploader: reduced to a single button ──
   The dashed dropzone, its "Drag and drop files here" copy and the size/format
   line are all hidden, leaving Streamlit's own browse button - which is still a
   drop target, so drag-and-drop keeps working, it is just no longer advertised.
   The button's label is a bare text node inside <button>, so it cannot be
   replaced by a child selector: font-size:0 collapses it and ::after supplies
   "Upload clips" instead. Pinned to 250x48px - the hero "Analyse a clip"
   button's own measured size (getBoundingClientRect), not this element's
   native content-driven sizing. */
[data-testid="stFileUploader"] section {
  background: transparent !important;
  border: none !important;
  padding: 0 !important;
  min-height: 0 !important;
  justify-content: flex-start !important;
}
[data-testid="stFileUploaderDropzoneInstructions"] { display: none !important; }
[data-testid="stFileUploader"] section [data-testid="stBaseButton-secondary"] {
  background: var(--navy) !important;
  color: var(--navy-foreground) !important;
  border: none !important;
  border-radius: var(--radius-md) !important;
  padding: 0 !important;
  width: 250px !important;
  height: 48px !important;
  display: flex !important;
  align-items: center !important;
  justify-content: center !important;
  font-size: 0 !important;
  box-shadow: none !important;
  transition: background 0.2s ease !important;
}
[data-testid="stFileUploader"] section
  [data-testid="stBaseButton-secondary"]::after {
  content: "Upload clips";
  font-family: 'Space Grotesk', sans-serif;
  font-size: 0.95rem;
  font-weight: 600;
  letter-spacing: 0.3px;
}
[data-testid="stFileUploader"] section
  [data-testid="stBaseButton-secondary"]:hover {
  background: color-mix(in srgb, var(--navy) 88%, white 12%) !important;
}

video {
  border-radius: var(--radius-md) !important;
  border: 1px solid var(--border) !important;
  box-shadow: none !important;
}

/* Mono, but not uppercased - it carries a filename. */
.video-preview-label {
  font-family: 'JetBrains Mono', ui-monospace, monospace;
  font-size: 0.72rem;
  letter-spacing: 0.1em;
  color: var(--text-muted);
  margin: 0 !important;
  padding: 0 0 12px;
}

/* Secondary / destructive buttons (e.g. a per-clip remove action) read as
   outline buttons rather than the navy primary fill. */
.stButton > button:not([kind="primary"]):not(:disabled) {
  background: var(--card) !important;
  color: var(--text-muted) !important;
  border: 1px solid var(--border) !important;
  border-radius: var(--radius-md) !important;
  font-size: 0.9rem !important;
  font-weight: 600 !important;
}
.stButton > button:not([kind="primary"]):not(:disabled):hover {
  background: rgba(218,69,40,0.10) !important;
  color: var(--red) !important;
  border-color: rgba(218,69,40,0.3) !important;
  transform: none !important;
}

.helper-text {
  text-align: left;
  color: var(--text-subtle);
  font-family: 'JetBrains Mono', ui-monospace, monospace;
  font-size: 0.7rem;
  text-transform: uppercase;
  letter-spacing: 0.14em;
  margin-top: 14px !important;
}

/* Keep number inputs as simple fields - hide the +/- steppers */
[data-testid="stNumberInput"] button { display: none !important; }
/* Number inputs wrap the actual field in an extra stNumberInputContainer div
   (for the now-hidden steppers) that carries its own near-invisible border
   (rgb(240,242,246), almost the page background) around the real input box -
   confirmed live via getComputedStyle, and absent on stTextInput, which is
   why Age/Height read as borderless next to Name/Sex despite the inner field
   itself having the identical border. Drop the redundant outer one so only
   the real input border remains, same as every other field. */
[data-testid="stNumberInputContainer"] { border: none !important; }
/* REVERTED (broke the widget): `[data-testid="stNumberInputContainer"] >
   *:not(input) { display: none }` hid Age/Height entirely, which means the
   real <input> isn't a direct child of that container - the rule's
   `:not(input)` matched the wrapper div the input actually lives inside,
   hiding it along with everything else. Not attempting another blind guess
   here without real DOM inspection; back to just the ::-ms-clear attempt
   below, which is at least additive/safe even if it doesn't fully fix the
   tab order (the circled "x" that eats a Tab press once the field has a
   value is still unresolved). */
input::-ms-clear, input::-ms-reveal { display: none; width: 0; height: 0; }

/* ── Analysis run sections ── */
.run-head{
  border-top: 1px solid var(--border);
  margin: 30px 0 0;
  padding-top: 22px;
}
.run-title{
  font-family: 'Space Grotesk', ui-sans-serif, system-ui, sans-serif;
  font-size: 1.05rem;
  font-weight: 600;
  color: var(--text);
  margin: 0 !important;
}
.run-meta{
  font-family: 'JetBrains Mono', ui-monospace, monospace;
  font-size: 0.7rem;
  text-transform: uppercase;
  letter-spacing: 0.14em;
  color: var(--text-muted);
  margin: 6px 0 12px !important;
}

/* Tinted callout used for PB-derived pace targets and runtime notices - same
   language as the shared .note-card. */
.prediction-box {
  background: var(--accent-tint);
  border: 1px solid var(--border);
  border-left: 3px solid var(--brand);
  border-radius: var(--radius-md);
  padding: 14px 18px;
  margin: 18px 0 14px;
}
.prediction-box p { margin: 2px 0 !important; }

/* Normalize st.caption to body rhythm */
[data-testid="stCaptionContainer"] p {
  font-size: 0.85rem !important;
  color: var(--text-muted) !important;
  line-height: 1.55 !important;
}
/* Supporting lines under the lead share one size (clean type scale) */
.up-note { font-size: 0.92rem; line-height: 1.6; margin: 8px 0 !important; color: var(--text); }
.up-note.muted { color: var(--text-muted); }
.up-note a { color: var(--brand-text); font-weight: 600; text-decoration: none; }
.up-note a:hover { text-decoration: underline; }
[data-testid="stPageLink"] a p { font-size: 0.92rem !important; }

/* L1 (2026-08-30): Upload's two zones. Header + Runner information stay on
   the page's default ground; everything from "Add up to three clips" through
   the analyze CTA is one continuous lightened band, the same two-zone shape
   the Guide page settled on. Same full-bleed values as the shared .band and
   as the Guide's own .capture-band-mark; a marker span plus :has() is used
   because the region contains real widgets (the uploader, the selects, the
   button), which cannot be nested inside a raw .band div. Per L1's own
   finding, this overlay reads LIGHTER than the default ground, so this is
   the lighter of Upload's two zones. */
[data-testid="stElementContainer"]:has(.up-band-mark){ display: none; }
[data-testid="stVerticalBlock"]:has(> [data-testid="stElementContainer"] .up-band-mark){
  margin: 24px calc(50% - 50vw) 0;
  width: 100vw;
  max-width: 100vw;
  box-sizing: border-box;
  padding: 34px max(20px, calc(50vw - 50%));
  border-top: 1px solid var(--border);
  border-bottom: 1px solid var(--border);
  background: color-mix(in srgb, var(--card) 60%, transparent);
}
@media (max-width: 700px){
  [data-testid="stVerticalBlock"]:has(> [data-testid="stElementContainer"] .up-band-mark){
    margin: 18px calc(50% - 50vw) 0;
    padding: 26px 20px;
  }
}
""")

render_navbar("Upload")

# Mirrors results.py's own RU2-era fix: switch_page is a client-side
# re-render, not a real navigation, so this page would otherwise inherit
# whatever scroll position the previous page (often Results, scrolled deep)
# was at. Only consumed on arrival via the flag set at each switch_page
# call site, not unconditionally - Upload's own widgets (pace/shoe selects,
# runner-info fields) trigger reruns on every interaction, and scrolling to
# top on those too would be jarring.
if st.session_state.pop("scroll_upload_top", False):
    scroll_to_top()

st.markdown(
    """
<p class="eyebrow">step 1 of 2</p>
<h1 class="up-title">Upload videos</h1>
<p class="up-lead">Up to three clips of the same runner, one at each effort:
<strong>threshold, 1500 m and 800 m</strong> race pace. Each clip should capture
the runner crossing the frame, about 2 to 5 seconds. <strong>Standing
height</strong> is the calibration reference behind every distance
measurement, so enter it as accurately as you can.</p>
""",
    unsafe_allow_html=True,
)

st.markdown("<br>", unsafe_allow_html=True)

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

with st.container(border=True):
    st.markdown(
        '<p class="card-h">Runner information</p>',
        unsafe_allow_html=True,
    )
    # Name gets extra width - a full first + last name is common and was
    # getting cramped in an equal quarter-width column (user feedback).
    r1c1, r1c2, r1c3, r1c4 = st.columns([1.6, 1, 1, 1])
    with r1c1:
        runner_name = st.text_input("Name", placeholder="Optional")
    with r1c2:
        runner_age = st.number_input(
            "Age", min_value=10, max_value=100, value=None, step=1
        )
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

    # PB → pace inference. Each entered (event, PB) with a known distance and
    # a parseable time seeds the predictor; two PBs personalise the curve
    # (profile + better target paces). These are filming suggestions only -
    # the measured hip velocity reported after analysis stays authoritative.
    _pb_pairs = []
    for _event, _pb_text in [
        (primary_event, primary_pb),
        (secondary_event, secondary_pb),
    ]:
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
            '<p class="up-note muted">Paces are extrapolated from your PB(s) '
            'based on the <a href="https://vdoto2.com" target="_blank">VDOT '
            "equivalency system</a> (Jack Daniels); the threshold estimate is "
            "approximate for middle-distance runners. The pace measured from "
            "your video is what is reported - these are only filming targets "
            "for the three paces.</p>",
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

# Shoe type (shown) → (SHOE_TYPES key | None, footwear_category). The key
# drives calibration sole thickness (SHOE_TYPES[key].sole_cm) and is the
# load-bearing half: it sets the spatial-scale denominator, so it moves
# stride length, velocity and oscillation.
#
# The footwear_category half used to map onto the strike classifier's
# forefoot/heel shoe sets. Those priors were deleted (the shoe prior measured
# 40.0% -- below chance, and a constant on our cohort since every runner
# wore spikes), so nothing in the app consumes this value any more. It is
# still recorded with the result as provenance: this dropdown is the only
# place the runner's footwear is captured. Note the CLI keeps a *separate,
# live* use of the same field name -- `FOOTWEAR_CATEGORY_TO_SHOE_TYPE` in
# `run_prevalidation_single.py` derives `shoe_type` from it, and that path
# feeds calibration. Do not delete that on the strength of this comment.
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
    # This read `0 < fps < 30`, so exactly 30 fps - the commonest
    # phone default - passed with no warning at all. The bar is now the
    # pipeline's actual 60 fps working assumption, with a 1% tolerance to match
    # ensure_cfr's. A slow clip is resampled to 60 fps on ingest rather than
    # being analysed as if it were already there, but resampling cannot invent
    # timing the camera never recorded, so the warning stays: GCT precision is
    # bounded by the source rate (one frame is 33 ms at 30 fps, against a 10 ms
    # target). Faster sources need no warning; they downsample cleanly.
    if 0 < fps < 59.4:
        warns.append(
            f"{fps:.0f} fps source - it will be converted to 60 fps for "
            "analysis, but ground contact time stays limited by the original "
            "frame rate. Record at 60 fps for the most accurate result."
        )
    if h > w:
        warns.append("Portrait orientation - landscape is recommended.")
    return True, None, warns


with st.container():
    st.markdown('<span class="up-band-mark"></span>', unsafe_allow_html=True)
    st.markdown(
        '<div class="clips-head"><p class="eyebrow">your clips</p>'
        '<h2 class="up-h2">Add up to three clips</h2></div>',
        unsafe_allow_html=True,
    )

    uploaded_files = st.file_uploader(
        "Upload clips",
        type=sorted(e.lstrip(".") for e in VALID_EXTS),
        accept_multiple_files=True,
        key="clip_uploader",
        label_visibility="collapsed",
    )

    st.markdown(
        '<p class="up-note muted" style="margin-bottom:24px !important;">'
        "Clips are analysed on the spot and are never "
        'stored or shared - see the <a href="/privacy_policy" target="_blank">'
        "Privacy Policy</a> for details.</p>",
        unsafe_allow_html=True,
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
                f'<p class="video-preview-label">Clip {_i + 1} · {uf.name} · '
                f"{size_mb:.1f} MB</p>",
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

    # A disabled st.button() here used to render the "can't proceed yet" reason,
    # which looked like an inert copy of the real button below it rather than a
    # status message. Plain text (same .helper-text style as the "ready" status
    # line further down) reads as a notice instead.
    if num_uploaded == 0:
        st.markdown(
            '<p class="helper-text">Please upload at least one clip to '
            "continue.</p>",
            unsafe_allow_html=True,
        )
        analyze_clicked = False
    elif not clips_ready:
        st.markdown(
            '<p class="helper-text">Select pace and shoe type for each clip to '
            "continue.</p>",
            unsafe_allow_html=True,
        )
        analyze_clicked = False
    else:
        # Constrained to a button-sized column rather than the full page width,
        # which read as a banner once the page moved to the new card layout.
        _action_col, _ = st.columns([1, 1.9])
        with _action_col:
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
# Output slots + footer (both precede the pipeline run below)
# ─────────────────────────────────────────────────────────────

# Streamlit renders in call order, and the analysis below takes minutes. If
# render_footer() were called after it, the footer would not exist in the DOM
# for the whole run (it only reappears when the script finishes). Reserving the
# container here and filling it afterwards keeps the footer on the page
# throughout, with the analysis output above it where it belongs.
analysis_area = st.container()

render_footer()

# ─────────────────────────────────────────────────────────────
# Pipeline execution
# ─────────────────────────────────────────────────────────────

if analyze_clicked and num_uploaded > 0:
    # Written into the slot reserved above so the footer, already rendered,
    # stays on the page for the whole run instead of vanishing until it ends.
    with analysis_area:
        # Fast path by default: BlazePose only, through the shipped
        # FiLM CNN (ONNX). RTMPose + the classical contact-detection pipeline
        # only run later, on request, via results.py's "Run Full Analysis".
        from src.preprocessing.calibration import SHOE_TYPES
        from pipeline_runner import run_fast_path

        st.markdown(
            '<div class="prediction-box"><p class="up-note">'
            "StrideoNet is reading your video's pose data. Please keep this "
            "tab open.</p></div>",
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
                f'<div class="run-head"><p class="run-title">Run {run_num}</p>'
                f'<p class="run-meta">{video_name} · {pace_label} · {shoe_label}</p>'
                "</div>",
                unsafe_allow_html=True,
            )

            progress = st.progress(0, text="Preparing video… (0%)")
            fast_result = None
            metrics_error = None

            # The pipeline now reports per frame rather than per stage, so the
            # bar creeps instead of sitting still for 13.5 s at a time. That is
            # ~600 callbacks a clip, so only push a rerender when the rendered
            # text would actually change (integer percent or stage label) -
            # ~50 websocket messages instead of 600, same visible motion.
            _last_render = {"pct": -1, "stage": ""}

            def _fast_progress(stage, frac, _p=progress, _last=_last_render):
                pct = min(int(frac * 100), 100)
                if pct == _last["pct"] and stage == _last["stage"]:
                    return
                _last["pct"], _last["stage"] = pct, stage
                _p.progress(pct, text=f"{stage} ({pct}%)")

            try:
                fast_result = run_fast_path(
                    video_path,
                    runner_height_cm=(
                        float(runner_height) if runner_height is not None else 175.0
                    ),
                    shoe_type=shoe_type,
                    shoe_sole_cm=shoe_sole_cm,
                    pace_level=pace_level,
                    progress=_fast_progress,
                )
            except Exception as e:  # noqa: BLE001 - surface any pipeline failure
                metrics_error = str(e)

            # Made eager per user feedback 2026-08-27): the annotated
            # video ships already-built alongside the rest of this clip's
            # results, not behind an on-demand button on the Results page.
            # A failure here must not fail the whole clip - it's a bonus
            # visual, not a metric.
            annotated_video_fast = None
            if fast_result is not None:
                # run_fast_path stops at 90%; the encode owns the last tenth,
                # which matches its measured share of the wall-clock (~3 s of
                # ~30 s, against 13.5 s for each of the two pose passes).
                _fast_progress("Building annotated video…", 0.90)
                try:
                    annotated_video_fast = encode_strideonet_video(
                        fast_result["overlay"],
                        fast_result["metrics"],
                        fast_result.get("preprocessing"),
                        progress=lambda f: _fast_progress(
                            "Building annotated video…", 0.90 + 0.10 * f
                        ),
                    )
                except Exception:  # noqa: BLE001 - video is a bonus, not core
                    annotated_video_fast = None

            progress.progress(100, text="Complete! (100%)")

            metrics_dict = fast_result["metrics"] if fast_result else None
            preprocessing = (
                fast_result["preprocessing"]
                if fast_result
                else {
                    "nn_input_shape": None,
                    "mean_visibility": 0.0,
                    "detection_rate": 0.0,
                    "torso_scale": 0.0,
                    "direction": None,
                }
            )
            # No per-clip result-card grid here any more - metrics_dict and
            # preprocessing (Cadence/GCT/Stride/Detection Rate and friends)
            # are shown once, on the Results page's own Calculated Metrics /
            # Pipeline Diagnostics sections, not duplicated here before RU2's
            # immediate switch_page (user feedback: redundant, briefly seen).

            if metrics_error:
                st.error(f"Analysis failed: {metrics_error}")

            # Resolution/fps for the Results header strip: prefer the
            # source-file probe (no pipeline object to fall back to any more
            # - the classical ClipAnalysis only exists once Full analysis runs).
            res_w = source_meta.get("width")
            res_h = source_meta.get("height")
            clip_fps = source_meta.get("fps")

            result_entry = {
                "video_name": video_name,
                "video_path": video_path,
                "pace_label": pace_label,
                "pace_level": pace_level,
                "shoe_label": shoe_label,
                "shoe_type": shoe_type,
                "shoe_sole_cm": shoe_sole_cm,
                "footwear_category": footwear_category,
                "shooting_datetime": shooting_datetime,
                "runner_info": st.session_state.get("runner_info"),
                "resolution": ({"width": res_w, "height": res_h} if res_w else None),
                "fps": clip_fps,
                "metrics_fast": metrics_dict,
                "metrics_error": metrics_error,
                "preprocessing_fast": preprocessing,
                # StrideoNet's own annotated video (skeleton + metrics HUD,
                # no contacts), already-encoded MP4 bytes - built eagerly
                # above, always available, unlike the full-analysis fields
                # below which are opt-in via the Deterministic Kinematics
                # Engine. None if encoding failed.
                "annotated_video_fast": annotated_video_fast,
                # Populated later, per-clip, by results.py's "Run Full
                # Analysis" (the classical pipeline is opt-in, not
                # automatic). None until then.
                "metrics_full": None,
                "preprocessing_full": None,
                "strike_pattern_full": None,
                "analysis_full": None,
                "success": metrics_dict is not None,
            }
            all_results.append(result_entry)

        # Final summary
        st.markdown("---")
        successes = [r for r in all_results if r["metrics_fast"] is not None]
        if successes:
            st.session_state["analysis_results"] = all_results
            if len(successes) < len(all_results):
                n_failed = len(all_results) - len(successes)
                clip_word = "clip" if n_failed == 1 else "clips"
                st.warning(f"{n_failed} {clip_word} couldn't be analyzed - see above.")
            # RU2: go straight to the results page - no extra click. The
            # warning above (if any) still renders in this run before the
            # redirect fires. Flag a one-shot scroll reset - switch_page is
            # a client-side re-render, not a real navigation, so Results
            # would otherwise open still scrolled down from Upload's own
            # progress bars.
            st.session_state["scroll_results_top"] = True
            st.switch_page("pages/results.py")
        else:
            st.error("No clips were analyzed successfully. Check the errors above.")
