# app/pages/results.py
import streamlit as st
import sys
import math
from pathlib import Path

app_dir = Path(__file__).parent.parent
project_root = Path(__file__).parent.parent.parent
for _p in (str(project_root), str(app_dir)):
    if _p not in sys.path:
        sys.path.insert(0, _p)

from ui.styles import (  # noqa: E402
    inject_shared_styles,
    inject_page_css,
    render_navbar,
    render_page_header,
    render_empty_state,
)
from ui.scrubber import render_gc_scrubber  # noqa: E402
from src.utils.pace_predictor import format_pace_per_km  # noqa: E402

st.set_page_config(page_title="Results | strideo.it", page_icon="🏃", layout="wide")

inject_shared_styles()

inject_page_css("""

/* Quality bar */
.quality-bar{
  height: 6px;
  background: var(--hover-fill);
  border-radius: 3px;
  overflow: hidden;
  margin-top: 6px;
}
.quality-fill{
  height: 100%;
  border-radius: 3px;
  transition: width 0.6s ease;
}

/* Comparison table */
.compare-table{
  width: 100%;
  border-collapse: separate;
  border-spacing: 0;
  margin-top: 16px;
}
.compare-table th{
  background: rgba(13, 148, 136, 0.12);
  color: var(--orange);
  font-size: 0.82rem;
  font-weight: 700;
  text-transform: uppercase;
  letter-spacing: 0.5px;
  padding: 12px 16px;
  text-align: left;
  border-bottom: 1px solid var(--border);
}
.compare-table th:first-child{ border-radius: var(--radius-sm) 0 0 0; }
.compare-table th:last-child{ border-radius: 0 var(--radius-sm) 0 0; }

.compare-table td{
  padding: 12px 16px;
  font-size: 0.92rem;
  color: var(--text-white);
  border-bottom: 1px solid var(--border);
}
.compare-table tr:hover td{
  background: var(--hover-fill);
}
.compare-table .pace-col{
  color: var(--orange);
  font-weight: 700;
}

[data-testid="stHorizontalBlock"]{
  align-items: stretch !important;
}
[data-testid="stColumn"] {
  display: flex !important;
}
[data-testid="stColumn"] > div {
  flex: 1 !important;
  display: flex !important;
  flex-direction: column !important;
}

/* Per-clip header strip */
.hdr-strip{
  display: flex;
  flex-wrap: wrap;
  gap: 10px;
  margin: 4px 0 20px;
}
.hdr-pill{
  display: flex;
  flex-direction: column;
  gap: 2px;
  background: linear-gradient(135deg, var(--bg-card) 0%, var(--card-2) 100%);
  border: 1px solid var(--border);
  border-radius: var(--radius-sm);
  padding: 8px 14px;
  min-width: 110px;
}
.hdr-pill-label{
  font-size: 0.68rem;
  font-weight: 600;
  color: var(--text-subtle);
  text-transform: uppercase;
  letter-spacing: 0.6px;
}
.hdr-pill-value{
  font-size: 0.95rem;
  font-weight: 700;
  color: var(--text-white);
}

/* Runner profile banner - flat blockquote (no rounded box) */
.profile-note{
  border-left: 3px solid var(--brand);
  background: rgba(13,148,136,0.05);
  padding: 10px 18px;
  margin: 0 0 22px;
  font-size: 0.95rem;
  color: var(--text);
  line-height: 1.7;
}
.profile-paces{ font-weight: 700; color: var(--brand); }
.profile-cav{ color: var(--text-subtle); font-size: 0.82rem; font-weight: 400; }

/* Section heading: teal rule sits ABOVE the label */
.section-heading{
  color: var(--orange);
  font-size: 1rem;
  font-weight: 700;
  margin: 32px 0 8px;
  padding-top: 12px;
  border-top: 1px solid rgba(13, 148, 136,0.25);
}
/* Dark heading variant — same border-top, dark text (used for Runner Information) */
.heading-dark{
  color: var(--text);
  font-size: 0.95rem;
  font-weight: 700;
  margin: 32px 0 8px;
  padding-top: 12px;
  border-top: 1px solid rgba(13, 148, 136,0.25);
}
.section-sub{
  color: var(--text-muted);
  font-size: 0.88rem;
  margin: 0 0 12px;
}

/* Scrubber frame image */
[data-testid="stImage"] img{
  border-radius: var(--radius-md);
  border: 1px solid var(--border);
  box-shadow: 0 4px 20px rgba(16,24,40,0.10);
}

/* ── Flat stat grid (replaces the bulky metric cards) ── */
.stat-grid{
  display: grid;
  grid-template-columns: repeat(auto-fit, minmax(120px, 1fr));
  gap: 6px 32px;
  margin: 10px 0 0;
}
.stat{ padding: 8px 0; }
.stat-label{
  font-size: 0.72rem; font-weight: 600; text-transform: uppercase;
  letter-spacing: 0.6px; color: var(--text-subtle); margin: 0;
}
.stat-value{
  font-size: 1.9rem; font-weight: 800; color: var(--text);
  margin: 2px 0 0; letter-spacing: -0.5px; line-height: 1.1;
}
.stat-unit{ font-size: 0.8rem; color: var(--text-muted); }
.stat-sub{ font-size: 0.72rem; color: var(--text-subtle); margin: 2px 0 0; }
.stat-flag{ color: #D97706; font-weight: 600; }
.derived-line{
  color: var(--text-muted); font-size: 0.92rem;
  margin: 14px 0 0; border-top: 1px solid var(--border); padding-top: 12px;
}
.derived-line strong{ color: var(--text); font-weight: 700; }

/* ── Target vs actual pace - flat blockquote (no rounded box) ── */
.pace-compare{
  border-left: 3px solid var(--brand);
  background: rgba(13,148,136,0.05);
  padding: 10px 18px; margin: 16px 0 0;
  font-size: 0.95rem; color: var(--text);
}
.pace-compare .pc-delta{ font-weight: 700; }
.pace-compare .pc-faster{ color: #DC2626; }
.pace-compare .pc-slower{ color: #2563EB; }
.pace-compare .pc-onpace{ color: #16A34A; }

/* ── Elite range bars ── */
.range-row{
  display: grid; grid-template-columns: 130px 1fr 140px;
  align-items: center; gap: 14px; margin: 12px 0;
}
.range-name{ font-weight: 700; font-size: 0.9rem; color: var(--text); }
.range-name .rn-sub{ font-weight: 400; color: var(--text-subtle); font-size: 0.78rem; }
.range-track{
  position: relative; height: 10px; background: var(--hover-fill);
  border-radius: 5px;
}
.range-band{
  position: absolute; top: 0; bottom: 0;
  background: rgba(13,148,136,0.30); border-radius: 5px;
}
.range-marker{
  position: absolute; top: -4px; width: 4px; height: 18px;
  background: #7C3AED; border-radius: 2px; transform: translateX(-2px);
}
.range-verdict{ font-size: 0.82rem; color: var(--text-muted); text-align: right; }
.range-scale{
  display: flex; justify-content: space-between;
  font-size: 0.66rem; color: var(--text-subtle); margin-top: 2px;
}
.range-sources{ font-size: 0.78rem; color: var(--text-subtle); margin: 14px 0 0; line-height: 1.6; }
.range-sources a{ color: var(--brand); }

/* ── Compact diagnostics list ── */
.diag-list{
  display: grid; grid-template-columns: repeat(auto-fit, minmax(220px, 1fr));
  gap: 0 32px;
}
.diag-item{
  display: flex; justify-content: space-between; gap: 12px;
  border-bottom: 1px solid var(--border-soft); padding: 7px 0; font-size: 0.88rem;
}
.diag-k{ color: var(--text-subtle); }
.diag-v{ color: var(--text); font-weight: 600; text-align: right; }

/* Clip filename label inside each tab */
.clip-filename{
  font-size: 0.95rem; font-weight: 700; color: var(--text);
  margin: 4px 0 10px; letter-spacing: -0.1px;
}

/* Style st.download_button to match regular buttons */
[data-testid="stDownloadButton"] button{
  background: linear-gradient(135deg, var(--orange) 0%, var(--accent) 100%) !important;
  color: white !important;
  border: none !important;
  border-radius: var(--radius-md) !important;
  padding: 12px 0 !important;
  width: 100% !important;
  font-weight: 700 !important;
  font-size: 0.95rem !important;
  font-family: 'Inter', sans-serif !important;
  box-shadow: 0 4px 16px var(--orange-glow) !important;
  transition: all 0.2s ease !important;
}
[data-testid="stDownloadButton"] button:hover{
  box-shadow: 0 8px 28px var(--orange-glow) !important;
  transform: translateY(-1px) !important;
}
[data-testid="stDownloadButton"] button *{ color: white !important; }

@media (max-width: 700px){
  .hdr-pill{ flex: 1 1 calc(50% - 10px); min-width: 0; }
  .stat-value{ font-size: 1.6rem; }
  .range-row{ grid-template-columns: 1fr; gap: 4px; }
  .range-verdict{ text-align: left; }
  /* Wide cross-pace table scrolls horizontally instead of overflowing */
  .compare-table{ display: block; overflow-x: auto; white-space: nowrap; }
}
""")

render_navbar("Results")


# ─────────────────────────────────────────────────────────────
# Chart theme constants
# ─────────────────────────────────────────────────────────────

# Light-theme Plotly palette (mirrors the CSS variables in styles.py).
_CHART_BG_CARD = "#FFFFFF"  # plot area
_CHART_BG_DARK = "#F4F6FA"  # paper / page
_CHART_TEXT = "rgba(22,26,35,0.75)"
_CHART_GRID = "rgba(16,24,40,0.10)"
_CHART_COLORS = ["#0D9488", "#2563EB", "#7C3AED", "#F59E0B"]  # teal·blue·violet·amber
_CHART_FONT = "Inter, sans-serif"


def _dark_layout(**overrides):
    """Return a Plotly layout dict with the dark theme applied."""
    base = dict(
        plot_bgcolor=_CHART_BG_CARD,
        paper_bgcolor=_CHART_BG_DARK,
        font=dict(family=_CHART_FONT, color=_CHART_TEXT),
        margin=dict(l=48, r=24, t=40, b=40),
        xaxis=dict(gridcolor=_CHART_GRID, zerolinecolor=_CHART_GRID),
        yaxis=dict(gridcolor=_CHART_GRID, zerolinecolor=_CHART_GRID),
    )
    base.update(overrides)
    return base


# ─────────────────────────────────────────────────────────────
# Check for analysis data
# ─────────────────────────────────────────────────────────────

results = st.session_state.get("analysis_results", None)

if not results:
    render_page_header("Analysis Results")
    render_empty_state(
        "No analysis results yet. Upload and analyze videos to see your biomechanical metrics here.",
        link_text="Go to Upload →",
        link_page="pages/upload.py",
    )
    st.stop()

# Filter to results that have metrics
results_with_metrics = [r for r in results if r.get("metrics") is not None]
results_without_metrics = [r for r in results if r.get("metrics") is None]

render_page_header(
    "Analysis Results",
    f"{len(results)} video{'s' if len(results) != 1 else ''} analyzed",
)


# ─────────────────────────────────────────────────────────────
# Helper to safely format a metric value
# ─────────────────────────────────────────────────────────────


def fmt(value, decimals=1, fallback="N/A"):
    """Format a numeric value, handling NaN and None."""
    if value is None:
        return fallback
    try:
        if math.isnan(value):
            return fallback
    except (TypeError, ValueError):
        return str(value)
    return f"{value:.{decimals}f}"


def quality_color(value, good=0.7, ok=0.4):
    """Return CSS color based on quality level."""
    if value is None or (isinstance(value, float) and math.isnan(value)):
        return "#DC2626"
    if value >= good:
        return "#16A34A"
    if value >= ok:
        return "#D97706"
    return "#DC2626"


def _safe_float(value, fallback=0.0):
    """Safely convert a value to float, returning fallback for None/NaN."""
    if value is None:
        return fallback
    try:
        f = float(value)
        return fallback if math.isnan(f) else f
    except (TypeError, ValueError):
        return fallback


def _render_header_strip(r, m):
    """Render the per-clip header strip: pace · date · resolution · strike · cal."""
    res = r.get("resolution") or {}
    res_w, res_h = res.get("width"), res.get("height")
    fps = r.get("fps")
    if res_w and res_h:
        res_txt = f"{res_w}×{res_h}"
        if fps:
            res_txt += f" · {fps:.0f}fps"
    elif fps:
        res_txt = f"{fps:.0f} fps"
    else:
        res_txt = "-"

    strike = (r.get("strike_pattern") or "-").capitalize()
    dt = r.get("shooting_datetime") or "-"
    # Keep the calendar date only (drop the time component of an ISO stamp).
    date_txt = dt.split("T")[0] if dt and "T" in dt else dt
    cal = m.get("calibration_confidence")
    cal_txt = fmt(cal, 2)
    cal_color = quality_color(cal if cal is not None else 0)

    pills = [
        ("Pace", r.get("pace_label", "-"), None),
        ("Footwear", r.get("shoe_label", "-"), None),
        ("Date", date_txt, None),
        ("Resolution", res_txt, None),
        ("Strike", strike, None),
        ("Calibration", cal_txt, cal_color),
    ]
    html = '<div class="hdr-strip">'
    for label, value, color in pills:
        style = f' style="color:{color}"' if color else ""
        html += (
            '<div class="hdr-pill">'
            f'<span class="hdr-pill-label">{label}</span>'
            f'<span class="hdr-pill-value"{style}>{value}</span>'
            "</div>"
        )
    html += "</div>"
    st.markdown(html, unsafe_allow_html=True)


def _render_runner_banner(info):
    """Render the runner identity + PB-derived pace estimates once, at the top.

    The profile/VDOT is a runner-level attribute (shared across clips), so it
    lives here rather than in the per-clip header strip.
    """
    if not info:
        return

    st.markdown(
        '<hr style="border:none;border-top:2px solid var(--border);margin:8px 0 20px;">',
        unsafe_allow_html=True,
    )
    st.markdown(
        '<p class="section-heading" style="border-top:none;padding-top:0;margin-top:0;">Runner Information</p>',
        unsafe_allow_html=True,
    )

    pills = []
    if info.get("name"):
        pills.append(("Runner", info["name"]))
    if info.get("age"):
        pills.append(("Age", str(info["age"])))
    if info.get("sex"):
        pills.append(("Sex", info["sex"]))
    if info.get("height_cm"):
        pills.append(("Height", f"{info['height_cm']:.0f} cm"))
    for evt_key, pb_key, label in [
        ("primary_event", "primary_pb", "Primary"),
        ("secondary_event", "secondary_pb", "Secondary"),
    ]:
        evt = info.get(evt_key)
        if evt:
            pb = info.get(pb_key)
            pills.append((label, f"{evt} ({pb})" if pb else evt))

    if pills:
        html = '<div class="hdr-strip">'
        for label, value in pills:
            html += (
                '<div class="hdr-pill">'
                f'<span class="hdr-pill-label">{label}</span>'
                f'<span class="hdr-pill-value">{value}</span>'
                "</div>"
            )
        html += "</div>"
        st.markdown(html, unsafe_allow_html=True)

    pred = info.get("predicted_paces")
    if not pred:
        return
    cp = pred.get("clip_paces_s_per_km") or {}
    bits = []
    if pred.get("profile"):
        bits.append(f"<strong>{pred['profile']}</strong>")
    if pred.get("vdot"):
        bits.append(f"VDOT&nbsp;≈&nbsp;{pred['vdot']:.0f}")
    if pred.get("confidence"):
        bits.append(f"confidence: {pred['confidence']}")
    paces = [
        f"{lbl} {format_pace_per_km(cp[lbl])}"
        for lbl in ("Threshold", "1500m", "800m")
        if lbl in cp
    ]
    pace_html = (
        f'<br><span class="profile-paces">Target paces &nbsp;'
        f'{"  ·  ".join(paces)}</span>'
        if paces
        else ""
    )
    st.markdown(
        f'<div class="profile-note">Estimated runner profile - {" · ".join(bits)}'
        f"{pace_html}"
        ' <span class="profile-cav">(estimated from entered PBs; the measured '
        "pace below is authoritative)</span></div>",
        unsafe_allow_html=True,
    )


_render_runner_banner((results[0] or {}).get("runner_info"))


def _render_pace_compare(r, m):
    """Compare the measured pace to the PB-derived target for this clip's pace."""
    info = r.get("runner_info") or {}
    pred = info.get("predicted_paces") or {}
    clip_paces = (pred or {}).get("clip_paces_s_per_km") or {}
    label = r.get("pace_label")
    target = clip_paces.get(label)
    vel = _safe_float(m.get("velocity_ms"))
    if not target or vel <= 0:
        return
    actual = 1000.0 / vel  # seconds per km
    delta = actual - target  # +ve → slower than target, -ve → faster
    if abs(delta) < 3:
        cls, word, tail = "pc-onpace", "on target", "matches your estimated target"
    elif delta < 0:
        cls = "pc-faster"
        word = f"{abs(delta):.0f} s/km faster"
        tail = "this clip was run faster than your estimated target"
    else:
        cls = "pc-slower"
        word = f"{delta:.0f} s/km slower"
        tail = "this clip was run slower than your estimated target"
    st.markdown(
        f'<div class="pace-compare">Measured pace '
        f"<strong>{format_pace_per_km(actual)}</strong> vs estimated {label} "
        f"target <strong>{format_pace_per_km(target)}</strong> - "
        f'<span class="pc-delta {cls}">{word}</span>. '
        f'<span style="color:var(--text-subtle);font-size:0.85rem;">{tail}.</span>'
        "</div>",
        unsafe_allow_html=True,
    )


# ─────────────────────────────────────────────────────────────
# Per-video tabs
# ─────────────────────────────────────────────────────────────

if results_with_metrics:
    run_labels = [f"Run {i + 1}" for i in range(len(results_with_metrics))]
    if len(run_labels) > 1:
        selected_run = st.selectbox("Run", run_labels, label_visibility="collapsed")
    else:
        selected_run = run_labels[0]
    run_idx = int(selected_run.split()[-1]) - 1
    r = results_with_metrics[run_idx]
    m = r["metrics"]
    pre = r["preprocessing"]

    # Run heading + filename subtitle
    st.markdown(
        f'<p style="font-weight:700;font-size:1.05rem;color:var(--text);margin:16px 0 2px;">'
        f"{selected_run}</p>"
        f'<p style="font-size:0.82rem;color:var(--text-muted);margin:0 0 8px;">{r["video_name"]}</p>',
        unsafe_allow_html=True,
    )
    # Video Information: clip metadata pills
    st.markdown(
        '<p class="section-heading" style="border-top:none;padding-top:4px;margin-top:0;">Video Information</p>',
        unsafe_allow_html=True,
    )
    _render_header_strip(r, m)
    st.markdown(
        '<div class="profile-note" style="margin:10px 0 0;font-size:0.88rem;">'
        "Calibration (0-1): how confidently the runner's height fixed "
        "the pixel-to-metre scale used for all distance metrics - higher "
        "is better.</div>",
        unsafe_allow_html=True,
    )

    st.markdown(
        '<p class="section-heading">Calculated Metrics</p>',
        unsafe_allow_html=True,
    )

    # Core metrics as pill boxes
    osc_leg = fmt(m.get("oscillation_leg_ratio", None), 3)
    st.markdown(
        '<div class="hdr-strip">'
        '<div class="hdr-pill">'
        '<span class="hdr-pill-label">Cadence</span>'
        f'<span class="hdr-pill-value">{fmt(m["cadence_spm"], 0)} spm</span>'
        f'<span class="hdr-pill-label">± {fmt(m["cadence_std"], 1)} spm</span>'
        "</div>"
        '<div class="hdr-pill">'
        '<span class="hdr-pill-label">Ground Contact</span>'
        f'<span class="hdr-pill-value">{fmt(m["gct_ms"], 0)} ms</span>'
        f'<span class="hdr-pill-label">± {fmt(m["gct_std"], 1)} ms</span>'
        "</div>"
        '<div class="hdr-pill">'
        '<span class="hdr-pill-label">Stride Length</span>'
        f'<span class="hdr-pill-value">{fmt(m["stride_length_m"], 2)} m</span>'
        f'<span class="hdr-pill-label">± {fmt(m["stride_length_std"], 2)} m</span>'
        "</div>"
        '<div class="hdr-pill">'
        '<span class="hdr-pill-label">Vertical Oscillation</span>'
        f'<span class="hdr-pill-value">{fmt(m["oscillation_cm"], 1)} cm</span>'
        f'<span class="hdr-pill-label stat-flag">unvalidated · {osc_leg} × leg</span>'
        "</div>"
        "</div>",
        unsafe_allow_html=True,
    )

    # Derived metrics - single inline line
    st.markdown(
        '<p class="derived-line">'
        f'<strong>Flight Time</strong> {fmt(m["flight_time_ms"], 0)} ms'
        " &nbsp;·&nbsp; "
        f'<strong>Duty factor</strong> {fmt(m["duty_factor"], 2)}'
        " &nbsp;·&nbsp; "
        f'<strong>Speed</strong> {fmt(m["velocity_kmh"], 1)} km/h '
        f'({fmt(m["velocity_ms"], 2)} m/s) &nbsp;·&nbsp; '
        f'<strong>Pace</strong> {m.get("pace_per_km", "N/A")}/km'
        " &nbsp;·&nbsp; "
        f'<strong>Economy</strong> {fmt(m["running_economy_index"], 1)}% '
        "(lower = more efficient)</p>",
        unsafe_allow_html=True,
    )

    # Target-vs-actual pace (PB-derived target for this clip's pace)
    _render_pace_compare(r, m)

    # ── Ground-contact verifier (interactive frame scrubber) ──
    analysis = r.get("analysis")
    st.markdown(
        '<p class="section-heading">Ground-Contact Verifier</p>',
        unsafe_allow_html=True,
    )
    if analysis is not None:
        st.markdown(
            '<p class="section-sub">Step through the clip to check the '
            "detected foot-strike (IC) and toe-off (TO) frames. Overlays "
            "are always on; nothing here re-runs the analysis.</p>",
            unsafe_allow_html=True,
        )
        render_gc_scrubber(analysis, key_prefix=f"scrub_{r['video_name']}")
    else:
        st.info(
            "Frame-by-frame review isn't available for this clip "
            "(analysis data was not retained in this session)."
        )

    # ── Pipeline diagnostics (always visible) ──
    st.markdown(
        '<p class="section-heading">Pipeline Diagnostics</p>',
        unsafe_allow_html=True,
    )
    direction = pre.get("direction", "N/A")
    dir_txt = "→ Right" if direction == "right" else "← Left"
    det_rate = pre.get("detection_rate", 0)
    n_contacts = m.get("n_contacts", 0)
    n_refined = m.get("n_refined_contacts", 0)
    refined_pct = (n_refined / n_contacts * 100) if n_contacts > 0 else 0
    det_c = quality_color(det_rate / 100)

    diag_items = [
        ("NN input shape", str(pre.get("nn_input_shape", "N/A"))),
        ("Torso scale (norm.)", f"{pre.get('torso_scale', 0):.4f}"),
        ("Running direction", dir_txt),
        ("Mean visibility", f"{pre.get('mean_visibility', 0):.3f}"),
        (
            "Pose detection rate",
            f'<span style="color:{det_c}">{det_rate:.1f}%</span>',
        ),
        (
            "Ground contacts",
            f"{n_contacts} total · {n_refined} refined " f"({refined_pct:.0f}%)",
        ),
    ]
    diag_pills = "".join(
        f'<div class="hdr-pill">'
        f'<span class="hdr-pill-label">{k}</span>'
        f'<span class="hdr-pill-value">{v}</span>'
        f"</div>"
        for k, v in diag_items
    )
    st.markdown(
        f'<div class="hdr-strip">{diag_pills}</div>',
        unsafe_allow_html=True,
    )

    # ── Elite Range Comparison (always shown for the selected run) ──
    st.markdown(
        '<p class="section-heading">Elite Range Comparison</p>',
        unsafe_allow_html=True,
    )
    st.markdown(
        "Your metrics against indicative reference ranges for world-class 800 m "
        "runners. The bands are approximate - orientation, not a pass/fail."
    )

    # (label, unit, value, (lo, hi), note) - bands are indicative (see sources).
    bars = [
        ("Cadence", "spm", _safe_float(m.get("cadence_spm")), (190.0, 210.0), ""),
        ("Ground contact", "ms", _safe_float(m.get("gct_ms")), (115.0, 130.0), ""),
        ("Stride length", "m", _safe_float(m.get("stride_length_m")), (3.8, 4.4), ""),
        (
            "Vertical oscillation",
            "cm",
            _safe_float(m.get("oscillation_cm")),
            (6.0, 10.0),
            "not yet validated",
        ),
    ]

    rows_html = []
    for name, unit, val, (lo, hi), note in bars:
        disp_lo = min(lo, val)
        disp_hi = max(hi, val)
        pad = (disp_hi - disp_lo or 1.0) * 0.15
        disp_lo -= pad
        disp_hi += pad
        span = disp_hi - disp_lo or 1.0
        band_left = (lo - disp_lo) / span * 100
        band_w = (hi - lo) / span * 100
        mark = max(0.0, min(100.0, (val - disp_lo) / span * 100))
        verdict = (
            "below range" if val < lo else "above range" if val > hi else "in range"
        )
        scale_fmt = "{:.1f}" if disp_hi < 20 else "{:.0f}"
        note_html = (
            f'<br><span class="stat-flag" style="font-size:0.78rem;">{note}</span>'
            if note
            else ""
        )
        rows_html.append(
            '<div class="range-row">'
            f'<div class="range-name">{name}<br>'
            f'<span class="rn-sub">elite {lo:g}-{hi:g} {unit}</span></div>'
            '<div><div class="range-track">'
            f'<div class="range-band" style="left:{band_left:.1f}%;'
            f'width:{band_w:.1f}%"></div>'
            f'<div class="range-marker" style="left:{mark:.1f}%"></div></div>'
            f'<div class="range-scale"><span>{scale_fmt.format(disp_lo)}</span>'
            f"<span>{scale_fmt.format(disp_hi)}</span></div></div>"
            f'<div class="range-verdict"><strong>{val:.1f} {unit}</strong><br>'
            f"{verdict}{note_html}</div></div>"
        )
    st.markdown("".join(rows_html), unsafe_allow_html=True)

    st.markdown(
        '<p class="range-sources">Reference ranges are indicative, drawn from '
        "championship middle-distance biomechanics: Hanley, Bissas &amp; Sheridan "
        "(2022), <em>Biomechanics of World-Class 800&nbsp;m Women</em>, "
        '<a href="https://pmc.ncbi.nlm.nih.gov/articles/PMC9047885/" '
        'target="_blank">Front. Sports Act. Living</a>; Hanley et&nbsp;al. (2023), '
        "<em>Men's 1500&nbsp;m</em>, "
        '<a href="https://onlinelibrary.wiley.com/doi/10.1111/sms.14331" '
        'target="_blank">Scand. J. Med. Sci. Sports</a>; Preece, Mason &amp; '
        'Bramah (2019), <a href="https://doi.org/10.1080/17461391.2018.1554707" '
        'target="_blank">Eur. J. Sport Sci.</a> Vertical oscillation is not yet '
        "validated against ground truth.</p>",
        unsafe_allow_html=True,
    )

# ─────────────────────────────────────────────────────────────
# Cross-pace comparison (if 2+ videos with metrics)
# ─────────────────────────────────────────────────────────────

if len(results_with_metrics) >= 2:
    st.markdown(
        '<p class="section-heading">Cross-Run Comparison</p>',
        unsafe_allow_html=True,
    )

    rows = ""
    for i, r in enumerate(results_with_metrics):
        m = r["metrics"]
        run_lbl = f"Run {i + 1}"
        rows += (
            f"<tr>"
            f'<td class="pace-col"><strong>{run_lbl}</strong><br>'
            f'<span style="font-size:0.78rem;font-weight:400;color:var(--text-muted);">'
            f"{r['video_name']}</span></td>"
            f"<td>{fmt(m['cadence_spm'], 0)} spm</td>"
            f"<td>{fmt(m['gct_ms'], 0)} ms</td>"
            f"<td>{fmt(m['stride_length_m'], 2)} m</td>"
            f"<td>{fmt(m['oscillation_cm'], 1)} cm</td>"
            f"<td>{fmt(m['velocity_kmh'], 1)} km/h</td>"
            f"<td>{fmt(m['duty_factor'], 2)}</td>"
            f"</tr>"
        )

    st.markdown(
        f'<table class="compare-table">'
        f"<thead><tr>"
        f"<th>Run</th><th>Cadence</th><th>GCT</th>"
        f"<th>Stride</th><th>Oscillation</th><th>Speed</th><th>Duty Factor</th>"
        f"</tr></thead>"
        f"<tbody>{rows}</tbody>"
        f"</table>",
        unsafe_allow_html=True,
    )

# ─────────────────────────────────────────────────────────────
# Videos without full metrics
# ─────────────────────────────────────────────────────────────

if results_without_metrics:
    st.markdown("<br>", unsafe_allow_html=True)
    for r in results_without_metrics:
        pre = r["preprocessing"]
        st.markdown(
            f"""
<div class="result-card warn" style="margin-bottom:12px;">
  <p class="result-label">{r['pace_label']} - {r['video_name']}</p>
  <p class="result-value" style="font-size:0.95rem;">Preprocessing only</p>
  <p class="result-detail">
    Visibility: {pre['mean_visibility']:.3f} · Detection: {pre['detection_rate']:.1f}%
    {(' · Error: ' + r['metrics_error']) if r.get('metrics_error') else ''}
  </p>
</div>
""",
            unsafe_allow_html=True,
        )

# ─────────────────────────────────────────────────────────────
# PDF Export (only when full metrics are available)
# ─────────────────────────────────────────────────────────────


def _generate_pdf(results_data):
    """Generate a PDF report of the biomechanical analysis."""
    from io import BytesIO

    from reportlab.lib import colors
    from reportlab.lib.enums import TA_CENTER
    from reportlab.lib.pagesizes import A4
    from reportlab.lib.styles import ParagraphStyle, getSampleStyleSheet
    from reportlab.lib.units import mm
    from reportlab.platypus import (
        HRFlowable,
        KeepTogether,
        Paragraph,
        SimpleDocTemplate,
        Spacer,
        Table,
        TableStyle,
    )

    # ── Palette ──────────────────────────────────────────────────
    TEAL = colors.HexColor("#0D9488")
    DARK = colors.HexColor("#1A1A2E")
    MUTED = colors.HexColor("#64748B")
    LIGHT = colors.HexColor("#F0F3F8")
    BORDER = colors.HexColor("#E0E0E0")
    WHITE = colors.white

    # ── Page layout ──────────────────────────────────────────────
    buf = BytesIO()
    PAGE_W, _ = A4
    LM = RM = 20 * mm
    CONTENT_W = PAGE_W - LM - RM
    doc = SimpleDocTemplate(
        buf,
        pagesize=A4,
        leftMargin=LM,
        rightMargin=RM,
        topMargin=18 * mm,
        bottomMargin=18 * mm,
    )
    styles = getSampleStyleSheet()

    # ── Text styles ───────────────────────────────────────────────
    def _style(name, **kw):
        base = kw.pop("parent", styles["Normal"])
        return ParagraphStyle(name, parent=base, **kw)

    app_title = _style(
        "AppTitle",
        fontSize=26,
        fontName="Helvetica-Bold",
        textColor=DARK,
        alignment=TA_CENTER,
        leading=30,
        spaceAfter=2,
    )
    app_subtitle = _style(
        "AppSubtitle",
        fontSize=14,
        fontName="Helvetica",
        textColor=TEAL,
        alignment=TA_CENTER,
        leading=18,
        spaceAfter=6,
    )
    body = _style("Body", fontSize=9, textColor=MUTED, leading=13, spaceAfter=6)
    note = _style(
        "Note",
        fontSize=8,
        fontName="Helvetica-Oblique",
        textColor=MUTED,
        leading=11,
        spaceAfter=4,
    )
    sec_head = _style(
        "SecHead",
        fontSize=11,
        fontName="Helvetica-Bold",
        textColor=TEAL,
        spaceBefore=10,
        spaceAfter=5,
    )
    clip_head = _style(
        "ClipHead",
        fontSize=13,
        fontName="Helvetica-Bold",
        textColor=DARK,
        spaceBefore=4,
        spaceAfter=3,
    )
    sub_head = _style(
        "SubHead",
        fontSize=9,
        fontName="Helvetica-Bold",
        textColor=TEAL,
        spaceBefore=8,
        spaceAfter=3,
    )

    def _hr(thick=1, color=BORDER, before=4, after=4):
        return HRFlowable(
            width="100%",
            thickness=thick,
            color=color,
            spaceBefore=before * mm,
            spaceAfter=after * mm,
        )

    # ── Pill table helper (label row + value row, N per row) ──────
    def _pill_table(pairs, per_row=3):
        """Render (label, value) pairs as a pill-style grid table."""
        col_w = CONTENT_W / per_row
        rows = []
        chunk = []
        for lbl, val in pairs:
            chunk.append((lbl, val))
            if len(chunk) == per_row:
                rows.append(chunk)
                chunk = []
        if chunk:
            while len(chunk) < per_row:
                chunk.append(("", ""))
            rows.append(chunk)

        data = []
        cmds = [
            ("LEFTPADDING", (0, 0), (-1, -1), 5),
            ("RIGHTPADDING", (0, 0), (-1, -1), 5),
            ("TOPPADDING", (0, 0), (-1, -1), 3),
            ("BOTTOMPADDING", (0, 0), (-1, -1), 3),
            ("INNERGRID", (0, 0), (-1, -1), 0.5, BORDER),
            ("BOX", (0, 0), (-1, -1), 0.5, BORDER),
        ]
        for ri, row in enumerate(rows):
            label_row = [r[0].upper() for r in row]
            value_row = [r[1] for r in row]
            data.append(label_row)
            data.append(value_row)
            li = ri * 2
            vi = li + 1
            cmds += [
                ("FONTSIZE", (0, li), (-1, li), 7),
                ("FONTNAME", (0, li), (-1, li), "Helvetica"),
                ("TEXTCOLOR", (0, li), (-1, li), MUTED),
                ("BACKGROUND", (0, li), (-1, li), LIGHT),
                ("FONTSIZE", (0, vi), (-1, vi), 10),
                ("FONTNAME", (0, vi), (-1, vi), "Helvetica-Bold"),
                ("TEXTCOLOR", (0, vi), (-1, vi), DARK),
                ("BACKGROUND", (0, vi), (-1, vi), WHITE),
            ]
        t = Table(data, colWidths=[col_w] * per_row)
        t.setStyle(TableStyle(cmds))
        return t

    # ── Simple data table helper ──────────────────────────────────
    def _data_table(rows, col_widths, has_header=True):
        cmds = [
            ("FONTSIZE", (0, 0), (-1, -1), 9),
            ("FONTNAME", (0, 0), (-1, -1), "Helvetica"),
            ("TEXTCOLOR", (0, 0), (-1, -1), DARK),
            ("VALIGN", (0, 0), (-1, -1), "MIDDLE"),
            ("TOPPADDING", (0, 0), (-1, -1), 5),
            ("BOTTOMPADDING", (0, 0), (-1, -1), 5),
            ("LEFTPADDING", (0, 0), (-1, -1), 6),
            ("RIGHTPADDING", (0, 0), (-1, -1), 6),
            ("GRID", (0, 0), (-1, -1), 0.5, BORDER),
            ("ROWBACKGROUNDS", (0, 1), (-1, -1), [WHITE, LIGHT]),
        ]
        if has_header:
            cmds += [
                ("BACKGROUND", (0, 0), (-1, 0), TEAL),
                ("TEXTCOLOR", (0, 0), (-1, 0), WHITE),
                ("FONTNAME", (0, 0), (-1, 0), "Helvetica-Bold"),
                ("FONTSIZE", (0, 0), (-1, 0), 9),
            ]
        t = Table(rows, colWidths=col_widths)
        t.setStyle(TableStyle(cmds))
        return t

    elements = []

    # ─────────────────────────────────────────────────────────────
    # Title block
    # ─────────────────────────────────────────────────────────────
    elements.append(Spacer(1, 4 * mm))
    elements.append(Paragraph("Strideo", app_title))
    elements.append(Paragraph("Video Analysis Report", app_subtitle))
    elements.append(_hr(thick=2, color=TEAL, before=4, after=6))

    # ── About ──────────────────────────────────────────────────────
    elements.append(
        Paragraph(
            "Strideo is an open-source toolkit that turns ordinary training footage of a "
            "middle-distance runner into objective biomechanical data: cadence, ground "
            "contact time, stride length, and vertical oscillation - with no markers, no "
            "wearables, and no lab. Built for 800 m and 1500 m athletes, it surfaces the "
            "mechanical details that coaches measure intuitively. Results are produced by a "
            "classical computer-vision pipeline; a pace-conditioned neural network is in "
            "training and does not yet power these results.",
            body,
        )
    )
    elements.append(_hr(thick=1, color=BORDER, before=4, after=4))

    # ─────────────────────────────────────────────────────────────
    # Runner Information
    # ─────────────────────────────────────────────────────────────
    info = (results_data[0].get("runner_info") if results_data else None) or {}
    pred = info.get("predicted_paces") or {}
    cp = pred.get("clip_paces_s_per_km") or {}

    runner_pills = []
    if info.get("name"):
        runner_pills.append(("Runner", info["name"]))
    if info.get("age"):
        runner_pills.append(("Age", str(info["age"])))
    if info.get("sex"):
        runner_pills.append(("Sex", info["sex"]))
    if info.get("height_cm"):
        runner_pills.append(("Height", f"{info['height_cm']:.0f} cm"))
    for evt_key, pb_key, lbl in [
        ("primary_event", "primary_pb", "Primary"),
        ("secondary_event", "secondary_pb", "Secondary"),
    ]:
        evt = info.get(evt_key)
        if evt:
            pb = info.get(pb_key)
            runner_pills.append((lbl, f"{evt} ({pb})" if pb else str(evt)))
    if pred.get("profile"):
        runner_pills.append(("Profile", str(pred["profile"])))
    if pred.get("vdot"):
        runner_pills.append(("VDOT", f"approx. {pred['vdot']:.0f}"))
    if pred.get("confidence"):
        runner_pills.append(("Confidence", str(pred["confidence"])))
    for lbl in ("Threshold", "1500m", "800m"):
        if lbl in cp:
            runner_pills.append((f"Target {lbl}", format_pace_per_km(cp[lbl])))

    if runner_pills:
        elements.append(
            KeepTogether(
                [
                    Paragraph("Runner Information", sec_head),
                    _pill_table(runner_pills, per_row=3),
                ]
            )
        )
        if pred:
            elements.append(
                Paragraph(
                    "Target paces estimated from entered PBs via the VDOT equivalency system "
                    "(Jack Daniels). The pace measured from video is authoritative.",
                    note,
                )
            )
        elements.append(Spacer(1, 4 * mm))

    # ─────────────────────────────────────────────────────────────
    # Per-clip sections
    # ─────────────────────────────────────────────────────────────
    ELITE_RANGES = [
        ("Cadence", "spm", "cadence_spm", 190.0, 210.0, ""),
        ("Ground Contact Time", "ms", "gct_ms", 115.0, 130.0, ""),
        ("Stride Length", "m", "stride_length_m", 3.8, 4.4, ""),
        (
            "Vertical Oscillation",
            "cm",
            "oscillation_cm",
            6.0,
            10.0,
            "not yet validated",
        ),
    ]

    for run_num, r in enumerate(results_data, start=1):
        m = r["metrics"]
        pre = r["preprocessing"]

        elements.append(_hr(thick=2, color=BORDER, before=4, after=4))
        elements.append(Paragraph(f"Run {run_num}", clip_head))
        elements.append(Paragraph(r["video_name"], note))

        # ── Video Information ──────────────────────────────────
        elements.append(Paragraph("Video Information", sub_head))
        res = r.get("resolution") or {}
        res_w, res_h = res.get("width"), res.get("height")
        fps_val = r.get("fps")
        if res_w and res_h:
            res_str = f"{res_w}x{res_h}"
            if fps_val:
                res_str += f" / {fps_val:.0f} fps"
        elif fps_val:
            res_str = f"{fps_val:.0f} fps"
        else:
            res_str = "-"
        dt = r.get("shooting_datetime") or "-"
        date_str = dt.split("T")[0] if dt and "T" in dt else dt
        cal = m.get("calibration_confidence")
        cal_str = f"{cal:.2f}" if cal is not None else "-"
        strike = (r.get("strike_pattern") or "-").capitalize()

        vid_pills = [
            ("Pace", r.get("pace_label", "-")),
            ("Footwear", r.get("shoe_label", "-")),
            ("Date", date_str),
            ("Resolution", res_str),
            ("Strike", strike),
            ("Calibration", cal_str),
        ]
        elements.append(_pill_table(vid_pills, per_row=3))

        # ── Calculated Metrics ────────────────────────────────
        elements.append(Paragraph("Calculated Metrics", sub_head))
        osc_leg = fmt(m.get("oscillation_leg_ratio"), 3)
        metrics_rows = [
            ["Metric", "Value", "+/- Std", "Notes"],
            [
                "Cadence",
                f"{fmt(m['cadence_spm'], 0)} spm",
                f"+/- {fmt(m['cadence_std'], 1)} spm",
                "",
            ],
            [
                "Ground Contact Time",
                f"{fmt(m['gct_ms'], 0)} ms",
                f"+/- {fmt(m['gct_std'], 1)} ms",
                "",
            ],
            [
                "Stride Length",
                f"{fmt(m['stride_length_m'], 2)} m",
                f"+/- {fmt(m['stride_length_std'], 2)} m",
                "",
            ],
            [
                "Vertical Oscillation",
                f"{fmt(m['oscillation_cm'], 1)} cm",
                "",
                f"unvalidated / {osc_leg}x leg",
            ],
            ["Flight Time", f"{fmt(m['flight_time_ms'], 0)} ms", "", ""],
            ["Duty Factor", fmt(m["duty_factor"], 2), "", ""],
            [
                "Speed",
                f"{fmt(m['velocity_kmh'], 1)} km/h ({fmt(m['velocity_ms'], 2)} m/s)",
                "",
                "",
            ],
            ["Pace", m.get("pace_per_km", "-") + "/km", "", ""],
            [
                "Economy Index",
                f"{fmt(m['running_economy_index'], 1)}%",
                "",
                "lower = more efficient",
            ],
        ]
        cw = [CONTENT_W * f for f in (0.36, 0.24, 0.22, 0.18)]
        elements.append(_data_table(metrics_rows, cw))

        # ── Pipeline Diagnostics ──────────────────────────────
        elements.append(Paragraph("Pipeline Diagnostics", sub_head))
        direction = pre.get("direction", "N/A")
        dir_txt = "Right" if direction == "right" else "Left"
        det_rate = pre.get("detection_rate", 0)
        n_contacts = m.get("n_contacts", 0)
        n_refined = m.get("n_refined_contacts", 0)
        refined_pct = (n_refined / n_contacts * 100) if n_contacts else 0
        diag_pills = [
            ("NN Input Shape", str(pre.get("nn_input_shape", "-"))),
            ("Torso Scale", f"{pre.get('torso_scale', 0):.4f}"),
            ("Direction", dir_txt),
            ("Mean Visibility", f"{pre.get('mean_visibility', 0):.3f}"),
            ("Detection Rate", f"{det_rate:.1f}%"),
            (
                "Ground Contacts",
                f"{n_contacts} total / {n_refined} refined ({refined_pct:.0f}%)",
            ),
        ]
        elements.append(_pill_table(diag_pills, per_row=3))

        # ── Elite Range Comparison ────────────────────────────
        elements.append(Paragraph("Elite Range Comparison", sub_head))
        elements.append(
            Paragraph(
                "Indicative reference ranges for world-class 800 m runners. "
                "Approximate - orientation, not a pass/fail.",
                note,
            )
        )
        range_rows = [["Metric", "Your Value", "Elite Range", "Verdict"]]
        for name, unit, key, lo, hi, rng_note in ELITE_RANGES:
            val = _safe_float(m.get(key))
            val_str = f"{val:.1f} {unit}" if val is not None else "-"
            range_str = f"{lo}-{hi} {unit}"
            if val is None:
                verdict = "-"
            elif val < lo:
                verdict = "below range"
            elif val > hi:
                verdict = "above range"
            else:
                verdict = "in range"
            if rng_note:
                verdict += f" ({rng_note})"
            range_rows.append([name, val_str, range_str, verdict])
        cw_r = [CONTENT_W * f for f in (0.30, 0.20, 0.22, 0.28)]
        elements.append(_data_table(range_rows, cw_r))
        elements.append(Spacer(1, 4 * mm))

    # ─────────────────────────────────────────────────────────────
    # Cross-clip comparison (2+ clips)
    # ─────────────────────────────────────────────────────────────
    if len(results_data) >= 2:
        elements.append(_hr(thick=2, color=TEAL, before=4, after=4))
        elements.append(Paragraph("Cross-Run Comparison", sec_head))
        header = ["Run", "Cadence", "GCT", "Stride", "Osc.", "Speed"]
        rows = [header]
        for rn, r in enumerate(results_data, start=1):
            m = r["metrics"]
            rows.append(
                [
                    f"Run {rn}  {r['video_name']}",
                    f"{fmt(m['cadence_spm'], 0)} spm",
                    f"{fmt(m['gct_ms'], 0)} ms",
                    f"{fmt(m['stride_length_m'], 2)} m",
                    f"{fmt(m['oscillation_cm'], 1)} cm",
                    f"{fmt(m['velocity_kmh'], 1)} km/h",
                ]
            )
        cw_c = [CONTENT_W * f for f in (0.34, 0.14, 0.12, 0.12, 0.12, 0.16)]
        elements.append(_data_table(rows, cw_c))

    doc.build(elements)
    return buf.getvalue()


# ─────────────────────────────────────────────────────────────
# Action buttons
# ─────────────────────────────────────────────────────────────

st.markdown("<br>", unsafe_allow_html=True)

if results_with_metrics:
    bc1, bc2, bc3 = st.columns(3)
else:
    bc1, bc2 = st.columns(2)
    bc3 = None

with bc1:
    if st.button("↩ Upload More Videos", use_container_width=True):
        st.switch_page("pages/upload.py")
with bc2:
    if st.button("🗑 Clear Results", use_container_width=True):
        del st.session_state["analysis_results"]
        st.rerun()
if bc3 is not None and results_with_metrics:
    with bc3:
        try:
            pdf_bytes = _generate_pdf(results_with_metrics)
            st.download_button(
                label="📄 Download PDF Report",
                data=pdf_bytes,
                file_name="strideo_report.pdf",
                mime="application/pdf",
                use_container_width=True,
            )
        except ImportError:
            st.button(
                "📄 PDF export requires reportlab",
                disabled=True,
                use_container_width=True,
            )
