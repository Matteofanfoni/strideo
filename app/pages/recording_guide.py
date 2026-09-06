# app/pages/recording_guide.py
import streamlit as st
import sys
from pathlib import Path

app_dir = Path(__file__).parent.parent
if str(app_dir) not in sys.path:
    sys.path.insert(0, str(app_dir))

from ui.styles import (  # noqa: E402
    inject_shared_styles,
    inject_page_css,
    render_navbar,
    render_footer,
    mark_svg,
)

st.set_page_config(
    page_title="Recording Guide | strideo.it",
    page_icon=str(app_dir / "assets" / "favicon.png"),
    layout="wide",
)

inject_shared_styles()
render_navbar("Guide")

# Every number on this page is a copy of a field in
# the data-collection protocol (v1.6) - keep the two in step when changing
# either one.
#
# The page is laid out as CSS grids inside single st.markdown blocks rather than
# st.columns: nothing here is a widget except the closing CTA button, and one
# HTML block per section keeps the card grids from depending on Streamlit's
# column DOM. Display headings restate size and reset padding with !important
# because Streamlit's own h1-h3 rules outrank a bare class selector.
inject_page_css("""
/* ── Hero ── */
.guide-title{
  font-family: 'Space Grotesk', ui-sans-serif, system-ui, sans-serif;
  font-size: clamp(2.1rem, 3.5vw, 3.2rem) !important;
  font-weight: 700;
  letter-spacing: -0.035em;
  line-height: 1.04 !important;
  color: var(--text);
  margin: 16px 0 0 !important;
  padding: 0 !important;
  max-width: 30ch;
}
.guide-lead{
  font-size: 1.1rem;
  line-height: 1.68;
  color: var(--text-muted);
  margin: 22px 0 0 !important;
}

/* ── Rules + camera geometry ── */
.guide-2col{
  display: grid;
  /* Matches .gv-grid's 1fr/1fr split (and its 22px gap) exactly, so the rule
     list lines up edge-to-edge with the good/avoid boxes below it instead of
     running its own, slightly narrower ratio. */
  grid-template-columns: 1fr 1fr;
  gap: 22px;
  margin-top: 40px;
}
.rule-list{
  background: var(--card);
  border: 1px solid var(--border);
  border-radius: var(--radius-lg);
  overflow: hidden;
  list-style: none !important;
  margin: 0 !important;
  padding: 0 !important;
}
.rule-row{
  display: flex;
  gap: 16px;
  padding: 18px 24px !important;
  margin: 0 !important;
}
.rule-row + .rule-row{ border-top: 1px solid var(--border); }
.rule-num{
  font-family: 'JetBrains Mono', ui-monospace, monospace;
  font-size: 0.85rem;
  color: var(--brand-text);
  padding-top: 2px;
}
.rule-title{
  font-family: 'Space Grotesk', ui-sans-serif, system-ui, sans-serif;
  font-size: 1rem !important;
  font-weight: 600;
  line-height: 1.35 !important;
  color: var(--text);
  margin: 0 !important;
  padding: 0 !important;
}
.rule-body{
  font-size: 0.9rem;
  line-height: 1.65;
  color: var(--text-muted);
  margin: 6px 0 0 !important;
}
.rule-body strong{ color: var(--text); font-weight: 600; }

/* Flex column so the diagram centres itself in whatever height the taller rule
   list gives this card, instead of leaving dead space underneath. */
.geo-card{
  background: var(--card);
  border: 1px solid var(--border);
  border-radius: var(--radius-lg);
  padding: 24px 26px;
  display: flex;
  flex-direction: column;
}
.geo-title{
  font-family: 'Space Grotesk', ui-sans-serif, system-ui, sans-serif;
  font-size: 1rem !important;
  font-weight: 600;
  color: var(--text);
  margin: 0 !important;
  padding: 0 !important;
}
.geo-card svg{
  width: 100%;
  height: auto;
  margin-top: 14px;
  display: block;
  flex: 1 1 auto;
  min-height: 0;
}
.geo-caption{
  font-family: 'JetBrains Mono', ui-monospace, monospace;
  font-size: 0.68rem;
  text-transform: uppercase;
  letter-spacing: 0.14em;
  color: var(--text-muted);
  margin: 14px 0 0 !important;
}

/* ── Good / avoid ──
   One outer box (L3 - was three visually distinct cards: good-tinted,
   avoid-tinted, and a separate note-card below). Good/avoid keep only their
   label colour and bullet-dot colour as the at-a-glance signal; a hairline
   divider replaces the avoid card's own border, and the ghost-lock note is
   now this box's own closing paragraph instead of a second card. */
.gv-box{
  background: var(--card);
  border: 1px solid var(--border);
  border-radius: var(--radius-lg);
  padding: 26px 28px;
  margin-top: 22px;
}
.gv-grid{
  display: grid;
  grid-template-columns: 1fr 1fr;
  gap: 28px;
}
.gv-card.avoid{
  border-left: 1px solid var(--border);
  padding-left: 28px;
}
.gv-label{
  font-family: 'JetBrains Mono', ui-monospace, monospace;
  font-size: 0.7rem;
  text-transform: uppercase;
  letter-spacing: 0.18em;
  margin: 0 !important;
}
.gv-card.good .gv-label{ color: var(--brand-text); }
.gv-card.avoid .gv-label{ color: var(--red); }
.gv-card ul{
  list-style: none !important;
  margin: 14px 0 0 !important;
  padding: 0 !important;
}
.gv-card li{
  position: relative;
  font-size: 0.9rem;
  line-height: 1.6;
  color: var(--text-muted);
  margin: 0 0 9px !important;
  padding: 0 0 0 18px !important;
}
.gv-card li:last-child{ margin-bottom: 0 !important; }
.gv-card li::before{
  content: '';
  position: absolute;
  left: 0;
  top: 8px;
  width: 6px;
  height: 6px;
  border-radius: 50%;
}
.gv-card.good li::before{ background: var(--brand); }
.gv-card.avoid li::before{ background: var(--red); }
.gv-note{
  border-top: 1px solid var(--border);
  margin-top: 24px;
  padding-top: 18px;
}
.gv-note p{
  margin: 0 !important;
  font-size: 0.92rem;
  line-height: 1.65;
  color: var(--text-muted);
}
.gv-note strong{ color: var(--text); }

/* ── Capture spec: the full protocol distillation, as mono key/value rows ── */
.spec-grid{
  display: grid;
  grid-template-columns: repeat(2, 1fr);
  gap: 22px;
  margin-top: 34px;
}
.spec-card{
  background: var(--card);
  border: 1px solid var(--border);
  border-radius: var(--radius-lg);
  padding: 22px 24px;
}
.spec-title{
  font-family: 'Space Grotesk', ui-sans-serif, system-ui, sans-serif;
  font-size: 1rem !important;
  font-weight: 600;
  color: var(--text);
  margin: 0 0 4px !important;
  padding: 0 !important;
}
.spec-row{
  display: flex;
  flex-wrap: wrap;
  align-items: baseline;
  gap: 4px 14px;
  justify-content: space-between;
  padding: 10px 0;
  border-bottom: 1px solid var(--border-soft);
}
.spec-row:last-child{ border-bottom: none; padding-bottom: 0; }
.spec-key{
  font-family: 'JetBrains Mono', ui-monospace, monospace;
  font-size: 0.68rem;
  text-transform: uppercase;
  letter-spacing: 0.14em;
  color: var(--text-subtle);
  flex-shrink: 0;
}
.spec-val{
  font-size: 0.9rem;
  line-height: 1.55;
  color: var(--text);
  text-align: right;
  margin-left: auto;
}

/* 250px matches the hero "Analyse a clip" button's own measured width
   (getBoundingClientRect, both buttons sharing the same text/font/padding
   now) - not an arbitrary cap like the previous 320px. */
.st-key-guide_cta{ max-width: 250px; margin-top: 30px; }

/* Capture-spec band, extended to include the CTA button below it (a real
   widget, so it can't just be nested inside the .band div like the spec grid
   is - same full-bleed/tint/border values as the shared .band class,
   applied via :has() to the container instead). */
[data-testid="stVerticalBlock"]:has(> [data-testid="stElementContainer"] .capture-band-mark){
  margin: 56px calc(50% - 50vw) 0;
  width: 100vw;
  max-width: 100vw;
  box-sizing: border-box;
  padding: 60px max(20px, calc(50vw - 50%));
  border-top: 1px solid var(--border);
  border-bottom: 1px solid var(--border);
  background: color-mix(in srgb, var(--card) 60%, transparent);
}
@media (max-width: 700px){
  [data-testid="stVerticalBlock"]:has(> [data-testid="stElementContainer"] .capture-band-mark){
    margin: 36px calc(50% - 50vw) 0;
    padding: 40px 20px;
  }
}

@media (max-width: 900px){
  .guide-2col, .gv-grid, .spec-grid{ grid-template-columns: 1fr; }
  .gv-card.avoid{ border-left: none; padding-left: 0; margin-top: 24px; }
}
@media (max-width: 700px){
  .guide-lead{ font-size: 1rem; }
  .rule-row{ padding: 16px 18px !important; }
  .geo-card, .spec-card{ padding: 18px 18px; }
  .gv-box{ padding: 20px 18px; }
  .spec-val{ text-align: left; margin-left: 0; }
}
""")


# ─────────────────────────────────────────────────────────────
# Hero
# ─────────────────────────────────────────────────────────────

st.markdown(
    """
<p class="eyebrow">recording guide</p>
<h1 class="guide-title">Get the clip right and the maths takes care of
itself.</h1>
<p class="guide-lead">Pose estimation is only as good as the geometry you hand
it, and four rules cover almost every failed analysis. Every distance below is
the setup that captured the validation dataset: follow it for the most accurate
result. Both of Strideo's analysis methods read the same pose data.</p>
""",
    unsafe_allow_html=True,
)


# ─────────────────────────────────────────────────────────────
# The four rules + the camera-geometry diagram
# ─────────────────────────────────────────────────────────────

_RULES = [
    (
        "01",
        "Stand 13 m to the side",
        "Perpendicular to the track, level with the middle of the capture zone, "
        "shooting the <strong>sagittal plane</strong>. Perspective error grows "
        "fast if you shoot from an angle.",
    ),
    (
        "02",
        "Lock 4K at 60 fps",
        "Ground contact lasts roughly <strong>120–140 ms</strong>, so frames "
        "matter as much as pixels. Stabilisation and HDR <strong>off</strong> so "
        "the camera cannot re-crop or re-expose mid-clip.",
    ),
    (
        "03",
        "Whole body, whole time",
        "Head and both feet in frame for the entire pass, from a "
        "<strong>fixed tripod at hip height</strong>. Zero camera movement: no "
        "panning, no zooming mid-clip.",
    ),
    (
        "04",
        "One steady-state pace per clip",
        "Film three passes, at <strong>threshold, 1500 m and 800 m</strong> "
        "effort. Accelerations distort cadence and contact time, so capture the "
        "runner already at pace.",
    ),
]

_rules_html = "".join(
    f'<li class="rule-row"><span class="rule-num">{num}</span><div>'
    f'<h2 class="rule-title">{title}</h2>'
    f'<p class="rule-body">{body}</p></div></li>'
    for num, title, body in _RULES
)

# Plan view of the capture setup, drawn to the protocol's own numbers: a 15.7 m
# usable zone across lanes 3-5, the tripod 13.0 m back on the perpendicular, and
# the runner drawn as the Strideo mark itself (nested SVG, see ui.styles.mark_svg)
# rather than a hand-drawn approximation of it.
#
# Written with no blank lines and no 4-space indentation on purpose: Streamlit
# renders markdown, and a blank line ends an HTML block while a 4-space indent
# starts a code block - either one turns the rest of this SVG into escaped text
# on the page. Colours are literal hex rather than var(--token) because the SVG
# is inlined into markdown, where the custom properties do resolve, but only if
# the block survives parsing; keeping both concerns separate is less fragile.
_GEOMETRY_SVG = (
    '<svg viewBox="0 0 520 300" role="img" aria-label="Plan view: a tripod'
    " placed 13 metres to the side of a 15.7 metre usable capture zone on lanes"
    ' 3 to 5, at hip height">'
    # track band + lane line
    '<text x="40" y="40" font-family="JetBrains Mono, monospace"'
    ' font-size="11" letter-spacing="1.4" fill="#8B93A1">TRACK · LANES'
    " 3–5</text>"
    '<text x="480" y="40" text-anchor="end"'
    ' font-family="JetBrains Mono, monospace" font-size="10"'
    ' letter-spacing="1.2" fill="#8B93A1">← either direction →</text>'
    '<rect x="40" y="52" width="440" height="62" rx="4" fill="#ECF1F5"/>'
    '<line x1="40" y1="83" x2="480" y2="83" stroke="rgba(217,222,229,0.9)"'
    ' stroke-dasharray="6 8"/>'
    # usable-zone dimension
    '<line x1="40" y1="128" x2="480" y2="128" stroke="#8B93A1"'
    ' stroke-width="1"/>'
    '<line x1="40" y1="122" x2="40" y2="134" stroke="#8B93A1"'
    ' stroke-width="1"/>'
    '<line x1="480" y1="122" x2="480" y2="134" stroke="#8B93A1"'
    ' stroke-width="1"/>'
    '<text x="260" y="148" text-anchor="middle"'
    ' font-family="JetBrains Mono, monospace" font-size="11"'
    ' letter-spacing="1.2" fill="#5B6471">15.7 m usable zone</text>'
    # runner, either direction: the brand mark itself, centred on the tripod's
    # own centreline (x=260) and vertically centred in the 52-114 track band
    + mark_svg("recording-guide-mark-grad", size=44, x=238, y=61) +
    # field of view
    '<path d="M260 236 L96 172 M260 236 L424 172" stroke="#2D69DE"'
    ' stroke-width="1" stroke-dasharray="4 6" opacity="0.55" fill="none"/>'
    # perpendicular distance
    '<line x1="260" y1="172" x2="260" y2="232" stroke="#2D69DE"'
    ' stroke-width="1.5"/>'
    '<path d="M256 178 L260 170 L264 178" stroke="#2D69DE" stroke-width="1.5"'
    ' fill="none"/>'
    '<text x="272" y="206" font-family="JetBrains Mono, monospace"'
    ' font-size="12" fill="#2D69DE">13.0 m</text>'
    # tripod
    '<rect x="228" y="236" width="64" height="34" rx="6" fill="#0B1423"/>'
    '<circle cx="260" cy="253" r="8" fill="#2D69DE"/>'
    '<text x="260" y="290" text-anchor="middle"'
    ' font-family="JetBrains Mono, monospace" font-size="11"'
    ' letter-spacing="1.2" fill="#5B6471">TRIPOD · 95–100 cm</text>'
    "</svg>"
)

# ─────────────────────────────────────────────────────────────
# Good / avoid
# ─────────────────────────────────────────────────────────────

_GOOD = [
    "Full body visible throughout, feet never cropped",
    "Even daylight, no backlighting, foot sharp at contact",
    "Plain matte backdrop at least 5 m wide",
    "Empty lane either side of the athlete",
    "The runner's standing height entered exactly",
]

_AVOID = [
    "Shooting from behind, head-on, or at an angle",
    "30 fps, or slow-motion with a variable frame rate",
    "Ultra-wide lens: barrel distortion breaks calibration",
    "Other runners crossing the frame (occlusion)",
    "Handheld footage, or stabilisation left on with a tripod",
]


def _list_items(items: list[str]) -> str:
    return "".join(f"<li>{item}</li>" for item in items)


# Plain (untinted) background, continuous with the Hero above it, all the way
# through the good/avoid box (user feedback 2026-08-27, corrected: the plain
# "darker" background - .band's white overlay actually reads LIGHTER than the
# page's own default ambient gradient, confirmed by direct pixel sampling -
# should run from the top through good/avoid, not be its own tinted island).
st.markdown(
    f"""
<div class="guide-2col">
    <ol class="rule-list">{_rules_html}</ol>
    <div class="geo-card">
      <h2 class="geo-title">Camera geometry</h2>
      {_GEOMETRY_SVG}
      <p class="geo-caption">perpendicular · 13.0 m · hip height · no zoom</p>
    </div>
  </div>
  <div class="gv-box">
    <div class="gv-grid">
      <div class="gv-card good">
        <p class="gv-label">good</p>
        <ul>{_list_items(_GOOD)}</ul>
      </div>
      <div class="gv-card avoid">
        <p class="gv-label">avoid</p>
        <ul>{_list_items(_AVOID)}</ul>
      </div>
    </div>
    <div class="gv-note">
      <p><strong>Why the plain background matters.</strong> It is the single
      biggest fix for <strong>ghost-lock</strong>, where the pose tracker
      latches onto background texture instead of the runner for the first
      60–110 frames of a clip. Bleachers, fences and spectators are the usual
      culprits.</p>
    </div>
  </div>
""",
    unsafe_allow_html=True,
)


# ─────────────────────────────────────────────────────────────
# Full capture spec (the protocol distillation, kept in full)
# ─────────────────────────────────────────────────────────────

_SPEC_CARDS = [
    (
        "Camera settings",
        [
            ("resolution", "4K @ 60 fps"),
            ("camera app", "Standard app, daylight"),
            ("lens", "Main 1× (never ultra-wide)"),
            ("orientation", "Landscape"),
            ("stabilisation", "Off (on if handheld)"),
            ("hdr", "Off"),
            ("grid overlay", "On, to level the horizon"),
        ],
    ),
    (
        "Camera position",
        [
            ("mounting", "Fixed tripod, zero movement"),
            ("angle", "Side view (sagittal plane)"),
            ("distance", "13.0 m, perpendicular"),
            ("height", "95–100 cm (hip height)"),
            ("visible width", "≈ 17.0 m, 15.7 m usable"),
            ("image scale", "≈ 226 px per metre"),
            ("runner height", "≈ 413 px (300 px floor)"),
        ],
    ),
    (
        "Background & lighting",
        [
            ("backdrop", "Plain, matte, single colour, ≥ 5 m"),
            ("direction", "Away from bleachers and fences"),
            ("lighting", "Good ambient light, no backlight"),
            ("clothing", "Tight-fitting"),
            ("timing", "Roll before entry, hold until clear"),
        ],
    ),
    (
        "What to upload",
        [
            ("framing", "Full body, no other runners crossing"),
            ("duration", "≈ 2 s fast, up to 5 s slow"),
            ("trimming", "Longer clips cut to the first 5 s"),
            ("clips", "Three paces: threshold, 1500 m, 800 m"),
            ("format", "MP4 or MOV (M4V, AVI, WebM accepted)"),
        ],
    ),
]

_spec_html = "".join(
    '<div class="spec-card">'
    f'<h2 class="spec-title">{title}</h2>'
    + "".join(
        f'<div class="spec-row"><span class="spec-key">{key}</span>'
        f'<span class="spec-val">{value}</span></div>'
        for key, value in rows
    )
    + "</div>"
    for title, rows in _SPEC_CARDS
)

# ─────────────────────────────────────────────────────────────
# Capture spec + CTA, one continuous tinted band (user feedback 2026-08-27:
# the capture-spec band should include the button, not end before it). The
# button is a real widget, so raw-div nesting doesn't work here - a marker
# span plus :has() lets the CSS style this container the same way .band
# styles a plain div, same technique as Home's about-mark/cta-mark.
# ─────────────────────────────────────────────────────────────

with st.container():
    st.markdown(
        f"""
<span class="capture-band-mark"></span>
<div class="section-head">
  <div>
    <p class="eyebrow">capture spec</p>
    <h2 class="section-h2">The whole checklist</h2>
  </div>
</div>
<div class="spec-grid">{_spec_html}</div>
""",
        unsafe_allow_html=True,
    )
    _cta, _ = st.columns([1, 2.4])
    with _cta:
        if st.button(
            "Analyse a clip",
            type="primary",
            use_container_width=True,
            key="guide_cta",
        ):
            st.session_state["scroll_upload_top"] = True
            st.switch_page("pages/upload.py")

render_footer()
