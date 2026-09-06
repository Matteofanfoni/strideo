# app/app.py
import streamlit as st
import sys
from pathlib import Path

# Add app/ directory to sys.path so `ui.styles` is importable
app_dir = Path(__file__).parent
if str(app_dir) not in sys.path:
    sys.path.insert(0, str(app_dir))

import preload  # noqa: F401, E402
from ui.styles import (  # noqa: E402
    inject_shared_styles,
    inject_page_css,
    render_navbar,
    render_footer,
)
from ui.recaptcha import render_recaptcha, verify_recaptcha  # noqa: E402

st.set_page_config(
    page_title="strideo.it",
    page_icon=str(app_dir / "assets" / "favicon.png"),
    layout="wide",
)

inject_shared_styles()
render_navbar("Home")

# Home-specific CSS. Section chrome (.eyebrow, .section-h2, .band, .stat-grid,
# .note-card) is shared and lives in ui/styles.py, and so is the CTA panel
# itself (.cta-mark/.cta-h2, moved to ui/styles.py's _CTA_PANEL_CSS so
# results.py's own "Save this session" panel can reuse the same navy-box
# language) - everything remaining here is unique to this page's other three
# blocks: hero, pipeline timeline, validation card.
#
# Three blocks need Streamlit widgets *inside* a styled surface (hero frame,
# about band, CTA panel), which raw HTML cannot contain. Those are keyed off a
# marker class with :has() - the same technique the navbar and footer already
# use - rather than by nth-child, so they survive Streamlit DOM changes.
inject_page_css("""
/* ── Hero ──
   Streamlit's built-in h1-h3 rules outrank a bare class selector and add their
   own vertical padding, so every display heading below states its size and
   resets that padding with !important. */
.hero-title{
  font-family: 'Space Grotesk', ui-sans-serif, system-ui, sans-serif;
  font-size: clamp(2.2rem, 3.9vw, 3.5rem) !important;
  font-weight: 700;
  letter-spacing: -0.035em;
  line-height: 1.0 !important;
  color: var(--text);
  margin: 16px 0 0 !important;
  padding: 0 !important;
}
.hero-grad{
  background: linear-gradient(90deg, var(--brand), var(--accent));
  -webkit-background-clip: text;
  background-clip: text;
  color: transparent;
}
.hero-lead{
  font-size: 1.1rem;
  line-height: 1.68;
  color: var(--text-muted);
  margin: 30px 0 28px !important;
  max-width: 54ch;
}
.hero-meta{
  display: flex;
  flex-wrap: wrap;
  gap: 14px 24px;
  margin: 26px 0 0 !important;
  font-family: 'JetBrains Mono', ui-monospace, monospace;
  font-size: 15px !important;
  text-transform: uppercase;
  letter-spacing: 0.14em;
}
.hero-meta dt{ color: var(--text-muted); }
.hero-meta dd{ margin: 5px 0 0 !important; color: var(--text); }

/* Hero frame: a navy, rounded, elevated plate holding the real pipeline frame
   grab with the metric HUD laid over its bottom edge. The plate is the
   st.container() that wraps the image - identified by .heroframe-mark, whose
   own element container is hidden (:has() still matches a display:none child).
   gap:0 keeps a navy strip from appearing between image and HUD. */
[data-testid="stElementContainer"]:has(.heroframe-mark){ display: none; }
[data-testid="stVerticalBlock"]:has(> [data-testid="stElementContainer"] .heroframe-mark){
  position: relative;
  border-radius: var(--radius-lg);
  overflow: hidden;
  background: var(--navy);
  box-shadow: var(--shadow-hero);
  gap: 0 !important;
  margin-top: 71px;
}
@media (max-width: 700px){
  [data-testid="stVerticalBlock"]:has(> [data-testid="stElementContainer"] .heroframe-mark){
    margin-top: 20px;
  }
}
[data-testid="stVerticalBlock"]:has(> [data-testid="stElementContainer"] .heroframe-mark)
  [data-testid="stImage"] img{
  display: block;
  border: none !important;
  border-radius: 0 !important;
}
/* Streamlit gives every element container `position: relative`, which would
   make the overlay position against the (zero-height) markdown cell sitting
   below the image instead of against the plate - the badge then lands under
   the frame and gets clipped by the plate's overflow. Force it static so the
   plate is the containing block. */
[data-testid="stVerticalBlock"]:has(> [data-testid="stElementContainer"] .heroframe-mark)
  [data-testid="stElementContainer"]:has(.hero-hud){
  margin: 0 !important;
  position: static !important;
}

/* Top-right, not top-left: the frame grab already has the pipeline's own HUD
   text burned into its top-left corner. */
.hero-badge{
  position: absolute;
  top: 12px;
  right: 12px;
  display: inline-flex;
  align-items: center;
  gap: 7px;
  background: rgba(11,20,35,0.8);
  backdrop-filter: blur(4px);
  border-radius: var(--radius-sm);
  padding: 4px 9px;
  font-family: 'JetBrains Mono', ui-monospace, monospace;
  font-size: 0.62rem;
  text-transform: uppercase;
  letter-spacing: 0.16em;
  color: var(--navy-foreground);
}
.hero-badge-dot{
  width: 6px; height: 6px; border-radius: 50%;
  background: var(--brand);
  flex-shrink: 0;
}
.hero-hud{
  position: absolute;
  inset: auto 0 0 0;
  display: flex;
  flex-wrap: wrap;
  gap: 8px;
  padding: 44px 12px 12px;
  background: linear-gradient(to top, rgba(11,20,35,0.92), transparent);
}
.hero-hud-chip{
  border: 1px solid rgba(243,245,248,0.15);
  background: rgba(11,20,35,0.7);
  backdrop-filter: blur(4px);
  border-radius: var(--radius-md);
  padding: 5px 10px;
}
.hero-hud-label{
  display: block;
  font-family: 'JetBrains Mono', ui-monospace, monospace;
  font-size: 0.58rem;
  text-transform: uppercase;
  letter-spacing: 0.16em;
  color: rgba(243,245,248,0.6);
}
.hero-hud-value{
  font-family: 'JetBrains Mono', ui-monospace, monospace;
  font-size: 0.85rem;
  color: var(--navy-foreground);
}
.hero-hud-unit{ font-size: 0.62rem; color: rgba(243,245,248,0.6); margin-left: 4px; }
.hero-caption{
  font-family: 'JetBrains Mono', ui-monospace, monospace;
  font-size: 15px !important;
  text-transform: uppercase;
  letter-spacing: 0.14em;
  margin: 26px 0 0 !important;
}
.hero-caption-label{
  color: var(--text-muted);
  margin: 0 !important;
}
.hero-caption-value{
  color: var(--text);
  margin: 5px 0 0 !important;
}

/* Secondary (outline) button - Streamlit renders every st.button with the
   same navy fill, so the one that must read as secondary is re-styled by
   its widget key (.st-key-<key> lands on the element container). */
.st-key-hero_guide button{
  background: var(--card) !important;
  color: var(--text) !important;
  border: 1px solid var(--border) !important;
}
.st-key-hero_guide button:hover{
  background: var(--secondary) !important;
}
/* "Get in touch" sits alone on the page's own light background (no dark
   primary button beside it, unlike the hero's pairing), so the same
   light-outline treatment read as nearly invisible there. The brand
   gradient (same teal-to-blue as .hero-grad and the wordmark) gives it its
   own contrast against the light page instead of borrowing the hero's
   pairing-dependent style. */
.st-key-about_contact button{
  background: linear-gradient(90deg, var(--brand), var(--accent)) !important;
  color: var(--navy-foreground) !important;
  border: none !important;
  /* Same fixed box as the CTA panel's three actions (233x48, measured
     live against the hero buttons) - the column-ratio trick used there
     doesn't reach the same number here since this button sits in a
     full-width column, not a narrower one, so pin the box directly
     instead of fighting ratios. */
  width: 233px !important;
  height: 48px !important;
}
.st-key-about_contact button:hover{
  /* Colour shift, not a lift - matches how every other action on the page
     responds to hover (page-links, the primary/secondary st.button
     styles). Brightening the gradient is the equivalent "colour change"
     for a gradient fill. The gradient itself has to be re-asserted here,
     not just the filter: the generic [data-testid="stButton"] button:hover
     rule (solid navy) is MORE specific than this button's own non-hover
     background rule, so without a same-specificity :hover rule of its own
     to compete with it, that generic rule silently won and blanked the
     gradient out to solid navy on hover (confirmed live). */
  background: linear-gradient(90deg, var(--brand), var(--accent)) !important;
  filter: brightness(1.08);
}

/* ── Pipeline timeline ── */
/* The heading now breaks on its own explicit <br>, not on the shared
   .section-h2's 34ch measure - remove that width cap here (scoped to
   .band only) so the second, longer line doesn't wrap a second time. */
.band .section-h2{ max-width: none; }
.timeline{ margin-top: 40px; }
.timeline-rail{
  position: relative;
  height: 1px;
  background: var(--border);
  margin-bottom: 0;
}
.timeline-rail-fill{
  position: absolute;
  inset: 0 auto 0 0;
  width: 100%;
  background: linear-gradient(90deg, var(--brand), var(--accent));
}
/* !important on the list resets too: Streamlit's markdown stylesheet indents
   every <ol> and would push the first step off the page column. */
.timeline-steps{
  display: grid;
  grid-template-columns: repeat(4, 1fr);
  gap: 22px;
  list-style: none !important;
  margin: 0 !important;
  padding: 0 !important;
}
/* Streamlit indents <li> with its own margin/padding - zero both so the step
   lines up with the rail dot and the section heading above it. */
.timeline-step{
  position: relative;
  margin: 0 !important;
  padding: 26px 0 0 !important;
}
.timeline-step::before{
  content: '';
  position: absolute;
  top: -5px;
  left: 0;
  width: 10px;
  height: 10px;
  border-radius: 50%;
  background: var(--accent);
}
.timeline-row{ display: flex; align-items: baseline; gap: 11px; }
.timeline-num{
  font-family: 'JetBrains Mono', ui-monospace, monospace;
  font-size: 0.85rem;
  color: var(--brand-text);
}
.timeline-title{
  font-family: 'Space Grotesk', ui-sans-serif, system-ui, sans-serif;
  font-size: 1rem !important;
  font-weight: 600;
  line-height: 1.35 !important;
  color: var(--text);
  margin: 0 !important;
  padding: 0 !important;
}
.timeline-body{
  font-size: 0.9rem;
  line-height: 1.65;
  color: var(--text-muted);
  margin: 9px 0 0 !important;
}
.timeline-meta{
  font-family: 'JetBrains Mono', ui-monospace, monospace;
  font-size: 0.68rem;
  text-transform: uppercase;
  letter-spacing: 0.14em;
  color: var(--text-subtle);
  margin: 12px 0 0 !important;
}

/* ── Validation evidence card (heading + photo + note in one bordered box) ── */
[data-testid="stVerticalBlock"]:has(> [data-testid="stElementContainer"] .valcard-mark){
  background: var(--card);
  border: 1px solid var(--border);
  border-radius: var(--radius-lg);
  padding: 24px 26px;
}
.valcard-head{
  display: flex;
  flex-wrap: wrap;
  align-items: baseline;
  justify-content: space-between;
  gap: 6px 14px;
  margin-bottom: 4px;
}
.valcard-title{
  font-family: 'Space Grotesk', ui-sans-serif, system-ui, sans-serif;
  font-size: 1rem !important;
  font-weight: 600;
  line-height: 1.4 !important;
  color: var(--text);
  margin: 0 !important;
  padding: 0 !important;
}
.valcard-tag{
  font-family: 'JetBrains Mono', ui-monospace, monospace;
  font-size: 0.68rem;
  text-transform: uppercase;
  letter-spacing: 0.14em;
  color: var(--text-muted);
  margin: 0 !important;
}
.valcard-body{
  font-size: 0.9rem;
  line-height: 1.7;
  color: var(--text-muted);
  margin: 14px 0 0 !important;
}
.valcard-body strong{ color: var(--text); font-weight: 600; }
.valcard-body .mono{
  font-family: 'JetBrains Mono', ui-monospace, monospace;
  color: var(--text);
}
div[data-testid="stImage"] img{
  border-radius: var(--radius-sm);
  border: 1px solid var(--border);
}
/* The calibration photo is near-4:3; at full column width it would make the
   evidence card twice the height of the stat grid beside it. Cap the height
   and centre it so the two columns stay in balance. */
[data-testid="stVerticalBlock"]:has(> [data-testid="stElementContainer"] .valcard-mark)
  div[data-testid="stImage"]{
  display: flex;
  justify-content: center;
}
[data-testid="stVerticalBlock"]:has(> [data-testid="stElementContainer"] .valcard-mark)
  div[data-testid="stImage"] img{
  max-height: 340px;
  width: auto !important;
  max-width: 100%;
}
div[data-testid="stImage"] figcaption{
  color: var(--text-subtle) !important;
  font-size: 0.8rem !important;
  text-align: center;
}

/* ── Roadmap + about prose ──
   No max-width: these sections run the full width of the page column rather
   than sitting in a narrow measure on the left. */
.prose{
  font-size: 1rem;
  line-height: 1.75;
  color: var(--text-muted);
  margin: 20px 0 0 !important;
}
.prose strong{ color: var(--text); font-weight: 600; }
.about-meta{
  font-family: 'JetBrains Mono', ui-monospace, monospace;
  font-size: 0.68rem;
  text-transform: uppercase;
  letter-spacing: 0.14em;
  color: var(--text-muted);
  margin: 26px 0 10px !important;
}

/* About: plain background (L1 - systematic section alternation). Pipeline
   and Roadmap are the tinted .band sections either side of it, so About
   stays a normal in-flow block rather than a second full-bleed tint. No extra
   top margin of its own - Roadmap's .band already carries a 56px bottom
   margin, same as how Validation (also plain, after Pipeline's .band) needs
   none either; adding one here doubled the gap above About. */
[data-testid="stHorizontalBlock"]:has(.about-mark){
  align-items: flex-start !important;
}

@media (max-width: 900px){
  .timeline-steps{ grid-template-columns: repeat(2, 1fr); }
}
@media (max-width: 700px){
  .timeline-rail{ display: none; }
  .timeline-steps{ grid-template-columns: 1fr; }
  .timeline-step::before{ display: none; }
  .timeline-step{ padding-top: 0; }
  [data-testid="stHorizontalBlock"]:has(.about-mark){ padding: 40px 20px !important; }
}
""")

# Public source repository (shown as a link on the home page). The public
# mirror goes live with the Day-6 Hugging Face deploy - update if the URL differs.
REPO_URL = "https://github.com/Matteofanfoni/strideo"

_ASSETS_DIR = Path(__file__).parent / "assets"


# ─────────────────────────────────────────────────────────────
# Contact dialog (server-side POST to FormSubmit; the endpoint lives in
# .streamlit/secrets.toml, gitignored, and never reaches the browser).
# ─────────────────────────────────────────────────────────────

try:
    _CONTACT_ENDPOINT = st.secrets.get("contact_endpoint", "")
    # FormSubmit refuses server-side posts that lack a web Referer/Origin
    # ("open this page through a web server"); send a plausible one.
    _CONTACT_REFERER = st.secrets.get("contact_referer", "https://strideo.it")
    _RECAPTCHA_SITE_KEY = st.secrets.get("recaptcha_site_key", "")
    _RECAPTCHA_SECRET_KEY = st.secrets.get("recaptcha_secret_key", "")
except Exception:
    _CONTACT_ENDPOINT = ""
    _CONTACT_REFERER = "https://strideo.it"
    _RECAPTCHA_SITE_KEY = ""
    _RECAPTCHA_SECRET_KEY = ""


def _send_contact(name: str, email: str, category: str, message: str):
    """POST the inquiry to the configured FormSubmit AJAX endpoint.

    Returns (ok, message). FormSubmit returns HTTP 200 even when it is asking
    you to confirm the address, so we inspect the JSON ``success`` flag and
    surface its ``message`` rather than trusting the status code alone.
    """
    import requests

    payload = {
        "name": name,
        "email": email,
        "_replyto": email,
        "category": category,
        "message": message,
        "_subject": f"Strideo contact - {category}",
        "_captcha": "false",
        "_template": "table",
    }
    resp = requests.post(
        _CONTACT_ENDPOINT,
        json=payload,
        headers={
            "Accept": "application/json",
            "Referer": _CONTACT_REFERER,
            "Origin": _CONTACT_REFERER,
        },
        timeout=15,
    )
    try:
        data = resp.json()
    except Exception:
        data = {}
    ok = str(data.get("success", "")).lower() == "true" if data else resp.ok
    return ok, data.get("message", "")


@st.dialog("Get in touch")
def _contact_dialog() -> None:
    st.write(
        "Questions, bug reports, or interest in collaborating on the research - "
        "send a message and it comes straight to me."
    )
    name = st.text_input("Name", key="ct_name", placeholder="Your name")
    email = st.text_input("Your email", key="ct_email", placeholder="you@example.com")
    category = st.selectbox(
        "Topic",
        [
            "General question",
            "Bug report",
            "Feature request",
            "Research / collaboration",
        ],
        key="ct_category",
    )
    message = st.text_area(
        "Message", key="ct_message", height=140, placeholder="What's on your mind?"
    )
    recaptcha_token = None
    if _RECAPTCHA_SITE_KEY:
        recaptcha_token = render_recaptcha(_RECAPTCHA_SITE_KEY)
    if st.button(
        "Send message", type="primary", use_container_width=True, key="ct_send"
    ):
        if not (name and email and message):
            st.warning("Please fill in your name, email, and a message.")
        elif not _CONTACT_ENDPOINT:
            st.info("The contact form isn't configured yet - please check back soon.")
        elif _RECAPTCHA_SITE_KEY and not verify_recaptcha(
            recaptcha_token, _RECAPTCHA_SECRET_KEY
        ):
            st.warning("Please complete the reCAPTCHA verification.")
        else:
            try:
                ok, msg = _send_contact(name, email, category, message)
            except Exception as exc:  # noqa: BLE001 - surface any transport error
                ok, msg = False, str(exc)
            if ok:
                st.success("Thanks - your message is on its way!")
            elif msg:
                # e.g. FormSubmit's "confirm your email" notice on first use.
                st.info(msg)
            else:
                st.error(
                    "Sorry, the message couldn't be sent right now. "
                    "Please try again later."
                )


# ─────────────────────────────────────────────────────────────
# Hero
# ─────────────────────────────────────────────────────────────

# Read off the burned-in HUD of app/assets/pose_overlay_strideonet.jpg - the
# chips restate that frame's own StrideoNet output legibly, they are not a
# second measurement. Rendered by scripts/render_strideonet_hero_overlay.py
# (S1_04_800m_1, frame 154) -- StrideoNet's own default result, not the
# opt-in Deterministic Kinematics Engine the old asset showed.
# Re-read 2026-08-31 when a metric change forced a regeneration (the HUD's "Economy" line
# became "VO ratio"). The four values below moved with it, from 198.1/146.4/
# 4.63/9.08: the asset predated a re-fit of the shipped ONNX, so the same
# frame now predicts slightly differently. These MUST be re-read from the image
# whenever that script is re-run, or the page contradicts the picture beside it.
_HERO_CHIPS = [
    ("CAD", "197.6", "spm"),
    ("GCT", "148.2", "ms"),
    ("STRIDE", "4.64", "m"),
    ("VO", "9.13", "cm"),
]


def _render_hero_frame() -> None:
    """The pose-overlay frame grab on a navy plate, with the metric HUD
    overlaid along its bottom edge.

    ``st.image`` is used (rather than an inline base64 ``<img>``) so the file
    is served once through Streamlit's media endpoint instead of being re-sent
    inside the page HTML on every rerun. Because Streamlit widgets cannot be
    nested inside raw HTML, the plate is the surrounding ``st.container()``,
    styled via the ``.heroframe-mark`` marker class.
    """
    chips = "".join(
        f'<div class="hero-hud-chip">'
        f'<span class="hero-hud-label">{label}</span>'
        f'<div class="hero-hud-value">{value}'
        f'<span class="hero-hud-unit">{unit}</span></div></div>'
        for label, value, unit in _HERO_CHIPS
    )
    with st.container():
        st.markdown('<span class="heroframe-mark"></span>', unsafe_allow_html=True)
        st.image(str(_ASSETS_DIR / "pose_overlay_strideonet.jpg"), width="stretch")
        st.markdown(
            '<div class="hero-badge"><span class="hero-badge-dot"></span>'
            "blazepose → StrideoNet</div>"
            f'<div class="hero-hud">{chips}</div>',
            unsafe_allow_html=True,
        )
    st.markdown(
        '<div class="hero-caption">'
        '<p class="hero-caption-label">output</p>'
        '<p class="hero-caption-value">800m · blazepose overlay · metrics hud</p>'
        "</div>",
        unsafe_allow_html=True,
    )


hero_left, hero_right = st.columns([1.05, 1], gap="large", vertical_alignment="top")

with hero_left:
    st.markdown(
        """
<p class="eyebrow">running gait analysis · open source</p>
<h1 class="hero-title">Your stride,<br>
<span class="hero-grad">measured</span> not guessed.</h1>
<p class="hero-lead">Strideo turns side-on smartphone clips into calibrated
biomechanics (cadence, ground contact time, stride length and
vertical oscillation) using pose estimation instead of lab hardware.</p>
""",
        unsafe_allow_html=True,
    )
    _b1, _b2, _ = st.columns([1, 1, 0.45])
    with _b1:
        if st.button(
            "Analyse a clip",
            type="primary",
            use_container_width=True,
            key="hero_upload",
        ):
            st.session_state["scroll_upload_top"] = True
            st.switch_page("pages/upload.py")
    with _b2:
        if st.button("How to film it", use_container_width=True, key="hero_guide"):
            st.switch_page("pages/recording_guide.py")
    # Same [1, 1, 0.45] column ratio as the button row above, so "input"
    # lands flush under "How to film it" instead of wherever a flex gap
    # after "metrics" happened to put it - text-length-proof, unlike a
    # hand-tuned margin would be.
    _m1, _m2, _ = st.columns([1, 1, 0.45])
    with _m1:
        st.markdown(
            '<dl class="hero-meta"><div><dt>metrics</dt>'
            "<dd>CAD · GCT · STRIDE · VO</dd></div></dl>",
            unsafe_allow_html=True,
        )
    with _m2:
        st.markdown(
            '<dl class="hero-meta"><div><dt>input</dt>'
            "<dd>landscape · 4k · 60 fps</dd></div></dl>",
            unsafe_allow_html=True,
        )

with hero_right:
    _render_hero_frame()


# ─────────────────────────────────────────────────────────────
# Pipeline timeline (full-bleed band, static HTML - no widgets inside)
# ─────────────────────────────────────────────────────────────

_PIPELINE_STEPS = [
    (
        "01",
        "Constant-rate conversion",
        "Variable-frame-rate phone footage is rewritten to a constant 60 fps "
        "so every timing measurement shares one clock.",
        "60 fps",
    ),
    (
        "02",
        "Pose estimation",
        "MediaPipe BlazePose, a pre-trained pose estimation computer-vision "
        "model, tracks 33 body landmarks and calibrates real-world scale "
        "from the runner's own height.",
        "33 landmarks",
    ),
    (
        "03",
        "StrideoNet",
        "An original FiLM pace-conditioned 2D CNN reads the landmarks "
        "directly and predicts cadence, ground contact time, stride length "
        "and vertical oscillation, without running a second pose model or "
        "detecting individual footstrikes.",
        "≤10ms GCT MAE",
    ),
    (
        "04",
        "Deterministic kinematics engine",
        "A second, independent computer-vision pre-trained model, "
        "RTMPose-x, adds hip and ankle landmarks and feeds a deterministic "
        "engine that detects initial contact and toe-off per footstrike "
        "using fixed rules (on request).",
        "opt-in",
    ),
]

_steps_html = "".join(
    f'<li class="timeline-step">'
    f'<div class="timeline-row"><span class="timeline-num">{num}</span>'
    f'<h3 class="timeline-title">{title}</h3></div>'
    f'<p class="timeline-body">{body}</p>'
    f'<p class="timeline-meta">{meta}</p></li>'
    for num, title, body, meta in _PIPELINE_STEPS
)

st.markdown(
    f"""
<div class="band">
  <p class="eyebrow">pipeline</p>
  <h2 class="section-h2">A hybrid approach,<br>one model that learns, one
  engine that doesn&rsquo;t.</h2>
  <div class="timeline">
    <div class="timeline-rail"><span class="timeline-rail-fill"></span></div>
    <ol class="timeline-steps">{_steps_html}</ol>
  </div>
</div>
""",
    unsafe_allow_html=True,
)


# ─────────────────────────────────────────────────────────────
# Validation
# ─────────────────────────────────────────────────────────────

# Every figure below is a copy of a documented fact, and the source of truth
# is the project's claims record: a claim not in it has
# not been checked and must not ship. Do not add a number here without a
# ledger row to point at, and carry a Tier B claim's caveat in the same breath
# as the claim itself. The content-drift mapping in
# The app content reference tracks what this file says; it is
# not the authority for whether the number is right.
#
# Two symmetric columns, not a shared lead + lopsided card/grid/note-card
# stack (a layout/density correction to an earlier pass, which got the content
# split right but kept too much chrome and let the two sides run to very
# different heights). Each column now carries the same three-part shape:
# one procedure sentence, its own result cards, one limitations paragraph.
# StrideoNet's and the engine's ground-contact figures are both MAE (mean
# absolute error) and, at v1.31, are genuinely computed over the same 45-clip
# cohort, which is why they're the sole result card on each side.
#
# This block was rewritten 2026-08-30. It previously paired "5.81 ms MAE
# (±6.74 ms SD) ... same cohort as StrideoNet's figure" with an 8-clip
# pre-validation cohort card, and every part of that was wrong: 5.81 was n=44
# against StrideoNet's n=45 so the cohort-equality claim was false; the figure
# was flattered by a gate that had been retired from production; the fix
# retired it outright rather than beating it; and the cohort card described a
# different shoot from the number printed beside it. The ledger names that
# exact string as a claim that must not be made. The engine's SD here is A5's
# ±3.60 ms. StrideoNet's card carries no SD because the ledger publishes none
# for it at v1.31 (A6), and inventing one is precisely the drift this block is
# the cautionary example of.
_STRIDEONET_STATS = [
    ("45 clips / 5 runners", "leave-one-runner-out cross-validation"),
    ("7.57 ms MAE", "ground contact time - meets the ≤10 ms target"),
]
_ENGINE_STATS = [
    (
        "45 clips / 5 runners",
        "the same cohort, against the same annotation",
    ),
    (
        "3.99 ±3.60 ms MAE",
        "ground contact time - constants fitted on this same cohort",
    ),
]

st.markdown(
    """
<p class="eyebrow">validation</p>
<h2 class="section-h2 validation-h2">How it&rsquo;s validated</h2>
""",
    unsafe_allow_html=True,
)

val_left, val_right = st.columns(2, gap="large")

with val_left:
    nn_cells = "".join(
        f'<div class="stat-cell"><p class="stat-value">{value}</p>'
        f'<p class="stat-label">{label}</p></div>'
        for value, label in _STRIDEONET_STATS
    )
    st.markdown(
        f"""
<span class="valcard-mark"></span>
<h3 class="valcard-title">StrideoNet</h3>
<p class="valcard-body">A trained machine learning model, checked by
leave-one-out cross-validation on a 5-runner cohort, holding out each runner
in turn so no result is measured on data the model has seen; cadence and
ground contact time train on Kinovea-annotated ground truth, and vertical
oscillation trains on real annotation where available, falling back to a
pipeline-fitted correction on the remaining clips.</p>
<div class="stat-grid" style="margin:14px 0;">{nn_cells}</div>
<p class="valcard-body">This is a feasibility demonstration on 5 runners, not
a full external validation. On ground contact time the deterministic
kinematics engine's figure is the lower of the two, but the two numbers are
not earned the same way: this one is held out runner by runner, while the
engine's constants were fitted on the very clips it is scored on. More
runners is the clearest lever, for both.</p>
""",
        unsafe_allow_html=True,
    )

with val_right:
    engine_cells = "".join(
        f'<div class="stat-cell"><p class="stat-value">{value}</p>'
        f'<p class="stat-label">{label}</p></div>'
        for value, label in _ENGINE_STATS
    )
    st.markdown(
        f"""
<span class="valcard-mark"></span>
<h3 class="valcard-title">Deterministic kinematics engine</h3>
<p class="valcard-body">A physics approach to deriving kinematic metrics
from velocity and displacement, with its output compared contact by contact
against manual Kinovea-annotated ground truth on the same 45-clip cohort,
against the same &plusmn;10&nbsp;ms ground-contact tolerance; it also runs
forefoot-vs-heel strike-pattern detection and features a two-pass ground
contact detector.</p>
<div class="stat-grid" style="margin:14px 0;">{engine_cells}</div>
<p class="valcard-body">Two caveats travel with that number. Every constant
producing its lead was fitted on this same cohort, and removing the
contact filter alone brings the two methods back to a tie. And the figure
is flattered by roughly 0.19&nbsp;ms: 18 of the 45 clips are scored on fewer
contacts than they contain, and the contacts that go missing are the hard
ones. Nothing here supports an accuracy claim for a different runner,
venue, camera geometry or frame rate.</p>
""",
        unsafe_allow_html=True,
    )


# ─────────────────────────────────────────────────────────────
# Where it's going
# ─────────────────────────────────────────────────────────────

st.markdown(
    """
<div class="band">
  <p class="eyebrow">roadmap</p>
  <h2 class="section-h2">Where it&rsquo;s going</h2>
  <p class="prose"><strong>StrideoNet</strong>, a FiLM pace-conditioned 2D
  CNN trained on a 5-runner dataset, now produces the app's default result.
  It reaches ground contact time within Strideo's own ≤10ms accuracy target
  without a second pose model (cutting per-clip processing time by about
  6x), without classifying strike pattern, and without detecting
  individual footstrikes. It also has the most room left to improve: more
  runners is the clearest lever on its remaining cadence and stride
  accuracy, which is why a second data-collection round is next.</p>
  <p class="prose">Strideo is built specifically for <strong>800 m and
  1500 m</strong> athletes, because middle-distance form is not static: cadence,
  contact time and stride length all shift as a runner moves from threshold
  toward race pace. Most consumer tools report one number per metric and average
  that shift away. Strideo is designed to surface it.</p>
</div>
""",
    unsafe_allow_html=True,
)


# ─────────────────────────────────────────────────────────────
# About (full-bleed band; contains the contact-dialog button)
# ─────────────────────────────────────────────────────────────

# One full-width column rather than a prose/button split: the marker class the
# band CSS keys off still lands inside an st.columns row, so the band styling
# works, but the text now runs the width of the page column.
(about_col,) = st.columns(1)

with about_col:
    st.markdown(
        """
<span class="about-mark"></span>
<p class="eyebrow">about</p>
<h2 class="section-h2">Why this exists</h2>
<p class="prose">I'm Matteo, an IB student with an interest in both running and
artificial intelligence, and this project is where those two worlds meet. As a
competitive 800 m and 1500 m runner, I've spent years feeling the small
mechanical details that separate a clean race from a sloppy one. GPS running
watches can already estimate cadence, ground contact time, stride length and
vertical oscillation from a wrist IMU, but the model behind those numbers is
closed and the hardware isn't cheap. I wanted an open-source alternative that
needs nothing but a phone: a computer-vision pipeline that tracks an athlete's
joints and stride patterns from ordinary training footage, with room to surface
more of the movement picture than a single wrist sensor ever could.</p>
<p class="prose">This started as curiosity about machine learning. It has
become a research project and may develop into an open-source product that
helps athletes and coaches, while pushing me to understand how AI can engage
with real physical performance, not just clean datasets on a screen.</p>
<p class="about-meta">questions · bug reports · research</p>
""",
        unsafe_allow_html=True,
    )
    if st.button("Get in touch", key="about_contact"):
        _contact_dialog()


# ─────────────────────────────────────────────────────────────
# CTA (navy panel; contains the upload button and the source link)
# ─────────────────────────────────────────────────────────────

# Single centred column: eyebrow, headline and the two actions all share one
# centre line. The panel styling keys off .cta-mark, which still sits inside an
# st.columns row here.
(cta_col,) = st.columns(1)

with cta_col:
    st.markdown(
        """
<div class="cta-mark">
  <p class="eyebrow">MIT licensed</p>
  <h2 class="cta-h2">Run it on your own athletes tonight.</h2>
</div>
""",
        unsafe_allow_html=True,
    )
    _, _cta_a, _cta_b, _cta_c, _ = st.columns([0.9, 1, 1, 1, 0.9])
    with _cta_a:
        if st.button("Analyse a clip", use_container_width=True, key="cta_upload"):
            st.session_state["scroll_upload_top"] = True
            st.switch_page("pages/upload.py")
    with _cta_b:
        st.page_link("pages/recording_guide.py", label="How to film it")
    with _cta_c:
        st.page_link(REPO_URL, label="View source")


render_footer()
