"""Shared CSS and UI components for the Strideo Streamlit app."""

import streamlit as st

# ─────────────────────────────────────────────────────────────
# CSS building blocks
# ─────────────────────────────────────────────────────────────

_BASE_CSS = """
@import url('https://fonts.googleapis.com/css2?family=Inter:wght@400;500;600;700;800&display=swap');

:root{
  /* ── Surfaces (crisp white + cool gray) ── */
  --bg: #F4F6FA;            /* very light cool-gray page  */
  --card: #FFFFFF;          /* card surface               */
  --card-2: #F0F3F8;        /* card gradient end / hover  */
  --card-hover: #F0F3F8;
  --border: #E2E7F0;        /* default hairline border    */
  --border-soft: rgba(15,23,42,0.06);
  --hover-fill: rgba(15,23,42,0.035);

  /* ── Text ── */
  --text: #161A23;          /* near-black                 */
  --text-muted: #545B6B;
  --text-subtle: #8A90A0;
  --on-brand: #FFFFFF;      /* text on brand-gradient surfaces */

  /* ── Brand (teal) + accent (blue) + violet (tertiary) ── */
  --brand: #0D9488;         /* teal-600 - primary         */
  --brand-hover: #14B8A6;   /* teal-500                   */
  --brand-glow: rgba(13,148,136,0.28);
  --accent: #2563EB;        /* blue-600 - gradient partner */
  --accent-glow: rgba(37,99,235,0.22);
  --violet: #7C3AED;        /* retained secondary accent  */
  --violet-glow: rgba(124,58,237,0.20);
  --green: #16A34A;
  --red: #DC2626;

  /* ── Shadows (soft, for light bg) ── */
  --shadow: 0 1px 3px rgba(16,24,40,0.06), 0 4px 14px rgba(16,24,40,0.06);
  --shadow-lg: 0 10px 30px rgba(16,24,40,0.12);

  --radius-lg: 24px;
  --radius-md: 16px;
  --radius-sm: 12px;

  /* ── Legacy aliases (old page CSS still references these names) ── */
  --bg-dark: var(--bg);
  --bg-card: var(--card);
  --bg-card-hover: var(--card-hover);
  --orange: var(--brand);
  --orange-hover: var(--brand-hover);
  --orange-glow: var(--brand-glow);
  --cyan: var(--violet);
  --cyan-glow: var(--violet-glow);
  --text-white: var(--text);
}

* {
  font-family: 'Inter', -apple-system, BlinkMacSystemFont, sans-serif;
}

html, body, [data-testid="stAppViewContainer"], [data-testid="stApp"]{
  background: var(--bg) !important;
  background-image:
    radial-gradient(ellipse at 20% 20%, rgba(13, 148, 136, 0.06) 0%, transparent 50%),
    radial-gradient(ellipse at 80% 80%, rgba(37, 99, 235, 0.05) 0%, transparent 55%) !important;
  color: var(--text);
}

[data-testid="stMain"]{ background: transparent !important; }
[data-testid="stSidebar"]{ display: none !important; }

.block-container{
  padding-top: 1rem !important;
  padding-bottom: 2rem !important;
  max-width: 1440px;
}

header[data-testid="stHeader"]{ background: transparent !important; }
div[data-testid="stToolbar"]{ visibility: hidden; height: 0px; }
footer{ visibility: hidden; height: 0px; }
[data-testid="stDecoration"]{ display: none; }

/* Default text colour for Streamlit-rendered markdown/labels on light bg */
[data-testid="stMarkdownContainer"], [data-testid="stCaptionContainer"]{
  color: var(--text);
}

[data-testid="stHorizontalBlock"]{
  gap: 20px !important;
}

/* ── Mobile: collapse every multi-column row to a single column ── */
@media (max-width: 700px){
  .block-container{
    padding-left: 0.6rem !important;
    padding-right: 0.6rem !important;
  }
  [data-testid="stHorizontalBlock"]{
    flex-wrap: wrap !important;
    gap: 12px !important;
  }
  [data-testid="stColumn"]{
    flex: 1 1 100% !important;
    width: 100% !important;
    min-width: 100% !important;
  }
}
"""

_ANIMATIONS_CSS = """
@keyframes pulse-glow {
  0%, 100% { box-shadow: 0 0 20px var(--orange-glow), 0 0 40px rgba(13, 148, 136, 0.2); }
  50% { box-shadow: 0 0 30px var(--orange-glow), 0 0 60px rgba(13, 148, 136, 0.3); }
}

@keyframes float {
  0%, 100% { transform: translateY(0px); }
  50% { transform: translateY(-6px); }
}

@keyframes border-glow {
  0%, 100% { opacity: 0.5; }
  50% { opacity: 1; }
}

@keyframes shimmer {
  0% { background-position: -200% center; }
  100% { background-position: 200% center; }
}

@keyframes gradient-shift {
  0%, 100% { background-position: 0% 50%; }
  50% { background-position: 100% 50%; }
}

@keyframes border-pulse {
  0%, 100% { border-color: var(--orange); opacity: 0.8; }
  50% { border-color: var(--accent); opacity: 1; }
}
"""

_NAVBAR_CSS = """
/* Top nav built from st.page_link (NOT raw <a href>): URL navigation spawns a
   new browser session and resets st.session_state, which would wipe uploads
   and results. st.page_link navigates client-side and preserves the session.

   We key the navbar styling off the brand lockup (.nav-brand-row), which only
   the navbar has - NOT :has(stPageLink), since body page links (e.g. the
   recording-guide pointer) also live in column rows and must not become bars. */
[data-testid="stHorizontalBlock"]:has(.nav-brand-row){
  background: linear-gradient(135deg, var(--card) 0%, var(--card-2) 100%);
  border: 1px solid var(--border);
  border-radius: var(--radius-md);
  box-shadow: var(--shadow);
  padding: 8px 18px !important;
  margin-bottom: 26px !important;
  align-items: center !important;
  position: relative;
  overflow: hidden;
}
/* Shimmer accent line along the top of the bar */
[data-testid="stHorizontalBlock"]:has(.nav-brand-row)::before{
  content: '';
  position: absolute;
  top: 0; left: 0; right: 0;
  height: 2px;
  background: linear-gradient(90deg, transparent, var(--brand), var(--accent), transparent);
  background-size: 200% 100%;
  animation: shimmer 3s infinite linear;
}

.nav-brand-row{
  display: flex;
  align-items: center;
  gap: 10px;
}
/* Root cause of the brand sitting ~5px low: the brand is an st.markdown cell,
   so Streamlit wraps it in stElementContainer/stMarkdown wrappers that carry
   default vertical margin. The page-link cells have no such wrapper, so
   align-items:center put the two groups on different centerlines. Zero the
   wrappers' margins and center the brand column's content directly - robust
   across breakpoints, no magic pixel offset. */
[data-testid="stColumn"]:has(.nav-brand-row){
  display: flex;
  flex-direction: column;
  justify-content: center;
}
[data-testid="stColumn"]:has(.nav-brand-row) [data-testid="stElementContainer"],
[data-testid="stColumn"]:has(.nav-brand-row) [data-testid="stMarkdown"],
[data-testid="stColumn"]:has(.nav-brand-row) [data-testid="stMarkdownContainer"]{
  margin: 0 !important;
  padding: 0 !important;
}
.nav-brand-badge{
  width: 22px;
  height: 22px;
  border-radius: 5px;
  background: linear-gradient(135deg, var(--brand) 0%, var(--accent) 100%);
  display: inline-flex;
  align-items: center;
  justify-content: center;
  flex-shrink: 0;
  box-shadow: 0 2px 8px var(--brand-glow);
}
.nav-brand-name{
  font-weight: 800;
  font-size: 1.25rem;
  color: var(--text);
  letter-spacing: -0.5px;
  /* Drop inherited body leading so the glyphs sit on the row's centerline
     instead of being pushed up by extra line-height above them. */
  line-height: 1;
}

[data-testid="stPageLink"]{ margin: 0 !important; }

/* Body page links (outside the navbar) read as plain text links - brand
   colour, left-aligned, underline on hover. */
[data-testid="stPageLink"] a{
  padding: 2px 0 !important;
  border-radius: var(--radius-sm) !important;
  transition: all 0.2s ease !important;
}
[data-testid="stPageLink"] a p{
  color: var(--brand) !important;
  font-weight: 600 !important;
  white-space: normal !important;  /* wrap long labels instead of clipping */
}
[data-testid="stPageLink"] a:hover p{ text-decoration: underline !important; }

/* Navbar page links - centred, muted, button-like (override the body style). */
[data-testid="stHorizontalBlock"]:has(.nav-brand-row) [data-testid="stPageLink"] a{
  display: flex !important;
  justify-content: center !important;
  padding: 7px 12px !important;
}
[data-testid="stHorizontalBlock"]:has(.nav-brand-row)
  [data-testid="stPageLink"] a:hover{ background: var(--hover-fill) !important; }
[data-testid="stHorizontalBlock"]:has(.nav-brand-row) [data-testid="stPageLink"] a p{
  color: var(--text-muted) !important;
  font-size: 0.9rem !important;
  white-space: nowrap !important;  /* nav items stay on one line */
}
[data-testid="stHorizontalBlock"]:has(.nav-brand-row)
  [data-testid="stPageLink"] a:hover p{
  color: var(--text) !important;
  text-decoration: none !important;
}
[data-testid="stHorizontalBlock"]:has(.nav-brand-row)
  [data-testid="stPageLink"] a[aria-current="page"]{
  background: rgba(13, 148, 136, 0.12) !important;
}
[data-testid="stHorizontalBlock"]:has(.nav-brand-row)
  [data-testid="stPageLink"] a[aria-current="page"] p{ color: var(--brand) !important; }

@media (max-width: 700px){
  /* Keep the navbar a horizontal row on mobile (the global rule stacks every
     column full-width; override it just here). Brand on its own line, the nav
     links flow in a wrapping row beneath it. */
  [data-testid="stHorizontalBlock"]:has(.nav-brand-row){
    flex-wrap: wrap !important;
    gap: 6px 3px !important;
    padding: 8px 12px !important;
  }
  [data-testid="stHorizontalBlock"]:has(.nav-brand-row)
    > [data-testid="stColumn"]:has(.nav-brand-row){
    flex: 0 0 100% !important;
    width: 100% !important;
    min-width: 0 !important;
  }
  [data-testid="stHorizontalBlock"]:has(.nav-brand-row)
    > [data-testid="stColumn"]:has([data-testid="stPageLink"]){
    flex: 0 0 auto !important;
    width: auto !important;
    min-width: 0 !important;
  }
  .nav-brand-name{ font-size: 1.1rem; }
  [data-testid="stHorizontalBlock"]:has(.nav-brand-row)
    [data-testid="stPageLink"] a{ padding: 4px 5px !important; }
  [data-testid="stHorizontalBlock"]:has(.nav-brand-row)
    [data-testid="stPageLink"] a p{ font-size: 0.75rem; }
}
"""

_PAGE_HEADER_CSS = """
.page-header{
  text-align: center;
  margin-bottom: 36px;
}

.page-title{
  font-size: 2.4rem;
  font-weight: 800;
  background: linear-gradient(135deg, var(--text) 0%, var(--text-muted) 100%);
  -webkit-background-clip: text;
  -webkit-text-fill-color: transparent;
  background-clip: text;
  margin: 0 0 12px 0;
  letter-spacing: -0.5px;
}

.page-subtitle{
  font-size: 1.1rem;
  color: var(--text-muted);
  margin: 0;
}

@media (max-width: 700px){
  .page-title{ font-size: 1.8rem; }
}
"""

_CARD_CSS = """
.blue-card{
  background: linear-gradient(135deg, var(--bg-card) 0%, var(--card-2) 100%);
  border-radius: var(--radius-lg);
  padding: 32px 28px;
  margin-bottom: 20px;
  border: 1px solid var(--border);
  position: relative;
  overflow: hidden;
  transition: all 0.4s ease;
}

.blue-card::before {
  content: '';
  position: absolute;
  top: 0; left: 0; right: 0;
  height: 2px;
  background: linear-gradient(90deg, transparent, var(--orange), transparent);
  animation: shimmer 3s infinite linear;
  background-size: 200% 100%;
}

.blue-card:hover {
  transform: translateY(-4px);
  box-shadow: var(--shadow-lg);
  border-color: rgba(13, 148, 136, 0.2);
}

.orange-card{
  background: linear-gradient(135deg, var(--orange) 0%, var(--accent) 100%);
  background-size: 200% 200%;
  animation: gradient-shift 6s ease infinite;
  border-radius: var(--radius-lg);
  padding: 32px 28px;
  border: 1px solid rgba(255,255,255,0.2);
  position: relative;
  overflow: hidden;
  transition: all 0.4s ease;
}

.orange-card::before {
  content: '';
  position: absolute;
  top: 0; left: 0; right: 0; bottom: 0;
  background: linear-gradient(135deg, transparent 0%, rgba(255,255,255,0.1) 50%, transparent 100%);
  pointer-events: none;
}

.orange-card:hover {
  transform: translateY(-4px) scale(1.01);
  box-shadow: 0 15px 40px var(--orange-glow);
}

.card-title{
  font-size: 1.6rem;
  font-weight: 700;
  color: var(--text-white);
  margin: 0 0 16px 0;
}

.card-text{
  color: var(--text-muted);
  font-size: 1.05rem;
  line-height: 1.7;
  margin: 0;
  max-width: 72ch;  /* keep prose readable even in a full-width card */
}

.orange-card .card-text{
  color: rgba(255,255,255,0.95);
}

@media (max-width: 700px){
  .blue-card, .orange-card{ padding: 22px 18px; }
  .card-title{ font-size: 1.3rem; }
  .card-text{ font-size: 0.98rem; }
}
"""

_RESULT_CARD_CSS = """
/* Responsive metric grid - cards reflow and wrap on narrow/mobile screens
   without relying on st.columns (which would leave awkward gaps). */
.result-grid {
  display: grid;
  grid-template-columns: repeat(auto-fit, minmax(150px, 1fr));
  gap: 14px;
  margin: 6px 0 4px;
}
.result-card {
  background: linear-gradient(135deg, var(--bg-card) 0%, var(--card-2) 100%);
  border-radius: var(--radius-md);
  padding: 18px 22px;
  border-left: 4px solid var(--green);
  border-top: 1px solid var(--border-soft);
  /* Equal-height cards in a row even when some have an extra detail line */
  height: 100%;
  box-sizing: border-box;
}
/* Make columns that contain a result-card stretch so height:100% resolves */
[data-testid="stColumn"]:has(.result-card){ align-items: stretch; }
[data-testid="stColumn"]:has(.result-card) [data-testid="stMarkdownContainer"]{
  height: 100%;
}
.result-card.error { border-left-color: var(--red); }
.result-card.warn { border-left-color: #FF9800; }
.result-card.blue { border-left-color: var(--orange); }
.result-label {
  color: var(--text-subtle);
  font-size: 0.75rem;
  margin: 0 0 4px;
  text-transform: uppercase;
  letter-spacing: 0.8px;
  font-weight: 600;
}
.result-value {
  color: var(--text-white);
  font-size: 1.15rem;
  font-weight: 700;
  margin: 0;
}
.result-detail {
  color: var(--text-muted);
  font-size: 0.8rem;
  margin: 4px 0 0;
}
"""

_BUTTON_CSS = """
.stButton > button[kind="primary"] {
  background: linear-gradient(135deg, var(--orange) 0%, var(--accent) 100%) !important;
  color: white !important;
  border: none !important;
  border-radius: var(--radius-md) !important;
  padding: 14px 32px !important;
  font-size: 1.05rem !important;
  font-weight: 700 !important;
  font-family: 'Inter', sans-serif !important;
  box-shadow: 0 4px 20px var(--orange-glow) !important;
  animation: pulse-glow 2s ease-in-out infinite;
  transition: all 0.3s ease !important;
}

.stButton > button[kind="primary"]:hover {
  transform: scale(1.03) !important;
  box-shadow: 0 8px 30px var(--orange-glow) !important;
}

.stButton > button:disabled {
  background: linear-gradient(135deg, var(--card-2) 0%, var(--border) 100%) !important;
  color: var(--text-subtle) !important;
  box-shadow: none !important;
  animation: none !important;
}

[data-testid="stButton"] button {
  background: linear-gradient(135deg, var(--orange) 0%, var(--accent) 100%) !important;
  color: white !important;
  border: none !important;
  border-radius: var(--radius-md) !important;
  padding: 12px 0 !important;
  width: 100% !important;
  font-weight: 700 !important;
  font-size: 0.95rem !important;
  font-family: 'Inter', sans-serif !important;
  letter-spacing: 0.3px;
  box-shadow: 0 4px 16px var(--orange-glow) !important;
  transition: all 0.2s ease !important;
}
[data-testid="stButton"] button:hover {
  box-shadow: 0 8px 28px var(--orange-glow) !important;
  transform: translateY(-1px) !important;
}

/* Streamlit colours the label element directly, so the button's own `color`
   never reaches the text. Force the label to inherit it - white on the
   gradient buttons, muted on the transparent secondary/remove button. */
[data-testid="stButton"] button p,
[data-testid="stButton"] button span,
[data-testid="stButton"] button div,
.stButton > button[kind="primary"] p,
.stButton > button[kind="primary"] span,
.stButton > button[kind="primary"] div {
  color: inherit !important;
}
"""

_FORM_CSS = """
/* Labels */
[data-testid="stTextInput"] label p,
[data-testid="stTextArea"] label p,
[data-testid="stSelectbox"] label p,
[data-testid="stNumberInput"] label p {
  color: var(--text) !important;
  font-size: 0.88rem !important;
  font-weight: 600 !important;
  letter-spacing: 0.2px;
}

/* Visible containers — one rule sets bg, border, radius for every input type */
[data-baseweb="input"],
[data-testid="stSelectbox"] [data-baseweb="select"] > div,
[data-testid="stTextArea"] textarea {
  background: var(--card) !important;
  border: 1px solid #C3CCDC !important;
  border-radius: var(--radius-sm) !important;
}

[data-baseweb="input"]:focus-within,
[data-testid="stSelectbox"] [data-baseweb="select"] > div:focus-within,
[data-testid="stTextArea"] textarea:focus {
  border-color: var(--brand) !important;
  box-shadow: 0 0 0 2px rgba(13, 148, 136, 0.2) !important;
}

/* Inner <input>: transparent bg + no border so the container styles show through */
[data-testid="stTextInput"] input,
[data-testid="stNumberInput"] input {
  background: transparent !important;
  border: none !important;
  box-shadow: none !important;
  color: var(--text) !important;
  font-size: 1rem !important;
  caret-color: var(--brand);
}

[data-testid="stTextInput"] input::placeholder,
[data-testid="stTextArea"] textarea::placeholder,
[data-testid="stNumberInput"] input::placeholder {
  color: var(--text-subtle) !important;
}

[data-testid="stSelectbox"] [data-baseweb="select"] span {
  color: var(--text) !important;
  font-size: 1rem !important;
}

[data-testid="stTextArea"] textarea {
  color: var(--text) !important;
  font-size: 1rem !important;
  caret-color: var(--brand);
}

/* Dropdown list */
[data-baseweb="popover"] [role="listbox"] {
  background: var(--card) !important;
  border: 1px solid #C3CCDC !important;
}
[data-baseweb="popover"] [role="option"] {
  background: var(--card) !important;
  color: var(--text-muted) !important;
}
[data-baseweb="popover"] [role="option"]:hover,
[data-baseweb="popover"] [aria-selected="true"] {
  background: rgba(13, 148, 136, 0.15) !important;
  color: var(--text) !important;
}
"""


# ─────────────────────────────────────────────────────────────
# Public API
# ─────────────────────────────────────────────────────────────


def inject_shared_styles() -> None:
    """Inject the base CSS shared by all pages."""
    css = (
        _BASE_CSS
        + _ANIMATIONS_CSS
        + _NAVBAR_CSS
        + _PAGE_HEADER_CSS
        + _CARD_CSS
        + _RESULT_CARD_CSS
        + _BUTTON_CSS
        + _FORM_CSS
    )
    st.markdown(f"<style>{css}</style>", unsafe_allow_html=True)


def inject_page_css(css: str) -> None:
    """Inject additional page-specific CSS."""
    st.markdown(f"<style>{css}</style>", unsafe_allow_html=True)


# (label, page-path-relative-to-entrypoint) - same paths st.switch_page uses.
_NAV_ITEMS = [
    ("Home", "app.py"),
    ("Upload", "pages/upload.py"),
    ("Results", "pages/results.py"),
    ("Guide", "pages/recording_guide.py"),
]

_NAV_BRAND_SVG = (
    '<svg width="12" height="12" viewBox="0 0 24 24" fill="white">'
    '<path d="M13.49 5.48c1.1 0 2-.9 2-2s-.9-2-2-2-2 .9-2 2 .9 2 2 2zm-3.6 '
    "13.9l1-4.4 2.1 2v6h2v-7.5l-2.1-2 .6-3c1.3 1.5 3.3 2.5 5.5 2.5v-2c-1.9 "
    "0-3.5-1-4.3-2.4l-1-1.6c-.4-.6-1-1-1.7-1-.3 0-.5.1-.8.1l-5.2 2.2v4.7h2v-"
    '3.4l1.8-.7-1.6 8.1-4.9-1-.4 2 7 1.4z"/></svg>'
)


def render_navbar(active: str = "") -> None:
    """Render the top navigation bar.

    Built with ``st.page_link`` rather than raw ``<a href>`` links: navigating
    by URL spawns a new browser session and resets ``st.session_state`` (wiping
    uploads/results), whereas ``st.page_link`` navigates client-side and keeps
    the session. ``active`` is accepted for backward compatibility (Streamlit
    highlights the current page automatically).
    """
    cols = st.columns([2.2, 1.0, 1.0, 1.0, 1.7], vertical_alignment="center")
    cols[0].markdown(
        f'<div class="nav-brand-row"><span class="nav-brand-badge">'
        f"{_NAV_BRAND_SVG}</span>"
        f'<span class="nav-brand-name">Strideo</span></div>',
        unsafe_allow_html=True,
    )
    for col, (label, page) in zip(cols[1:], _NAV_ITEMS):
        col.page_link(page, label=label)


def render_page_header(title: str, subtitle: str = "") -> None:
    """Render a centred page header."""
    sub = f'<p class="page-subtitle">{subtitle}</p>' if subtitle else ""
    st.markdown(
        f"""
<div class="page-header">
  <h1 class="page-title">{title}</h1>
  {sub}
</div>
""",
        unsafe_allow_html=True,
    )


def render_empty_state(message: str, link_text: str = "", link_page: str = "") -> None:
    """Render a card for when there is no data to show.

    The optional link uses ``st.page_link`` (not an ``<a href>``) so navigating
    from it preserves ``st.session_state``.
    """
    st.markdown(
        f"""
<div class="blue-card" style="text-align:center;padding:48px 28px;">
  <svg width="48" height="48" viewBox="0 0 24 24" fill="none" stroke="var(--text-subtle)"
       stroke-width="1.5" style="margin-bottom:16px;">
    <circle cx="12" cy="12" r="10"></circle>
    <line x1="12" y1="8" x2="12" y2="12"></line>
    <line x1="12" y1="16" x2="12.01" y2="16"></line>
  </svg>
  <p style="color:var(--text-muted);font-size:1.1rem;margin:0;">{message}</p>
</div>
""",
        unsafe_allow_html=True,
    )
    if link_text and link_page:
        st.page_link(link_page, label=link_text)
