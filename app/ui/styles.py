"""Shared CSS and UI components for the Strideo Streamlit app."""

import streamlit as st

# ─────────────────────────────────────────────────────────────
# CSS building blocks
# ─────────────────────────────────────────────────────────────

_BASE_CSS = """
@import url('https://fonts.googleapis.com/css2?family=Space+Grotesk:wght@600;700&family=Inter:wght@400;500;600;700&family=JetBrains+Mono:wght@400;500;600&display=swap');

:root{
  /* ── Surfaces ── */
  --bg: #F8FAFD;             /* light page background      */
  --card: #FFFFFF;           /* card surface (flat, no gradient) */
  --card-2: #ECF1F5;         /* hover / inset fill (== secondary) */
  --card-hover: #ECF1F5;
  --secondary: #ECF1F5;      /* inset rows, active nav item, table header fill */
  --accent-tint: #E0F3F1;    /* tinted teal highlight blocks */
  --border: rgba(217,222,229,0.7);   /* hairline border, 70% per design tokens */
  --border-soft: rgba(217,222,229,0.4);
  --hover-fill: rgba(16,24,39,0.035);

  /* ── Text ── */
  --text: #101827;           /* foreground                 */
  --text-muted: #5B6471;     /* muted-foreground            */
  /* was #8B93A1 - contrasted only 2.96:1 on --card-2/inset surfaces (WCAG AA
     needs 4.5:1) despite being used as real caption/metadata text in ~13
     places. Darkened, same hue/saturation, to 4.79:1 on that worst-case
     surface (5.2-5.4:1 on --bg/--card). */
  --text-subtle: #626A78;
  --on-brand: #FFFFFF;       /* text on brand-gradient surfaces */

  /* ── Brand (teal) + accent (blue) ── */
  --brand: #299A8B;          /* brand-teal - primary        */
  --brand-hover: #22857A;
  --brand-glow: rgba(41,154,139,0.28);
  /* --brand at small/normal text sizes on --bg contrasts only 3.30:1 (WCAG AA
     needs 4.5:1) - fine for buttons/icons/borders/gradients (UI-component
     3:1 rule, or on-brand text on a filled surface), not for readable text.
     Use this darkened variant (5.81:1) wherever --brand colors actual text
     on a light surface. */
  --brand-text: #1C6E62;
  --accent: #2D69DE;         /* brand-blue - gradient partner */
  --accent-glow: rgba(45,105,222,0.22);
  --logo-grad-a: #0D9488;    /* fixed hex, mark/lockup SVGs only */
  --logo-grad-b: #2563EB;
  --green: var(--brand);     /* OK / valid state uses brand teal */
  /* was #DA4528 - contrasted 4.13:1 on --bg/--card, just short of WCAG AA's
     4.5:1. Darkened, same hue, to 4.67:1. */
  --red: #CC3F23;            /* destructive / REVIEW state */
  --warn: #DF911A;           /* experimental / unvalidated state */

  /* ── Dark sections (hero frame, footer) ── */
  --navy: #0B1423;
  --navy-foreground: #F3F5F8;

  /* ── Shadows (flat cards; elevation reserved for the nav pill + hero frame) ── */
  --shadow: none;
  --shadow-lg: 0 10px 40px -24px rgba(20,28,48,0.50);
  --shadow-hero: 0 40px 80px -50px rgba(20,28,48,0.70);

  --radius-lg: 16px;   /* cards, panels, nav pill, hero frame, dropzone */
  --radius-md: 10px;   /* buttons, inputs, list rows */
  --radius-sm: 6px;    /* status tags, small chips */

  /* ── Legacy aliases (old page CSS still references these names) ── */
  --bg-dark: var(--bg);
  --bg-card: var(--card);
  --bg-card-hover: var(--card-hover);
  --orange: var(--brand);
  --orange-hover: var(--brand-hover);
  --orange-glow: var(--brand-glow);
  --cyan: var(--accent);
  --cyan-glow: var(--accent-glow);
  --text-white: var(--text);
}

* {
  font-family: 'Inter', -apple-system, BlinkMacSystemFont, sans-serif;
}

html, body, [data-testid="stAppViewContainer"], [data-testid="stApp"]{
  background: var(--bg) !important;
  background-image:
    radial-gradient(ellipse at 20% 20%, rgba(41, 154, 139, 0.05) 0%, transparent 50%),
    radial-gradient(ellipse at 80% 80%, rgba(45, 105, 222, 0.04) 0%, transparent 55%) !important;
  color: var(--text);
}

[data-testid="stMain"]{ background: transparent !important; }
[data-testid="stSidebar"]{ display: none !important; }

/* inject_shared_styles()/inject_page_css() inject raw <style> blocks via
   st.markdown, which still land as a normal flex child of the page's
   top-level stVerticalBlock - zero height, but the block's own `gap` still
   opens a visible slot before/after it (most visible as a stray strip above
   the now-full-bleed navbar). display:none removes it from flex flow
   entirely; a <style> tag still applies while hidden, same as in <head>.
   `:only-child` on `style` is load-bearing, not decoration: without it this
   also matches render_legal_document()'s markdown block, which embeds a
   <style> tag (Termly's own custom-class CSS) ahead of the real policy
   text - a broader match hid the entire Cookie/Privacy/Terms page behind
   this same rule (found 2026-08-25, introduced here 2026-08-23). */
[data-testid="stElementContainer"]:has(
  > [data-testid="stMarkdown"] [data-testid="stMarkdownContainer"] > style:only-child
){
  display: none;
}

/* The page column is a full-height flex column with no bottom padding, so the
   footer (margin-top:auto, see _FOOTER_CSS) is always the last thing on the
   page: flush to the viewport bottom on short pages, flush to the end of the
   content on long ones. Any bottom padding here would show as a white strip
   under the navy bar. */
.block-container{
  padding-top: 0 !important;
  padding-bottom: 0 !important;
  max-width: 1440px;
  min-height: 100vh;
  display: flex;
  flex-direction: column;
}
.block-container > [data-testid="stVerticalBlock"]{
  flex: 1 1 auto;
}

/* Streamlit's own fixed header sits above everything by z-index and, even
   fully transparent, still captures pointer events across its full band -
   it silently swallowed clicks on the navbar once the navbar became a
   full-bleed sticky bar flush at top:0 (previously the floating pill sat
   lower, at top:12px, mostly clear of it). Nothing in it is left
   interactive (stToolbar/stDecoration are already hidden below), so it's
   safe to take it out of hit-testing entirely. */
header[data-testid="stHeader"]{
  background: transparent !important;
  pointer-events: none;
}
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

_TYPE_CSS = """
.font-display, h1, h2, h3,
.page-title, .card-title, .nav-brand-name, .eyebrow-label {
  font-family: 'Space Grotesk', ui-sans-serif, system-ui, sans-serif;
  letter-spacing: -0.02em;
}

.font-mono, .eyebrow, .mono-label, .mono-value {
  font-family: 'JetBrains Mono', ui-monospace, monospace;
}

.eyebrow {
  font-size: 0.7rem;
  font-weight: 500;
  text-transform: uppercase;
  letter-spacing: 0.2em;
  color: var(--brand-text);
  margin: 0 0 4px 0;
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
  /* Full-bleed via a ::before background layer, NOT negative margins on the
     block itself - see the stLayoutWrapper rule below for why sticky lives
     there instead of here. */
  position: relative;
  padding: 20px 20px !important;
  margin-bottom: 26px !important;
  align-items: center !important;
}
[data-testid="stHorizontalBlock"]:has(.nav-brand-row)::before{
  content: '';
  position: absolute;
  top: 0;
  bottom: 0;
  left: calc(50% - 50vw);
  right: calc(50% - 50vw);
  background: var(--navy);
  box-shadow: 0 4px 16px rgba(11,20,35,0.15);
  z-index: -1;
}
/* Streamlit 1.54 wraps every block in a stLayoutWrapper that's sized to
   exactly its child's height (same reason the footer's margin-top:auto has
   to target this wrapper too, see _FOOTER_CSS). That leaves the navbar zero
   room to travel if position:sticky sits on the stHorizontalBlock itself:
   sticky is capped at not moving past its containing block's edge, and a
   containing block exactly as tall as the sticky element gives it ~0px of
   stick range before it "runs out" and resumes scrolling with the page
   (confirmed live: it visibly detaches after ~16px of scroll). Putting
   sticky on the wrapper instead makes its containing block the tall
   top-level stVerticalBlock, which has the whole page's worth of travel
   room. */
[data-testid="stLayoutWrapper"]:has(> [data-testid="stHorizontalBlock"] .nav-brand-row){
  position: sticky;
  top: 0;
  z-index: 999;
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
  width: 38px;
  height: 38px;
  display: inline-flex;
  align-items: center;
  justify-content: center;
  flex-shrink: 0;
}
.nav-brand-badge svg{ width: 100%; height: 100%; }
.nav-brand-name{
  font-weight: 700;
  font-size: 1.7rem;
  /* Same teal-to-blue gradient text as the hero headline's "measured"
     (.hero-grad in app.py) - kept as its own rule since navbar/footer are
     styled here, not in the page-specific CSS module. */
  background: linear-gradient(90deg, var(--brand), var(--accent));
  -webkit-background-clip: text;
  background-clip: text;
  color: transparent;
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
  color: var(--brand-text) !important;
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
  [data-testid="stPageLink"] a:hover{ background: rgba(255,255,255,0.08) !important; }
/* Same mono/uppercase treatment as the eyebrow labels and the footer's own
   nav-style links (GitHub, strideo.org, legal pages) - ties the header into
   the label language used everywhere else on the site instead of the
   default body sans. */
[data-testid="stHorizontalBlock"]:has(.nav-brand-row) [data-testid="stPageLink"] a p{
  font-family: 'JetBrains Mono', ui-monospace, monospace;
  text-transform: uppercase;
  letter-spacing: 0.1em;
  color: rgba(243,245,248,0.6) !important;
  font-size: 15px !important;
  white-space: nowrap !important;  /* nav items stay on one line */
}
[data-testid="stHorizontalBlock"]:has(.nav-brand-row)
  [data-testid="stPageLink"] a:hover p{
  color: var(--navy-foreground) !important;
  text-decoration: none !important;
}
/* Active-page signal: an invisible per-column marker set by render_navbar's
   own `active` argument, NOT Streamlit's href="" self-link convention - that
   was tried first (confirmed live at the time) but Streamlit 1.54 marks
   the MAIN script's own page_link as href="" unconditionally, on every page,
   not only when actually on it (confirmed live: "Home" stayed bold on
   /upload, /results, /recording_guide after a real client-side nav click,
   not just a hard reload). [aria-current="page"] is also never set, so
   neither built-in signal is usable here. Bold + full-bright is the visual
   treatment, kept from the original attempt. */
[data-testid="stColumn"]:has(.nav-active-mark)
  [data-testid="stPageLink"] a p{
  font-weight: 700 !important;
  color: var(--navy-foreground) !important;
}
/* The marker is a second element in the column's own vertical stack, so
   without this it (or the column's inter-element gap) pushed the page_link
   below it down, breaking the row's vertical centring - only the active
   item was affected, since every other column has just the one child.
   display:none removes it from flex flow entirely; :has() still matches
   through a hidden descendant (same technique the hero frame's own marker
   uses, app.py's .heroframe-mark). */
[data-testid="stElementContainer"]:has(.nav-active-mark){ display: none; }

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
  font-size: 2.6rem;
  font-weight: 700;
  color: var(--text);
  margin: 0 0 12px 0;
}

.page-subtitle{
  font-size: 1.1rem;
  color: var(--text-muted);
  margin: 0;
}

@media (max-width: 700px){
  .page-title{ font-size: 1.9rem; }
}
"""

_SECTION_CSS = """
/* ── Section chrome: the eyebrow / display-heading / lead trio the new design
   system opens every content section with, plus the two layout primitives
   those sections are built from (a full-bleed band and a hairline stat grid).
   Shared by Home and the Recording Guide - keep page-specific rules in the
   page's own inject_page_css() call. ── */
.section-head{
  display: flex;
  flex-wrap: wrap;
  align-items: flex-end;
  justify-content: space-between;
  gap: 18px 32px;
}
/* Streamlit ships its own h1-h3 rules (font-size plus generous top/bottom
   padding) that outrank a bare class selector, so every display heading we
   render inside st.markdown has to state its size and reset that padding
   explicitly. Same reason the page-level heading classes do it. */
.section-h2{
  font-family: 'Space Grotesk', ui-sans-serif, system-ui, sans-serif;
  font-size: 2.15rem !important;
  font-weight: 700;
  letter-spacing: -0.025em;
  line-height: 1.14 !important;
  color: var(--text);
  margin: 14px 0 0 !important;
  padding: 0 !important;
  max-width: 34ch;
}
/* "How it's validated" has no .section-lead paragraph between the heading
   and the two-column content below it (unlike every other .section-h2 use
   on this page, where the lead's own margin-top supplies the gap) - without
   its own bottom margin the heading sat almost flush against "StrideoNet" /
   "Deterministic kinematics engine" underneath it. */
.validation-h2{
  margin-bottom: 24px !important;
}
/* Full width by design - these leads run the width of the page column rather
   than sitting in a narrow measure with empty space beside them. */
.section-lead{
  font-size: 1rem;
  line-height: 1.7;
  color: var(--text-muted);
  margin: 16px 0 0 !important;
}
/* The right-hand aside that sits opposite a .section-h2 inside .section-head */
.section-note{
  font-size: 0.9rem;
  line-height: 1.65;
  color: var(--text-muted);
  margin: 0 !important;
  max-width: 40ch;
}

/* Full-bleed band. Same escape-the-column trick as the footer: pull the
   element out to the viewport edges, then re-inset its content so the text
   still lines up with the page column.
   The horizontal padding is `50vw - 50%`: the exact width that was just pulled
   out on each side (50% resolves against the parent's content box), so the
   band's own text lands back on the page column's left edge instead of
   guessing a gutter from a hardcoded max-width. */
.band{
  margin: 56px calc(50% - 50vw);
  width: 100vw;
  max-width: 100vw;
  box-sizing: border-box;
  padding: 60px max(20px, calc(50vw - 50%));
  border-top: 1px solid var(--border);
  border-bottom: 1px solid var(--border);
  background: color-mix(in srgb, var(--card) 60%, transparent);
}

/* Inset stat grid: each cell is its own tinted card (--card-2, the shared
   inset/hover fill) with a real gap between them, so they read as separate
   but close-together results rather than one seamless bordered block - matters
   when the grid sits inside another --card-background box (e.g. the Home
   validation columns), where the old plain --card cells were indistinguishable
   from their parent box. */
.stat-grid{
  display: grid;
  grid-template-columns: 1fr;
  gap: 10px;
}
.stat-cell{
  background: var(--card-2);
  border-radius: var(--radius-md);
  padding: 17px 22px;
}
.stat-value{
  font-family: 'Space Grotesk', ui-sans-serif, system-ui, sans-serif;
  font-size: 1.85rem;
  font-weight: 700;
  letter-spacing: -0.03em;
  color: var(--text);
  margin: 0 !important;
  line-height: 1.1;
}
.stat-label{
  font-size: 0.86rem;
  line-height: 1.5;
  color: var(--text-muted);
  margin: 5px 0 0 !important;
}

/* Tinted callout for caveats and scope limits (research-preview notices, the
   unvalidated-metric warning). */
.note-card{
  background: var(--accent-tint);
  border: 1px solid var(--border);
  border-left: 3px solid var(--brand);
  border-radius: var(--radius-md);
  padding: 16px 20px;
}
.note-card p{
  margin: 0 !important;
  font-size: 0.92rem;
  line-height: 1.65;
  color: var(--text-muted);
}
.note-card p + p{ margin-top: 10px !important; }
.note-card strong{ color: var(--text); }

@media (max-width: 700px){
  .section-h2{ font-size: 1.7rem; }
  .band{ margin: 36px calc(50% - 50vw); padding: 40px 20px; }
  .stat-value{ font-size: 1.6rem; }
}
"""

_CARD_CSS = """
/* Flat, bordered cards - no gradient fill, no hover-lift/glow, matching the
   new design system's "elevation: none, border only" spec. */
.blue-card{
  background: var(--card);
  border-radius: var(--radius-lg);
  padding: 28px 26px;
  margin-top: 20px;
  margin-bottom: 20px;
  border: 1px solid var(--border);
}

.navy-card{
  background: var(--navy);
  border-radius: var(--radius-lg);
  padding: 28px 26px;
  border: none;
}

.card-title{
  font-size: 1.15rem;
  font-weight: 600;
  color: var(--text);
  margin: 0 0 16px 0;
}

.card-text{
  color: var(--text-muted);
  font-size: 1rem;
  line-height: 1.7;
  margin: 0;
  max-width: 72ch;  /* keep prose readable even in a full-width card */
}

.navy-card .card-title,
.navy-card .card-text{
  color: var(--navy-foreground);
}
.navy-card .card-text{
  color: rgba(243,245,248,0.6);
}

@media (max-width: 700px){
  .blue-card, .navy-card{ padding: 20px 18px; }
  .card-title{ font-size: 1.05rem; }
  .card-text{ font-size: 0.95rem; }
}
"""

_RESULT_CARD_CSS = """
.result-card {
  background: var(--card);
  border-radius: var(--radius-lg);
  padding: 18px 22px;
  border: 1px solid var(--border);
  /* Equal-height cards in a row even when some have an extra detail line */
  height: 100%;
  box-sizing: border-box;
}
/* Make columns that contain a result-card stretch so height:100% resolves */
[data-testid="stColumn"]:has(.result-card){ align-items: stretch; }
[data-testid="stColumn"]:has(.result-card) [data-testid="stMarkdownContainer"]{
  height: 100%;
}
.result-card.error { border-color: var(--red); }
.result-card.warn { border-color: var(--warn); }
.result-card.blue { border-color: var(--accent); }
.result-label {
  color: var(--text-muted);
  font-size: 0.68rem;
  margin: 0 0 4px;
  text-transform: uppercase;
  letter-spacing: 0.14em;
  font-weight: 500;
  font-family: 'JetBrains Mono', ui-monospace, monospace;
}
.result-value {
  color: var(--text);
  font-size: 1.3rem;
  font-weight: 600;
  margin: 0;
  font-family: 'JetBrains Mono', ui-monospace, monospace;
}
.result-detail {
  color: var(--brand-text);
  font-size: 0.75rem;
  margin: 4px 0 0;
  font-family: 'JetBrains Mono', ui-monospace, monospace;
}
"""

_BUTTON_CSS = """
.stButton > button[kind="primary"] {
  background: var(--navy) !important;
  color: var(--navy-foreground) !important;
  border: none !important;
  border-radius: var(--radius-md) !important;
  /* Matches the shared [data-testid="stButton"] button padding below -
     this rule's higher specificity (kind="primary") was silently winning
     with a taller 14px/32px value, making every primary button a couple
     px taller than its secondary sibling in the same row (hero, contact
     form). */
  padding: 12px 0 !important;
  font-size: 0.95rem !important;
  font-weight: 600 !important;
  font-family: 'Space Grotesk', sans-serif !important;
  box-shadow: none !important;
  transition: background 0.2s ease !important;
}

/* A colour shift on hover, not a lift - every other action on the page
   (page-links, .st-key-hero_guide's secondary button) reads hover as a
   colour change, so a translateY here was the odd one out, not a bug in
   itself. */
.stButton > button[kind="primary"]:hover {
  background: color-mix(in srgb, var(--navy) 88%, white 12%) !important;
  box-shadow: none !important;
}

.stButton > button:disabled {
  background: var(--secondary) !important;
  color: var(--text-subtle) !important;
  box-shadow: none !important;
}

[data-testid="stButton"] button {
  background: var(--navy) !important;
  color: var(--navy-foreground) !important;
  border: none !important;
  border-radius: var(--radius-md) !important;
  padding: 12px 0 !important;
  width: 100% !important;
  font-weight: 600 !important;
  font-size: 0.95rem !important;
  font-family: 'Space Grotesk', sans-serif !important;
  letter-spacing: 0.3px;
  box-shadow: none !important;
  transition: background 0.2s ease !important;
}
[data-testid="stButton"] button:hover {
  box-shadow: none !important;
  background: color-mix(in srgb, var(--navy) 88%, white 12%) !important;
}

/* Streamlit renders the label as its own <p> inside nested <span>/<div>
   wrappers, and the global `* { font-family: 'Inter' }` rule (_BASE_CSS)
   matches that <p> directly - a same-element match always wins over an
   inherited value, no matter the ancestor rule's specificity, so the
   button's own font-family/weight/size never reached the visible text
   (confirmed live: the <button> computed Space Grotesk/600/15.2px, but its
   <p> computed Inter/400/16px). Same fix needed for `color` for the same
   structural reason - Streamlit colours the label element directly, so the
   button's own `color` never reaches the text either. */
[data-testid="stButton"] button p,
[data-testid="stButton"] button span,
[data-testid="stButton"] button div,
.stButton > button[kind="primary"] p,
.stButton > button[kind="primary"] span,
.stButton > button[kind="primary"] div {
  color: inherit !important;
  font-family: inherit !important;
  font-weight: inherit !important;
  font-size: inherit !important;
  letter-spacing: inherit !important;
}
"""

_CTA_PANEL_CSS = """
/* ── CTA panel (navy, rounded, holds a row of widgets) ── */
/* Shared surface: Home's pre-footer CTA band and Results' "Save this
   session" export panel both key off .cta-mark. The button-inversion rule
   below matches stButton AND stDownloadButton (Home only ever used the
   former; Results' panel is all download buttons) - purely additive, so
   Home's own rendering is unchanged. */
[data-testid="stHorizontalBlock"]:has(.cta-mark){
  background: var(--navy);
  border-radius: 24px;
  padding: 46px 42px !important;
  margin-top: 56px !important;
}
.cta-mark{ text-align: center; }
.cta-mark .eyebrow{ color: #5CC9BA; }
.cta-h2{
  font-family: 'Space Grotesk', ui-sans-serif, system-ui, sans-serif;
  font-size: 2.05rem !important;
  font-weight: 700;
  letter-spacing: -0.025em;
  line-height: 1.16 !important;
  color: var(--navy-foreground) !important;
  margin: 14px auto 26px !important;
  padding: 0 !important;
  max-width: 44ch;
}
[data-testid="stHorizontalBlock"]:has(.cta-mark) [data-testid="stButton"] button,
[data-testid="stHorizontalBlock"]:has(.cta-mark) [data-testid="stDownloadButton"] button{
  background: var(--navy-foreground) !important;
  color: var(--navy) !important;
  border: none !important;
  /* Matching padding alone still left a few px of height difference from
     the sibling page-link anchors - a <button> and an <a> resolve their
     default line-height differently even at the same font-size. Pin the
     height directly instead of chasing font metrics - 48px, matching the
     hero "Analyse a clip"/"How to film it" pair's measured height, so
     this row reads as the same size as the hero, not larger. */
  padding: 0 12px !important;
  height: 48px !important;
  display: flex !important;
  align-items: center !important;
  justify-content: center !important;
}
/* Colour-shift hover, matching the sibling page-links exactly (same
   var(--card-hover) fill) instead of the generic navy-hover lighten - this
   button's resting background is light, not navy, so that rule would be
   the wrong colour here, and a transform would put it out of step with
   "How to film it" / "View source" right beside it. */
[data-testid="stHorizontalBlock"]:has(.cta-mark) [data-testid="stButton"] button:hover,
[data-testid="stHorizontalBlock"]:has(.cta-mark) [data-testid="stDownloadButton"] button:hover{
  background: var(--card-hover) !important;
}
/* The whole chain down to the anchor is shrink-to-fit (the element container
   that holds a page link does not stretch to its column), so width:100% has to
   be set from the element container inwards - otherwise the link stays narrower
   than the button beside it and the centred pair looks off-axis. */
[data-testid="stHorizontalBlock"]:has(.cta-mark)
  [data-testid="stElementContainer"]:has([data-testid="stPageLink"]),
[data-testid="stHorizontalBlock"]:has(.cta-mark) [data-testid="stPageLink"],
[data-testid="stHorizontalBlock"]:has(.cta-mark) [data-testid="stPageLink"] > div{
  display: block !important;
  width: 100% !important;
}
/* Streamlit's own default gives a page-link's element container a -6px 0
   margin (a built-in gap-collapse trick) that the button's container doesn't
   have, so the link sat 6px higher than the button beside it even with
   matching heights everywhere else - confirmed live via getBoundingClientRect
   (button top vs link top differed by exactly 6px). Zero it out so both
   containers sit flush with the column top. */
[data-testid="stHorizontalBlock"]:has(.cta-mark)
  [data-testid="stElementContainer"]:has([data-testid="stPageLink"]){
  margin: 0 !important;
}
[data-testid="stHorizontalBlock"]:has(.cta-mark) [data-testid="stPageLink"] a{
  display: flex !important;
  align-items: center !important;
  justify-content: center !important;
  /* Full column width, so all three actions are the same size and sit
     on the panel's centre line. Solid light fill (not an outline on navy)
     to match the st.button beside them - same light-on-dark contrast for
     every action, not just the primary one. */
  width: 100% !important;
  box-sizing: border-box;
  background: var(--navy-foreground);
  border: none;
  border-radius: var(--radius-md) !important;
  /* Fixed height (not padding) so it matches the sibling st.button exactly -
     an <a> and a <button> resolve line-height differently at the same
     font-size, so padding alone left a few px of drift. 48px to match the
     hero button pair above. */
  padding: 0 12px !important;
  height: 48px !important;
  /* Streamlit's own default puts a 2px top/bottom margin on this anchor
     (stPageLink-NavLink) that the button doesn't have, so the link sat
     2px lower than "Analyse a clip" even with matching container tops
     and heights - confirmed live via getBoundingClientRect. */
  margin: 0 !important;
}
[data-testid="stHorizontalBlock"]:has(.cta-mark) [data-testid="stPageLink"] a:hover{
  background: var(--card-hover);
}
[data-testid="stHorizontalBlock"]:has(.cta-mark) [data-testid="stPageLink"] a p{
  color: var(--navy) !important;
  font-family: 'Space Grotesk', sans-serif !important;
  font-size: 0.95rem !important;
  font-weight: 600 !important;
  /* Matches the sibling st.button's global letter-spacing (0.3px) exactly -
     it's the one property that still differed between the two element
     types at otherwise-identical font/size/weight. */
  letter-spacing: 0.3px !important;
}
[data-testid="stHorizontalBlock"]:has(.cta-mark) [data-testid="stPageLink"] a:hover p{
  text-decoration: none !important;
}
@media (max-width: 700px){
  .cta-h2{ font-size: 1.7rem; }
  [data-testid="stHorizontalBlock"]:has(.cta-mark){ padding: 32px 24px !important; }
}
"""

_FOOTER_CSS = """
/* Full-bleed navy footer bar, keyed off .footer-brand-row (unique to the
   footer, same marker-class technique the navbar uses for .nav-brand-row -
   :has() matches it at any depth so it reaches the outer st.columns row). */
[data-testid="stHorizontalBlock"]:has(.footer-brand-row){
  background: var(--navy);
  /* `auto` (not a fixed margin) so the bar is pushed to the bottom of the page
     column on short pages - see the flex setup on .block-container in
     _BASE_CSS. The gap above it comes from the transparent top border below,
     because margin-top:auto collapses to 0 once the page is tall enough to
     leave no free space. background-clip keeps the navy out of that border. */
  margin-top: auto !important;
  margin-bottom: 0 !important;
  border-top: 44px solid transparent;
  background-clip: padding-box;
  margin-left: calc(50% - 50vw) !important;
  margin-right: calc(50% - 50vw) !important;
  width: 100vw !important;
  max-width: 100vw !important;
  /* Re-inset by exactly what the negative margins pulled out (50% resolves
     against the parent's content box), so the footer's columns line up with
     the page column above instead of with a guessed max-width gutter. */
  padding: 40px max(20px, calc(50vw - 50%)) !important;
  align-items: center !important;
}

/* Streamlit 1.54 wraps every block in a stLayoutWrapper, so the footer row is a
   grandchild of the page column and its own margin-top:auto has no free space
   to absorb. The push has to happen on the wrapper, which IS a flex child of
   the full-height page column. The rule on the row above is kept for older
   Streamlit builds that render without the wrapper. */
[data-testid="stLayoutWrapper"]:has(> [data-testid="stHorizontalBlock"] .footer-brand-row){
  margin-top: auto !important;
}

/* Defensive: confirmed live and reproducible (3/3 tries) that Streamlit
   transiently renders a second copy of this footer for the ENTIRE duration
   of the first long-running pipeline execution in a fresh browser session
   (upload's fast path, results' full-analysis run) - one footer while idle,
   two while that first run's progress bar is active, back to one once it
   completes. Root cause not conclusively pinned down (suspected WebSocket
   reconnect / forward-message-cache replay triggered by a long synchronous
   gap with no st.* traffic during cold model-load, before the first
   progress callback fires) and not reproducible on any later run in the
   same session, so a real upstream fix isn't tractable here. This footer
   never legitimately repeats, so hiding any instance beyond the first is
   safe regardless of the exact trigger. */
[data-testid="stHorizontalBlock"]:has(.footer-brand-row)
  ~ [data-testid="stHorizontalBlock"]:has(.footer-brand-row){
  display: none !important;
}
[data-testid="stLayoutWrapper"]:has(.footer-brand-row)
  ~ [data-testid="stLayoutWrapper"]:has(.footer-brand-row){
  display: none !important;
}

.footer-brand-row{
  display: flex;
  align-items: center;
  gap: 10px;
}
.footer-brand-mark{ width: 38px; height: 38px; display: inline-flex; }
.footer-brand-mark svg{ width: 100%; height: 100%; }
.footer-brand-name{
  font-family: 'Space Grotesk', ui-sans-serif, system-ui, sans-serif;
  font-weight: 700;
  font-size: 1.7rem;
  background: linear-gradient(90deg, var(--brand), var(--accent));
  -webkit-background-clip: text;
  background-clip: text;
  color: transparent;
}
/* Right column: GitHub link, "strideo.org" meta line, legal-doc links. Keyed
   off .footer-meta-line (unique to this column) rather than the GitHub href,
   which also appears in the home page's CTA panel.

   All three lines are left-aligned to the column's own left edge so they share
   one start point - right-aligning them made each line begin at a different x
   depending on its length. */
[data-testid="stColumn"]:has(.footer-meta-line){
  text-align: left;
}
[data-testid="stColumn"]:has(.footer-meta-line)
  [data-testid="stPageLink"] a{
  display: inline-flex !important;
}
[data-testid="stColumn"]:has(.footer-meta-line)
  [data-testid="stPageLink"] a p,
[data-testid="stColumn"]:has(.footer-meta-line)
  .footer-meta-line{
  font-family: 'JetBrains Mono', ui-monospace, monospace;
  font-size: 0.7rem !important;
  text-transform: uppercase;
  letter-spacing: 0.14em;
  color: rgba(243,245,248,0.5) !important;
  white-space: nowrap !important;
}
[data-testid="stColumn"]:has(.footer-meta-line)
  [data-testid="stPageLink"] a:hover p{
  color: var(--navy-foreground) !important;
  text-decoration: none !important;
}
[data-testid="stColumn"]:has(.footer-meta-line)
  .footer-meta-line{
  display: block;
  margin: 8px 0;
}
/* Legal-links mini-row: three page_links in their own nested columns, starting
   at the same left edge as the two lines above, with mid-dot separators. */
[data-testid="stColumn"]:has(.footer-meta-line)
  [data-testid="stHorizontalBlock"]{
  justify-content: flex-start !important;
  gap: 18px !important;
}
[data-testid="stColumn"]:has(.footer-meta-line)
  [data-testid="stHorizontalBlock"] [data-testid="stColumn"]{
  width: auto !important;
  flex: 0 0 auto !important;
  position: relative;
}
[data-testid="stColumn"]:has(.footer-meta-line)
  [data-testid="stHorizontalBlock"] [data-testid="stColumn"]:not(:first-child)::before{
  content: '·';
  position: absolute;
  left: -10px;
  color: rgba(243,245,248,0.35);
}

@media (max-width: 700px){
  [data-testid="stHorizontalBlock"]:has(.footer-brand-row){
    flex-wrap: wrap !important;
    gap: 20px !important;
    padding: 30px 20px !important;
    border-top-width: 32px;
    align-items: flex-start !important;
  }
  /* The global mobile rule stacks every column full-width, which would break
     the legal links into three dot-prefixed lines. Keep that mini-row inline
     (it wraps on its own if it has to). */
  [data-testid="stColumn"]:has(.footer-meta-line)
    [data-testid="stHorizontalBlock"] [data-testid="stColumn"]{
    flex: 0 0 auto !important;
    width: auto !important;
    min-width: 0 !important;
  }
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

/* BaseWeb nests a second wrapper inside that container carrying its own grey
   fill (#F0F2F6), which covered the white surface above and made every field
   look disabled. Clear it so the styled container is what you see. */
[data-baseweb="base-input"] {
  background: transparent !important;
}

[data-baseweb="input"]:focus-within,
[data-testid="stSelectbox"] [data-baseweb="select"] > div:focus-within,
[data-testid="stTextArea"] textarea:focus {
  border-color: var(--brand) !important;
  box-shadow: 0 0 0 2px rgba(41, 154, 139, 0.2) !important;
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
  background: rgba(41, 154, 139, 0.15) !important;
  color: var(--text) !important;
}
"""

_PROGRESS_CSS = """
/* The analysis bar is driven by per-frame callbacks, so it advances in ~1%
   steps a few times a second. Easing the fill's width turns that staircase
   into a glide; the duration is deliberately just longer than the typical
   gap between updates. Targeting the inner divs generically (rather than a
   BaseWeb class name) keeps this from silently dying on a Streamlit bump -
   width is the only property any of them animates. */
[data-testid="stProgress"] div {
  transition: width 300ms linear;
}
"""


# ─────────────────────────────────────────────────────────────
# Public API
# ─────────────────────────────────────────────────────────────


def inject_shared_styles() -> None:
    """Inject the base CSS shared by all pages."""
    css = (
        _BASE_CSS
        + _TYPE_CSS
        + _NAVBAR_CSS
        + _PAGE_HEADER_CSS
        + _SECTION_CSS
        + _CARD_CSS
        + _RESULT_CARD_CSS
        + _BUTTON_CSS
        + _CTA_PANEL_CSS
        + _FOOTER_CSS
        + _FORM_CSS
        + _PROGRESS_CSS
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


_MARK_BONES = (
    "M54 34 L54.4 30.2",
    "M54 34 L51 56",
    "M54 34 L60 44.4 L70.4 38.4",
    "M54 34 L42.7 38.1 L45.8 49.7",
    "M51 56 L65.1 70.1 L71.1 88.1",
    "M71.1 88.1 L69.8 92.2 L79.3 89 L71.1 88.1",
    "M51 56 L38.7 71.7 L22 62.6",
    "M22 62.6 L20.2 58.7 L15.4 67.5 L22 62.6",
)
_MARK_JOINTS = (
    (54, 34),
    (60, 44.4),
    (70.4, 38.4),
    (42.7, 38.1),
    (45.8, 49.7),
    (51, 56),
    (65.1, 70.1),
    (38.7, 71.7),
    (71.1, 88.1),
    (22, 62.6),
)
_MARK_HEAD = (55.8, 22.6)


_MARK_JOINT_COLOR = "#FFC800"  # fixed hex, mark/lockup SVGs only - see --logo-grad-a/b


def mark_svg(
    gradient_id: str,
    size: int = 26,
    x: float | None = None,
    y: float | None = None,
) -> str:
    """The Strideo mark: a running pose-keypoint figure (head, bones as
    strokes, joints as keypoint dots), teal-to-blue gradient. No container
    shape - matches the standalone assets in ``app/assets/logo/``.

    Joints are a flat yellow dot (fill and edge the same color, no outline)
    at the as-designed radius (3.2 of the 100-unit viewBox) - a solid,
    high-contrast hue reads at small sizes on its own, unlike the earlier
    white-fill/navy-outline attempt, where the outline was the only thing
    carrying contrast and vanished at 16-28px real-world sizes.

    ``x``/``y`` position this as a nested ``<svg>`` inside a larger parent
    SVG's own coordinate space (e.g. dropping the mark into a diagram) -
    omitted by default, which renders a normal top-level, unpositioned mark
    exactly as every existing caller (navbar, footer) already expects.
    """
    bones = "".join(f'<path d="{d}"/>' for d in _MARK_BONES)
    joints = "".join(
        f'<circle cx="{jx}" cy="{jy}" r="3.2"/>' for jx, jy in _MARK_JOINTS
    )
    pos = f' x="{x}" y="{y}"' if x is not None and y is not None else ""
    return (
        f'<svg{pos} width="{size}" height="{size}" viewBox="-2.65 5.35 100 100" '
        f'role="img" aria-label="Strideo">'
        f'<defs><linearGradient id="{gradient_id}" x1="-2.65" y1="105.35" '
        f'x2="97.35" y2="5.35" gradientUnits="userSpaceOnUse">'
        f'<stop offset="0" stop-color="#0D9488"/>'
        f'<stop offset="1" stop-color="#2563EB"/></linearGradient></defs>'
        f'<g stroke="url(#{gradient_id})" stroke-width="4.6" fill="none" '
        f'stroke-linecap="round" stroke-linejoin="round">{bones}</g>'
        f'<circle cx="{_MARK_HEAD[0]}" cy="{_MARK_HEAD[1]}" r="7.2" '
        f'fill="url(#{gradient_id})"/>'
        f'<g fill="{_MARK_JOINT_COLOR}">{joints}</g>'
        f"</svg>"
    )


def render_navbar(active: str = "") -> None:
    """Render the top navigation bar.

    Built with ``st.page_link`` rather than raw ``<a href>`` links: navigating
    by URL spawns a new browser session and resets ``st.session_state`` (wiping
    uploads/results), whereas ``st.page_link`` navigates client-side and keeps
    the session. ``active`` names the current page (matched against
    ``_NAV_ITEMS``' own labels) and drives the bold/bright active-page style -
    Streamlit's own href="" self-link convention looked like it could do this
    automatically, but page_link marks the MAIN script's link as href=""
    unconditionally on every page, not only when actually on it (Streamlit
    1.54, confirmed live), so "Home" stayed bold everywhere. Pages with no nav
    entry (e.g. the legal pages) pass no ``active`` and nothing highlights.
    """
    cols = st.columns([2.2, 1.0, 1.0, 1.0, 1.7], vertical_alignment="center")
    cols[0].markdown(
        f'<div class="nav-brand-row"><span class="nav-brand-badge">'
        f'{mark_svg("strideo-nav-grad")}</span>'
        f'<span class="nav-brand-name">Strideo</span></div>',
        unsafe_allow_html=True,
    )
    for col, (label, page) in zip(cols[1:], _NAV_ITEMS):
        if label == active:
            col.markdown(
                '<span class="nav-active-mark"></span>', unsafe_allow_html=True
            )
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


_FOOTER_LEGAL_ITEMS = [
    ("Privacy Policy", "pages/privacy_policy.py"),
    ("Terms of Service", "pages/terms_of_service.py"),
    ("Cookie Policy", "pages/cookie_policy.py"),
]

_GITHUB_URL = "https://github.com/Matteofanfoni/strideo"


def scroll_to_top() -> None:
    """Force the page to the top on the *next* render only.

    ``st.switch_page`` is a client-side re-render, not a real browser
    navigation, so the scroll position from whatever page (and however far
    down it) the user was on carries straight over — the actual bug this
    fixes: landing on Results already scrolled down because Upload had been
    scrolled through its progress bars. ``st.markdown`` strips ``<script>``
    tags, so this goes through ``components.v1.html`` instead, whose iframe
    can still reach ``window.parent`` (same origin). Call once, right after
    setting the session-state flag that gates it, not on every rerun - a
    page the user has manually scrolled (e.g. re-selecting a Run) must not
    keep jumping back to the top under them.
    """
    import streamlit.components.v1 as components

    components.html(
        """<script>
        var d = window.parent.document;
        window.parent.scrollTo(0, 0);
        d.documentElement.scrollTop = 0;
        d.body.scrollTop = 0;
        var main = d.querySelector('[data-testid="stAppViewContainer"]');
        if (main) { main.scrollTop = 0; }
        </script>""",
        height=0,
    )


def render_footer() -> None:
    """Render the shared footer: a full-bleed navy bar with the brand lockup on
    the left and GitHub / strideo.org / the three legal-doc pages on the right,
    the two halves centred on one line. The three legal pages use
    ``st.page_link`` (not raw ``<a href>``) for the same session-preserving
    reason the navbar does (see ``render_navbar``); the GitHub link does too
    since ``st.page_link`` accepts external URLs directly.
    """
    left, right = st.columns(2, gap="large")
    with left:
        st.markdown(
            f"""
<div class="footer-brand-row">
  <span class="footer-brand-mark">{mark_svg("strideo-footer-grad")}</span>
  <span class="footer-brand-name">Strideo</span>
</div>
""",
            unsafe_allow_html=True,
        )
    with right:
        right.page_link(_GITHUB_URL, label="GitHub / Strideo")
        st.markdown(
            '<span class="footer-meta-line">strideo.org · research ' "preview</span>",
            unsafe_allow_html=True,
        )
        legal_cols = st.columns(len(_FOOTER_LEGAL_ITEMS), gap="small")
        for col, (label, page) in zip(legal_cols, _FOOTER_LEGAL_ITEMS):
            col.page_link(page, label=label)


def render_legal_document(html_text: str) -> None:
    """Render a Termly-exported legal document (Privacy Policy, Terms of
    Service, Cookie Policy) inline, as part of the normal page flow.

    Earlier versions embedded this via ``components.html`` in an iframe, but
    Streamlit bakes that iframe's *container* height from the Python-side
    ``height=`` argument into a fixed-size box independent of the iframe's
    real content - no in-page JS can resize it. A guessed height either
    clipped short content (an inner scrollbar next to the page's own) or left
    a gap after tall content, and anything rendered after it (the footer)
    landed at the bottom of that guessed box, overlapping the actual content
    if the guess was wrong. None of the three documents has a `<script>` tag,
    so there is nothing an iframe was buying here - `st.markdown` renders the
    same static HTML/CSS directly into the page, sized to its real content,
    on the page's own (light) background, with the footer following in
    normal flow immediately after.

    The source file's own indentation is stripped first: Streamlit's markdown
    renderer runs content through CommonMark before allowing raw HTML
    through, and a line indented 4+ spaces is CommonMark's syntax for an
    indented code block - exactly what this Termly export's formatting
    triggers, rendering the whole document as a literal code block instead
    of markup. Collapsing to one line (harmless for HTML, which does not
    treat inter-tag whitespace as meaningful here) sidesteps every line-start
    rule at once.

    The export also leaves Termly's own template markup in place - `<bdt>`
    wrapper tags (merge-fields, conditional-block markers) that Termly's own
    site hides/unwraps with its own CSS, absent here. `<bdt>` is not a real
    HTML element, so an unstyled browser defaults it to `display: inline`;
    when one wraps a large run of block-level `<div>` paragraphs (as one does
    in the Terms of Service export, ~80 children under a single
    `block-container if` node), each inline/block boundary the browser has to
    resolve adds its own stray vertical space, compounding into a multi-
    thousand-pixel gap in the middle of the document. `display: contents`
    makes the wrapper generate no box of its own - its children lay out
    exactly as if `<bdt>` were not there - which is what Termly's own CSS
    effectively does. Measured on Terms of Service: 25047px -> 9219px.
    """
    st.markdown(
        "<style>[data-testid='stMarkdownContainer'] bdt{display:contents}</style>"
        + " ".join(html_text.splitlines()),
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
