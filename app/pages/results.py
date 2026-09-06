# app/pages/results.py
import streamlit as st
import sys
import json
import math
from dataclasses import asdict
from datetime import datetime
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
    render_empty_state,
    render_footer,
    scroll_to_top,
)
from src.utils.pace_predictor import format_pace_per_km  # noqa: E402
from src.utils.vo_correction import (  # noqa: E402
    correct_engine_metrics_for_display,
)
from utils.elite_reference import (  # noqa: E402
    METRICS,
    TIER_LABELS,
    get_elite_ranges,
)

st.set_page_config(
    page_title="Results | strideo.it",
    page_icon=str(app_dir / "assets" / "favicon.png"),
    layout="wide",
)

inject_shared_styles()

# Every section of this page keeps its structure; this is the new design
# system's surface language applied to it. Text classes carry !important on
# margins/sizes because Streamlit's own markdown rules (`.st-emotion-… p`,
# `h1-h3`) outrank a bare class selector.
inject_page_css("""
/* ── Page + run headings ── */
.res-title{
  font-family: 'Space Grotesk', ui-sans-serif, system-ui, sans-serif;
  font-size: clamp(2.1rem, 3.5vw, 3.1rem) !important;
  font-weight: 700;
  letter-spacing: -0.035em;
  line-height: 1.04 !important;
  color: var(--text);
  margin: 16px 0 0 !important;
  padding: 0 !important;
}
.res-lead{
  font-size: 1.05rem;
  line-height: 1.65;
  color: var(--text-muted);
  margin: 18px 0 0 !important;
}
/* 250x48px matches the hero "Analyse a clip" button's own measured size
   (getBoundingClientRect) - same text, same size, not a full-width banner. */
.st-key-results_empty_cta{ max-width: 250px; margin-top: 22px; }
.run-title{
  font-family: 'Space Grotesk', ui-sans-serif, system-ui, sans-serif;
  font-size: 1.15rem;
  font-weight: 600;
  color: var(--text);
  margin: 18px 0 0 !important;
}
.run-meta{
  font-family: 'JetBrains Mono', ui-monospace, monospace;
  font-size: 0.7rem;
  letter-spacing: 0.1em;
  color: var(--text-muted);
  margin: 6px 0 0 !important;
}

/* ── Section heading: mono eyebrow sitting under a hairline rule ── */
.section-heading{
  font-family: 'JetBrains Mono', ui-monospace, monospace;
  font-size: 0.7rem;
  font-weight: 500;
  text-transform: uppercase;
  letter-spacing: 0.2em;
  color: var(--brand-text);
  /* L1 (2026-08-31, user): no hairline rule between sections. Once the bands
     run edge to edge, the alternating background is the separator and a
     divider on top of it reads as clutter.

     Headroom cut from 52px to 16px later the same day (user: reduce the
     headroom above Calculated Metrics, Elite Range Comparison and Cross-Run
     Comparison). The 18px padding was the clearance under that hairline rule
     and had no job left once the rule went; the 34px margin was set to keep
     the rule's old rhythm, which the band backgrounds now provide instead.
     This reaches exactly the plain sections, those three plus Runner
     Information: a banded section's heading is already zeroed by the
     .res-band-mark rule below, taking its headroom from the band's own
     padding. */
  margin: 16px 0 14px !important;
  padding: 0 !important;
}
.section-sub{
  color: var(--text-muted);
  font-size: 0.9rem;
  line-height: 1.6;
  margin: 0 0 14px !important;
}

/* ── Pill strip: the workhorse for Video Information, Calculated Metrics,
   Runner Information and Pipeline Diagnostics ──

   A grid of fixed 200px tracks, not the flex-wrap it was until 2026-08-31.
   Flex sized every pill to its own content, so a strip wrapped into ragged
   rows whose boxes lined up with nothing above or below them (user: "I want
   the boxes aligned and to form a rectangle"). Equal fixed tracks make every
   box the same width and every column continuous down the strip.

   auto-fill + justify-content:start rather than 1fr tracks: the same user
   asked for the block to stay anchored left rather than stretch to the right
   edge, so the row takes as many 200px boxes as fit and stops. 200px is set
   by the longest label a pill carries at this size, "Vertical Oscillation"
   in Calculated Metrics, plus the 32px of horizontal padding.

   Video Information overrides the track list (see .vinfo-panel-mark below):
   it sits in a narrow column beside the video, where fixed tracks would
   leave a ragged margin against the video's edge. ── */
.hdr-strip{
  display: grid;
  grid-template-columns: repeat(auto-fill, 200px);
  justify-content: start;
  gap: 12px;
  margin: 4px 0 20px;
}
.hdr-pill{
  display: flex;
  flex-direction: column;
  gap: 3px;
  background: var(--card);
  border: 1px solid var(--border);
  border-radius: var(--radius-md);
  padding: 12px 16px;
  /* The grid track sets the width now; a min-width would only let a long
     value push its own box out of the column and break the rectangle. */
  min-width: 0;
}
.hdr-pill-label{
  font-family: 'JetBrains Mono', ui-monospace, monospace;
  font-size: 0.63rem;
  text-transform: uppercase;
  letter-spacing: 0.14em;
  color: var(--text-subtle);
}
/* Small circled "?" with a real CSS hover tooltip (content: attr(data-tip))
   rather than the native title= attribute - user-reported: native tooltips
   weren't appearing (browser-dependent delay/hover-precision quirks on a
   13px target). This version renders immediately and is fully styled, not
   dependent on OS/browser tooltip behavior. Couldn't find a prior instance
   of "the session explorer's" own tooltip pattern in dev_tools to match
   exactly (grepped for help=/tooltip/circled markers, none found). */
.info-tip{
  position: relative;
  display: inline-flex; align-items: center; justify-content: center;
  width: 13px; height: 13px; border-radius: 50%;
  background: var(--text-subtle); color: var(--card);
  font-family: ui-sans-serif, system-ui, sans-serif;
  font-size: 0.62rem; font-weight: 700; line-height: 1;
  margin-left: 5px; cursor: help; user-select: none;
  vertical-align: middle;
}
.info-tip::after{
  content: attr(data-tip);
  position: absolute;
  bottom: calc(100% + 8px);
  left: 50%;
  transform: translateX(-50%);
  background: var(--text);
  color: var(--card);
  padding: 8px 10px;
  border-radius: var(--radius-md);
  font-family: ui-sans-serif, system-ui, sans-serif;
  font-size: 0.72rem;
  font-weight: 400;
  text-transform: none;
  letter-spacing: normal;
  line-height: 1.45;
  white-space: normal;
  width: max-content;
  max-width: 240px;
  text-align: left;
  opacity: 0;
  visibility: hidden;
  pointer-events: none;
  transition: opacity 0.12s ease;
  z-index: 60;
}
.info-tip:hover::after{ opacity: 1; visibility: visible; }
.hdr-pill-value{
  font-family: 'JetBrains Mono', ui-monospace, monospace;
  font-size: 1rem;
  font-weight: 600;
  color: var(--text);
  /* A file name or an event+PB is the one value too long for its track, so
     it wraps; the grid stretches its row-mates to match rather than letting
     one box grow wider than its column. */
  overflow-wrap: anywhere;
}
/* Unvalidated-metric marker (vertical oscillation, and its range row) */
.metric-flag{ color: var(--warn); font-weight: 600; }

/* ── Tinted callouts: runner profile, calibration note, pace comparison ── */
.profile-note, .pace-compare{
  background: var(--accent-tint);
  border: 1px solid var(--border);
  border-left: 3px solid var(--brand);
  border-radius: var(--radius-md);
  padding: 14px 18px;
  margin: 0 0 22px !important;
  font-size: 0.92rem;
  line-height: 1.65;
  color: var(--text-muted);
}
.profile-note strong{ color: var(--text); }
.pace-compare{ margin: 16px 0 0 !important; }
.profile-paces{ font-weight: 600; color: var(--brand-text); }
.profile-cav{ color: var(--text-subtle); font-size: 0.82rem; font-weight: 400; }
.pace-compare strong{ color: var(--text); }
.pace-compare .pc-delta{ font-weight: 600; }
.pace-compare .pc-faster{ color: var(--red); }
.pace-compare .pc-slower{ color: var(--accent); }
.pace-compare .pc-onpace{ color: var(--brand-text); }

/* ── Calculated Metrics card: one bordered/backed card holding
   every value (core + derived) as pills in the same format, rather than
   several separate floating pill-boxes plus a differently-styled derived
   text line. Nested pills use --card-2 (the shared inset/hover fill, same
   convention as .stat-cell) for contrast against the card's own --card
   background.

   REVISED 2026-08-31 (user): the card box itself is gone. This section sits
   on one of the page's darker bands, where a white container wrapping white-
   ish pills read as a box inside a box; the pills alone carry the section
   now. Background, border, radius and padding go, the margin stays so the
   spacing either side is unchanged, and the pills drop their --card-2
   override so they fall back to the plain white .hdr-pill. The rule that
   used to separate the two strips goes with the box: constrained to the
   card's width it read as a divider, but the strips are left-anchored and it
   would now run the full width of the page. The 16px gap does that job. ── */
.metrics-card{
  margin: 4px 0 20px;
}
.metrics-card .hdr-strip{ margin: 0; }
.metrics-card .hdr-strip + .hdr-strip{ margin-top: 16px; }

/* ── Elite range bars ── */
.range-row{
  display: grid; grid-template-columns: 150px 1fr 150px;
  align-items: center; gap: 16px; margin: 14px 0;
}
.range-name{ font-weight: 600; font-size: 0.9rem; color: var(--text); }
.range-name .rn-sub{
  font-family: 'JetBrains Mono', ui-monospace, monospace;
  font-weight: 400; color: var(--text-subtle); font-size: 0.68rem;
  text-transform: uppercase; letter-spacing: 0.1em;
}
.range-track{
  position: relative; height: 10px; background: var(--secondary);
  border-radius: 5px;
}
.range-band{
  position: absolute; top: 0; bottom: 0;
  background: rgba(41,154,139,0.28); border-radius: 5px;
}
.range-marker{
  position: absolute; top: -4px; width: 4px; height: 18px;
  background: var(--accent); border-radius: 2px; transform: translateX(-2px);
}
.range-verdict{
  font-family: 'JetBrains Mono', ui-monospace, monospace;
  font-size: 0.72rem; color: var(--text-muted); text-align: right;
  line-height: 1.7;
}
.range-verdict strong{ color: var(--text); font-size: 0.85rem; }
.range-scale{
  display: flex; justify-content: space-between;
  font-family: 'JetBrains Mono', ui-monospace, monospace;
  font-size: 0.62rem; color: var(--text-subtle); margin-top: 4px;
}
.range-sources{
  font-size: 0.8rem; color: var(--text-subtle);
  margin: 16px 0 0 !important; line-height: 1.65;
}
.range-sources a{ color: var(--brand-text); text-decoration: none; }
.range-sources a:hover{ text-decoration: underline; }

/* ── Cross-run comparison table ── */
.compare-table{
  width: 100%;
  border-collapse: separate;
  border-spacing: 0;
  margin-top: 16px;
  background: var(--card);
  border: 1px solid var(--border);
  border-radius: var(--radius-lg);
  overflow: hidden;
}
.compare-table th{
  background: var(--secondary);
  font-family: 'JetBrains Mono', ui-monospace, monospace;
  color: var(--text-muted);
  font-size: 0.65rem;
  font-weight: 500;
  text-transform: uppercase;
  letter-spacing: 0.14em;
  padding: 12px 16px;
  text-align: left;
  border-bottom: 1px solid var(--border);
}
.compare-table td{
  padding: 13px 16px;
  font-size: 0.9rem;
  color: var(--text);
  border-bottom: 1px solid var(--border-soft);
}
.compare-table tr:last-child td{ border-bottom: none; }
.compare-table tr:hover td{ background: var(--hover-fill); }
.compare-table .pace-col{ color: var(--text); font-weight: 600; }

/* ── Contact Tracker subsection heading: a lighter mono eyebrow
   nested under .section-heading's "Deterministic kinematics engine", not a
   second top-level section of its own ── */
.subsection-heading{
  font-family: 'JetBrains Mono', ui-monospace, monospace;
  font-size: 0.68rem;
  font-weight: 600;
  text-transform: uppercase;
  letter-spacing: 0.16em;
  color: var(--text-muted);
  margin: 22px 0 10px !important;
}

/* Equal-height columns (the Video Information / annotated-video hero row
   relies on this) */
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

/* st.download_button matches the navy action buttons beside it */
[data-testid="stDownloadButton"] button{
  background: var(--navy) !important;
  color: var(--navy-foreground) !important;
  border: none !important;
  border-radius: var(--radius-md) !important;
  padding: 12px 0 !important;
  width: 100% !important;
  font-family: 'Space Grotesk', sans-serif !important;
  font-weight: 600 !important;
  font-size: 0.95rem !important;
  box-shadow: none !important;
  transition: background 0.2s ease !important;
}
[data-testid="stDownloadButton"] button:hover{
  box-shadow: none !important;
  background: color-mix(in srgb, var(--navy) 88%, white 12%) !important;
}
[data-testid="stDownloadButton"] button *{
  color: inherit !important;
  font-family: inherit !important;
  font-weight: inherit !important;
  font-size: inherit !important;
}

/* Annotated-video plate - same navy/rounded/shadow language as the Home
   hero frame, so the shipped video reads consistently with the rest of the
   site. Sits top-right beside the Video Information column (user feedback
   2026-08-27: match Home's "up right" hero placement, not a standalone
   block further down the page). */
/* Video Information panel: a tinted background (not the shared
   .section-heading divider line) sets it apart as the page's first block -
   same reusable mark/:has() pattern as .res-vidframe-mark below. */
/* L1: the page's alternating band - Video Information tinted, Calculated
   Metrics plain, the engine section tinted, Elite Range plain, Pace Analysis
   tinted, Cross-Run plain.

   REVISED 2026-08-31 after seeing it rendered (user: "the bands should be
   end-to-end"). The first pass reused .vinfo-panel-mark's *contained* panel,
   reasoning that a full-bleed tint would fight the cards inside it. On the
   real page that was wrong: a rounded, inset panel reads as one more card in
   a page already full of cards, so it never registered as a band at all. It
   now uses the same full-bleed escape-the-column trick as Home's .band and
   the Guide's .capture-band-mark, so the tint runs edge to edge and the
   alternation is legible as page structure rather than as another container.

   No border-top/bottom, unlike .band: the same feedback asked for the fine
   lines between sections to go, and once a band runs edge to edge its own
   background is the separator. Note the Guide-page lesson (L1's own row): on
   this palette the tinted surface reads LIGHTER than the page's default
   ground, so "banded" means lighter here. */
/* Empty state (2026-08-31, user): "Load saved results" must read as the peer
   of "Analyse a clip", not as a differently-shaped control next to it.
   Measured rather than eyeballed - the popover trigger was 428x40 at y=488
   against the button's 250x48 at y=510, because the popover fills its whole
   column while the shared button CSS caps at 250px, and Streamlit's trigger
   carries its own height and 16px font. Scoped through the marker span so it
   cannot reach any other popover on the page. */
[data-testid="stElementContainer"]:has(.empty-load-mark){ display: none; }
[data-testid="stVerticalBlock"]:has(> [data-testid="stElementContainer"] .empty-load-mark)
  [data-testid="stPopoverButton"]{
  /* width:100% + the same 250px cap the shared button CSS applies, so the two
     controls stay identical at any viewport rather than matching only at the
     width this was measured at. */
  width: 100% !important;
  max-width: 250px !important;
  height: 48px !important;
  font-size: 0.95rem !important;
  font-weight: 600 !important;
  justify-content: center !important;
  border-radius: var(--radius-md) !important;
}
/* ...and the same 22px top margin the button's own element container gets
   from the shared button spacing. Without it the two columns are 70px and
   48px tall, so vertical_alignment="center" splits the difference and leaves
   the pair 10px out. Matching the margin rather than stripping it keeps the
   gap below the empty-state card unchanged. */
[data-testid="stVerticalBlock"]:has(> [data-testid="stElementContainer"] .empty-load-mark)
  [data-testid="stPopover"]{
  margin-top: 22px;
}
.section-heading.no-rule{ border-top: none; margin-top: 0 !important; padding-top: 0 !important; }
[data-testid="stElementContainer"]:has(.vinfo-panel-mark),
[data-testid="stElementContainer"]:has(.res-band-mark){ display: none; }
[data-testid="stVerticalBlock"]:has(> [data-testid="stElementContainer"] .vinfo-panel-mark),
[data-testid="stVerticalBlock"]:has(> [data-testid="stElementContainer"] .res-band-mark){
  background: color-mix(in srgb, var(--card) 60%, transparent);
  margin: 30px calc(50% - 50vw);
  width: 100vw;
  max-width: 100vw;
  box-sizing: border-box;
  padding: 34px max(20px, calc(50vw - 50%));
}
@media (max-width: 700px){
  [data-testid="stVerticalBlock"]:has(> [data-testid="stElementContainer"] .vinfo-panel-mark),
  [data-testid="stVerticalBlock"]:has(> [data-testid="stElementContainer"] .res-band-mark){
    margin: 22px calc(50% - 50vw);
    padding: 26px 20px;
  }
}
/* A banded section's heading already has the band's own top padding above it,
   so its top margin would double the gap. */
[data-testid="stVerticalBlock"]:has(> [data-testid="stElementContainer"] .res-band-mark) .section-heading{
  margin-top: 0 !important;
  padding-top: 0 !important;
}
/* Video Information is the one strip that stretches, in both axes. Equal
   tracks rather than the shared fixed 200px ones, because it sits beside the
   annotated video and any leftover gutter reads as a mistake against the
   video's edge; each pill then claims a span of those tracks through its own
   w<n> class, giving the 3 / 2 / 3 / 4 rows the user specified 2026-08-31
   (see _render_header_strip's docstring for the order and the spans).

   48 tracks, not 12, so the frame-statistics row can be split 11/13/13/11
   instead of four even quarters: its two middle labels are the long ones.
   See _render_header_strip's docstring for why 48 and not 24.

   The vertical half is what the absolute positioning below is for: 1fr rows
   in a full-height grid make every box exactly the same height AND end the
   block level with the video, whatever the clip's aspect ratio (user,
   2026-08-31: "make the height of the boxes all the same to fit in any case
   the total height of the video").

   Stretching alone does not get there. The two columns are already
   equal-height (see the stretch rule above), but that height is the taller
   column's *content*, and with twelve boxes over four rows the strip is the
   taller one - measured 381px against the video's 346px, so it overhung by
   39px and the stretch had nothing left to do. Taking the strip out of flow
   drops the vinfo column's content height to zero, which leaves the video
   as the only thing setting the row height; the grid then fills exactly
   that. The element container is the positioning parent and the only child
   in this column, so flex:1 gives it the column's full height.

   Every rule that takes the strip out of flow is gated on the panel
   actually CONTAINING a video (:has(.res-vidframe-mark)), because out-of-flow
   only works while something else sets the row height. Load a saved results
   file and there is no video to load - results and video are exported as
   separate downloads, so results-without-video is the supported path, not
   a corruption - and the row then collapsed to the height of the one-line
   placeholder that stands in for it. Measured on the real page before the
   fix: the strip was 56px tall holding 101px of content, and its twelve
   pills were 25px each against the 105px and 89px the same pill gets in the
   sections below, so labels rendered on top of their own values. Gating on
   the video rather than adding a no-video override is deliberate: a third
   branch in the video column would inherit the right behaviour instead of
   silently reintroducing this. */
[data-testid="stVerticalBlock"]:has(> [data-testid="stElementContainer"] .vinfo-panel-mark)
  .hdr-strip{
  grid-template-columns: repeat(48, minmax(0, 1fr));
}
[data-testid="stVerticalBlock"]:has(> [data-testid="stElementContainer"] .vinfo-panel-mark):has(.res-vidframe-mark)
  .hdr-strip{
  /* minmax(0, 1fr), not a bare 1fr: 1fr floors a row at its own max-content
     height, so a row with a wrapping label stayed 86px while the rest shared
     what was left at 74px. A zero minimum makes all four rows equal. */
  grid-auto-rows: minmax(0, 1fr);
  position: absolute;
  inset: 0;
  margin: 0;
}
/* No video: back into normal flow, sizing to its own content, exactly as the
   narrow-viewport branch below already does for the same reason (there, the
   columns stack and there is no video beside the strip either). 1fr rather
   than minmax(0, 1fr) for the same reason it gives there: with no fixed
   height to divide up, rows should equalise to the tallest content, not to
   zero. */
[data-testid="stVerticalBlock"]:has(> [data-testid="stElementContainer"] .vinfo-panel-mark):not(:has(.res-vidframe-mark))
  .hdr-strip{
  grid-auto-rows: 1fr;
  margin: 0 0 4px;
}
[data-testid="stVerticalBlock"]:has(> [data-testid="stElementContainer"] .vinfo-panel-mark):has(.res-vidframe-mark)
  [data-testid="stColumn"]:has(.hdr-strip) [data-testid="stElementContainer"]{
  position: relative;
  flex: 1;
  min-height: 0;
}
/* Both text rows inherit a 1.6 line-height from the page, which is right for
   prose and far too loose for a 10px uppercase label stacked on one number.
   It cost 12px a box, and the frame-statistics row could not afford it: its
   boxes are the narrow ones, and a label that wraps to two lines came to
   85px of content inside a 77px row. Scoped here rather than applied to
   .hdr-pill everywhere because
   Video Information is the only strip whose row height is fixed from
   outside; the others size to their own content and are already settled. */
/* Darker fill here (user, 2026-08-31), the same --card-2 inset the metrics
   pills used to carry. Video Information's band is the light one, so white
   pills on it barely separated from their own background; --card-2 gives
   them an edge that does not depend on the hairline border alone. Runner
   Information keeps the plain white pill, as does Calculated Metrics, which
   sits on a darker band and needs the contrast the other way round. */
[data-testid="stVerticalBlock"]:has(> [data-testid="stElementContainer"] .vinfo-panel-mark)
  .hdr-pill{ background: var(--card-2); }
[data-testid="stVerticalBlock"]:has(> [data-testid="stElementContainer"] .vinfo-panel-mark)
  .hdr-pill-label{ line-height: 1.25; }
[data-testid="stVerticalBlock"]:has(> [data-testid="stElementContainer"] .vinfo-panel-mark)
  .hdr-pill-value{ line-height: 1.3; }
[data-testid="stVerticalBlock"]:has(> [data-testid="stElementContainer"] .vinfo-panel-mark)
  .hdr-pill.w11{ grid-column: span 11; }
[data-testid="stVerticalBlock"]:has(> [data-testid="stElementContainer"] .vinfo-panel-mark)
  .hdr-pill.w13{ grid-column: span 13; }
[data-testid="stVerticalBlock"]:has(> [data-testid="stElementContainer"] .vinfo-panel-mark)
  .hdr-pill.w16{ grid-column: span 16; }
[data-testid="stVerticalBlock"]:has(> [data-testid="stElementContainer"] .vinfo-panel-mark)
  .hdr-pill.w24{ grid-column: span 24; }
[data-testid="stElementContainer"]:has(.res-vidframe-mark){ display: none; }
[data-testid="stVerticalBlock"]:has(> [data-testid="stElementContainer"] .res-vidframe-mark){
  border-radius: var(--radius-lg);
  overflow: hidden;
  background: var(--navy);
  box-shadow: var(--shadow-hero);
  gap: 0 !important;
}
[data-testid="stVerticalBlock"]:has(> [data-testid="stElementContainer"] .res-vidframe-mark) video{
  display: block;
  width: 100%;
  border: none !important;
  border-radius: 0 !important;
  box-shadow: none !important;
}

@media (max-width: 700px){
  /* Two stretched columns everywhere below 700px: a 200px track no longer
     fits two-up once the viewport is this narrow, and a single column of
     wide boxes wastes the row. Replaces the old .hdr-pill flex-basis rule,
     which a grid container ignores. Video Information needs its own copy of
     this: a media query does not raise specificity, so its own rules would
     otherwise keep winning and squeeze its values into ~70px of content
     width on a phone. It keeps the 48-track grid and restates the spans
     instead, since that is what its row layout is built on - two boxes a
     row, with the file name full width. */
  .hdr-strip{ grid-template-columns: repeat(2, minmax(0, 1fr)); }
  /* Back into normal flow below the breakpoint: the columns stack here, so
     there is no video height to match and an out-of-flow strip would sit in
     a zero-height container and overlap whatever follows. */
  [data-testid="stVerticalBlock"]:has(> [data-testid="stElementContainer"] .vinfo-panel-mark)
    .hdr-strip{
    position: static;
    /* 1fr, not minmax(0, 1fr): with no video to match there is no height to
       divide up, so the rows should equalise to the tallest content rather
       than to zero. Boxes stay uniform, nothing overflows. */
    grid-auto-rows: 1fr;
    margin: 4px 0 20px;
  }
  [data-testid="stVerticalBlock"]:has(> [data-testid="stElementContainer"] .vinfo-panel-mark)
    .hdr-pill.w11,
  [data-testid="stVerticalBlock"]:has(> [data-testid="stElementContainer"] .vinfo-panel-mark)
    .hdr-pill.w13,
  [data-testid="stVerticalBlock"]:has(> [data-testid="stElementContainer"] .vinfo-panel-mark)
    .hdr-pill.w16{ grid-column: span 24; }
  [data-testid="stVerticalBlock"]:has(> [data-testid="stElementContainer"] .vinfo-panel-mark)
    .hdr-pill.w24{ grid-column: span 48; }
  .range-row{ grid-template-columns: 1fr; gap: 4px; }
  .range-verdict{ text-align: left; }
  /* Wide cross-pace table scrolls horizontally instead of overflowing */
  .compare-table{ display: block; overflow-x: auto; white-space: nowrap; }
}
""")

render_navbar("Results")

if st.session_state.pop("scroll_results_top", False):
    scroll_to_top()


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
    # Same left-aligned header as the populated page below, so the empty state
    # is not a differently-shaped page.
    # Same CTA-style heading as the populated page (RP1), and no lead
    # paragraph: it used to read "Nothing to show yet. Analyse a clip and its
    # metrics land here", which the empty-state card below now says in full.
    # RP1 originally kept "Analysis results" here on the grounds that "view
    # your results" does not fit when there is nothing to view; reversed
    # 2026-08-31 at the user's direction, so the page has one heading in both
    # states instead of changing identity depending on session state.
    st.markdown(
        """
<p class="eyebrow">step 2 of 2</p>
<h1 class="res-title">View your results</h1>
""",
        unsafe_allow_html=True,
    )
    render_empty_state(
        "Nothing here yet. Analyse a clip to measure your running form, or "
        "load a results file you saved from an earlier session."
    )
    # [1, 1, 3] rather than [1, 1, 0.9]: both controls are capped at 250px, so
    # wider columns left them ~180px apart and reading as unrelated. This puts
    # each column at roughly the button's own width, so the pair sits together
    # with only the column gap between them.
    _empty_cta, _empty_load, _ = st.columns([1, 1, 3], vertical_alignment="center")
    with _empty_cta:
        if st.button(
            "Analyse a clip",
            type="primary",
            use_container_width=True,
            key="results_empty_cta",
        ):
            st.session_state["scroll_upload_top"] = True
            st.switch_page("pages/upload.py")
    with _empty_load:
        # Re-import a "Save results" export from an earlier
        # session. No server-side retention either way - the file only ever
        # lives in the browser upload and this session's memory.
        st.markdown('<span class="empty-load-mark"></span>', unsafe_allow_html=True)
        with st.popover("Load saved results", use_container_width=True):
            st.caption('Re-import a results file saved earlier with "Save results".')
            _uploaded = st.file_uploader(
                "Results JSON",
                type="json",
                key="results_reimport_uploader",
                label_visibility="collapsed",
            )
            if _uploaded is not None:
                try:
                    _payload = json.loads(_uploaded.read().decode("utf-8"))
                    _imported = _payload.get("results")
                    if not isinstance(_imported, list) or not _imported:
                        raise ValueError("File has no results to load.")
                    # A file saved before 2026-08-31 carries
                    # running_economy_index: the same vertical oscillation over
                    # a full stride rather than a step, so x2 lands it on the
                    # current definition exactly, with nothing re-measured.
                    # Without this the page raises KeyError on the renamed
                    # field rather than degrading, since every metrics read
                    # below subscripts it directly.
                    for _entry in _imported:
                        for _key in ("metrics_fast", "metrics_full"):
                            _mx = _entry.get(_key)
                            if not isinstance(_mx, dict):
                                continue
                            if "vertical_oscillation_ratio" in _mx:
                                continue
                            _legacy = _mx.get("running_economy_index")
                            _mx["vertical_oscillation_ratio"] = (
                                _legacy * 2 if _legacy is not None else float("nan")
                            )
                        # Engine metrics from a file saved before the display
                        # correction shipped are on the raw pipeline scale.
                        # correct_engine_metrics_for_display is idempotent via
                        # its own flag, so a newer file passes through here
                        # untouched rather than being corrected twice.
                        _entry["metrics_full"] = correct_engine_metrics_for_display(
                            _entry.get("metrics_full")
                        )
                    st.session_state["analysis_results"] = _imported
                    st.rerun()
                except Exception as e:  # noqa: BLE001 - surface a malformed file
                    st.error(f"Couldn't load that file: {e}")
    render_footer()
    st.stop()

# Each entry carries an always-present fast-path result plus an
# optional classical ("Full analysis") one, added later per-clip via the
# Ground-Contact Verifier's own trigger. Resolve "best available" once here
# so the rest of this page's existing metrics/preprocessing/strike_pattern/
# analysis reads keep working unchanged — Full analysis, once run, wins for
# every section except the Calculated Metrics pills, which show both,
# source-labeled (add-only, not a silent overwrite).
for r in results:
    r["metrics"] = r.get("metrics_full") or r.get("metrics_fast")
    r["preprocessing"] = r.get("preprocessing_full") or r.get("preprocessing_fast")
    r["strike_pattern"] = r.get("strike_pattern_full")
    r["analysis"] = r.get("analysis_full")
    r["has_full_analysis"] = r.get("metrics_full") is not None
    # Why there is no annotated video, which the placeholder below has to
    # say correctly. `upload.py`'s result_entry always writes the key, None if
    # encoding failed, so a missing KEY (not a falsy value) means this entry
    # came back through "Load saved results": `annotated_video_fast` is in
    # `_EXPORT_EXCLUDE_KEYS`, the video having its own separate download.
    # "Couldn't be built" is a real failure; "not in this file"
    # is the normal shape of a re-imported session, and telling the user the
    # first when it is the second misdiagnoses their own file to them.
    r["video_omitted_by_export"] = "annotated_video_fast" not in r

# Filter to results that have metrics
results_with_metrics = [r for r in results if r.get("metrics") is not None]
results_without_metrics = [r for r in results if r.get("metrics") is None]

st.markdown(
    f"""
<p class="eyebrow">step 2 of 2</p>
<h1 class="res-title">View your results</h1>
<p class="res-lead">{len(results)} video{'s' if len(results) != 1 else ''}
analyzed. Metrics below are StrideoNet's result by default. For any clip, run
a second, independent check with the deterministic kinematics engine:
contact-by-contact timing, strike-pattern detection, full anatomical
parameters, and a second opinion on every metric.</p>
""",
    unsafe_allow_html=True,
)


# ─────────────────────────────────────────────────────────────
# Helper to safely format a metric value
# ─────────────────────────────────────────────────────────────


def _info_tip(text: str) -> str:
    """A small circled "?" that shows ``text`` as a CSS hover tooltip.

    Uses ``data-tip`` + ``content: attr(data-tip)`` (see the ``.info-tip``
    CSS) rather than the native ``title`` attribute - title tooltips proved
    unreliable in practice (browser hover-delay/precision on a 13px target).
    """
    safe = text.replace('"', "&quot;")
    return f'<span class="info-tip" data-tip="{safe}">?</span>'


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


def _cross_clip_metrics(r):
    """Metrics to use when comparing ACROSS clips (Pace Analysis, Cross-Run
    Comparison) - always StrideoNet's fast-path result, never the per-clip
    "best available" ``r["metrics"]`` the rest of the page uses.

    Running the deterministic kinematics engine on just one clip swaps that
    clip's ``r["metrics"]`` from fast to full (see the resolution loop above),
    and the two methods disagree systematically on every metric, speed
    included. A cross-clip trend or table built from whichever clips happen
    to have been re-checked mixes two measurement methods silently, which
    reads as the runner's numbers having changed rather than as a difference
    in method - on S1_04 this desyncs speed enough to reorder an 800 m clip
    behind its own threshold-pace one. Holding every clip to the one method
    that is always available keeps the trend meaningful regardless of which
    clips have had a second check run. Falls back to whatever is resolved if
    metrics_fast is absent (an older saved-results import, pre-dating the
    fast path).
    """
    return r.get("metrics_fast") or r["metrics"]


def _preprocessing_from_full(full_analysis, preprocessing_fast, direction):
    """The ``preprocessing`` diagnostics for a clip once the deterministic
    engine has run.

    Most fields are recomputed from the engine's own ``pose_quality`` and
    should overwrite the fast path's, since the two extractions really do
    differ. ``torso_scale`` is the exception and is **carried forward**: it
    comes out of ``preprocess_for_nn``'s hip-centred normalisation, which is
    StrideoNet's input pipeline and something the engine never runs, so there
    is no engine-side value to supply. It used to be hardcoded ``0.0`` here,
    and because the page resolves ``preprocessing_full or preprocessing_fast``
    the placeholder won outright: every clip the engine had been run on showed
    "Torso scale 0.0000" while every other clip showed the real number. Torso
    length is a property of the clip's landmarks rather than of which
    estimator ran, so the fast path's value is the right one to keep.

    Lives at module level, rather than inline where it is used, so it can be
    tested: the call site sits inside a fragment callback that only runs after
    a multi-minute RTMPose pass, which is why the placeholder shipped
    unnoticed.
    """
    pq = full_analysis.pose_quality
    presence = pq.get("presence_quality") or {}
    return {
        "nn_input_shape": None,
        "mean_visibility": pq["mean_hip_visibility"],
        "detection_rate": pq["detection_rate"] * 100.0,
        "torso_scale": (preprocessing_fast or {}).get("torso_scale", 0.0),
        "direction": direction,
        # Frame statistics for the Video Information section (same
        # fields as the fast path's own).
        "total_frames": pq.get("total_frames"),
        "detected_frames": pq.get("detected_frames"),
        "frames_with_fill": pq.get("frames_with_fill"),
        "presence_window_start": presence.get("window_start"),
        "presence_window_end": presence.get("window_end"),
    }


def _render_header_strip(r, m, pre):
    """Render the whole Video Information strip: twelve pills, one grid.

    Rows are fixed by the ``w<n>`` class on each pill, ``n`` being its width
    in forty-eighths of the strip's 48-column grid, so each row's classes sum
    to 48 and the rows break where they are listed below rather than wherever
    the content happened to wrap. Layout as specified by the user 2026-08-31,
    left to right:

        pace · footwear · date               (16 + 16 + 16)
        file · resolution                    (24 + 24: two wide boxes, a
                                              file name needs the room)
        running direction · torso scale · calibration    (16 + 16 + 16)
        total · detected · presence · filled  (11 + 13 + 13 + 11)

    The last row is why the grid is 48 tracks and not 12: at an even quarter
    each its two long labels wrapped to two lines, so they take a share from
    the two short-labelled boxes either side (user 2026-08-31). 48 is the
    coarsest grid where all four fit on one line: measured single-line needs
    are 122 / 144 / 162 / 129 px, and the 11/13/13/11 split gives 139 / 166 /
    166 / 139. A 24-track 5/7/7/5 was tried first and left "Filled frames"
    4px short of its own label. Rows 1-3 are unaffected by the change of
    denominator: 16 and 24 of 48 are the same widths 4 and 6 of 12 were.

    The frame statistics (that last row) used to render as a second,
    separate ``.hdr-strip`` below this one, which is why ``pre`` is a
    parameter now. Two containers could never align into one rectangle:
    Streamlit puts each ``st.markdown`` in its own element container with
    the column's own flex gap between them. They are also the one row that
    is conditional - a clip with no ``pose_quality`` drops it and leaves
    three rows.

    Strike pattern is engine-only (StrideoNet doesn't classify it) - removed
    from here, now shown only in the Contact Tracker section.
    """
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

    dt = r.get("shooting_datetime") or "-"
    # Keep the calendar date only (drop the time component of an ISO stamp).
    date_txt = dt.split("T")[0] if dt and "T" in dt else dt
    cal = m.get("calibration_confidence")
    cal_txt = fmt(cal, 2)
    cal_color = quality_color(cal if cal is not None else 0)

    # Calibration's explanation used to run as its own full-width line below
    # this whole strip - now a small "?" hover tooltip next to the label
    # itself (user feedback). File name gets its own pill here too (was an
    # always-visible paragraph above this strip, duplicating the run
    # picker) - pace is deliberately not repeated in it, that's the "Pace"
    # pill's own job.
    cal_tip = _info_tip(
        "How confidently the runner's declared height fixed the pixel-to-"
        "metre scale used for every distance metric on this clip. 0-1, "
        "higher is better."
    )
    # The one number in this strip that had no explanation at all, and
    # the label's "(norm.)" reads as "normalised" when the value is the
    # divisor used FOR normalisation, so the tooltip says so outright.
    torso_tip = _info_tip(
        "The length every landmark coordinate is divided by before StrideoNet "
        "sees it: the median hip-to-shoulder distance on the near side, in "
        "pixels. Scaling by it is what lets one model read runners of "
        "different sizes filmed at different distances. It is the reference "
        "for that normalisation, not a normalised value itself."
    )
    direction = pre.get("direction")
    dir_txt = {"LR": "→ Right", "RL": "← Left"}.get(direction, "N/A")

    # (label, value, color, tip, width class)
    pills = [
        ("Pace", r.get("pace_label", "-"), None, "", "w16"),
        ("Footwear", r.get("shoe_label", "-"), None, "", "w16"),
        ("Date", date_txt, None, "", "w16"),
        ("File", r.get("video_name", "-"), None, "", "w24"),
        ("Resolution", res_txt, None, "", "w24"),
        ("Running direction", dir_txt, None, "", "w16"),
        (
            "Torso scale (norm.)",
            f"{pre.get('torso_scale', 0):.4f}",
            None,
            torso_tip,
            "w16",
        ),
        ("Calibration", cal_txt, cal_color, cal_tip, "w16"),
    ]

    # Frame statistics - total/detected/presence-window/filled, from
    # pose_quality (src/preprocessing/pipeline.py) via pipeline_runner.py.
    total_f = pre.get("total_frames")
    if total_f is not None:
        win_start = pre.get("presence_window_start")
        win_end = pre.get("presence_window_end")
        window_txt = (
            f"{int(win_start)} - {int(win_end)}"
            if win_start is not None and win_end is not None
            else "N/A"
        )
        presence_tip = _info_tip(
            "First to last frame where a hip landmark AND an ankle landmark "
            "are both detected (coarse presence check, not a full-body check "
            "- in practice BlazePose detects or misses a frame's landmarks "
            "together)."
        )
        pills += [
            ("Total frames", int(total_f), None, "", "w11"),
            ("Detected frames", int(pre.get("detected_frames")), None, "", "w13"),
            ("Presence window", window_txt, None, presence_tip, "w13"),
            ("Filled frames", int(pre.get("frames_with_fill")), None, "", "w11"),
        ]

    html = '<div class="hdr-strip">'
    for label, value, color, tip, width in pills:
        style = f' style="color:{color}"' if color else ""
        html += (
            f'<div class="hdr-pill {width}">'
            f'<span class="hdr-pill-label">{label}{tip}</span>'
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
        '<p class="section-heading">Runner Information</p>',
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

    # Estimated runner profile (targets/profile/VDOT/confidence) - own row,
    # below identity, in that order left-to-right (user feedback).
    pred = info.get("predicted_paces") or {}
    cp = pred.get("clip_paces_s_per_km") or {}
    pred_pills = []
    for lbl in ("Threshold", "1500m", "800m"):
        if lbl in cp:
            pred_pills.append((f"Target · {lbl}", format_pace_per_km(cp[lbl])))
    if pred.get("profile"):
        pred_pills.append(("Profile (est.)", pred["profile"]))
    if pred.get("vdot"):
        pred_pills.append(("VDOT (est.)", f"≈ {pred['vdot']:.0f}"))
    if pred.get("confidence"):
        pred_pills.append(("Confidence", str(pred["confidence"])))

    if pred_pills:
        html = '<div class="hdr-strip">'
        for label, value in pred_pills:
            html += (
                '<div class="hdr-pill">'
                f'<span class="hdr-pill-label">{label}</span>'
                f'<span class="hdr-pill-value">{value}</span>'
                "</div>"
            )
        html += "</div>"
        st.markdown(html, unsafe_allow_html=True)

    if pred:
        st.markdown(
            '<p class="up-note muted" style="margin-top:-8px;">Profile/VDOT/'
            "confidence/target paces above are estimated from entered PBs; "
            "the measured pace reported per clip below is authoritative.</p>",
            unsafe_allow_html=True,
        )


_render_runner_banner((results[0] or {}).get("runner_info"))


# The two pace pills sit side by side in Calculated Metrics and do not always
# agree, so each says where its own number comes from (user, 2026-08-31).
# They are different measurements, not two roundings of one: "Calculated" is
# reconstructed from stride length and cadence by calculate_velocity()
# (src/utils/metrics.py), while "Measured" inverts the hip-x velocity that
# estimate_velocity_from_hipx() (src/utils/pace_estimator.py) fits directly
# from the runner's displacement. app/pipeline_runner.py keeps the measured
# one on `velocity_ms` deliberately, calling it the authoritative estimate.
# A second, smaller source of disagreement is still open: calculate_velocity
# truncates the seconds while format_pace_per_km rounds them, so the pair can
# differ by 1 s/km even on identical input. The user has that one parked.
calculated_pace_tip = _info_tip(
    "Reconstructed from two other metrics on this card: 1000 / (stride "
    "length x cadence / 120). It inherits whatever error those two carry, "
    "so it can differ slightly from Measured pace."
)
measured_pace_tip = _info_tip(
    "Measured from the runner's own displacement: the mean hip position is "
    "tracked across the longest run of frames where both hips are visible, "
    "fitted to a line for horizontal speed, and converted to seconds per "
    "kilometre. It does not depend on stride length or cadence."
)


def _pace_compare_pills_html(r, m):
    """Measured-vs-target pace as pill HTML (no wrapper div) - spliced into
    the same metrics-card as the rest of Calculated Metrics, rather than a
    separate box below it (user feedback: "aligned with the other Strideo
    calculated metrics").

    Measured pace stands on its own: it needs only this clip's measured
    velocity, so it renders whenever that velocity is valid. The target and
    the comparison against it need a PB-derived target for the clip's
    declared pace, and are dropped when there is none. Returns "" only when
    there is no usable velocity at all.
    """
    vel = _safe_float(m.get("velocity_ms"))
    if vel <= 0:
        return ""
    actual = 1000.0 / vel  # seconds per km
    measured_html = (
        '<div class="hdr-pill">'
        f'<span class="hdr-pill-label">Measured pace{measured_pace_tip}</span>'
        f'<span class="hdr-pill-value">{format_pace_per_km(actual)}</span>'
        "</div>"
    )

    info = r.get("runner_info") or {}
    pred = info.get("predicted_paces") or {}
    clip_paces = (pred or {}).get("clip_paces_s_per_km") or {}
    label = r.get("pace_label")
    target = clip_paces.get(label)
    if not target:
        return measured_html

    delta = actual - target  # +ve → slower than target, -ve → faster
    pct = delta / target * 100
    if abs(delta) < 3:
        color, word = "var(--brand-text)", "on target"
    elif delta < 0:
        color = "var(--red)"
        word = f"{abs(delta):.0f} s/km faster"
    else:
        color = "var(--accent)"
        word = f"{delta:.0f} s/km slower"
    return (
        measured_html + '<div class="hdr-pill">'
        f'<span class="hdr-pill-label">Target · {label}</span>'
        f'<span class="hdr-pill-value">{format_pace_per_km(target)}</span>'
        "</div>"
        '<div class="hdr-pill">'
        '<span class="hdr-pill-label">Vs. target</span>'
        f'<span class="hdr-pill-value" style="color:{color}">{word}</span>'
        f'<span class="hdr-pill-label" style="color:{color}">'
        f"{'+' if pct >= 0 else ''}{pct:.1f}%</span>"
        "</div>"
    )


def _render_contact_tracker(analysis, clip_strike_pattern):
    """Per-footstrike IC/TO timing + strike pattern, as a table.

    Same underlying per-contact data (``analysis.contacts``' contact_frame/
    toe_off_frame/gct_ms) the old interactive frame scrubber (since
    removed - user: "I don't see the value of it as it is") overlaid on the
    clip image; shown here as a table instead of a frame-by-frame viewer.
    """
    contacts = list(getattr(analysis, "contacts", None) or [])
    fps = float(getattr(analysis, "fps", 0) or 0)
    n_total = len(contacts)
    n_refined = sum(1 for c in contacts if c.detection_method == "refined")
    refined_pct = (n_refined / n_total * 100) if n_total else 0
    strike_txt = (clip_strike_pattern or "unknown").capitalize()

    st.markdown(
        '<p class="subsection-heading">Contact Tracker</p>'
        '<div class="hdr-strip">'
        '<div class="hdr-pill">'
        '<span class="hdr-pill-label">Contacts Detected</span>'
        f'<span class="hdr-pill-value">{n_total}</span>'
        "</div>"
        '<div class="hdr-pill">'
        '<span class="hdr-pill-label">Refined</span>'
        f'<span class="hdr-pill-value">{n_refined} ({refined_pct:.0f}%)</span>'
        "</div>"
        '<div class="hdr-pill">'
        '<span class="hdr-pill-label">Strike Pattern</span>'
        f'<span class="hdr-pill-value">{strike_txt}</span>'
        "</div>"
        "</div>",
        unsafe_allow_html=True,
    )

    if not contacts or fps <= 0:
        st.markdown(
            '<p class="section-sub">No individual contacts to list for '
            "this clip.</p>",
            unsafe_allow_html=True,
        )
        return

    leg_names = {"L": "Left", "R": "Right"}
    rows = []
    for i, c in enumerate(contacts, start=1):
        ic = c.contact_frame if c.contact_frame is not None else float(c.frame)
        to = c.toe_off_frame
        ic_txt = f"{ic:.1f} ({ic / fps:.2f}s)"
        to_txt = f"{to:.1f} ({to / fps:.2f}s)" if to is not None else "N/A"
        gct_txt = f"{fmt(c.gct_ms, 0)} ms" if c.gct_ms is not None else "N/A"
        leg_txt = leg_names.get(c.leg, c.leg)
        contact_strike = (c.strike_pattern or "-").capitalize()
        rows.append(
            f"<tr><td>{i}</td><td>{leg_txt}</td><td>{ic_txt}</td>"
            f"<td>{to_txt}</td><td>{gct_txt}</td><td>{contact_strike}</td></tr>"
        )

    st.markdown(
        '<table class="compare-table"><thead><tr>'
        "<th>#</th><th>Leg</th><th>Contact (IC)</th><th>Toe-off (TO)</th>"
        "<th>GCT</th><th>Strike</th>"
        "</tr></thead><tbody>" + "".join(rows) + "</tbody></table>",
        unsafe_allow_html=True,
    )


def _render_metric_comparison_charts(m_fast, m_full, video_name):
    """One StrideoNet-vs-engine bar chart per calculated metric."""
    import plotly.graph_objects as go

    specs = [
        ("Cadence", "cadence_spm", "spm", 0),
        ("Ground Contact Time", "gct_ms", "ms", 0),
        ("Stride Length", "stride_length_m", "m", 2),
        ("Vertical Oscillation", "oscillation_cm", "cm", 1),
    ]
    st.markdown(
        '<p class="subsection-heading">StrideoNet vs. deterministic '
        "kinematics engine</p>",
        unsafe_allow_html=True,
    )
    cols = st.columns(4)
    for col, (label, key, unit, dec) in zip(cols, specs):
        fv = m_fast.get(key)
        pv = m_full.get(key)
        if fv is None or pv is None:
            continue
        fig = go.Figure(
            data=[
                go.Bar(
                    x=["StrideoNet", "Engine"],
                    y=[fv, pv],
                    marker_color=_CHART_COLORS[:2],
                    text=[f"{fv:.{dec}f}", f"{pv:.{dec}f}"],
                    textposition="outside",
                    # Without this Plotly clips an outside label at the plot
                    # edge rather than letting it sit over the margin.
                    cliponaxis=False,
                )
            ]
        )
        # Plotly's default range stops at the taller bar, which leaves the
        # "outside" value label nowhere to go: it was drawn over the bar's own
        # top or dropped (user, 2026-08-31: the vertical axis is too short to
        # show the bars fully). 22% of headroom fits the label at every value
        # these four metrics take. The gridcolor/zerolinecolor have to be
        # repeated because _dark_layout's overrides replace a whole axis dict
        # rather than merging into it.
        y_top = max(fv, pv)
        fig.update_layout(
            **_dark_layout(
                title=dict(text=f"{label} ({unit})", font=dict(size=13)),
                height=260,
                showlegend=False,
                yaxis=dict(
                    range=[0, y_top * 1.22 if y_top > 0 else 1],
                    gridcolor=_CHART_GRID,
                    zerolinecolor=_CHART_GRID,
                ),
            )
        )
        with col:
            st.plotly_chart(
                fig, use_container_width=True, key=f"dke_chart_{key}_{video_name}"
            )


def _render_pace_analysis_charts(results):
    """Cadence/GCT/stride/VO/duty factor vs. speed, one chart per
    metric, across every analyzed clip (needs 2+ clips with a valid speed to
    be worth a trend line).

    Duty factor is the fifth, added last. `app/utils/elite_reference.py`'s
    ``METRICS`` had carried five for some time, so until this landed the
    Pace Analysis section and the Elite Range Comparison below it disagreed
    about what the metric set was. It is expected to fall as speed rises, and
    three independent sources agree on roughly how fast: two published models
    give -0.005 and -0.0053 per km/h, while a linear fit over our own 45-clip
    cohort at the stride-based definition gives -0.0043 per km/h (r = -0.570)
    across 19-28 km/h. None of those is a published claim, so they stay here
    as the reason for the chart rather than as anything the page tells the
    user.
    """
    import plotly.graph_objects as go

    specs = [
        ("Cadence", "cadence_spm", "spm", 0),
        ("Ground Contact Time", "gct_ms", "ms", 0),
        ("Stride Length", "stride_length_m", "m", 2),
        ("Vertical Oscillation", "oscillation_cm", "cm", 1),
        ("Duty Factor", "duty_factor", "", 3),
    ]
    points = [
        (mc["velocity_kmh"], mc, r["video_name"])
        for r in results
        for mc in [_cross_clip_metrics(r)]
        if mc.get("velocity_kmh") is not None
    ]
    points.sort(key=lambda p: p[0])
    if len(points) < 2:
        st.markdown(
            '<p class="section-sub">Not enough clips with a valid speed to '
            "chart yet.</p>",
            unsafe_allow_html=True,
        )
        return

    cols = st.columns(len(specs))
    for col, (label, key, unit, dec) in zip(cols, specs):
        rows = [(v, m[key], name) for v, m, name in points if m.get(key) is not None]
        if len(rows) < 2:
            continue
        xs, ys, names = zip(*rows)
        fig = go.Figure(
            data=[
                go.Scatter(
                    x=xs,
                    y=ys,
                    mode="markers+lines",
                    marker=dict(color=_CHART_COLORS[0], size=9),
                    line=dict(color=_CHART_COLORS[0], width=1),
                    text=names,
                    hovertemplate=(
                        "%{text}<br>%{x:.1f} km/h<br>%{y:."
                        + str(dec)
                        + "f}<extra></extra>"
                    ),
                )
            ]
        )
        # Duty factor is the one dimensionless metric here, so the unit
        # parenthetical is dropped rather than rendered as an empty "()".
        title_text = f"{label} ({unit}) vs. speed" if unit else f"{label} vs. speed"
        fig.update_layout(
            **_dark_layout(
                title=dict(text=title_text, font=dict(size=13)),
                height=240,
                showlegend=False,
                xaxis=dict(
                    title="Speed (km/h)",
                    gridcolor=_CHART_GRID,
                    zerolinecolor=_CHART_GRID,
                ),
            )
        )
        with col:
            st.plotly_chart(fig, use_container_width=True, key=f"pace_chart_{key}")


# ─────────────────────────────────────────────────────────────
# Per-video tabs
# ─────────────────────────────────────────────────────────────

if results_with_metrics:
    # Pace + filename, not "Run 1"/"Run 2" - a bare ordinal told the user
    # nothing about which clip they were picking (user feedback).
    run_labels = [
        f"{_r.get('pace_label') or f'Clip {_i + 1}'} — {_r['video_name']}"
        for _i, _r in enumerate(results_with_metrics)
    ]
    if len(run_labels) > 1:
        selected_run = st.selectbox("Run", run_labels, label_visibility="collapsed")
    else:
        selected_run = run_labels[0]
    run_idx = run_labels.index(selected_run)
    r = results_with_metrics[run_idx]
    m = r["metrics"]
    pre = r["preprocessing"]

    # Whole block sits on one tinted panel (.vinfo-panel-mark/:has(), same
    # reusable pattern as .res-vidframe-mark below) instead of the shared
    # divider line other section headings use - the heading is full-width,
    # above the two-column split, so the pill row inside vinfo_col and the
    # video inside vid_col both start at the same y. The "always-there
    # run-title" paragraph that used to sit here (pace + filename,
    # duplicating both the picker above and the Pace pill below) is gone;
    # the filename now lives in its own pill in the strip instead (user
    # feedback).
    with st.container():
        st.markdown('<span class="vinfo-panel-mark"></span>', unsafe_allow_html=True)
        st.markdown(
            '<p class="section-heading no-rule">Video Information</p>',
            unsafe_allow_html=True,
        )

        vinfo_col, vid_col = st.columns([1.05, 1], gap="large")

        with vinfo_col:
            _render_header_strip(r, m, pre)

        with vid_col:
            # StrideoNet's own annotated video (skeleton + metrics
            # HUD, no contacts - those are an engine concept), shipped already-
            # built alongside the rest of the fast-path result (encoded
            # eagerly in upload.py, not on demand here). Its download
            # button lives in the "Save this session" panel at the bottom
            # of the page, alongside Save results/Save PDF, rather than
            # here - one export panel instead of buttons scattered across
            # the page (user feedback).
            video_bytes = r.get("annotated_video_fast")
            if video_bytes:
                with st.container():
                    st.markdown(
                        '<span class="res-vidframe-mark"></span>',
                        unsafe_allow_html=True,
                    )
                    st.video(video_bytes)
            elif r.get("video_omitted_by_export"):
                # A saved results file never carries the video, so this
                # is the expected state after "Load saved results", not a
                # failure. Name the export that does carry it, since the two
                # are separate on purpose - and say "in the
                # original session", because this page's own "Save video"
                # button is disabled here for exactly the same reason.
                st.info(
                    "This saved results file doesn't include the annotated "
                    "video. It's a separate download, saved with "
                    '"Save video" in the original session.'
                )
            else:
                st.info("The annotated video couldn't be built for this clip.")

    has_full = r.get("has_full_analysis", False)

    # Pipeline Diagnostics section removed entirely (user feedback) - "NN
    # windows used" and "Mean visibility (hip)" dropped as not useful;
    # "Torso scale" and "Running direction" moved into Video Information
    # above (see _render_header_strip); "Ground contacts" already lives in
    # the Contact Tracker section below once the engine has run.

    st.markdown(
        '<p class="section-heading">StrideoNet Calculated Metrics</p>',
        unsafe_allow_html=True,
    )

    m_fast = r.get("metrics_fast")

    # Every value (core + derived) as pills in one bordered card,
    # instead of separate floating pill-boxes plus a differently-formatted
    # derived-metrics text line.
    osc_leg = fmt(m.get("oscillation_leg_ratio", None), 3)
    # This said "the complement of ground contact time within a stride",
    # which is the same step-vs-stride slip that once made the duty factor
    # read 2x high. `calculate_flight_time` is `60000 / cadence_spm - gct`
    # and cadence is in STEPS per minute, so the interval it completes is a
    # step. Within a stride the complement of contact is swing, and
    # `swing = contact + 2 * flight` (see `calculate_duty_factor`'s docstring),
    # so the two are not interchangeable.
    flight_tip = _info_tip(
        "Time in the air between two consecutive footstrikes, the complement "
        "of ground contact time within a step. A stride is two steps, so this "
        "is not one leg's swing phase."
    )
    duty_tip = _info_tip(
        "Fraction of a full stride (two steps) that one foot spends on the "
        "ground. Lower means proportionally more time airborne."
    )
    # Two caveated claims, two caveats that must travel with them. Oscillation:
    # the engine reads it 54.5%
    # low against annotated reference on 15 of 15 annotated clips, so what is
    # displayed for the engine carries a fitted correction onto that scale,
    # applied at display only (src/utils/vo_correction.py). StrideoNet needs no
    # such note: it predicts on the annotated scale already, which is exactly
    # why the two disagreed by ~2x before this.
    # "Through one stride" was checked against the elite band's own
    # "whole-cycle"/"total stride" framing rather than assumed to match.
    # They do: the reference literature records pelvis total-stride
    # oscillation as the same quantity this pose pipeline
    # computes, and for symmetric gait the total-stride range equals the
    # per-cycle excursion `calculate_vertical_oscillation` averages. So the
    # NUMBER was never wrong here; the wording implied the hip rises and falls
    # once per stride, which is the wrong period (it is once per step, and
    # `_vo_smoothing_params` sizes its peak separation on exactly that).
    osc_tip = _info_tip(
        "Peak-to-trough rise and fall of the hip. The hip completes this "
        "cycle once per step, twice per stride. "
        + (
            "The in-depth engine reads this well below annotated reference, "
            "so the figure shown carries a correction fitted on 15 annotated "
            "clips. A calibration, not a validated measurement."
            if m.get("oscillation_display_corrected")
            else "StrideoNet predicts this on the annotated reference scale."
        )
    )
    # The caveat clause is not optional decoration: this metric may appear
    # anywhere only with its caveat in the same breath. It has never been
    # scored against a reference, and both its inputs carry caveats of their
    # own (the oscillation scale, and stride length's own bias).
    vo_ratio_tip = _info_tip(
        "Vertical oscillation as a percentage of step length (half a stride): "
        "how far you rise and fall for every unit travelled forward. Lower "
        "means less of the motion is vertical. Descriptive only, it has not "
        "been checked against a reference measurement."
    )
    pace_compare_html = _pace_compare_pills_html(r, m)
    # Row order set by the user 2026-08-31: the six directly-measured gait
    # metrics first, then the six speed/pace ones. "Pace" is now "Calculated
    # Pace", so that it and "Measured pace" beside it say which is which
    # rather than leaving the reader to guess why two paces disagree.
    # pace_compare_html supplies Measured pace, and Target / Vs. target when
    # this clip has a PB-derived target; it is spliced in before the Vertical
    # Oscillation Ratio so the second row reads speed, then pace, then ratio.
    st.markdown(
        '<div class="metrics-card">'
        '<div class="hdr-strip">'
        '<div class="hdr-pill">'
        '<span class="hdr-pill-label">Cadence</span>'
        f'<span class="hdr-pill-value">{fmt(m["cadence_spm"], 0)} spm</span>'
        f'<span class="hdr-pill-label">± {fmt(m["cadence_std"], 1)} spm</span>'
        "</div>"
        '<div class="hdr-pill">'
        '<span class="hdr-pill-label">Stride Length</span>'
        f'<span class="hdr-pill-value">{fmt(m["stride_length_m"], 2)} m</span>'
        f'<span class="hdr-pill-label">± {fmt(m["stride_length_std"], 2)} m</span>'
        "</div>"
        '<div class="hdr-pill">'
        f'<span class="hdr-pill-label">Vertical Oscillation{osc_tip}</span>'
        f'<span class="hdr-pill-value">{fmt(m["oscillation_cm"], 1)} cm</span>'
        f'<span class="hdr-pill-label">{osc_leg} × leg</span>'
        "</div>"
        '<div class="hdr-pill">'
        '<span class="hdr-pill-label">Ground Contact Time</span>'
        f'<span class="hdr-pill-value">{fmt(m["gct_ms"], 0)} ms</span>'
        f'<span class="hdr-pill-label">± {fmt(m["gct_std"], 1)} ms</span>'
        "</div>"
        '<div class="hdr-pill">'
        f'<span class="hdr-pill-label">Flight Time{flight_tip}</span>'
        f'<span class="hdr-pill-value">{fmt(m["flight_time_ms"], 0)} ms</span>'
        "</div>"
        '<div class="hdr-pill">'
        f'<span class="hdr-pill-label">Duty Factor{duty_tip}</span>'
        f'<span class="hdr-pill-value">{fmt(m["duty_factor"], 2)}</span>'
        "</div>"
        "</div>"
        '<div class="hdr-strip">'
        '<div class="hdr-pill">'
        '<span class="hdr-pill-label">Speed</span>'
        f'<span class="hdr-pill-value">{fmt(m["velocity_kmh"], 1)} km/h</span>'
        f'<span class="hdr-pill-label">{fmt(m["velocity_ms"], 2)} m/s</span>'
        "</div>"
        '<div class="hdr-pill">'
        f'<span class="hdr-pill-label">Calculated Pace{calculated_pace_tip}</span>'
        f'<span class="hdr-pill-value">{m.get("pace_per_km", "N/A")}/km</span>'
        "</div>"
        f"{pace_compare_html}"
        '<div class="hdr-pill">'
        f'<span class="hdr-pill-label">Vertical Oscillation Ratio{vo_ratio_tip}</span>'
        f'<span class="hdr-pill-value">{fmt(m["vertical_oscillation_ratio"], 1)}%</span>'
        "</div>"
        "</div>"
        "</div>",
        unsafe_allow_html=True,
    )

    # "Result: StrideoNet" label removed entirely (user feedback) - the
    # section heading above already says so; kept just the substance below.
    with st.expander("How accurate is this?"):
        st.markdown(
            "This is StrideoNet's result: a trained machine learning "
            "model, specifically a FiLM pace-conditioned 2D CNN, that "
            "reads your video's BlazePose landmarks directly, without "
            "running a second pose model, classifying strike pattern, or "
            "detecting individual footstrikes. Its ground contact time "
            "meets Strideo's own accuracy target: 7.57 ms mean absolute "
            "error against a ≤10 ms bar, measured leave-one-runner-out, so "
            "no figure comes from a runner the model had already seen. The "
            "deterministic kinematics engine's own figure is lower "
            "(3.99±3.60 ms), though not earned the same way: its constants "
            "were fitted on the same clips it is scored on. Cadence and "
            "vertical oscillation come from the same network."
            + (
                ""
                if has_full
                else " Want a second, independent check? Run it below: "
                "it adds its own numbers alongside this result, "
                "source-labeled, so you can see how closely the two agree."
            )
        )
    if has_full and m_fast is not None:
        m_full = m  # resolved to metrics_full when has_full is True (line ~369)
        _deltas = [
            ("Cadence", "cadence_spm", 0, "spm"),
            ("GCT", "gct_ms", 0, "ms"),
            ("Stride", "stride_length_m", 2, "m"),
            ("Oscillation", "oscillation_cm", 1, "cm"),
        ]
        _bits = []
        for _label, _key, _dec, _unit in _deltas:
            _fv = m_fast.get(_key)
            _pv = m_full.get(_key)
            if _fv is not None and _pv is not None:
                _bits.append(f"{_label} Δ {fmt(abs(_pv - _fv), _dec)} {_unit}")
        st.markdown(
            '<p class="up-note muted">Agreement with the second check: '
            + " · ".join(_bits)
            + " (a plain difference, not a verdict on which is right)</p>",
            unsafe_allow_html=True,
        )

    # ── Deterministic kinematics engine (replaces the old "Second
    # Check" framing, and carries a Contact Tracker table of per-
    # footstrike IC/TO timing + strike pattern, replacing the interactive
    # frame scrubber the user said had no value as it was) ──
    #
    # Wrapped in its own fragment. Running the engine is a multi-
    # minute, CPU-bound RTMPose pass, and without a fragment boundary the
    # button click below reruns the WHOLE page script. Streamlit then
    # streams the page top-to-bottom as that rerun executes; everything past
    # this container (Elite Range Comparison, Pace Analysis, Cross-Run
    # Comparison, the export panel) hasn't been reached yet by the new run,
    # so the frontend keeps showing last run's version of it - greyed out
    # and disabled - for the whole multi-minute wait (user-reported: the
    # second half of the results page goes grey when this button is
    # pushed). Scoping the rerun to just this fragment leaves the rest of
    # the page untouched and interactive while the engine runs;
    # ``st.rerun(scope="app")`` below (not the bare, fragment-scoped
    # default) is what then refreshes those outside sections with the new
    # metrics_full once the run actually finishes.
    @st.fragment
    def _render_dke_section(r, m, m_fast):
        with st.container():
            st.markdown('<span class="res-band-mark"></span>', unsafe_allow_html=True)
            analysis = r.get("analysis")
            st.markdown(
                '<p class="section-heading">Deterministic kinematics engine</p>',
                unsafe_allow_html=True,
            )
            if analysis is not None:
                st.markdown(
                    '<p class="section-sub">A second, independent computer-vision '
                    "method - RTMPose-x pose estimation plus per-contact ground-"
                    "contact detection - run separately from StrideoNet's neural "
                    "network. Its own numbers are shown alongside StrideoNet's "
                    "throughout Calculated Metrics above, source-labeled. Below: "
                    "contact-by-contact timing and strike-pattern detection for "
                    "every footstrike it found.</p>",
                    unsafe_allow_html=True,
                )
                _render_contact_tracker(analysis, r.get("strike_pattern"))
                if m_fast is not None:
                    _render_metric_comparison_charts(m_fast, m, r["video_name"])
            else:
                if r.get("full_analysis_excluded"):
                    st.warning(
                        "The deterministic kinematics engine could not produce a "
                        "trustworthy result for this clip: the runner's hips and "
                        "ankles were lost for a stretch of frames while they were "
                        "still crossing the shot, so part of the run was never "
                        "measured. No second-check numbers are shown for this "
                        "clip. Re-filming with the runner fully in frame and "
                        "unobstructed throughout usually fixes it."
                    )
                st.markdown(
                    '<p class="section-sub">Run the deterministic kinematics '
                    "engine, a second, independent computer-vision method "
                    "(RTMPose-x + per-contact detection), to see how closely it "
                    "agrees with StrideoNet's result above, and to get contact-by-"
                    "contact timing and strike-pattern detection for every "
                    "footstrike.</p>",
                    unsafe_allow_html=True,
                )
                # A bare st.button() with no column around it shrink-wraps to its
                # own content instead of the page width, so the shared button CSS's
                # `width: 100%` had nothing real to fill - text sat flush against
                # the edges with zero horizontal padding (user feedback: "too
                # narrow"). Wrapping in a real-width column, same pattern the
                # hero/hero-CTA buttons already use, gives it real room.
                _dke_btn_col, _ = st.columns([1, 2])
                with _dke_btn_col:
                    _dke_clicked = st.button(
                        "Run in-depth analysis",
                        key=f"full_analysis_{r['video_name']}",
                        use_container_width=True,
                    )
                if _dke_clicked:
                    from pipeline_runner import run_full_analysis

                    full_progress = st.progress(0, text="Preparing video…")

                    # Same per-frame reporting as the upload page's bar, so only
                    # rerender when the rendered text would change (see the note
                    # on upload.py's _fast_progress).
                    _full_last = {"pct": -1, "stage": ""}

                    def _full_progress(stage, frac, _p=full_progress, _last=_full_last):
                        pct = min(int(frac * 100), 100)
                        if pct == _last["pct"] and stage == _last["stage"]:
                            return
                        _last["pct"], _last["stage"] = pct, stage
                        _p.progress(pct, text=stage)

                    runner_info = r.get("runner_info") or {}
                    try:
                        full_analysis = run_full_analysis(
                            r["video_path"],
                            runner_height_cm=float(
                                runner_info.get("height_cm") or 175.0
                            ),
                            shoe_type=r.get("shoe_type"),
                            shoe_sole_cm=r.get("shoe_sole_cm", 2.5),
                            progress=_full_progress,
                        )
                        if full_analysis.reliability.excluded:
                            # The runner's presence window is
                            # undefined, or tracking was lost mid-transit for longer
                            # than gap-fill repairs -- numbers here would not be
                            # trustworthy, so the clip's output is withheld rather
                            # than displayed. metrics_full/analysis_full stay unset,
                            # so has_full_analysis stays False. Fragment-scoped
                            # rerun (bare st.rerun()) is enough: the warning this
                            # sets up is rendered by this same fragment, and
                            # nothing outside it depends on this flag.
                            r["full_analysis_excluded"] = True
                            full_progress.progress(100, text="Tracking lost mid-run")
                            st.rerun()
                        else:
                            from src.preprocessing.nn_preprocessing import (
                                detect_running_direction,
                            )

                            # The engine measures oscillation 54.5% low against
                            # annotated reference on 15 of 15 annotated clips,
                            # and the fit that corrects it had only ever been
                            # applied to StrideoNet's training labels -- so the
                            # two estimators were showing this metric on scales
                            # ~2x apart. The correction is applied HERE, at
                            # display, and never in the pipeline: a corrected
                            # stored value would have the validation report
                            # scoring the fit against the clips it was fitted
                            # on. See src/utils/vo_correction.py.
                            r["metrics_full"] = correct_engine_metrics_for_display(
                                asdict(full_analysis.metrics)
                            )
                            r["preprocessing_full"] = _preprocessing_from_full(
                                full_analysis,
                                r.get("preprocessing_fast"),
                                detect_running_direction(
                                    full_analysis.selected_landmarks
                                ),
                            )
                            r["strike_pattern_full"] = full_analysis.clip_strike_pattern
                            r["analysis_full"] = full_analysis
                            full_progress.progress(100, text="Complete!")
                            # App-scoped, not the fragment default: Calculated
                            # Metrics, the Agreement note, Elite Range
                            # Comparison, Pace Analysis and Cross-Run
                            # Comparison all live outside this fragment and
                            # only pick up metrics_full on a full rerun.
                            st.rerun(scope="app")
                    except (
                        Exception
                    ) as e:  # noqa: BLE001 - surface any pipeline failure
                        full_progress.progress(100, text="Failed")
                        st.error(f"Second check failed: {e}")

    _render_dke_section(r, m, m_fast)

    # ── Elite Range Comparison (pace-tier + sex aware) ──
    _er_runner_info = r.get("runner_info") or {}
    # Bands are built at the clip's own measured speed where one is
    # available, not at the declared tier's nominal speed. The tier is a
    # label the runner picked before running; every metric in the band
    # varies with the speed they actually ran, and our own cohort's pace
    # error is entirely one-directional (0 of 45 clips undershot).
    _er_ranges = get_elite_ranges(
        r.get("pace_level"),
        _er_runner_info.get("sex"),
        _safe_float(m.get("velocity_kmh"), None),
        _safe_float(_er_runner_info.get("height_cm"), None),
    )
    if _er_ranges is None:
        st.markdown(
            '<p class="section-heading">Elite Range Comparison</p>',
            unsafe_allow_html=True,
        )
        st.markdown(
            '<p class="section-sub">Set sex (Male/Female) and a declared pace '
            "on Upload to see how this clip compares against elite reference "
            "ranges for that pace.</p>",
            unsafe_allow_html=True,
        )
    else:
        _er_tier_label = TIER_LABELS.get(r.get("pace_level"), r.get("pace_level"))
        _er_sex_label = (_er_runner_info.get("sex") or "").lower()
        st.markdown(
            '<p class="section-heading">Elite Range Comparison</p>',
            unsafe_allow_html=True,
        )
        # Say which speed the band was built at. The tier is what the
        # runner declared before running; the band is only meaningful at the
        # speed they actually ran, and the two routinely differ.
        _er_by_speed = any(rng.basis == "speed" for rng in _er_ranges.values())
        if _er_by_speed:
            _er_basis_copy = (
                f"built for elite {_er_sex_label} middle-distance runners at "
                f"the {_safe_float(m.get('velocity_kmh')):.1f} km/h you actually "
                f"ran, not at a nominal {_er_tier_label} pace"
            )
        else:
            _er_basis_copy = (
                f"built for elite {_er_sex_label} middle-distance runners across "
                f"the whole {_er_tier_label} pace range, because this clip's speed "
                "falls outside the range the reference model covers. Running "
                "faster or slower than your target moves where you sit against "
                "these bands, so read them loosely"
            )
        st.markdown(
            f'<p class="section-sub">Your metrics against indicative reference '
            f"ranges {_er_basis_copy}. The bands span roughly 95% of elite "
            "runners, so they are wide on purpose - orientation, not a "
            "pass/fail.</p>",
            unsafe_allow_html=True,
        )

        modeled_tip = _info_tip(
            "Modeled, not directly measured - no whole-cycle vertical "
            "oscillation measurement exists above 22 km/h in the published "
            "literature this reference is built from."
        )
        bars = []
        for key, name in METRICS:
            rng = _er_ranges[key]
            val = _safe_float(m.get(key))
            bars.append((name, rng.unit, val, (rng.lo, rng.hi), rng.modeled))

        rows_html = []
        for name, unit, val, (lo, hi), modeled in bars:
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
            val_decimals = 2 if not unit else 1
            scale_fmt = (
                "{:.2f}" if disp_hi < 1 else "{:.1f}" if disp_hi < 20 else "{:.0f}"
            )
            name_tip = modeled_tip if modeled else ""
            unit_suffix = f" {unit}" if unit else ""
            rows_html.append(
                '<div class="range-row">'
                f'<div class="range-name">{name}{name_tip}<br>'
                f'<span class="rn-sub">elite {lo:g}-{hi:g}{unit_suffix}</span></div>'
                '<div><div class="range-track">'
                f'<div class="range-band" style="left:{band_left:.1f}%;'
                f'width:{band_w:.1f}%"></div>'
                f'<div class="range-marker" style="left:{mark:.1f}%"></div></div>'
                f'<div class="range-scale"><span>{scale_fmt.format(disp_lo)}</span>'
                f"<span>{scale_fmt.format(disp_hi)}</span></div></div>"
                f'<div class="range-verdict"><strong>{val:.{val_decimals}f}{unit_suffix}</strong><br>'
                f"{verdict}</div></div>"
            )
        st.markdown("".join(rows_html), unsafe_allow_html=True)

        st.markdown(
            '<p class="range-sources">Reference ranges are indicative, drawn from '
            "championship middle-distance biomechanics: Hanley, Merlino &amp; "
            "Bissas (2022), <em>Biomechanics of World-Class 800&nbsp;m Women</em>, "
            '<a href="https://doi.org/10.3389/fspor.2022.834813" '
            'target="_blank">Front. Sports Act. Living</a>; Hanley et&nbsp;al. '
            "(2023), <em>Men's 1500&nbsp;m</em>, "
            '<a href="https://doi.org/10.1111/sms.14331" '
            'target="_blank">Scand. J. Med. Sci. Sports</a>; Burns et&nbsp;al. '
            "(2021), <em>Bouncing Behavior of Sub-Four Minute Milers</em>, "
            '<a href="https://doi.org/10.1038/s41598-021-89858-1" '
            'target="_blank">Sci. Rep.</a>; Preece, Bramah &amp; Mason (2019), '
            '<a href="https://doi.org/10.1080/17461391.2018.1554707" '
            'target="_blank">Eur. J. Sport Sci.</a>; Folland et&nbsp;al. (2017), '
            '<a href="https://doi.org/10.1249/MSS.0000000000001245" '
            'target="_blank">Med. Sci. Sports Exerc.</a></p>',
            unsafe_allow_html=True,
        )

# ─────────────────────────────────────────────────────────────
# Pace Analysis: metric vs. speed across runs (2+ clips)
# ─────────────────────────────────────────────────────────────

if len(results_with_metrics) >= 2:
    with st.container():
        st.markdown('<span class="res-band-mark"></span>', unsafe_allow_html=True)
        st.markdown(
            '<p class="section-heading">Pace Analysis</p>',
            unsafe_allow_html=True,
        )
        st.markdown(
            '<p class="section-sub">How each metric moves with speed, across '
            "every clip analyzed so far.</p>",
            unsafe_allow_html=True,
        )
        _render_pace_analysis_charts(results_with_metrics)

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
        m = _cross_clip_metrics(r)
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
            "mechanical details that coaches measure intuitively. By default, results come "
            "from StrideoNet - a trained machine learning model, specifically a FiLM "
            "pace-conditioned 2D CNN - that reads pose data directly, without a second pose "
            "model, strike-pattern classification, or per-contact detection. Where a clip "
            "was also run through the deterministic kinematics engine as a second check, "
            "this report notes it below, alongside how closely the two agree.",
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
        direction = pre.get("direction")
        dir_txt = {"LR": "Right", "RL": "Left"}.get(direction, "N/A")

        vid_pills = [
            ("Pace", r.get("pace_label", "-")),
            ("Footwear", r.get("shoe_label", "-")),
            ("File", r["video_name"]),
            ("Date", date_str),
            ("Resolution", res_str),
            ("Strike", strike),
            ("Calibration", cal_str),
            ("Torso Scale", f"{pre.get('torso_scale', 0):.4f}"),
            ("Direction", dir_txt),
        ]
        elements.append(_pill_table(vid_pills, per_row=3))

        # ── Calculated Metrics ────────────────────────────────
        source_label = (
            "Second check (deterministic kinematics engine)"
            if r.get("has_full_analysis")
            else "Result (StrideoNet)"
        )
        elements.append(Paragraph(f"Calculated Metrics: {source_label}", sub_head))
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
                f"{osc_leg}x leg",
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
                "Vertical Oscillation Ratio",
                f"{fmt(m['vertical_oscillation_ratio'], 1)}%",
                "",
                "of step length; lower = less vertical motion",
            ],
        ]
        cw = [CONTENT_W * f for f in (0.36, 0.24, 0.22, 0.18)]
        elements.append(_data_table(metrics_rows, cw))

        m_fast_pdf = r.get("metrics_fast")
        if r.get("has_full_analysis") and m_fast_pdf is not None:
            _pdf_deltas = [
                ("Cadence", "cadence_spm", 0, "spm"),
                ("GCT", "gct_ms", 0, "ms"),
                ("Stride", "stride_length_m", 2, "m"),
                ("Oscillation", "oscillation_cm", 1, "cm"),
            ]
            _pdf_bits = []
            for _label, _key, _dec, _unit in _pdf_deltas:
                _fv = m_fast_pdf.get(_key)
                _pv = m.get(_key)
                if _fv is not None and _pv is not None:
                    _pdf_bits.append(f"{_label} {fmt(abs(_pv - _fv), _dec)} {_unit}")
            if _pdf_bits:
                elements.append(
                    Paragraph(
                        "Agreement with StrideoNet's result - "
                        + " / ".join(_pdf_bits)
                        + " (plain differences, not a verdict on which is right)",
                        note,
                    )
                )

        # Pipeline Diagnostics section removed entirely, matching the live
        # page (user feedback) - Torso Scale/Direction folded into Video
        # Information above instead; Mean Visibility/NN Windows Used dropped.

        # ── Elite Range Comparison (pace-tier + sex aware) ──
        _pdf_runner_info = r.get("runner_info") or {}
        _pdf_ranges = get_elite_ranges(
            r.get("pace_level"),
            _pdf_runner_info.get("sex"),
            _safe_float(m.get("velocity_kmh"), None),
            _safe_float(_pdf_runner_info.get("height_cm"), None),
        )
        if _pdf_ranges is None:
            elements.append(Paragraph("Elite Range Comparison", sub_head))
            elements.append(
                Paragraph(
                    "Not available - requires sex (Male/Female) and a "
                    "declared pace on Upload.",
                    note,
                )
            )
        else:
            _pdf_tier_label = TIER_LABELS.get(r.get("pace_level"), r.get("pace_level"))
            _pdf_sex_label = (_pdf_runner_info.get("sex") or "").lower()
            elements.append(Paragraph("Elite Range Comparison", sub_head))
            # Same basis note as the live page, so a saved PDF cannot
            # claim a tier band was a measured-speed one or the reverse.
            if any(rng.basis == "speed" for rng in _pdf_ranges.values()):
                _pdf_basis_copy = (
                    f"at the {_safe_float(m.get('velocity_kmh')):.1f} km/h "
                    "actually run, not at a nominal "
                    f"{_pdf_tier_label} pace"
                )
            else:
                _pdf_basis_copy = (
                    f"across the whole {_pdf_tier_label} pace range, this clip's "
                    "speed falling outside the reference model. Running faster or "
                    "slower than the target moves where you sit"
                )
            elements.append(
                Paragraph(
                    f"Indicative reference ranges for elite {_pdf_sex_label} "
                    f"middle-distance runners {_pdf_basis_copy}. The bands span "
                    "roughly 95% of elite runners - orientation, not a pass/fail.",
                    note,
                )
            )
            range_rows = [["Metric", "Your Value", "Elite Range", "Verdict"]]
            for key, name in METRICS:
                rng = _pdf_ranges[key]
                unit = rng.unit
                lo, hi = rng.lo, rng.hi
                val = _safe_float(m.get(key))
                val_decimals = 2 if not unit else 1
                unit_suffix = f" {unit}" if unit else ""
                val_str = (
                    f"{val:.{val_decimals}f}{unit_suffix}" if val is not None else "-"
                )
                range_str = f"{lo:g}-{hi:g}{unit_suffix}"
                if val is None:
                    verdict = "-"
                elif val < lo:
                    verdict = "below range"
                elif val > hi:
                    verdict = "above range"
                else:
                    verdict = "in range"
                if rng.modeled:
                    verdict += " (modeled)"
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
# Save/re-import results, so a session survives a
# deliberate save even though (unlike a token-link) it does not
# auto-survive an unplanned refresh. Zero server-side retention: the
# export is a plain JSON download, re-imported client-side above.
# ─────────────────────────────────────────────────────────────

#: Keys dropped from a "Save results" export.
#:
#: The last six are the *derived aliases* this page writes onto each result
#: entry at render time ("best available" resolution, ~line 500). They are
#: recomputed from `metrics_fast`/`metrics_full` on every run, including
#: straight after a re-import, so exporting them would only duplicate state
#: that is rebuilt anyway.
#:
#: `analysis` is the one that actually bit (2026-08-31): it aliases
#: `analysis_full`, a `ClipAnalysis`, which is not JSON-serializable. With no
#: in-depth run it is None and serializes cleanly, which is the state the
#: feature was verified in -- so "Save results" worked until someone ran the
#: engine,
#: then `json.dumps` raised and took the whole export panel down with it.
#: Excluding the derived set rather than just that one key is deliberate: the
#: next alias added at line 500 would otherwise reintroduce the same bug.
_EXPORT_EXCLUDE_KEYS = {
    "analysis_full",
    "annotated_video_fast",
    "analysis",
    "metrics",
    "preprocessing",
    "strike_pattern",
    "has_full_analysis",
    "video_omitted_by_export",
}
_EXPORT_VERSION = 1


def _build_export_payload(results_data):
    """JSON-safe snapshot of ``analysis_results`` for the "Save results"
    download. Every field on a result entry except the two excluded here
    is already a plain str/float/bool/dict/None (see ``upload.py``'s
    ``result_entry`` and ``results.py``'s "Run Full Analysis" block) -
    ``ClipAnalysis`` (``analysis_full``) isn't JSON-serializable, and the
    annotated-video bytes (``annotated_video_fast``) already have their own
    per-clip download button, so re-exporting them here would only
    bloat the file.
    """
    clean_results = [
        {k: v for k, v in r.items() if k not in _EXPORT_EXCLUDE_KEYS}
        for r in results_data
    ]
    return {
        "strideo_export_version": _EXPORT_VERSION,
        "exported_at": datetime.now().isoformat(timespec="seconds"),
        "results": clean_results,
    }


# ─────────────────────────────────────────────────────────────
# Save panel - same navy CTA-panel language as Home's pre-footer band
# (.cta-mark, shared CSS), so the three export actions read as one
# grouped block instead of buttons scattered across the page. "Save
# video" downloads the currently-selected run's annotated video (the
# per-clip button that used to sit in the video column) -
# "Save results" and "Save PDF" cover every analyzed clip.
# ─────────────────────────────────────────────────────────────

if results_with_metrics:
    (save_col,) = st.columns(1)
    with save_col:
        st.markdown(
            """
<div class="cta-mark">
  <p class="eyebrow">export</p>
  <h2 class="cta-h2">Save this session</h2>
</div>
""",
            unsafe_allow_html=True,
        )
        _, sc1, sc2, sc3, _ = st.columns([0.9, 1, 1, 1, 0.9])
        with sc1:
            _video_bytes = r.get("annotated_video_fast")
            if _video_bytes:
                st.download_button(
                    label="Save video",
                    data=_video_bytes,
                    file_name=f"vid_{run_idx}_strideonet_annotated.mp4",
                    mime="video/mp4",
                    key=f"vid_{run_idx}_dlvid_bottom",
                    use_container_width=True,
                )
            else:
                st.button(
                    "Save video",
                    disabled=True,
                    use_container_width=True,
                    key="save_video_disabled",
                )
        with sc2:
            _export_payload = _build_export_payload(results_with_metrics)
            st.download_button(
                label="Save results",
                data=json.dumps(_export_payload, indent=2),
                file_name=(
                    f"strideo_results_{datetime.now().strftime('%Y%m%d_%H%M%S')}.json"
                ),
                mime="application/json",
                use_container_width=True,
            )
        with sc3:
            try:
                # Cached rather than rebuilt on every script rerun (every
                # widget interaction on this page reruns the whole script,
                # and ReportLab generation over every clip is not cheap) -
                # user-reported symptom: the whole export panel visibly
                # flickering/vanishing on interaction, most likely this
                # unconditional PDF rebuild slowing every rerun down.
                # Invalidated only when the actual result set changes.
                _pdf_cache_key = tuple(
                    (r["video_name"], r.get("pace_level"), r.get("has_full_analysis"))
                    for r in results_with_metrics
                )
                if st.session_state.get("_pdf_cache_key") != _pdf_cache_key:
                    st.session_state["_pdf_bytes"] = _generate_pdf(results_with_metrics)
                    st.session_state["_pdf_cache_key"] = _pdf_cache_key
                pdf_bytes = st.session_state["_pdf_bytes"]
                st.download_button(
                    label="Save PDF",
                    data=pdf_bytes,
                    file_name="strideo_report.pdf",
                    mime="application/pdf",
                    use_container_width=True,
                )
            except ImportError:
                st.button(
                    "PDF export requires reportlab",
                    disabled=True,
                    use_container_width=True,
                    key="pdf_disabled",
                )

# ─────────────────────────────────────────────────────────────
# Start a new analysis - a single action instead of two ("Upload more
# videos" / "Clear results"), since the second was really the first
# result of the first anyway (returning to Upload clears nothing on its
# own, but leaving without saving loses these results all the same).
# Placed after the export panel so the warning below reads in context.
# ─────────────────────────────────────────────────────────────

st.markdown("<br>", unsafe_allow_html=True)
# The warning sits outside the column, at full page width: inside the [1, 2.4]
# column that holds the button it had ~370px to work with and broke across
# three lines (user, 2026-08-31: do not wrap it). The button keeps the narrow
# column, since it is sized to itself rather than to the sentence above it.
st.markdown(
    '<p class="up-note muted" style="margin-bottom:8px !important;">'
    "Starting a new analysis clears these results - save video, "
    "results or a PDF above first if you want to keep them.</p>",
    unsafe_allow_html=True,
)
_new_col, _ = st.columns([1, 2.4])
with _new_col:
    if st.button(
        "Start a new analysis",
        type="primary",
        use_container_width=True,
        key="start_new_analysis",
    ):
        if "analysis_results" in st.session_state:
            del st.session_state["analysis_results"]
        st.session_state["scroll_upload_top"] = True
        st.switch_page("pages/upload.py")

render_footer()
