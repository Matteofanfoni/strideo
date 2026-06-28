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
)

st.set_page_config(
    page_title="Recording Guide | strideo.it", page_icon="🏃", layout="wide"
)

inject_shared_styles()
render_navbar("Recording Guide")

# Article styling, mirrored from the home page: a clean, well-typeset markdown
# document in a centred reading column. No cards or icons.
inject_page_css("""
[data-testid="stMarkdownContainer"] h1 {
  font-size: 2.6rem; font-weight: 800; letter-spacing: -1px;
  color: var(--text); margin: 8px 0 2px;
}
[data-testid="stMarkdownContainer"] h2 {
  font-size: 1.5rem; font-weight: 700; color: var(--text);
  margin: 38px 0 10px;
}
[data-testid="stMarkdownContainer"] p,
[data-testid="stMarkdownContainer"] li {
  font-size: 1.06rem; line-height: 1.8; color: var(--text-muted);
}
[data-testid="stMarkdownContainer"] li { margin-bottom: 6px; }
[data-testid="stMarkdownContainer"] strong { color: var(--text); font-weight: 700; }
[data-testid="stMarkdownContainer"] a { color: var(--brand); font-weight: 600; text-decoration: none; }
[data-testid="stMarkdownContainer"] a:hover { text-decoration: underline; }
[data-testid="stMarkdownContainer"] blockquote {
  border-left: 3px solid var(--brand);
  background: rgba(13,148,136,0.06);
  margin: 18px 0; padding: 12px 18px; border-radius: 0 8px 8px 0;
}
[data-testid="stMarkdownContainer"] blockquote p {
  margin: 0; font-size: 0.98rem; color: var(--text-muted);
}
/* Camera-position diagram: full-width box, art centred inside (unchanged) */
.diagram {
  display: block;
  text-align: center;
  margin: 14px 0;
  background: var(--hover-fill);
  border: 1px solid var(--border);
  border-radius: var(--radius-sm);
  padding: 16px 18px;
  overflow-x: auto;
}
.diagram-art {
  display: inline-block;
  text-align: left;
  white-space: pre;
  font-family: 'Courier New', monospace;
  font-size: 0.82rem;
  line-height: 1.6;
  color: var(--text-muted);
}
[data-testid="stMarkdownContainer"] hr {
  border: none; border-top: 1px solid var(--border); margin: 30px 0;
}
.lead { font-size: 1.2rem !important; color: var(--text) !important; font-weight: 500; }
""")

_, mid, _ = st.columns([1, 3, 1])

with mid:
    st.markdown(
        """
# Recording Guide

## Camera settings

The capture settings used for the validation dataset. Follow them
for the most accurate results; some will be relaxed once the neural network is
trained.

- **Resolution & frame rate:** **4K @ 60 FPS**.
- **Camera app:** the standard phone camera app. No third-party app is needed; just film in good daylight so the foot stays sharp at
  ground contact.
- **Lens:** the main **1× lens**. Avoid ultra-wide, which adds barrel distortion.
- **Orientation:** **landscape**, holding the phone horizontally.
- **Stabilisation:** **OFF**, so the camera does not re-crop mid-clip. Keep stablisation On if you're not using a tripod.
- **HDR:** **OFF**, for consistent exposure across frames.
- **Grid overlay:** **ON**, to help align the camera with the horizon.

## Camera position

Distances and tripod height come from the same setup used to capture the validation dataset.

- **Mounting:** a fixed tripod, with zero camera movement.
- **Angle:** side view (the sagittal plane).
- **Distance:** **15 m** from the runner, perpendicular to the track.
- **Height:** **95-100 cm**, roughly the runner's hip height.
- **Field of view:** about **21 m** of visible width, 19.7 m of it usable.
- **Runner size:** about **335 px** tall, comfortably above BlazePose's 300 px floor.

<div class="diagram"><div class="diagram-art">═════════════  TRACK  (Lanes 3-5)  ══════════════
│<───────────  19.7 m usable zone  ───────────>│
&nbsp;
        <───  Runner, either direction  ───>
&nbsp;
                       │
                    15.0 m
                       │
                       ▼
                  [ TRIPOD ]
                  hip-height
                  (95-100 cm)</div></div>

## Background & lighting

A plain background is the single biggest fix for **ghost-lock**, where the
pose tracker latches onto background texture instead of the runner for the first
60-110 frames of the clip.

- **Background:** plain, matte, ideally single colour.
- **Shooting direction:** avoid bleachers, fences, or spectators.
- **Lighting:** good ambient light, with no backlighting.
- **Clothing:** tight-fitting.
- **Timing:** start recording just before the runner enters frame, and keep the camera still until they've fully crossed.

## What to upload

- **Framing:** the full body visible throughout, with no other runners crossing.
- **Duration:** film the whole time the runner is in frame. That is about **2 s** at fast paces and up to **5 s** at slower ones; longer clips are trimmed to the first **5 s**.
- **Clips:** three paces, at **threshold**, **1500m**, and **800m** effort.
- **Format:** **MP4** or **MOV** (what a phone records). M4V, AVI, and WebM are also accepted.
""",
        unsafe_allow_html=True,
    )
