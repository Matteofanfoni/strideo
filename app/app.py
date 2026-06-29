# app/app.py
import streamlit as st
import sys
from pathlib import Path

# Add app/ directory to sys.path so `ui.styles` is importable
app_dir = Path(__file__).parent
if str(app_dir) not in sys.path:
    sys.path.insert(0, str(app_dir))

import preload  # noqa: F401
from ui.styles import inject_shared_styles, inject_page_css, render_navbar  # noqa: E402

st.set_page_config(page_title="strideo.it", page_icon="🏃", layout="wide")

inject_shared_styles()
render_navbar("Home")

# Article / long-form prose styling (applies only to this page). No cards or
# icons - a clean, well-typeset document in a readable centre column.
inject_page_css("""
[data-testid="stMarkdownContainer"] h1 {
  font-size: 2.6rem; font-weight: 800; letter-spacing: -1px;
  color: var(--text); margin: 8px 0 2px;
}
[data-testid="stMarkdownContainer"] h2 {
  font-size: 1.5rem; font-weight: 700; color: var(--text);
  margin: 36px 0 10px;
}
[data-testid="stMarkdownContainer"] h3 {
  font-size: 1.1rem; font-weight: 700; color: var(--text); margin: 22px 0 6px;
}
[data-testid="stMarkdownContainer"] p,
[data-testid="stMarkdownContainer"] li {
  font-size: 1.06rem; line-height: 1.8; color: var(--text-muted);
}
[data-testid="stMarkdownContainer"] strong { color: var(--text); font-weight: 700; }
[data-testid="stMarkdownContainer"] a {
  color: var(--brand); font-weight: 600; text-decoration: none;
}
[data-testid="stMarkdownContainer"] a:hover { text-decoration: underline; }
[data-testid="stMarkdownContainer"] code {
  background: rgba(13,148,136,0.10); color: var(--brand);
  padding: 1px 6px; border-radius: 6px; font-size: 0.88em;
}
[data-testid="stMarkdownContainer"] blockquote {
  border-left: 3px solid var(--brand);
  background: rgba(13,148,136,0.06);
  margin: 18px 0; padding: 12px 18px; border-radius: 0 8px 8px 0;
}
[data-testid="stMarkdownContainer"] blockquote p {
  margin: 0; font-size: 0.98rem; color: var(--text-muted);
}
[data-testid="stMarkdownContainer"] hr {
  border: none; border-top: 1px solid var(--border); margin: 30px 0;
}
.lead { font-size: 1.2rem !important; color: var(--text) !important; font-weight: 500; }
.byline { color: var(--text-subtle); font-size: 0.95rem; margin-top: 0; }
/* About-me follows the CTA widgets, which add an element gap on top of the h2
   margin; trim the heading's top margin so its spacing matches the prose
   sections above. */
.about-me h2 { margin-top: 16px; }
""")

# Public source repository (shown as a link on the home page). The public
# mirror goes live with the Day-6 Hugging Face deploy - update if the URL differs.
REPO_URL = "https://github.com/Matteofanfoni/strideo"


# ─────────────────────────────────────────────────────────────
# Contact dialog (server-side POST to FormSubmit; the endpoint lives in
# .streamlit/secrets.toml, gitignored, and never reaches the browser).
# ─────────────────────────────────────────────────────────────

try:
    _CONTACT_ENDPOINT = st.secrets.get("contact_endpoint", "")
    # FormSubmit refuses server-side posts that lack a web Referer/Origin
    # ("open this page through a web server"); send a plausible one.
    _CONTACT_REFERER = st.secrets.get("contact_referer", "https://strideo.it")
except Exception:
    _CONTACT_ENDPOINT = ""
    _CONTACT_REFERER = "https://strideo.it"


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
    if st.button(
        "Send message", type="primary", use_container_width=True, key="ct_send"
    ):
        if not (name and email and message):
            st.warning("Please fill in your name, email, and a message.")
        elif not _CONTACT_ENDPOINT:
            st.info("The contact form isn't configured yet - please check back soon.")
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
# The story (centre reading column)
# ─────────────────────────────────────────────────────────────

_, mid, _ = st.columns([1, 3, 1])

with mid:
    st.markdown(
        """
# Strideo

<p class="lead">Running biomechanics from a single smartphone clip.</p>

Strideo is an open-source toolkit that turns ordinary training footage of a
middle-distance runner into objective biomechanical data - **cadence, ground
contact time, stride length, and vertical oscillation** - with no markers, no
wearables, and no lab. Point a phone at the track, film a few seconds from the
side, and the pipeline measures the mechanics that usually live only in a
coach's intuition. 

[View source on GitHub →](https://github.com/Matteofanfoni/strideo)

It is built specifically for **800 m and 1500 m** athletes, because
middle-distance form is not static: cadence, contact time, and stride length
all shift as a runner moves from threshold toward race pace. Most consumer
tools report one number per metric and average that shift away. Strideo is
designed to surface it.

> Strideo is a research project, not a medical or coaching product. Today it
> measures biomechanics with **classical computer vision**; a pace-conditioned
> model is in training and does **not** power these results.

## How it works

A clip is first converted from variable to a constant **60 fps**, then passed
through a **hybrid pose stage**. MediaPipe BlazePose tracks 33 body landmarks
for canonicalisation and metric extraction, while RTMPose-x (COCO-Body)
supplies hip and ankle landmarks into coarse ground-contact detection - where
it reaches near-100% reliability versus BlazePose's 30-70% in occluded windows.

From those landmarks Strideo derives **initial-contact and toe-off timing** from
ankle vertical velocity (with thresholds that scale to the runner's horizontal
hip speed), **stride length** from same-leg hip displacement across a full gait
cycle, and **vertical oscillation** - all calibrated to real-world units using
the runner's own height. No markers, no manual digitising.

The output is a per-clip report: the four core metrics with quality indicators,
cross-pace comparisons, an interactive frame-by-frame **ground-contact
verifier**, and a downloadable PDF.

## How it's validated

Every pipeline release is checked against a Kinovea-annotated ground-truth set,
and the headline target is deliberately tight - **ground contact time within
±10 ms** of the reference per stride. Spatial calibration passes with all
segment errors under 3%, and the current baseline earns **13 of 15** metric
checks across the annotated clips. Final evaluation will be leave-one-out
cross-validation across 18 subjects once the full dataset is collected.

> **Vertical oscillation is not yet validated.** Unlike cadence, ground
> contact time, and stride length, this metric has no ground-truth
> annotations in the current set - its values are still being calibrated and
> should be treated as **experimental**. Don't rely on the vertical
> oscillation figure until a future release confirms it against reference
> measurements.

## Where it's going

The longer-term goal is a **pace-conditioned predictor** that refines the
classical extraction using a convolutional neural network trained on a multi-runner dataset.
The architecture is in place and waiting on data collection; until it is
trained, every number you see comes from the classical pipeline above.
Strideo is MIT-licensed. Detailed capture and validation protocols are
available on request and will be the subject of an accompanying research
paper in preparation.
""",
        unsafe_allow_html=True,
    )

    st.markdown("")
    _cta, _ = st.columns([2, 3])
    with _cta:
        if st.button("Analyze your run", type="primary", use_container_width=True):
            st.switch_page("pages/upload.py")
    st.page_link(
        "pages/recording_guide.py", label="First time? Read the recording guide →"
    )
    st.markdown(
        """
<div class="about-me">
<h2>About me</h2>
<p>I'm Matteo, an IB student with a lifelong pull toward both running and
artificial intelligence - this project is where those two worlds finally meet.
As a competitive 800m and 1500m runner, I've spent years feeling the small
mechanical details that separate a clean race from a sloppy one, but I never
had a precise way to measure them - only instinct and a stopwatch. So I built a
computer vision pipeline that tracks an athlete's joints and stride patterns
from ordinary training footage, turning hours of running into objective
biomechanical data instead of guesswork. What started as curiosity about
machine learning has become a tool I genuinely want to use: something that
helps athletes and coaches catch inefficiencies and injury risks early, while
pushing me to understand how AI can engage with real physical performance, not
just clean datasets on a screen.</p>
</div>
""",
        unsafe_allow_html=True,
    )

    st.markdown("")
    _gt, _ = st.columns([2, 3])
    with _gt:
        if st.button("Get in touch", use_container_width=True):
            _contact_dialog()
