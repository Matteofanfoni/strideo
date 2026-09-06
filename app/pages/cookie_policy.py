# app/pages/cookie_policy.py
import streamlit as st
import sys
from pathlib import Path

app_dir = Path(__file__).parent.parent
if str(app_dir) not in sys.path:
    sys.path.insert(0, str(app_dir))

from ui.styles import (  # noqa: E402
    inject_shared_styles,
    render_navbar,
    render_footer,
    render_legal_document,
)

st.set_page_config(
    page_title="Cookie Policy | strideo.it",
    page_icon=str(app_dir / "assets" / "favicon.png"),
    layout="wide",
)

inject_shared_styles()
render_navbar()

# Termly's free-tier "HTML Format" export, rendered as-is in an iframe rather
# than embedded via their JS snippet: Streamlit doesn't expose a place to
# inject arbitrary <head>/<body> scripts without a custom component, so the
# static HTML export (self-contained styles included) is the lower-friction
# fit. No public Termly-hosted URL exists on the free tier; this
# page's own URL is the canonical link used elsewhere (Privacy Policy
# cross-link, footer). Content lives in a sibling asset file (same pattern
# as privacy_policy.py / terms_of_service.py), not inlined here.
_CONTENT_PATH = app_dir / "assets" / "legal" / "cookie_policy.html"
_COOKIE_POLICY_HTML = _CONTENT_PATH.read_text(encoding="utf-8")

_, mid, _ = st.columns([1, 3, 1])

with mid:
    render_legal_document(_COOKIE_POLICY_HTML)

render_footer()
