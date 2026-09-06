# app/pages/privacy_policy.py
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
    page_title="Privacy Policy | strideo.it",
    page_icon=str(app_dir / "assets" / "favicon.png"),
    layout="wide",
)

inject_shared_styles()
render_navbar()

# Termly's free-tier "HTML Format" export, rendered as-is in an iframe -
# same approach as cookie_policy.py, but loaded from a static asset file
# rather than inlined as a Python string, since this document is ~170KB
# (much larger than the Cookie Policy). No public Termly-hosted URL exists
# on the free tier; this page's own URL is the canonical link used
# elsewhere (footer, the Cookie Policy's cross-reference, etc.).
_CONTENT_PATH = app_dir / "assets" / "legal" / "privacy_policy.html"
_PRIVACY_POLICY_HTML = _CONTENT_PATH.read_text(encoding="utf-8")

_, mid, _ = st.columns([1, 3, 1])

with mid:
    render_legal_document(_PRIVACY_POLICY_HTML)

render_footer()
