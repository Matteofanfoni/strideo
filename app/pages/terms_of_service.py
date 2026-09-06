# app/pages/terms_of_service.py
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
    page_title="Terms of Service | strideo.it",
    page_icon=str(app_dir / "assets" / "favicon.png"),
    layout="wide",
)

inject_shared_styles()
render_navbar()

# Termly's free-tier "HTML Format" export, rendered as-is - same pattern as
# cookie_policy.py / privacy_policy.py. No public Termly-hosted URL exists
# on the free tier; this page's own URL is the canonical link
# used elsewhere.
_CONTENT_PATH = app_dir / "assets" / "legal" / "terms_of_service.html"
_TERMS_OF_SERVICE_HTML = _CONTENT_PATH.read_text(encoding="utf-8")

_, mid, _ = st.columns([1, 3, 1])

with mid:
    render_legal_document(_TERMS_OF_SERVICE_HTML)

render_footer()
