"""Microsoft Clarity session analytics, off unless ``CLARITY_PROJECT_ID`` is set.

The ``?src=`` count
(``usage_count.py``) says how many analyses complete; Clarity shows where and
why the other visits stop.

**Why it is injected into Streamlit's own ``index.html``.** strideo.org only
iframes the Space, and Clarity cannot record inside a third-party iframe; a
``components.html`` snippet would live in its own sandboxed iframe too. So
``docker-entrypoint.sh`` runs this module as a script, before Streamlit
starts, when the Space variable is set: it writes :data:`HEAD_TEMPLATE` into
``streamlit/static/index.html`` once, before ``</head>``. The image is the
same with Clarity on or off; turning it off is unsetting the variable and
restarting (the container's filesystem is rebuilt on restart).

**Consent (user decision 2026-11-21): the Clarity script loads only after
Allow.** Until then the page has a consent bar and nothing else from
Microsoft; "No thanks" loads nothing at all, rather than Clarity's
cookie-free no-consent mode. Consent is ``consentv2`` with ``ad_Storage``
always denied (Strideo has no ads). A footer link (``#analytics-preferences``)
withdraws it: ``clarity('consent', false)`` erases Clarity's cookies and ends
the session, and the bar comes back.

**Funnel events** (:func:`clarity_event`) are fired from Python through a
zero-height ``components.v1.html`` iframe that calls ``window.parent.clarity``
(same origin, the route ``styles.scroll_to_top`` already relies on). Each
event also sets the session's ``src`` tag, so recordings filter by campaign.
Before consent ``window.parent.clarity`` does not exist and the call is a
no-op.

Stdlib only at import time: the entrypoint runs this file with the bare
container Python before the app exists.
"""

import os
import re
import sys
from pathlib import Path
from typing import Optional

ENV_VAR = "CLARITY_PROJECT_ID"
MARKER = "<!-- strideo-clarity -->"
PLACEHOLDER = "__CLARITY_PROJECT_ID__"
PREFERENCES_HREF = "#analytics-preferences"

#: Clarity project IDs are short lowercase alphanumerics (about 10 chars).
_ID_RE = re.compile(r"^[a-z0-9]{6,16}$")

#: The funnel, in order. Anything else is refused, so a typo cannot create a
#: stray event name in the dashboard.
FUNNEL_EVENTS = ("upload_started", "analysis_started", "results_shown")

# Light-only, like the app: tokens from app/ui/styles.py's _BASE_CSS.
HEAD_TEMPLATE = MARKER + """
<style>
#strideo-consent { position: fixed; left: 16px; right: 16px; bottom: 16px; z-index: 1000000;
  max-width: 760px; margin: 0 auto; box-sizing: border-box; display: flex; flex-wrap: wrap;
  align-items: center; gap: 12px 16px; padding: 14px 16px; background: #FFFFFF; color: #101827;
  border: 1px solid #DCE3EA; border-radius: 16px; font-family: "Inter", system-ui, sans-serif;
  font-size: 14px; line-height: 1.5; }
#strideo-consent p { margin: 0; flex: 1 1 320px; }
#strideo-consent a { color: #1C6E62; }
#strideo-consent .sc-actions { display: flex; gap: 8px; }
#strideo-consent button { font: inherit; font-weight: 600; border-radius: 10px; padding: 8px 16px;
  cursor: pointer; border: 1px solid #DCE3EA; background: #ECF1F5; color: #101827; }
#strideo-consent button.sc-allow { background: #1C6E62; border-color: #1C6E62; color: #FFFFFF; }
</style>
<script>
(function () {
  var KEY = "strideo_analytics_consent";
  var ID = "__CLARITY_PROJECT_ID__";
  var loaded = false;
  function read() { try { return window.localStorage.getItem(KEY); } catch (e) { return null; } }
  function write(v) {
    try { if (v === null) { window.localStorage.removeItem(KEY); } else { window.localStorage.setItem(KEY, v); } }
    catch (e) {}
  }
  function load() {
    if (!loaded) {
      loaded = true;
      (function (c, l, a, r, i, t, y) {
        c[a] = c[a] || function () { (c[a].q = c[a].q || []).push(arguments); };
        t = l.createElement(r); t.async = 1; t.src = "https://www.clarity.ms/tag/" + i;
        y = l.getElementsByTagName(r)[0]; y.parentNode.insertBefore(t, y);
      })(window, document, "clarity", "script", ID);
    }
    window.clarity("consentv2", { ad_Storage: "denied", analytics_Storage: "granted" });
  }
  function hide() { var b = document.getElementById("strideo-consent"); if (b) { b.remove(); } }
  function show() {
    if (document.getElementById("strideo-consent")) { return; }
    var bar = document.createElement("div");
    bar.id = "strideo-consent";
    bar.setAttribute("role", "region");
    bar.setAttribute("aria-label", "Analytics choice");
    bar.innerHTML =
      '<p>Strideo can record how this page is used (clicks and scrolling, never your video) ' +
      'to find where people get stuck. <a href="/cookie_policy" target="_blank">Cookie Policy</a></p>' +
      '<div class="sc-actions"><button type="button" class="sc-deny">No thanks</button>' +
      '<button type="button" class="sc-allow">Allow</button></div>';
    bar.querySelector(".sc-allow").addEventListener("click", function () { write("granted"); hide(); load(); });
    bar.querySelector(".sc-deny").addEventListener("click", function () { write("denied"); hide(); });
    document.body.appendChild(bar);
  }
  function reset() {
    write(null);
    if (loaded && window.clarity) { window.clarity("consent", false); }
    show();
  }
  document.addEventListener("click", function (e) {
    var a = e.target && e.target.closest ? e.target.closest('a[href$="#analytics-preferences"]') : null;
    if (a) { e.preventDefault(); e.stopPropagation(); reset(); }
  }, true);
  function init() { var v = read(); if (v === "granted") { load(); } else if (v !== "denied") { show(); } }
  if (document.body) { init(); } else { document.addEventListener("DOMContentLoaded", init); }
})();
</script>
"""


def project_id() -> Optional[str]:
    """The configured Clarity project ID, or None when Clarity is off."""
    raw = os.environ.get(ENV_VAR, "").strip()
    return raw or None


def clarity_enabled() -> bool:
    """True when the Space has a Clarity project ID set."""
    return project_id() is not None


def render_head(pid: str) -> str:
    """:data:`HEAD_TEMPLATE` with the project ID filled in.

    Raises:
        ValueError: ``pid`` is not a plausible Clarity project ID, so nothing
            unexpected is ever written into a ``<script>``.
    """
    if not _ID_RE.match(pid):
        raise ValueError(f"not a Clarity project ID: {pid!r}")
    return HEAD_TEMPLATE.replace(PLACEHOLDER, pid)


def inject(index_html: str, pid: str) -> str:
    """``index_html`` with the Clarity head inserted once before ``</head>``.

    Idempotent: a page that already carries :data:`MARKER` is returned
    unchanged.

    Raises:
        ValueError: bad ``pid``, or no ``</head>`` (a Streamlit upgrade changed
            ``index.html``; fail loudly rather than ship without analytics
            silently).
    """
    head = render_head(pid)
    if MARKER in index_html:
        return index_html
    if index_html.count("</head>") != 1:
        raise ValueError("expected exactly one </head> in Streamlit's index.html")
    return index_html.replace("</head>", head + "</head>", 1)


def streamlit_index_path() -> Path:
    """Streamlit's served ``static/index.html``."""
    import streamlit

    return Path(streamlit.__file__).parent / "static" / "index.html"


def inject_file(path: Path, pid: str) -> bool:
    """Inject into the file at ``path``. Returns True if it was changed."""
    before = path.read_text(encoding="utf-8")
    after = inject(before, pid)
    if after == before:
        return False
    with open(path, "w", encoding="utf-8", newline="\n") as fh:
        fh.write(after)
    return True


def clarity_event(name: str) -> None:
    """Fire one funnel event (and the session's ``src`` tag) in the browser.

    A no-op when Clarity is off, so local runs and tests add nothing to the
    page. In the browser it is also a no-op until the visitor has allowed
    analytics.

    Raises:
        ValueError: ``name`` is not in :data:`FUNNEL_EVENTS`.
    """
    if name not in FUNNEL_EVENTS:
        raise ValueError(f"unknown funnel event: {name!r}")
    if not clarity_enabled():
        return
    import streamlit as st
    import streamlit.components.v1 as components

    from utils.usage_count import SESSION_KEY, clean_tag

    tag = clean_tag(st.session_state.get(SESSION_KEY))
    components.html(
        "<script>try{var c=window.parent.clarity;if(c){"
        f'c("set","src","{tag}");c("event","{name}");'
        "}}catch(e){}</script>",
        height=0,
    )


def _main(argv: list) -> int:
    pid = argv[1] if len(argv) > 1 else (project_id() or "")
    path = streamlit_index_path()
    try:
        changed = inject_file(path, pid)
    except ValueError as exc:
        print(f"clarity: not injected: {exc}", file=sys.stderr)
        return 1
    print(f"clarity: {'injected into' if changed else 'already in'} {path}")
    return 0


if __name__ == "__main__":
    sys.exit(_main(sys.argv))
