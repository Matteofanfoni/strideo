"""Google reCAPTCHA v2 checkbox for the contact form, via Streamlit's
``st.components.v2`` bidirectional-component API (no separate JS build step).

Mirrors the pattern used elsewhere in ``app/ui``: a small, self-contained
module exposing plain functions that a page imports directly.
"""

from __future__ import annotations

import streamlit as st

_RECAPTCHA_JS = """
export default function(component) {
  const { data: siteKey, setStateValue, parentElement } = component;
  const container = parentElement.querySelector('#recaptcha-box');

  function renderWidget() {
    if (container.dataset.rendered === 'true') return;
    container.dataset.rendered = 'true';
    window.grecaptcha.render(container, {
      sitekey: siteKey,
      callback: (token) => setStateValue('token', token),
      'expired-callback': () => setStateValue('token', null),
    });
  }

  if (window.grecaptcha && window.grecaptcha.render) {
    renderWidget();
  } else {
    window.__strideoRecaptchaLoad = renderWidget;
    if (!document.getElementById('recaptcha-api-script')) {
      const script = document.createElement('script');
      script.id = 'recaptcha-api-script';
      script.src = 'https://www.google.com/recaptcha/api.js?onload=__strideoRecaptchaLoad&render=explicit';
      script.async = true;
      script.defer = true;
      document.head.appendChild(script);
    }
  }
}
"""

_recaptcha_widget = st.components.v2.component(
    "strideo_recaptcha",
    html='<div id="recaptcha-box"></div>',
    js=_RECAPTCHA_JS,
    # Google's widget positions its challenge popup by walking the DOM to
    # compute the checkbox's on-screen coordinates. Mounted inside the
    # default shadow root (isolate_styles=True), that calculation breaks
    # silently: the popup iframe gets created but parked off-screen
    # (visibility: hidden; top: -9999px) and never repositioned, so the
    # checkbox spins forever. Mounting directly into the light DOM fixes it.
    isolate_styles=False,
)


def render_recaptcha(site_key: str) -> str | None:
    """Mount the reCAPTCHA v2 checkbox and return the solved token, or None."""
    result = _recaptcha_widget(data=site_key, on_token_change=lambda: None)
    return result.token


def verify_recaptcha(token: str | None, secret_key: str) -> bool:
    """Verify a solved token server-side against Google's siteverify endpoint."""
    if not token or not secret_key:
        return False
    import requests

    try:
        resp = requests.post(
            "https://www.google.com/recaptcha/api/siteverify",
            data={"secret": secret_key, "response": token},
            timeout=10,
        )
        return bool(resp.json().get("success"))
    except Exception:
        return False
