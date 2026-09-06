#!/usr/bin/env bash
# HF Spaces (Docker SDK) injects "Repository secrets" as plain OS environment
# variables — it does NOT write a .streamlit/secrets.toml the way the native
# streamlit SDK does. Without this step, st.secrets.get(...) silently returns
# "" for every secret in production, even when configured in the Space
# settings, and every secret-gated feature (contact form, reCAPTCHA) degrades
# to its "unconfigured" state with no visible error.
set -euo pipefail

mkdir -p .streamlit
: > .streamlit/secrets.toml

for key in contact_endpoint contact_referer recaptcha_site_key recaptcha_secret_key; do
    val="${!key:-}"
    if [ -n "$val" ]; then
        esc=$(printf '%s' "$val" | sed 's/\\/\\\\/g; s/"/\\"/g')
        echo "${key} = \"${esc}\"" >> .streamlit/secrets.toml
    fi
done

exec streamlit run app/app.py --server.port=7860 --server.address=0.0.0.0
