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

# The nvidia-*-cu12 pip packages (nvidia-cublas-cu12, nvidia-cudnn-cu12,
# etc.) that back onnxruntime-gpu's CUDA execution provider do NOT
# self-register their lib/ directories anywhere the dynamic linker looks —
# their __init__.py files are empty stubs (confirmed by inspecting the
# actual wheel contents). onnxruntime's own pip package can preload some of
# these at import time via a build-time-generated capi/_ld_preload.py, but
# only for the packages its own [cuda]/[cudnn] extras declare -
# nvidia-cublas-cu12 isn't one of them (confirmed via its METADATA), so
# nothing ever preloads libcublasLt.so.12 without this. Building
# LD_LIBRARY_PATH explicitly from whatever nvidia-*/lib directories are
# actually installed makes this independent of that internal, undocumented
# mechanism entirely.
NVIDIA_LIB_DIRS="$(python -c "
import glob, os, site
bases = list(site.getsitepackages())
try:
    bases.append(site.getusersitepackages())
except AttributeError:
    pass
dirs = []
for base in bases:
    dirs.extend(glob.glob(os.path.join(base, 'nvidia', '*', 'lib')))
print(':'.join(dirs))
")"
if [ -n "$NVIDIA_LIB_DIRS" ]; then
    export LD_LIBRARY_PATH="${NVIDIA_LIB_DIRS}:${LD_LIBRARY_PATH:-}"
fi

exec streamlit run app/app.py --server.port=7860 --server.address=0.0.0.0
