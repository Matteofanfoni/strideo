#!/usr/bin/env bash
# Run the Strideo Streamlit app from any directory.
# Activate your virtual environment before running this script.
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
python -m streamlit run "$SCRIPT_DIR/app/app.py" "$@"
