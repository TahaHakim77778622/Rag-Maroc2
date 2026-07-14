#!/usr/bin/env bash
# Client Streamlit (FastAPI doit tourner sur le port indiqué par RAG_API_BASE).
set -euo pipefail
cd "$(dirname "$0")/.."
export RAG_API_BASE="${RAG_API_BASE:-http://127.0.0.1:8000}"
exec streamlit run app/streamlit_app.py "$@"
