#!/usr/bin/env bash
# Lance l’API FastAPI (UI Jinja + endpoints pour Streamlit).
# Options --http h11 et --loop asyncio réduisent les segfaults sur certains Mac (uvloop / httptools).
set -euo pipefail
cd "$(dirname "$0")/.."
export WEBAPP_SECRET_KEY="${WEBAPP_SECRET_KEY:-dev-change-me-in-production}"
export TOKENIZERS_PARALLELISM="${TOKENIZERS_PARALLELISM:-false}"
export OMP_NUM_THREADS="${OMP_NUM_THREADS:-1}"
export MKL_NUM_THREADS="${MKL_NUM_THREADS:-1}"
export KMP_DUPLICATE_LIB_OK="${KMP_DUPLICATE_LIB_OK:-TRUE}"
# Utile si un jour tu utilises des workers / fork sur macOS
export OBJC_DISABLE_INITIALIZE_FORK_SAFETY="${OBJC_DISABLE_INITIALIZE_FORK_SAFETY:-YES}"

exec uvicorn webapp.main:app \
  --host 127.0.0.1 \
  --port 8000 \
  --http h11 \
  --loop asyncio \
  "$@"
