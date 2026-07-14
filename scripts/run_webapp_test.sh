#!/usr/bin/env bash
# Serveur webapp pour tests d'intégration pytest (réponses LLM mockées, rapide).
set -euo pipefail
cd "$(dirname "$0")/.."
export LLM_MOCK=1
export WEBAPP_SECRET_KEY="${WEBAPP_SECRET_KEY:-dev-change-me-in-production}"
exec ./scripts/run_webapp.sh "$@"
