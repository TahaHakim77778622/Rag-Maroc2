#!/usr/bin/env bash
# « Compilation » du projet RAG-MAROC2 : dépendances + index vectoriel.
# Usage (depuis la racine) : ./scripts/build.sh
# Options : ./scripts/build.sh --full  (recollecte + merge + index)
set -euo pipefail
cd "$(dirname "$0")/.."

PY="${PY:-.venv/bin/python}"
if [[ ! -x "$PY" ]]; then
  echo "Créer le venv : python3 -m venv .venv && .venv/bin/pip install -r requirements.txt"
  exit 1
fi

echo "=== Installation des dépendances ==="
"$PY" -m pip install -q -r requirements.txt

if [[ "${1:-}" == "--full" ]]; then
  echo "=== Build complet (corpus + index) ==="
  "$PY" scripts/rebuild_corpus_and_index.py
else
  echo "=== Build index (embeddings + FAISS, corpus existant) ==="
  "$PY" scripts/rebuild_corpus_and_index.py --embeddings-only
fi

echo "=== Tests rapides ==="
LLM_MOCK=1 "$PY" -m pytest tests/ -q --tb=line

echo ""
echo "Build terminé. Lancer l'app : ./scripts/run_webapp.sh"
