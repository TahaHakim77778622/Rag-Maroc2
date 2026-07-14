#!/usr/bin/env bash
# Installation des dépendances sur macOS sans compiler faiss/pyarrow.
# faiss et pyarrow viennent de conda (binaires) ; le reste via pip.
set -euo pipefail
cd "$(dirname "$0")/.."

PY="${PYTHON:-python3}"
if command -v conda &>/dev/null && [[ -n "${CONDA_PREFIX:-}" ]]; then
  PY="python"
fi

echo "==> Vérification faiss + pyarrow (conda recommandé)..."
if ! "$PY" -c "import faiss" 2>/dev/null; then
  echo "Installation faiss-cpu via conda..."
  if command -v conda &>/dev/null; then
    conda install -y -c pytorch faiss-cpu
  else
    echo "Erreur: installez conda puis: conda install -c pytorch faiss-cpu"
    exit 1
  fi
fi
if ! "$PY" -c "import pyarrow" 2>/dev/null; then
  echo "Installation pyarrow via conda..."
  if command -v conda &>/dev/null; then
    conda install -y -c conda-forge pyarrow
  else
    echo "Erreur: conda install -c conda-forge pyarrow"
    exit 1
  fi
fi
"$PY" -c "import faiss, pyarrow; print('faiss + pyarrow OK')"

echo "==> Pip (sans faiss-cpu — évite CMake/SWIG)..."
"$PY" -m pip install --default-timeout=120 --retries 5 -r requirements-pip.txt

echo "==> OK. Lancez: ./scripts/run_webapp.sh"
