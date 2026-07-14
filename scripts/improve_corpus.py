#!/usr/bin/env python3
"""
Pipeline d'amélioration du corpus : filtrage → fusion → rebuild FAISS.

Usage (racine du projet) :
    python scripts/improve_corpus.py
"""

from __future__ import annotations

import subprocess
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent


def main() -> int:
    py = sys.executable
    scripts = ROOT / "scripts"

    print("Étape 1 : Filtrage chunks inutiles...")
    subprocess.run([py, str(scripts / "filter_chunks.py")], cwd=str(ROOT), check=True)

    print("\nÉtape 2 : Fusion chunks courts...")
    subprocess.run([py, str(scripts / "merge_short_chunks.py")], cwd=str(ROOT), check=True)

    print("\nÉtape 3 : Rebuild FAISS...")
    subprocess.run(
        [py, str(scripts / "build_embeddings.py"), "--batch-size", "64"],
        cwd=str(ROOT),
        check=True,
    )
    subprocess.run([py, str(scripts / "build_faiss.py")], cwd=str(ROOT), check=True)

    print("\n✅ Corpus amélioré et index FAISS rebuildé")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
