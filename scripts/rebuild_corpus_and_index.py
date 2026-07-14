#!/usr/bin/env python3
"""
Pipeline complet : admin (optionnel) → merge → consulat → embeddings → FAISS → inventaire.

Usage (racine du projet) :
    python scripts/rebuild_corpus_and_index.py
    python scripts/rebuild_corpus_and_index.py --skip-collect
    python scripts/rebuild_corpus_and_index.py --embeddings-only
"""

from __future__ import annotations

import argparse
import subprocess
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent


def run(cmd: list[str], *, label: str) -> None:
    print(f"\n=== {label} ===")
    print(" ", " ".join(cmd))
    r = subprocess.run(cmd, cwd=str(ROOT))
    if r.returncode != 0:
        raise SystemExit(f"Échec : {label} (code {r.returncode})")


def main() -> int:
    p = argparse.ArgumentParser(description="Rebuild corpus RAG-MAROC2 + index FAISS")
    p.add_argument("--skip-collect", action="store_true", help="Ne pas relancer collect_admin_docs")
    p.add_argument("--skip-admin-chunk", action="store_true", help="Ne pas regénérer admin_chunks.jsonl")
    p.add_argument("--embeddings-only", action="store_true", help="Seulement embeddings + FAISS")
    p.add_argument(
        "--ingest-web-queue",
        action="store_true",
        help="Intégrer data/processed/web_additions_queue.jsonl dans final_chunks (avant embeddings)",
    )
    args = p.parse_args()
    py = sys.executable

    if not args.embeddings_only:
        if not args.skip_collect:
            run([py, "scripts/collect_admin_docs.py"], label="Collecte portails admin")
        if not args.skip_admin_chunk:
            run([py, "scripts/chunk_admin_pages_smart.py"], label="Chunking admin (50 tokens overlap)")
        run([py, "scripts/merge_chunks.py"], label="Fusion final_chunks.jsonl")
        run([py, "scripts/append_passport_consulat_chunks.py"], label="Chunks consulat.ma (passeport)")
        if args.ingest_web_queue:
            run(
                [py, "scripts/ingest_web_queue.py", "--apply", "--prune-queue"],
                label="Ingestion file web fallback",
            )

    run([py, "scripts/build_embeddings.py"], label="Embeddings sentence-transformers")
    run([py, "scripts/build_faiss.py"], label="Index FAISS IndexFlatIP")
    run([py, "scripts/inventory_final_chunks.py"], label="Inventaire corpus")
    print("\nTerminé. Redémarrez uvicorn pour recharger l’index.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
