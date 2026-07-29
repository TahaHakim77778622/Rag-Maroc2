#!/usr/bin/env python3
"""
Intègre la file web fallback dans final_chunks.jsonl (idempotent).

La file est remplie automatiquement par le RAG quand WEB fallback est utilisé
(voir app/web_fallback.save_web_hits_to_queue).

Usage (racine du projet) :
    python scripts/ingest_web_queue.py              # dry-run par défaut : affiche le plan
    python scripts/ingest_web_queue.py --apply      # append dans final_chunks.jsonl
    python scripts/ingest_web_queue.py --apply --prune-queue   # retire de la file les lignes ingérées
    python scripts/ingest_web_queue.py --apply --rebuild       # + embeddings + FAISS

Ensuite (si pas --rebuild) :
    python scripts/build_embeddings.py
    python scripts/build_faiss.py
    # redémarrer uvicorn
"""

from __future__ import annotations

import argparse
import os
import subprocess
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from app.Rag_classique.web_queue_ingest import (  # noqa: E402
    FINAL_CHUNKS_PATH,
    QUEUE_PATH,
    _iter_jsonl,
    append_chunks,
    log_ingest,
    plan_ingest,
    rewrite_queue,
)


def _run_rebuild() -> int:
    py = sys.executable
    for script, label in (
        ("scripts/build_embeddings.py", "Embeddings"),
        ("scripts/build_faiss.py", "FAISS"),
    ):
        print(f"\n=== {label} ===")
        r = subprocess.run([py, script], cwd=str(ROOT))
        if r.returncode != 0:
            return r.returncode
    print("\nIndex à jour. Redémarrez uvicorn pour recharger l’index.")
    return 0


def main() -> int:
    p = argparse.ArgumentParser(description="Ingestion file web → final_chunks.jsonl")
    p.add_argument(
        "--apply",
        action="store_true",
        help="Écrit réellement dans final_chunks.jsonl (sinon dry-run)",
    )
    p.add_argument(
        "--prune-queue",
        action="store_true",
        help="Retire de web_additions_queue.jsonl les lignes ingérées ou déjà en corpus",
    )
    p.add_argument(
        "--rebuild",
        action="store_true",
        help="Après --apply, lance build_embeddings.py + build_faiss.py",
    )
    p.add_argument(
        "--min-text-len",
        type=int,
        default=int(os.environ.get("WEB_INGEST_MIN_TEXT_LEN", "120")),
        help="Longueur minimale du texte (défaut 120)",
    )
    args = p.parse_args()

    if not QUEUE_PATH.is_file():
        print(f"Aucune file web : {QUEUE_PATH}")
        print("Le fallback web remplit ce fichier lors des questions non couvertes.")
        return 0

    queue_rows = _iter_jsonl(QUEUE_PATH)
    if not queue_rows:
        print(f"File vide : {QUEUE_PATH}")
        return 0

    to_add, rejected, stats = plan_ingest(queue_rows, min_text_len=args.min_text_len)

    print(f"File : {QUEUE_PATH} ({stats['queue_rows']} lignes)")
    print(f"Corpus : {FINAL_CHUNKS_PATH}")
    print(
        f"Plan : +{stats['added']} chunk(s), "
        f"{stats['skipped_duplicate']} doublon(s) ignoré(s), "
        f"{stats['rejected']} rejeté(s) (texte court / stub / URL non .ma)"
    )

    if to_add:
        print("\nChunks à ajouter :")
        for c in to_add[:20]:
            print(f"  - {c['chunk_id']} | {c.get('source_url', '')[:70]}")
        if len(to_add) > 20:
            print(f"  … et {len(to_add) - 20} autre(s)")

    if not args.apply:
        if stats["added"]:
            print("\nDry-run : relancez avec --apply pour écrire dans le corpus.")
        return 0

    n = append_chunks(to_add)
    log_ingest(to_add, stats)
    print(f"\nÉcrit {n} chunk(s) dans {FINAL_CHUNKS_PATH}.")

    if args.prune_queue:
        from app.Rag_classique.web_queue_ingest import _norm_url, load_corpus_index, queue_row_to_chunk  # noqa: PLC0415

        urls_in_corpus, _, _ = load_corpus_index()
        keep: list[dict] = []
        for rec in queue_rows:
            ch = queue_row_to_chunk(rec, min_text_len=args.min_text_len)
            if ch is None:
                keep.append(rec)
                continue
            if _norm_url(str(rec.get("source_url") or "")) in urls_in_corpus:
                continue
            keep.append(rec)
        rewrite_queue(keep)
        print(f"File nettoyée : {len(keep)} ligne(s) restante(s) dans {QUEUE_PATH}")

    if args.rebuild and n > 0:
        return _run_rebuild()
    if n > 0 and not args.rebuild:
        print("\nÉtape suivante :")
        print("  python scripts/build_embeddings.py")
        print("  python scripts/build_faiss.py")
        print("  # puis redémarrer uvicorn")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
