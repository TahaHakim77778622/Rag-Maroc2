"""
Inventaire du corpus RAG : agrège final_chunks.jsonl (doc_id, source_org, etc.).

Usage (depuis la racine du projet) :
  python scripts/inventory_final_chunks.py

Sorties :
  data/processed/corpus_inventory.json
  data/processed/corpus_by_doc.csv
"""

from __future__ import annotations

import csv
import json
import sys
from collections import Counter, defaultdict
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from app.Rag_classique.corpus_io import load_chunks_jsonl  # noqa: E402

FINAL_CHUNKS = PROJECT_ROOT / "data" / "processed" / "final_chunks.jsonl"
OUT_JSON = PROJECT_ROOT / "data" / "processed" / "corpus_inventory.json"
OUT_CSV = PROJECT_ROOT / "data" / "processed" / "corpus_by_doc.csv"
MANIFEST = PROJECT_ROOT / "vector_store" / "faiss_manifest.json"


def main() -> None:
    texts, metas = load_chunks_jsonl(FINAL_CHUNKS)
    total = len(texts)
    if total != len(metas):
        raise RuntimeError("Textes et métadonnées désalignés.")

    by_org: Counter[str] = Counter()
    by_type: Counter[str] = Counter()
    by_category: Counter[str] = Counter()
    by_doc: defaultdict[str, int] = defaultdict(int)
    doc_meta: dict[str, dict] = {}

    for m in metas:
        doc_id = str(m.get("doc_id") or "(sans doc_id)")
        by_org[str(m.get("source_org") or "(inconnu)")] += 1
        by_type[str(m.get("source_type") or "(inconnu)")] += 1
        by_category[str(m.get("category") or "(inconnu)")] += 1
        by_doc[doc_id] += 1
        if doc_id not in doc_meta:
            doc_meta[doc_id] = {
                "source_org": m.get("source_org"),
                "source_type": m.get("source_type"),
                "title": m.get("title"),
                "filename": m.get("filename"),
            }

    # Comparaison avec l’index FAISS (si manifeste présent)
    faiss_info: dict[str, str | int | bool | None] = {
        "manifest_path": str(MANIFEST),
        "ntotal": None,
        "matches_corpus_count": None,
    }
    if MANIFEST.is_file():
        man = json.loads(MANIFEST.read_text(encoding="utf-8"))
        nvec = int(man.get("ntotal", -1))
        faiss_info["ntotal"] = nvec
        faiss_info["matches_corpus_count"] = nvec == total

    report = {
        "corpus_path": str(FINAL_CHUNKS),
        "total_chunks": total,
        "unique_doc_id": len(by_doc),
        "by_source_org": dict(by_org.most_common()),
        "by_source_type": dict(by_type.most_common()),
        "by_category": dict(by_category.most_common()),
        "faiss": faiss_info,
        "note": (
            "Le RAG charge final_chunks.jsonl + faiss.index. Le nombre de chunks doit "
            "aligner ntotal (sinon : rebuild embeddings + faiss)."
        ),
    }

    OUT_JSON.parent.mkdir(parents=True, exist_ok=True)
    OUT_JSON.write_text(json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8")
    print(f"Écrit : {OUT_JSON}")

    # CSV : un document (doc_id) par ligne, trié par nombre de chunks
    rows: list[dict] = []
    for doc_id, n in sorted(by_doc.items(), key=lambda x: (-x[1], x[0])):
        meta = doc_meta.get(doc_id, {})
        rows.append(
            {
                "doc_id": doc_id,
                "chunk_count": n,
                "source_org": meta.get("source_org") or "",
                "source_type": meta.get("source_type") or "",
                "title": (meta.get("title") or "")[:200],
                "filename": meta.get("filename") or "",
            }
        )
    with OUT_CSV.open("w", encoding="utf-8", newline="") as f:
        w = csv.DictWriter(
            f,
            fieldnames=[
                "doc_id",
                "chunk_count",
                "source_org",
                "source_type",
                "title",
                "filename",
            ],
        )
        w.writeheader()
        w.writerows(rows)
    print(f"Écrit : {OUT_CSV} ({len(rows)} documents)")


if __name__ == "__main__":
    main()
