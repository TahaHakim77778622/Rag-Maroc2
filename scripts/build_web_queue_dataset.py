"""
[Déprécié] Préférez scripts/ingest_web_queue.py qui fusionne directement dans final_chunks.jsonl.

Construit un fichier intermédiaire web_queue_chunks.jsonl (legacy).

Utilisation recommandée :
  python scripts/ingest_web_queue.py --apply --prune-queue --rebuild
"""

from __future__ import annotations

import json
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parent.parent
QUEUE_PATH = PROJECT_ROOT / "data" / "processed" / "web_additions_queue.jsonl"
OUT_PATH = PROJECT_ROOT / "data" / "processed" / "web_queue_chunks.jsonl"


def main() -> int:
    if not QUEUE_PATH.is_file():
        print(f"Aucune queue web: {QUEUE_PATH}")
        return 0

    rows = []
    with QUEUE_PATH.open("r", encoding="utf-8") as f:
        for i, line in enumerate(f, start=1):
            line = line.strip()
            if not line:
                continue
            try:
                rec = json.loads(line)
            except json.JSONDecodeError:
                continue

            url = str(rec.get("source_url") or "").strip()
            title = str(rec.get("title") or "Source web officielle")
            text = str(rec.get("text") or "").strip()
            if not url or len(text) < 80:
                continue

            rows.append(
                {
                    "chunk_id": f"WEBQ_{i:06d}",
                    "doc_id": f"WEBQ_{i:06d}",
                    "title": title,
                    "label": "Web fallback",
                    "source_org": "Web officiel",
                    "source_type": "web_fallback",
                    "source_url": url,
                    "text": text,
                }
            )

    OUT_PATH.parent.mkdir(parents=True, exist_ok=True)
    with OUT_PATH.open("w", encoding="utf-8") as out:
        for r in rows:
            out.write(json.dumps(r, ensure_ascii=False) + "\n")

    print(f"Dataset web genere: {OUT_PATH} ({len(rows)} chunks)")
    print("Recommandé : python scripts/ingest_web_queue.py --apply --rebuild")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
