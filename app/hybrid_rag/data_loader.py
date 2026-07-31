
import json
import sys
from pathlib import Path
from typing import List, Dict

REQUIRED_FIELDS = [
    "chunk_id", "doc_id", "title", "source_org", "source_url",
    "filename", "page_start", "source_type", "category", "label", "text"
]


def load_corpus(path: Path) -> List[Dict]:
    """Charge un fichier JSONL et retourne la liste des chunks valides."""
    if not path.exists():
        raise FileNotFoundError(f"Corpus introuvable: {path}")

    entries = []
    n_invalid = 0

    with open(path, "r", encoding="utf-8") as f:
        for i, line in enumerate(f, start=1):
            line = line.strip()
            if not line:
                continue
            try:
                obj = json.loads(line)
            except json.JSONDecodeError:
                print(f"[data_loader] Ligne {i} invalide (JSON malformé), ignorée.", file=sys.stderr)
                n_invalid += 1
                continue

            missing = [field for field in REQUIRED_FIELDS if field not in obj]
            if missing:
                print(f"[data_loader] Ligne {i} ignorée, champs manquants: {missing}", file=sys.stderr)
                n_invalid += 1
                continue

            if not obj.get("text", "").strip():
                print(f"[data_loader] Ligne {i} ignorée, champ 'text' vide.", file=sys.stderr)
                n_invalid += 1
                continue

            entries.append(obj)

    print(f"[data_loader] {len(entries)} chunks valides chargés, {n_invalid} ignorés.")
    return entries


def get_texts_and_ids(chunks: List[Dict]):
    """Retourne (chunk_ids, texts) alignés par index — utile pour FAISS/BM25."""
    chunk_ids = [c["chunk_id"] for c in chunks]
    texts = [c["text"] for c in chunks]
    return chunk_ids, texts


def get_chunk_by_id(chunks: List[Dict], chunk_id: str) -> Dict:
    """Retrouve un chunk complet à partir de son chunk_id (pour affichage des sources)."""
    for c in chunks:
        if c["chunk_id"] == chunk_id:
            return c
    return None