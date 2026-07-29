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
