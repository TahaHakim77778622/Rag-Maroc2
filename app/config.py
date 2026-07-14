"""Chemins et paramètres par défaut (racine du dépôt = répertoire parent de app/)."""

import os
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parent.parent

VECTOR_STORE_DIR = PROJECT_ROOT / "vector_store"
FAISS_INDEX_PATH = VECTOR_STORE_DIR / "faiss.index"
METADATA_PATH = VECTOR_STORE_DIR / "metadata.jsonl"
MANIFEST_PATH = VECTOR_STORE_DIR / "faiss_manifest.json"

FINAL_CHUNKS_PATH = PROJECT_ROOT / "data" / "processed" / "final_chunks.jsonl"

# Même défaut que build_embeddings si le manifeste n’a pas encore été généré
DEFAULT_EMBEDDING_MODEL = os.environ.get(
    "EMBEDDING_MODEL", "intfloat/multilingual-e5-small"
)


def embedding_encode_query(query: str, model_name: str | None = None) -> str:
    """Préfixe E5 pour les requêtes (retrieval question→document)."""
    name = model_name or DEFAULT_EMBEDDING_MODEL
    if "e5" in name.lower():
        q = query.strip()
        if not q.lower().startswith("query:"):
            return f"query: {q}"
    return query


def embedding_encode_passage(text: str, model_name: str | None = None) -> str:
    """Préfixe E5 pour les passages indexés."""
    name = model_name or DEFAULT_EMBEDDING_MODEL
    if "e5" in name.lower():
        t = text.strip()
        if not t.lower().startswith("passage:"):
            return f"passage: {t}"
    return text


DEFAULT_TOP_K = 5
