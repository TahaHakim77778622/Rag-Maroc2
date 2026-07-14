"""
Construit un index FAISS à partir de vector_store/embeddings.npy.

Les vecteurs ont été normalisés L2 dans build_embeddings.py : on utilise
IndexFlatIP (produit scalaire) pour équivaloir à la similarité cosinus.

Usage (depuis la racine du projet) :
    python scripts/build_faiss.py

Prérequis :
    pip install faiss-cpu numpy
    # ou : conda install -c pytorch faiss-cpu
"""

from __future__ import annotations

import json
import logging
import sys
from pathlib import Path

import numpy as np

# ---------------------------------------------------------------------------
# Constantes
# ---------------------------------------------------------------------------
PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

try:
    from dotenv import load_dotenv

    load_dotenv(PROJECT_ROOT / ".env", override=False)
except ImportError:
    pass

from app.config import DEFAULT_EMBEDDING_MODEL

VECTOR_STORE_DIR = PROJECT_ROOT / "vector_store"
EMBEDDINGS_PATH = VECTOR_STORE_DIR / "embeddings.npy"
METADATA_PATH = VECTOR_STORE_DIR / "metadata.jsonl"
FAISS_INDEX_PATH = VECTOR_STORE_DIR / "faiss.index"
MANIFEST_PATH = VECTOR_STORE_DIR / "faiss_manifest.json"

EMBEDDING_MODEL_NAME = DEFAULT_EMBEDDING_MODEL

logging.basicConfig(
    level=logging.INFO,
    format="%(levelname)s: %(message)s",
)
logger = logging.getLogger(__name__)


def _load_embeddings(path: Path) -> np.ndarray:
    if not path.is_file():
        raise FileNotFoundError(
            f"Embeddings introuvables : {path}. Lancez d'abord : python scripts/build_embeddings.py"
        )
    vectors = np.load(path)
    if vectors.ndim != 2:
        raise ValueError(f"Attendu une matrice 2D, reçu shape {vectors.shape}.")
    if vectors.dtype != np.float32:
        logger.info("Conversion float32 pour FAISS (était %s).", vectors.dtype)
        vectors = vectors.astype(np.float32, copy=False)
    if not vectors.flags["C_CONTIGUOUS"]:
        vectors = np.ascontiguousarray(vectors)
    return vectors


def _check_metadata_count(n_vectors: int) -> None:
    if not METADATA_PATH.is_file():
        logger.warning("metadata.jsonl absent : %s (vérifiez l'alignement manuellement).", METADATA_PATH)
        return
    n_meta = sum(1 for line in METADATA_PATH.open(encoding="utf-8") if line.strip())
    if n_meta != n_vectors:
        raise ValueError(
            f"Incohérence : {n_vectors} vecteurs dans embeddings.npy mais {n_meta} lignes dans metadata.jsonl."
        )
    logger.info("Alignement OK : %s vecteurs = %s lignes metadata.", n_vectors, n_meta)


def main() -> int:
    try:
        import faiss  # noqa: PLC0415 — import tardif pour message d'erreur clair
    except ImportError as exc:
        raise RuntimeError(
            "Paquet 'faiss-cpu' manquant ou non importable.\n"
            "  Recommandé (Mac / conda) : conda install -c pytorch faiss-cpu\n"
            "  Avec pip si erreur 'swig' : brew install swig cmake puis pip install faiss-cpu"
        ) from exc

    logger.info("Chargement : %s", EMBEDDINGS_PATH)
    vectors = _load_embeddings(EMBEDDINGS_PATH)
    n, dim = vectors.shape
    logger.info("Matrice : %s vecteurs × %s dimensions.", f"{n:,}", dim)

    _check_metadata_count(n)

    # Contrôle léger : avec normalisation L2, les normes doivent être ~1.
    sample = vectors[: min(1024, n)]
    norms = np.linalg.norm(sample, axis=1)
    if norms.min() < 0.99 or norms.max() > 1.01:
        logger.warning(
            "Normes L2 hors [0.99, 1.01] sur un échantillon — "
            "IndexFlatIP ne correspondra pas exactement au cosinus. "
            "Relancez build_embeddings avec normalize_embeddings=True."
        )

    index = faiss.IndexFlatIP(dim)
    index.add(vectors)
    if index.ntotal != n:
        raise RuntimeError(f"FAISS ntotal={index.ntotal} != {n}.")

    VECTOR_STORE_DIR.mkdir(parents=True, exist_ok=True)
    faiss.write_index(index, str(FAISS_INDEX_PATH))
    logger.info("Index sauvegardé : %s", FAISS_INDEX_PATH)

    manifest = {
        "index_type": "IndexFlatIP",
        "metric": "inner_product (vecteurs L2-normalisés ≈ cosinus)",
        "dimension": int(dim),
        "ntotal": int(n),
        "embedding_model": EMBEDDING_MODEL_NAME,
        "embeddings_file": EMBEDDINGS_PATH.name,
        "metadata_file": METADATA_PATH.name,
        "index_file": FAISS_INDEX_PATH.name,
    }
    MANIFEST_PATH.write_text(json.dumps(manifest, ensure_ascii=False, indent=2), encoding="utf-8")
    logger.info("Manifeste : %s", MANIFEST_PATH)

    logger.info("Terminé. Prochaine étape : python scripts/search_faiss.py \"votre requête\"")
    return 0


if __name__ == "__main__":
    try:
        sys.exit(main())
    except (FileNotFoundError, ValueError, RuntimeError) as e:
        logger.error("%s", e)
        sys.exit(1)
