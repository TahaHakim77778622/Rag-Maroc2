"""
Recherche sémantique top-k : encode la requête avec le même modèle que les chunks,
interroge l'index FAISS, affiche score + métadonnées (+ texte si présent dans meta).

Usage (depuis la racine du projet) :
    python scripts/search_faiss.py "CNIE délais de délivrance"
    python scripts/search_faiss.py "licence crédits" -k 8
"""

from __future__ import annotations

import argparse
import json
import logging
import sys
from pathlib import Path

import numpy as np

PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

from app.config import embedding_encode_query

VECTOR_STORE_DIR = PROJECT_ROOT / "vector_store"
FAISS_INDEX_PATH = VECTOR_STORE_DIR / "faiss.index"
METADATA_PATH = VECTOR_STORE_DIR / "metadata.jsonl"
MANIFEST_PATH = VECTOR_STORE_DIR / "faiss_manifest.json"

logging.basicConfig(level=logging.INFO, format="%(levelname)s: %(message)s")
logger = logging.getLogger(__name__)


def load_manifest() -> dict:
    if not MANIFEST_PATH.is_file():
        raise FileNotFoundError(
            f"Manifeste introuvable : {MANIFEST_PATH}. Lancez : python scripts/build_faiss.py"
        )
    return json.loads(MANIFEST_PATH.read_text(encoding="utf-8"))


def load_metadata_rows() -> list[dict]:
    if not METADATA_PATH.is_file():
        raise FileNotFoundError(f"Métadonnées introuvables : {METADATA_PATH}")
    rows: list[dict] = []
    with METADATA_PATH.open(encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            rows.append(json.loads(line))
    return rows


def main() -> int:
    parser = argparse.ArgumentParser(description="Recherche FAISS top-k sur le corpus indexé.")
    parser.add_argument("query", type=str, help="Question ou mots-clés")
    parser.add_argument("-k", type=int, default=5, help="Nombre de résultats (défaut : 5)")
    args = parser.parse_args()

    try:
        import faiss  # noqa: PLC0415
        from sentence_transformers import SentenceTransformer  # noqa: PLC0415
    except ImportError as exc:
        logger.error(
            "Dépendances manquantes (faiss-cpu, sentence-transformers). "
            "pip install -r requirements.txt"
        )
        raise SystemExit(1) from exc

    manifest = load_manifest()
    model_name = manifest.get("embedding_model")
    if not model_name:
        raise RuntimeError("Champ 'embedding_model' absent du manifeste — régénérez l'index.")

    dim_expected = int(manifest["dimension"])
    meta_rows = load_metadata_rows()
    if len(meta_rows) != int(manifest["ntotal"]):
        logger.warning(
            "Manifeste ntotal=%s mais %s lignes metadata — risque d'indices incorrects.",
            manifest["ntotal"],
            len(meta_rows),
        )

    logger.info("Chargement index : %s", FAISS_INDEX_PATH)
    index = faiss.read_index(str(FAISS_INDEX_PATH))
    if index.d != dim_expected:
        raise ValueError(f"Dimension index ({index.d}) != manifeste ({dim_expected}).")

    logger.info("Modèle requête : %s", model_name)
    model = SentenceTransformer(model_name)
    q = model.encode(
        [embedding_encode_query(args.query, model_name)],
        convert_to_numpy=True,
        normalize_embeddings=True,
    ).astype(np.float32)
    if q.shape != (1, dim_expected):
        raise ValueError(f"Vecteur requête shape {q.shape}, attendu (1, {dim_expected}).")

    scores, indices = index.search(q, min(args.k, index.ntotal))

    print()
    for rank, (idx, score) in enumerate(zip(indices[0], scores[0]), start=1):
        idx = int(idx)
        if idx < 0:
            continue
        meta = meta_rows[idx] if idx < len(meta_rows) else {}
        title = meta.get("title", "")
        label = meta.get("label", "")
        chunk_id = meta.get("chunk_id", "")
        print(f"--- #{rank}  score={score:.4f}  chunk_id={chunk_id}")
        print(f"    title={title!r}  label={label!r}")
        if meta.get("source_url"):
            print(f"    url={meta['source_url']}")
        text = meta.get("text")
        if text:
            preview = text.replace("\n", " ")[:400]
            print(f"    texte: {preview}{'…' if len(text) > 400 else ''}")
        else:
            print("    (pas de champ 'text' dans metadata — normal si build_embeddings l'a retiré)")
        print()

    return 0


if __name__ == "__main__":
    try:
        sys.exit(main())
    except (FileNotFoundError, ValueError, RuntimeError, json.JSONDecodeError) as e:
        logger.error("%s", e)
        sys.exit(1)
