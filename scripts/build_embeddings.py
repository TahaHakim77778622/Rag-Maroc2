"""
Génère les embeddings pour tous les chunks de final_chunks.jsonl.

Lecture JSONL (une ligne par objet) ou JSON concaténé (ex. premier objet multi-lignes)
→ encodage batch avec sentence-transformers →
sauvegarde alignée : vector_store/embeddings.npy + vector_store/metadata.jsonl

Usage (depuis la racine du projet) :
    python scripts/build_embeddings.py
    python scripts/build_embeddings.py --batch-size 64

Prérequis :
    pip install -r requirements.txt
"""

from __future__ import annotations

import argparse
import json
import logging
import sys
from pathlib import Path

import numpy as np
from sentence_transformers import SentenceTransformer

# ---------------------------------------------------------------------------
# Constantes (chemins relatifs à la racine du projet)
# ---------------------------------------------------------------------------
PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

try:
    from dotenv import load_dotenv

    load_dotenv(PROJECT_ROOT / ".env", override=False)
except ImportError:
    pass

from app.Rag_classique.config import DEFAULT_EMBEDDING_MODEL, embedding_encode_passage

FINAL_CHUNKS_PATH = PROJECT_ROOT / "data" / "processed" / "final_chunks.jsonl"
VECTOR_STORE_DIR = PROJECT_ROOT / "vector_store"
EMBEDDINGS_PATH = VECTOR_STORE_DIR / "embeddings.npy"
METADATA_PATH = VECTOR_STORE_DIR / "metadata.jsonl"

EMBEDDING_MODEL_NAME = DEFAULT_EMBEDDING_MODEL

BATCH_SIZE = 32
SHOW_PROGRESS_BAR = True

logging.basicConfig(
    level=logging.INFO,
    format="%(levelname)s: %(message)s",
)
logger = logging.getLogger(__name__)


def _iter_json_objects(raw: str, path: Path):
    """
    Parse un fichier contenant plusieurs objets JSON à la suite (JSONL strict
    OU objets multi-lignes « pretty-print » concaténés).

    Utilise raw_decode pour ne pas exiger « une ligne = un objet ».
    """
    decoder = json.JSONDecoder()
    idx = 0
    n = len(raw)
    obj_index = 0
    while idx < n:
        while idx < n and raw[idx].isspace():
            idx += 1
        if idx >= n:
            break
        try:
            record, end = decoder.raw_decode(raw, idx)
        except json.JSONDecodeError as exc:
            raise ValueError(
                f"Objet JSON #{obj_index + 1} invalide près du caractère {idx} dans {path}"
            ) from exc
        obj_index += 1
        yield record
        idx = end


def load_chunks_jsonl(path: Path) -> tuple[list[str], list[dict]]:
    """
    Lit le fichier chunks et retourne (textes, métadonnées) dans le même ordre.
    Chaque objet JSON doit avoir au minimum la clé 'text'.
    """
    if not path.is_file():
        raise FileNotFoundError(
            f"Fichier introuvable : {path}. Lancez les scripts de préparation du corpus."
        )

    with path.open("r", encoding="utf-8") as f:
        content = f.read()

    texts: list[str] = []
    metadata_rows: list[dict] = []

    for obj_index, record in enumerate(_iter_json_objects(content, path), start=1):
        if not isinstance(record, dict):
            raise TypeError(
                f"Objet #{obj_index} : attendu un objet JSON {{...}}, reçu {type(record)!r}."
            )

        text = record.get("text")
        if text is None:
            logger.warning("Objet #%s : clé 'text' absente, chunk ignoré.", obj_index)
            continue
        if not isinstance(text, str):
            raise TypeError(
                f"Objet #{obj_index} : 'text' doit être une chaîne, reçu {type(text)!r}."
            )

        text = text.strip()
        if not text:
            logger.warning("Objet #%s : texte vide après nettoyage, chunk ignoré.", obj_index)
            continue

        meta = {k: v for k, v in record.items() if k != "text"}
        texts.append(text)
        metadata_rows.append(meta)

    if not texts:
        raise ValueError(f"Aucun chunk valide lu depuis {path}.")

    return texts, metadata_rows


def main() -> int:
    parser = argparse.ArgumentParser(description="Génère embeddings.npy pour final_chunks.jsonl")
    parser.add_argument(
        "--batch-size",
        type=int,
        default=32,
        help="Taille des lots d'encodage (défaut: 32)",
    )
    args = parser.parse_args()
    batch_size = max(1, int(args.batch_size))

    logger.info("Modèle : %s", EMBEDDING_MODEL_NAME)
    logger.info("Batch size : %s", batch_size)
    logger.info("Chunks : %s", FINAL_CHUNKS_PATH)

    texts, metadata_rows = load_chunks_jsonl(FINAL_CHUNKS_PATH)
    n = len(texts)
    logger.info("%s chunks à encoder.", n)

    if len(metadata_rows) != n:
        raise RuntimeError("Incohérence interne : textes et métadonnées ne sont pas alignés.")

    try:
        model = SentenceTransformer(EMBEDDING_MODEL_NAME)
    except Exception as exc:
        logger.exception("Échec du chargement du modèle sentence-transformers.")
        raise RuntimeError(
            "Vérifiez la connexion réseau (premier téléchargement) et l'installation : "
            "pip install sentence-transformers torch"
        ) from exc

    passage_texts = [embedding_encode_passage(t, EMBEDDING_MODEL_NAME) for t in texts]

    try:
        embeddings = model.encode(
            passage_texts,
            batch_size=batch_size,
            show_progress_bar=SHOW_PROGRESS_BAR,
            convert_to_numpy=True,
            normalize_embeddings=True,
        )
    except Exception as exc:
        logger.exception("Erreur pendant l'encodage.")
        raise

    if not isinstance(embeddings, np.ndarray):
        embeddings = np.asarray(embeddings)

    embeddings = embeddings.astype(np.float32, copy=False)
    if embeddings.shape[0] != n:
        raise RuntimeError(
            f"Nombre de lignes embeddings ({embeddings.shape[0]}) != nombre de chunks ({n})."
        )

    VECTOR_STORE_DIR.mkdir(parents=True, exist_ok=True)
    np.save(EMBEDDINGS_PATH, embeddings)
    logger.info("Embeddings sauvegardés : %s (shape=%s)", EMBEDDINGS_PATH, embeddings.shape)

    with METADATA_PATH.open("w", encoding="utf-8") as out:
        for meta in metadata_rows:
            out.write(json.dumps(meta, ensure_ascii=False) + "\n")
    logger.info("Métadonnées sauvegardées : %s (%s lignes)", METADATA_PATH, n)

    logger.info("Terminé. Prochaine étape : python scripts/build_faiss.py")
    return 0


if __name__ == "__main__":
    try:
        sys.exit(main())
    except (FileNotFoundError, ValueError, RuntimeError) as e:
        logger.error("%s", e)
        sys.exit(1)
