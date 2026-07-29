"""
Lecture du fichier chunks (JSONL ou JSON concaténé), même logique que scripts/build_embeddings.py.
Garantit l’alignement texte ↔ métadonnées ↔ lignes d’embeddings / FAISS.
"""

from __future__ import annotations

import json
import logging
from pathlib import Path

logger = logging.getLogger(__name__)


def _iter_json_objects(raw: str, path: Path):
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
    """Retourne (textes, métadonnées sans 'text') dans l’ordre du fichier."""
    if not path.is_file():
        raise FileNotFoundError(f"Corpus introuvable : {path}")

    with path.open("r", encoding="utf-8") as f:
        content = f.read()

    texts: list[str] = []
    metadata_rows: list[dict] = []

    for obj_index, record in enumerate(_iter_json_objects(content, path), start=1):
        if not isinstance(record, dict):
            raise TypeError(
                f"Objet #{obj_index} : attendu un dict, reçu {type(record)!r}."
            )
        text = record.get("text")
        if text is None:
            logger.warning("Objet #%s : pas de 'text', ignoré.", obj_index)
            continue
        if not isinstance(text, str):
            raise TypeError(f"Objet #{obj_index} : 'text' doit être str.")
        text = text.strip()
        if not text:
            logger.warning("Objet #%s : texte vide, ignoré.", obj_index)
            continue
        meta = {k: v for k, v in record.items() if k != "text"}
        texts.append(text)
        metadata_rows.append(meta)

    if not texts:
        raise ValueError(f"Aucun chunk valide dans {path}.")
    return texts, metadata_rows
