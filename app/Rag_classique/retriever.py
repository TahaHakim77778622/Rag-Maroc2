"""
Charge l’index FAISS + le manifeste, relit les textes depuis final_chunks.jsonl
(dans le même ordre que lors du calcul des embeddings) et expose search(query, k).
"""

from __future__ import annotations

import json
import logging
from pathlib import Path
from typing import Any

import numpy as np

from app.Rag_classique.config import (
    DEFAULT_EMBEDDING_MODEL,
    FAISS_INDEX_PATH,
    FINAL_CHUNKS_PATH,
    MANIFEST_PATH,
    embedding_encode_query,
)
from app.Rag_classique.corpus_io import load_chunks_jsonl

logger = logging.getLogger(__name__)


class Retriever:
    def __init__(
        self,
        index_path: Path | None = None,
        manifest_path: Path | None = None,
        chunks_path: Path | None = None,
    ) -> None:
        self._index_path = index_path or FAISS_INDEX_PATH
        self._manifest_path = manifest_path or MANIFEST_PATH
        self._chunks_path = chunks_path or FINAL_CHUNKS_PATH

        self._manifest = json.loads(self._manifest_path.read_text(encoding="utf-8"))
        self._model_name = self._manifest.get("embedding_model") or DEFAULT_EMBEDDING_MODEL
        self._dim = int(self._manifest["dimension"])
        self._ntotal = int(self._manifest["ntotal"])

        try:
            import faiss  # noqa: PLC0415
        except ImportError as exc:
            raise RuntimeError(
                "faiss-cpu manquant. conda install -c pytorch faiss-cpu"
            ) from exc

        self._index = faiss.read_index(str(self._index_path))
        if self._index.ntotal != self._ntotal:
            logger.warning(
                "FAISS ntotal=%s ≠ manifeste ntotal=%s",
                self._index.ntotal,
                self._ntotal,
            )

        logger.info("Chargement des textes : %s", self._chunks_path)
        self._texts, self._metas = load_chunks_jsonl(self._chunks_path)
        self._usable_n = min(len(self._texts), int(self._index.ntotal))
        if len(self._texts) != self._index.ntotal:
            logger.warning(
                "%s chunks dans le JSONL mais l'index contient %s vecteurs. "
                "Le RAG continue avec %s éléments alignés; regénérez embeddings + FAISS.",
                len(self._texts),
                self._index.ntotal,
                self._usable_n,
            )
        if self._usable_n <= 0:
            raise ValueError("Aucun vecteur/text disponible pour la recherche.")

        from sentence_transformers import SentenceTransformer  # noqa: PLC0415

        logger.info("Modèle requêtes : %s", self._model_name)
        self._model = SentenceTransformer(self._model_name)

    @property
    def embedding_model(self) -> str:
        return self._model_name

    @property
    def n_vectors(self) -> int:
        return int(self._usable_n)

    def search(self, query: str, k: int = 5) -> list[dict[str, Any]]:
        """Top-k : score (produit scalaire = cosinus si vecteurs normalisés), métadonnées, texte."""
        if self._usable_n <= 0:
            return []

        q = self._model.encode(
            [embedding_encode_query(query, self._model_name)],
            convert_to_numpy=True,
            normalize_embeddings=True,
        ).astype(np.float32)
        if q.shape != (1, self._dim):
            raise ValueError(f"Vecteur requête {q.shape}, attendu (1, {self._dim}).")

        k_eff = min(max(1, k), self._usable_n)
        scores, indices = self._index.search(q, k_eff)

        hits: list[dict[str, Any]] = []
        for idx, score in zip(indices[0], scores[0]):
            i = int(idx)
            if i < 0 or i >= self._usable_n:
                continue
            hits.append(
                {
                    "index": i,
                    "score": float(score),
                    "metadata": dict(self._metas[i]),
                    "text": self._texts[i],
                }
            )
        return hits
