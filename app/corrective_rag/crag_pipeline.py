"""
Orchestration CRAG : retriever -> évaluateur -> raffinement.

Le retriever est injecté en paramètre (fonction query -> liste de doc_id),
pour que le même code fonctionne avec BM25, Hybrid RAG ou tout autre retriever.
"""

from typing import Callable, Dict, List

from . import config
from .evaluator import evaluate
from .refiner import refine


def crag_retrieve(
    query: str,
    retriever_fn: Callable[[str, int], List],
    texts_by_id: Dict,
):
    """
    query        : la question
    retriever_fn : fonction (query, n) -> liste de doc_id
    texts_by_id  : mapping doc_id -> texte

    Retourne (final_ranking, action, confidence).
    """
    candidates = retriever_fn(query, config.CANDIDATE_POOL)
    action, scored, confidence = evaluate(query, candidates, texts_by_id)
    final = refine(action, scored, candidates)
    return final, action, confidence