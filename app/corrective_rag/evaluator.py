"""
Évaluateur de pertinence — le cœur de CRAG.

Le papier original entraîne un T5 dédié. On réutilise ici le cross-encoder
bge-reranker-v2-m3 : déterministe, donc résultats reproductibles et testables
par bootstrap, comme pour GraphRAG.
"""

from typing import List, Tuple, Dict

from . import config

_reranker = None


def get_reranker():
    """Charge le cross-encoder une seule fois (singleton)."""
    global _reranker
    if _reranker is None:
        from FlagEmbedding import FlagReranker
        print(f"[crag] Chargement du reranker {config.RERANKER_MODEL} ...")
        _reranker = FlagReranker(config.RERANKER_MODEL, use_fp16=False)
    return _reranker


def score_candidates(query: str, candidate_ids: List, texts_by_id: Dict) -> List[Tuple]:
    """
    Score chaque candidat par rapport à la question.
    Retourne [(doc_id, score), ...] trié par score décroissant.
    """
    valid_ids = [cid for cid in candidate_ids if cid in texts_by_id]
    if not valid_ids:
        return []

    reranker = get_reranker()
    pairs = [[query, str(texts_by_id[cid])] for cid in valid_ids]
    scores = reranker.compute_score(pairs, normalize=config.NORMALIZE_SCORES)
    if isinstance(scores, float):
        scores = [scores]

    return sorted(zip(valid_ids, scores), key=lambda x: x[1], reverse=True)


def aggregate_confidence(scored: List[Tuple]) -> float:
    """Résume les scores individuels en un seul indice de confiance."""
    if not scored:
        return 0.0

    scores = [s for _, s in scored]
    mode = config.SCORE_AGGREGATION

    if mode == "max":
        return max(scores)
    if mode == "mean":
        return sum(scores) / len(scores)
    if mode == "top3":
        top = sorted(scores, reverse=True)[:3]
        return sum(top) / len(top)

    raise ValueError(f"SCORE_AGGREGATION inconnu : {mode!r}")


def decide(confidence: float) -> str:
    """Applique les deux seuils pour choisir la branche."""
    if confidence >= config.TAU_CORRECT:
        return "correct"
    if confidence <= config.TAU_INCORRECT:
        return "incorrect"
    return "ambiguous"


def evaluate(query: str, candidate_ids: List, texts_by_id: Dict):
    """
    Point d'entrée.
    Retourne (action, scored, confidence).
    """
    scored = score_candidates(query, candidate_ids, texts_by_id)
    confidence = aggregate_confidence(scored)
    return decide(confidence), scored, confidence