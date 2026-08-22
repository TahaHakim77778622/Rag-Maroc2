"""
Évaluateur de pertinence — le cœur de CRAG.

Décide, pour une question donnée, si les documents récupérés sont globalement
pertinents. Trois issues : correct / ambiguous / incorrect.
"""

from typing import List, Tuple, Dict

from . import config

_reranker = None


def get_reranker():
    """Charge le cross-encoder une seule fois (singleton)."""
    global _reranker
    if _reranker is None:
        from sentence_transformers import CrossEncoder
        print(f"[crag] Chargement du reranker {config.RERANKER_MODEL} ...")
        _reranker = CrossEncoder(config.RERANKER_MODEL, max_length=512)
    return _reranker


def score_candidates(query: str, candidate_ids: List, texts_by_id: Dict) -> List[Tuple]:
    """
    Score chaque candidat par rapport à la question.
    Retourne [(doc_id, score), ...] trié par score décroissant, scores dans [0,1].
    """
    valid_ids = [cid for cid in candidate_ids if cid in texts_by_id]
    if not valid_ids:
        return []

    reranker = get_reranker()
    pairs = [[query, str(texts_by_id[cid])] for cid in valid_ids]

    if config.NORMALIZE_SCORES:
        import torch
        scores = reranker.predict(pairs, activation_fct=torch.nn.Sigmoid())
    else:
        scores = reranker.predict(pairs)

    return sorted(zip(valid_ids, [float(s) for s in scores]),
                  key=lambda x: x[1], reverse=True)


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