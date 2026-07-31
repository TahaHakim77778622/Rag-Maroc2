
from typing import List, Tuple, Dict
from . import config

_reranker = None


def get_reranker():
    """Charge le modèle de reranking une seule fois (singleton)."""
    global _reranker
    if _reranker is None:
        from FlagEmbedding import FlagReranker
        print(f"[reranker] Chargement du modèle {config.RERANKER_MODEL} ...")
        _reranker = FlagReranker(config.RERANKER_MODEL, use_fp16=True)
    return _reranker


def rerank(
    query: str,
    candidates: List[Tuple[str, float]],
    chunks_by_id: Dict[str, dict],
    top_k: int = None,
) -> List[Tuple[str, float]]:
    """
    Reclasse les candidats (chunk_id, rrf_score) via cross-encoder.
    Retourne [(chunk_id, rerank_score), ...] trié, tronqué à top_k.
    """
    top_k = top_k or config.TOP_K_FINAL

    if not config.RERANKER_ENABLED:
        return candidates[:top_k]

    reranker = get_reranker()

    valid_ids = [chunk_id for chunk_id, _ in candidates if chunk_id in chunks_by_id]
    pairs = [[query, chunks_by_id[chunk_id]["text"]] for chunk_id in valid_ids]

    if not pairs:
        return []

    scores = reranker.compute_score(pairs, normalize=True)
    if isinstance(scores, float):
        scores = [scores]

    reranked = sorted(zip(valid_ids, scores), key=lambda x: x[1], reverse=True)
    return reranked[:top_k]