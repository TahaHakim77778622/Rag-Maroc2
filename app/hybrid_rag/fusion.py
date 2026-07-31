
from typing import List, Tuple, Dict
from . import config


def reciprocal_rank_fusion(
    faiss_results: List[Tuple[str, float]],
    bm25_results: List[Tuple[str, float]],
    k: int = None,
    top_k: int = None,
) -> List[Tuple[str, float]]:
    """
    Fusionne deux listes classées (chunk_id, score) via RRF.
    RRF_score(d) = somme( 1 / (k + rank(d)) ) sur chaque système où d apparaît.
    """
    k = k or config.RRF_K
    top_k = top_k or config.TOP_K_HYBRID

    rrf_scores: Dict[str, float] = {}

    for rank, (chunk_id, _) in enumerate(faiss_results, start=1):
        rrf_scores[chunk_id] = rrf_scores.get(chunk_id, 0.0) + 1.0 / (k + rank)

    for rank, (chunk_id, _) in enumerate(bm25_results, start=1):
        rrf_scores[chunk_id] = rrf_scores.get(chunk_id, 0.0) + 1.0 / (k + rank)

    fused = sorted(rrf_scores.items(), key=lambda x: x[1], reverse=True)
    return fused[:top_k]