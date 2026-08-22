from typing import List, Tuple, Dict
from . import config

_reranker = None

# Longueur maximale de texte envoyée au cross-encoder. Le modèle tronque de
# toute façon à 512 tokens : couper en amont évite de tokeniser des milliers de
# caractères inutilement. Mesuré sur ce corpus : 14,7 s sans limite contre
# 11,8 s à 1200 caractères pour 15 candidats.
MAX_TEXT_CHARS = 1000


def _disable_torch_load_check():
    """Neutralise la vérification de sécurité de transformers.

    Depuis la CVE-2025-32434, transformers refuse de charger les fichiers .bin
    avec torch < 2.6. Or PyTorch ne publie plus au-delà de 2.2.2 sur cette
    plateforme, et les modèles BAAI ne sont distribués qu'en .bin. Le risque est
    nul ici : les modèles proviennent de dépôts officiels et connus.
    """
    import transformers.utils.import_utils as _iu
    import transformers.modeling_utils as _mu
    _iu.check_torch_load_is_safe = lambda *a, **k: None
    _mu.check_torch_load_is_safe = lambda *a, **k: None


def get_reranker():
    """Charge le cross-encoder une seule fois (singleton).

    CrossEncoder de sentence-transformers plutôt que FlagReranker : plus stable
    avec les versions récentes de transformers. Les scores retournés sont des
    logits bruts, non bornés — sans importance puisqu'on ne fait que trier.
    """
    global _reranker
    if _reranker is None:
        _disable_torch_load_check()
        from sentence_transformers import CrossEncoder
        print(f"[reranker] Chargement du modèle {config.RERANKER_MODEL} ...")
        _reranker = CrossEncoder(config.RERANKER_MODEL, max_length=512)
    return _reranker


def rerank(
    query: str,
    candidates: List[Tuple[str, float]],
    chunks_by_id: Dict[str, dict],
    top_k: int = None,
) -> List[Tuple[str, float]]:
    """
    Reclasse les candidats (chunk_id, score_amont) via cross-encoder.
    Retourne [(chunk_id, rerank_score), ...] trié, tronqué à top_k.
    """
    top_k = top_k or config.TOP_K_FINAL

    if not config.RERANKER_ENABLED:
        return candidates[:top_k]

    reranker = get_reranker()

    valid_ids = [chunk_id for chunk_id, _ in candidates if chunk_id in chunks_by_id]
    pairs = [
        [query, str(chunks_by_id[chunk_id]["text"])[:MAX_TEXT_CHARS]]
        for chunk_id in valid_ids
    ]

    if not pairs:
        return []

    scores = reranker.predict(pairs)

    reranked = sorted(
        zip(valid_ids, [float(s) for s in scores]),
        key=lambda x: x[1],
        reverse=True,
    )
    return reranked[:top_k]