"""
Raffinement — que faire des documents selon la branche décidée par l'évaluateur.

Le papier découpe les documents en segments ("decompose-then-recompose").
Sur BSARD, l'évaluation attend des identifiants d'articles entiers, donc le
filtrage opère au niveau de l'article. En juridique, tronquer un article
risquerait par ailleurs de supprimer une condition ou une réserve.

Sur le padding : les documents écartés ne disparaissent pas, ils sont replacés
en fin de liste par ordre de score décroissant. Le filtrage agit donc comme un
réordonnancement plutôt que comme une suppression — ce qui garantit que CRAG
ne peut pas faire chuter le rappel sous celui du retriever, tout en restant
comparable aux autres architectures mesurées à K=10.
"""

from typing import List, Tuple

from . import config


def _pad(kept: List, scored: List[Tuple], base_ranking: List) -> List:
    """
    Complète `kept` jusqu'à TOP_K.

    Ordre de priorité :
      1. les documents scorés écartés, du meilleur score au moins bon
      2. le reste du classement de base (candidats non scorés)
    """
    seen = set(kept)
    out = list(kept)

    # 1. documents scorés mais écartés — déjà triés par score décroissant
    for doc_id, _score in scored:
        if len(out) >= config.TOP_K:
            return out
        if doc_id not in seen:
            out.append(doc_id)
            seen.add(doc_id)

    # 2. reste du classement de base
    for doc_id in base_ranking:
        if len(out) >= config.TOP_K:
            break
        if doc_id not in seen:
            out.append(doc_id)
            seen.add(doc_id)

    return out


def refine(action: str, scored: List[Tuple], base_ranking: List) -> List:
    """
    Applique le raffinement selon la branche.

    action       : "correct" | "ambiguous" | "incorrect"
    scored       : [(doc_id, score), ...] trié par pertinence décroissante
    base_ranking : classement du retriever, utilisé en secours et pour le padding

    Retourne une liste de doc_id, de longueur TOP_K si PAD_TO_TOP_K est actif.
    """
    # --- Branche incorrect : garde-fou, on ne touche à rien ---
    if action == "incorrect":
        if config.INCORRECT_STRATEGY == "drop":
            return []
        return list(base_ranking)[:config.TOP_K]

    # --- Sélection selon le mode de raffinement ---
    if config.REFINEMENT_MODE == "none":
        kept = [d for d, _ in scored]

    elif config.REFINEMENT_MODE == "conservative":
        # On n'écarte qu'un document dont le score est vraiment bas.
        kept = [d for d, s in scored if s >= config.MIN_KEEP_SCORE]
        if not kept:                                  # jamais de liste vide
            kept = [d for d, _ in scored]

    elif config.REFINEMENT_MODE == "standard":
        n = (config.KEEP_TOP_N_CORRECT if action == "correct"
             else config.KEEP_TOP_N_AMBIGUOUS)
        kept = [d for d, _ in scored[:n]]

    else:
        raise ValueError(f"REFINEMENT_MODE inconnu : {config.REFINEMENT_MODE!r}")

    # --- Complétion jusqu'à TOP_K ---
    if config.PAD_TO_TOP_K and len(kept) < config.TOP_K:
        kept = _pad(kept, scored, base_ranking)

    return kept[:config.TOP_K]