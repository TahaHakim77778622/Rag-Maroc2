"""
Politique globale : comprendre la phrase → chercher dans le dataset → web seulement si insuffisant.

Point d'entrée unique pour préparer les extraits locaux et décider du fallback web.
"""

from __future__ import annotations

import logging
from typing import Any

logger = logging.getLogger(__name__)


def _filter_by_dataset_domains(
    question: str, hits: list[dict[str, Any]]
) -> list[dict[str, Any]]:
    """Filtre hors-sujet selon tous les domaines du dataset (registre global)."""
    try:
        from app.dataset_registry import filter_hits_by_domains

        return filter_hits_by_domains(question, hits)
    except ImportError:
        return hits


def _filter_by_phrase_context(question: str, hits: list[dict[str, Any]]) -> list[dict[str, Any]]:
    """Retire les extraits d'un autre document quand le sens de la phrase est clair."""
    try:
        from app.phrase_context import analyze_phrase, hit_matches_primary_subject

        ctx = analyze_phrase(question)
        if not ctx.primary_subject:
            return hits
        kept = [h for h in hits if hit_matches_primary_subject(question, h)]
        return kept if kept else hits
    except ImportError:
        return hits


def _filter_education_noise(question: str, hits: list[dict[str, Any]]) -> list[dict[str, Any]]:
    try:
        from app.query_understanding import analyze_query

        qa = analyze_query(question)
        if not (qa.education_doctorate_intent or qa.education_master_intent):
            return hits
    except ImportError:
        return hits
    out = []
    for h in hits:
        txt = (h.get("text") or "").lower()
        if any(
            x in txt
            for x in (
                "marchés publics",
                "marches publics",
                "appel d'offres",
                "soumissionnaire",
            )
        ) and "master" not in txt and "doctorat" not in txt:
            continue
        out.append(h)
    return out if out else hits


def prepare_local_hits(
    question: str,
    hits: list[dict[str, Any]],
    *,
    top_k: int = 8,
) -> list[dict[str, Any]]:
    """
    Prépare les meilleurs extraits du dataset selon le contexte de la phrase
    (sujet principal, négations, domaine travail/CNIE/passeport/Watiqa, etc.).
    """
    if not hits:
        hits = []

    out = list(hits)
    try:
        from app.text_sanitize import sanitize_hit, text_is_usable_for_llm

        out = [sanitize_hit(h) for h in out if text_is_usable_for_llm(str(h.get("text") or ""))]
    except ImportError:
        pass
    out = _filter_by_dataset_domains(question, out)
    out = _filter_by_phrase_context(question, out)
    out = _filter_education_noise(question, out)

    try:
        from app.labor_corpus import is_labor_code_question, merge_labor_hits

        if is_labor_code_question(question):
            out = merge_labor_hits(question, out, top_k=top_k)
    except ImportError:
        pass

    try:
        from app.phrase_context import analyze_phrase

        ctx = analyze_phrase(question)
        portal = ctx.primary_subject
        if portal in (None, "passeport") and "passeport" in ctx.subject_scores:
            if ctx.subject_scores.get("passeport", 0) > 0:
                portal = "passeport"
    except ImportError:
        portal = None

    try:
        from app.web_fallback import _portal_intent

        portal = portal or _portal_intent(question)
    except ImportError:
        pass

    if portal == "passeport":
        try:
            from app.passeport_corpus import merge_passeport_hits

            out = merge_passeport_hits(question, out, top_k=top_k)
        except ImportError:
            pass
    elif portal == "cnie":
        try:
            from app.portal_cases import cnie_case_from_question, top_hits_match_cnie_case
            from app.web_fallback import _curated_cnie_hits, _web_hit_is_passeport_not_cnie

            cleaned = [h for h in out if not _web_hit_is_passeport_not_cnie(h)]
            if cleaned:
                out = cleaned
            if cnie_case_from_question(question) and not top_hits_match_cnie_case(question, out):
                curated = _curated_cnie_hits(question)
                cid = curated[0]["metadata"]["chunk_id"]
                out = [h for h in out if (h.get("metadata") or {}).get("chunk_id") != cid]
                out = (curated + out)[:top_k]
        except ImportError:
            pass
    elif portal == "watiqa":
        try:
            from app.web_fallback import (
                _curated_watiqa_hits,
                _prioritize_watiqa_hits,
                _web_hit_is_cnie_not_watiqa,
            )

            cleaned = [h for h in out if not _web_hit_is_cnie_not_watiqa(h)]
            if cleaned:
                out = cleaned
            out = _prioritize_watiqa_hits(out)
            if not out:
                out = _curated_watiqa_hits()
        except ImportError:
            pass

    return out[:top_k]


def local_corpus_covers(question: str, hits: list[dict[str, Any]], *, top_k: int = 8) -> bool:
    """Le dataset (après préparation contextuelle) suffit-il pour répondre ?"""
    from app.corpus_coverage import corpus_covers_question

    prepared = prepare_local_hits(question, hits, top_k=top_k)
    return corpus_covers_question(question, prepared)


def should_use_web_after_corpus(
    question: str, hits: list[dict[str, Any]], *, top_k: int = 8
) -> tuple[bool, list[dict[str, Any]]]:
    """
    Retourne (need_web, hits_préparés).
    need_web=True seulement si le corpus local ne couvre pas la question.
    """
    try:
        from app.web_fallback import is_morocco_admin_query
    except ImportError:

        def is_morocco_admin_query(_q: str) -> bool:
            return True

    prepared = prepare_local_hits(question, hits, top_k=top_k)
    if not is_morocco_admin_query(question):
        return False, prepared

    try:
        from app.web_fallback import (
            _is_passeport_fee_question,
            passeport_fee_hits_substantive,
        )

        if _is_passeport_fee_question(question) and not passeport_fee_hits_substantive(
            question, prepared
        ):
            logger.info(
                "Web fallback forcé (timbre/e-timbre passeport, extraits BO non pertinents)"
            )
            return True, prepared
    except ImportError:
        pass

    from app.corpus_coverage import corpus_covers_question, explain_coverage

    if corpus_covers_question(question, prepared):
        logger.info("Dataset suffisant (corpus_first) — %s", explain_coverage(question, prepared))
        return False, prepared

    logger.info(
        "Dataset insuffisant → web — %s",
        explain_coverage(question, prepared),
    )
    return True, prepared
