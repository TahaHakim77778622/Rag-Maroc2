"""
Réécriture de requête par LLM (~20 % des questions) — complète l'analyse par règles (~80 %).

Activé avec QUERY_REWRITE_LLM=1 et un LLM réel (pas LLM_MOCK).
"""

from __future__ import annotations

import json
import logging
import os
import re
from typing import Any

from app.llm_client import LLMClient, MockLLMClient
from app.query_understanding import QueryAnalysis, _history_user_text, _normalize, _tokens

logger = logging.getLogger(__name__)


def _bool_env(name: str, default: bool = False) -> bool:
    raw = os.environ.get(name)
    if raw is None:
        return default
    return raw.strip().lower() in ("1", "true", "yes", "on")


def _float_env(name: str, default: float) -> float:
    try:
        return float(os.environ.get(name, str(default)))
    except ValueError:
        return default


def ambiguity_score(analysis: QueryAnalysis, question: str) -> float:
    """
    Score 0–1 : au-dessus du seuil → passage LLM (cible ~20 % du trafic réel).
    Questions à sujet unique et objectif procedure/info clair restent sous le seuil.
    """
    q = (question or "").strip()
    toks = _tokens(q)
    s = 0.0

    if analysis.use_conversation_history:
        s += 0.38
    if analysis.goal == "transition":
        s += 0.42
    elif analysis.goal == "general":
        s += 0.22
    if len(analysis.topics) >= 2:
        s += 0.28
    if not analysis.topics and len(toks) >= 5:
        s += 0.32
    if len(toks) >= 14:
        s += 0.12
    low = _normalize(q)
    if any(x in low for x in ("je veux", "moi je", "passage", "plutot", "plutôt", "pas le", "pas la")):
        s += 0.15
    if "master" in low and "doctorat" in low:
        s += 0.2

    # Pénalités : requêtes « classiques » → rester sur corpus seul (~80 %)
    if len(analysis.topics) == 1 and analysis.goal in ("procedure", "info"):
        s -= 0.45
    if analysis.topics and analysis.topics[0] in (
        "cnie",
        "passeport",
        "watiqa",
        "labor",
        "construction",
    ) and analysis.goal == "procedure" and len(toks) <= 14:
        s -= 0.5
    if _has_specific_bo_ref(low):
        s -= 0.35

    return max(0.0, min(1.0, s))


def _has_specific_bo_ref(low: str) -> bool:
    return bool(
        re.search(r"\b(?:bo|bulletin)\s*n[°o]?\s*\d", low)
        or re.search(r"\barticle\s+\d+", low)
    )


def should_use_llm_query_rewrite(analysis: QueryAnalysis, question: str) -> bool:
    if not _bool_env("QUERY_REWRITE_LLM", False):
        return False
    if _bool_env("LLM_MOCK", False):
        return False
    threshold = _float_env("QUERY_REWRITE_LLM_THRESHOLD", 0.52)
    return ambiguity_score(analysis, question) >= threshold


def _build_rewrite_prompt(
    question: str,
    analysis: QueryAnalysis,
    history: list[dict[str, Any]] | None,
) -> str:
    hist = _history_user_text(history, max_turns=4)
    hist_block = f"\nHistorique récent:\n{hist}\n" if hist else ""
    topics = ", ".join(analysis.topics) if analysis.topics else "(aucun détecté)"
    return f"""Tu prépares une recherche dans un corpus juridique et administratif MAROCAIN (Bulletin officiel, portails .ma).

Question utilisateur:
{question}

Question reformulée (règles):
{analysis.resolved}

Objectif détecté: {analysis.goal}
Sujets détectés: {topics}
{hist_block}
Consignes:
- Produis une requête de recherche vectorielle en français (1–2 phrases, termes présents dans des textes officiels marocains).
- Ajoute 6 à 12 mots-clés séparés par des espaces dans "keywords".
- Dans "exclude", liste les thèmes à NE PAS confondre (ex. "normes pedagogiques master" si la question vise le doctorat).
- Ne invente pas de numéros d'articles ou de BO non mentionnés par l'utilisateur.
- Réponds UNIQUEMENT avec un objet JSON valide, sans markdown:

{{"search_query": "...", "keywords": "mot1 mot2 mot3", "exclude": "theme1 theme2"}}"""


def _parse_rewrite_response(text: str) -> dict[str, str] | None:
    raw = (text or "").strip()
    if not raw:
        return None
    # Extraire bloc JSON si le modèle ajoute du texte autour
    m = re.search(r"\{[^{}]*\}", raw, re.DOTALL)
    if m:
        raw = m.group(0)
    try:
        data = json.loads(raw)
    except json.JSONDecodeError:
        return None
    if not isinstance(data, dict):
        return None
    out: dict[str, str] = {}
    for key in ("search_query", "keywords", "exclude"):
        val = data.get(key)
        if val is None:
            continue
        if isinstance(val, list):
            out[key] = " ".join(str(x) for x in val)
        else:
            out[key] = str(val).strip()
    return out if out.get("search_query") or out.get("keywords") else None


def rewrite_query_with_llm(
    llm: LLMClient,
    question: str,
    analysis: QueryAnalysis,
    *,
    history: list[dict[str, Any]] | None = None,
) -> dict[str, str] | None:
    if isinstance(llm, MockLLMClient):
        return None
    prompt = _build_rewrite_prompt(question, analysis, history)
    try:
        raw = llm.complete(prompt)
    except Exception:
        logger.warning("Réécriture LLM échouée", exc_info=True)
        return None
    parsed = _parse_rewrite_response(raw)
    if not parsed:
        logger.warning("Réécriture LLM : JSON non parsable")
    return parsed


def enrich_analysis_with_llm_rewrite(
    llm: LLMClient,
    analysis: QueryAnalysis,
    question: str,
    *,
    history: list[dict[str, Any]] | None = None,
) -> QueryAnalysis:
    """
    Enrichit l'analyse corpus (~80 %) avec une réécriture LLM si ambiguïté élevée (~20 %).
    """
    if not should_use_llm_query_rewrite(analysis, question):
        return analysis

    parsed = rewrite_query_with_llm(llm, question, analysis, history=history)
    if not parsed:
        return analysis

    hints = list(analysis.retrieval_hints)
    sq = parsed.get("search_query", "").strip()
    if sq:
        hints.insert(0, sq)
        analysis.resolved = sq
    kw = parsed.get("keywords", "").strip()
    if kw:
        hints.append(kw)
    ex = parsed.get("exclude", "").strip()
    if ex:
        hints.append(f"exclure: {ex}")

    seen: set[str] = set()
    dedup: list[str] = []
    for h in hints:
        hh = " ".join(h.split())
        if hh and hh not in seen:
            seen.add(hh)
            dedup.append(hh)
    analysis.retrieval_hints = dedup
    analysis.retrieval_path = "corpus+llm"
    logger.info(
        "Réécriture LLM activée (score=%.2f) pour: %s",
        ambiguity_score(analysis, question),
        question[:80],
    )
    return analysis
