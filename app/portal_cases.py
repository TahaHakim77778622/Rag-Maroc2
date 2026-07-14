"""
Alignement question ↔ procédure portail (CNIE : première demande, renouvellement, perte/vol).
"""

from __future__ import annotations

import re
import unicodedata
from typing import Any

_CNIE_CASES = ("premiere", "renouvellement", "perte_vol")


def _fold(s: str) -> str:
    return "".join(
        c for c in unicodedata.normalize("NFD", s or "") if unicodedata.category(c) != "Mn"
    ).lower()


def cnie_case_from_question(question: str) -> str | None:
    """Cas CNIE explicite dans la question, ou None si générique."""
    try:
        from app.phrase_context import analyze_phrase, subject_mentioned_but_not_target

        ctx = analyze_phrase(question)
        if ctx.primary_subject and ctx.primary_subject != "cnie":
            return None
        if subject_mentioned_but_not_target(question, "cnie"):
            return None
    except ImportError:
        pass

    q = _fold(question)
    if not any(k in q for k in ("cnie", "carte nationale", "identite electronique", "cine")):
        if "carte" in q and "identite" in q:
            pass
        elif "identite" not in q and "carte" not in q:
            return None

    if any(
        k in q
        for k in (
            "perte",
            "perdu",
            "perdue",
            "vol",
            "vole",
            "volee",
            "volee",
            "vole ",
            "declarer la perte",
            "declaration de perte",
            "duplicata",
            "carte perdue",
            "carte volee",
        )
    ):
        return "perte_vol"

    if any(k in q for k in ("renouvel", "renouvellement", "expiration", "expire", "expir")):
        return "renouvellement"

    if any(
        k in q
        for k in (
            "premiere demande",
            "première demande",
            "premier demande",
            "premiere fois",
            "première fois",
            "premiere cnie",
            "première cnie",
            "jamais eu",
            "sans carte",
        )
    ):
        return "premiere"

    return None


def _hit_blob(hit: dict[str, Any]) -> str:
    meta = hit.get("metadata") or {}
    parts = [
        str(hit.get("text") or ""),
        str(meta.get("label") or ""),
        str(meta.get("title") or ""),
        str(meta.get("chunk_id") or ""),
    ]
    return _fold(" ".join(parts))


def chunk_cnie_case(hit: dict[str, Any]) -> str | None:
    """Cas couvert par un extrait (chunk_id, label, texte)."""
    blob = _hit_blob(hit)
    cid = str((hit.get("metadata") or {}).get("chunk_id") or "")

    if "perte_vol" in cid or "perte_vol" in blob.replace(" ", "_"):
        return "perte_vol"
    if re.search(r"\b(perte|vol|vole|volee|duplicata|declaration de perte)\b", blob):
        if re.search(r"premiere demande|première demande|premiere fois", blob):
            if "perte" in blob or "vol" in blob:
                return "perte_vol"
        else:
            return "perte_vol"

    if "renouvellement" in cid or re.search(r"\brenouvel", blob):
        if "premiere demande" not in blob and "première demande" not in blob:
            return "renouvellement"
        if "perte" not in blob and " vol " not in f" {blob} ":
            return "renouvellement"

    if re.search(r"premiere demande|première demande|premiere fois|première fois", blob):
        return "premiere"
    if cid.startswith("curated::cnie_premiere"):
        return "premiere"
    if cid.startswith("cnie_procedure_premiere"):
        return "premiere"

    return None


def cnie_case_aligned(question: str, hit: dict[str, Any]) -> bool:
    """L'extrait correspond-il au cas posé par la question ?"""
    wanted = cnie_case_from_question(question)
    if not wanted:
        return True
    got = chunk_cnie_case(hit)
    if got is None:
        return False
    if wanted == got:
        return True
    # Renouvellement mentionne parfois perte en une phrase — pas suffisant pour une question perte/vol
    if wanted == "perte_vol" and got == "renouvellement":
        blob = _hit_blob(hit)
        if "perte" in blob or " vol " in f" {blob} " or "declaration" in blob:
            return True
    return False


def top_hits_match_cnie_case(question: str, hits: list[dict[str, Any]], n: int = 3) -> bool:
    wanted = cnie_case_from_question(question)
    if not wanted or not hits:
        return True
    for h in hits[:n]:
        if cnie_case_aligned(question, h):
            return True
    return False
