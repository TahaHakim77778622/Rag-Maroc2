"""
Chunks et injection passeport (consulat.ma) — priorité dataset avant web.
"""

from __future__ import annotations

from typing import Any

_CONSULAT_MINEUR_SANS_CNIE = "consulat_passeport_biometrique_presentation_1"
_CONSULAT_PIECES_MINEURS = "consulat_passeport_biometrique_pieces_tutelle_mineurs_1"
_CONSULAT_PERTE_VOL = (
    "consulat_passeport_biometrique_presentation_1",
    "consulat_passeport_biometrique_formulaires_1",
    "consulat_passeport_mre_maroc_procedure_1",
    "consulat_passeport_retrait_pieces_1",
    "consulat_passeport_biometrique_pieces_majeur_1",
)
_CONSULAT_RENOUVELLEMENT = (
    "consulat_passeport_biometrique_pieces_majeur_1",
    "consulat_passeport_biometrique_presentation_1",
)
_CONSULAT_DEFAULT = (
    "consulat_passeport_biometrique_pieces_majeur_1",
    "consulat_passeport_biometrique_presentation_1",
)


def is_passeport_primary_question(question: str) -> bool:
    try:
        from app.phrase_context import analyze_phrase

        return analyze_phrase(question).primary_subject == "passeport"
    except ImportError:
        q = (question or "").lower()
        return "passeport" in q and "cnie" not in q.replace("sans cnie", "")


def _load_chunks_by_id() -> dict[str, dict[str, Any]]:
    from pathlib import Path
    import json

    path = Path(__file__).resolve().parent.parent / "data" / "processed" / "final_chunks.jsonl"
    if not path.is_file():
        return {}
    out: dict[str, dict[str, Any]] = {}
    for line in path.read_text(encoding="utf-8").splitlines():
        line = line.strip()
        if not line:
            continue
        try:
            row = json.loads(line)
        except json.JSONDecodeError:
            continue
        cid = str(row.get("chunk_id") or "")
        if cid.startswith("consulat_passeport_") or cid.startswith("curated::passeport"):
            out[cid] = row
    return out


def _row_to_hit(row: dict[str, Any], *, score: float = 0.98) -> dict[str, Any]:
    return {
        "index": -1,
        "score": score,
        "rerank_score": score,
        "metadata": {
            "chunk_id": row.get("chunk_id"),
            "doc_id": row.get("doc_id"),
            "title": row.get("title"),
            "label": row.get("label"),
            "source_type": row.get("source_type", "admin"),
            "source_url": row.get("source_url"),
            "category": row.get("category") or "passeport",
            "source_org": row.get("source_org"),
        },
        "text": str(row.get("text") or ""),
    }


def curated_passeport_hits_for(question: str) -> list[dict[str, Any]]:
    from app.web_fallback import _curated_passeport_hits

    return _curated_passeport_hits(question)


def best_passeport_hits_for(question: str) -> list[dict[str, Any]]:
    """Meilleur extrait dataset ou curated pour la question."""
    try:
        from app.phrase_context import analyze_phrase

        ctx = analyze_phrase(question)
        mineur = ctx.age_hint == "mineur"
        sans_cnie = "cnie" in ctx.negated_subjects
    except ImportError:
        q = (question or "").lower()
        mineur = "mineur" in q
        sans_cnie = "sans cnie" in q

    try:
        from app.passeport_cases import passeport_case_from_question

        case = passeport_case_from_question(question)
    except ImportError:
        case = None

    by_id = _load_chunks_by_id()
    preferred: list[str] = []
    if case == "perte_vol":
        preferred = list(_CONSULAT_PERTE_VOL)
    elif case == "renouvellement":
        preferred = list(_CONSULAT_RENOUVELLEMENT)
    elif mineur and sans_cnie:
        preferred = [_CONSULAT_MINEUR_SANS_CNIE, _CONSULAT_PIECES_MINEURS]
    elif mineur:
        preferred = [_CONSULAT_PIECES_MINEURS, _CONSULAT_MINEUR_SANS_CNIE]
    else:
        preferred = list(_CONSULAT_DEFAULT)

    hits: list[dict[str, Any]] = []
    for cid in preferred:
        if cid in by_id:
            hits.append(_row_to_hit(by_id[cid]))
    if hits:
        return hits
    return curated_passeport_hits_for(question)


def merge_passeport_hits(
    question: str, hits: list[dict[str, Any]], *, top_k: int = 8
) -> list[dict[str, Any]]:
    """Place en tête les extraits passeport alignés au contexte de la phrase."""
    if not is_passeport_primary_question(question):
        return hits[:top_k]

    injected = best_passeport_hits_for(question)
    if not injected:
        return hits[:top_k]

    seen: set[str] = set()
    out: list[dict[str, Any]] = []
    for h in injected:
        cid = str((h.get("metadata") or {}).get("chunk_id") or "")
        if cid and cid not in seen:
            seen.add(cid)
            out.append(h)
    for h in hits:
        cid = str((h.get("metadata") or {}).get("chunk_id") or "")
        if cid in seen:
            continue
        try:
            from app.phrase_context import hit_matches_primary_subject

            if not hit_matches_primary_subject(question, h):
                continue
        except ImportError:
            meta = h.get("metadata") or {}
            if str(meta.get("category") or "").lower() == "cnie":
                continue
        seen.add(cid or id(h))
        out.append(h)
    return out[:top_k]
