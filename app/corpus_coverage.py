"""
Décide si le corpus local (JSONL + FAISS) couvre vraiment la question.
Sinon → le pipeline active le fallback web (app/web_fallback.py).
"""

from __future__ import annotations

import os
import re
import unicodedata
from typing import Any

# Phrases métier : si présentes dans la question, doivent apparaître dans les extraits.
_REQUIRED_PHRASES: tuple[tuple[str, str], ...] = (
    ("heures supplementaires", "heures suppl"),
    ("heures supplémentaires", "heures suppl"),
    ("heure supplementaire", "heure suppl"),
    ("code du travail", "code du travail"),
    ("salaire minimum", "salaire minimum"),
    ("conge annuel", "conge annuel"),
    ("congé annuel", "congé annuel"),
    ("permis de construire", "permis de construire"),
    ("autorisation de construire", "autorisation de construire"),
    ("carte nationale", "carte nationale"),
    ("identite electronique", "identite electronique"),
    ("identité électronique", "identité électronique"),
    ("perte ou vol", "perte"),
    ("perte de la cnie", "perte"),
    ("vol de la cnie", "vol"),
    ("carte perdue", "perte"),
    ("carte volee", "vol"),
    ("acte de naissance", "acte de naissance"),
    ("licenciement", "licenciement"),
    ("delai de preavis", "preavis"),
    ("délai de préavis", "préavis"),
    ("bulletin officiel", "bulletin officiel"),
    ("cycle de master", "cycle de master"),
    ("normes pedagogiques", "normes pedagog"),
    ("normes pédagogiques", "normes pédagog"),
)

_STOP = frozenset(
    """
    pour les des une un une la le et ou mais avec sans son sa ses est sont été être
    avoir fait faire plus moins très tout tous toute dans sur par que qui dont où
    quel quels quelle quelles comme tel tels lors ainsi chez aux ces ses leur leurs
    faut peut doit même aussi bien non oui pas ne ni maroc marocaine marocain
    quelle quelles comment combien lorsque lorsqu
    """.split()
)

_WEAK = frozenset(
    {
        "avec",
        "dans",
        "pour",
        "vous",
        "votre",
        "question",
        "sujet",
        "savoir",
        "donner",
        "donne",
        "suis",
        "sommes",
        "es",
        "est",
        "mon",
        "ma",
        "mes",
        "ton",
        "ta",
        "veux",
        "voudrais",
        "souhaite",
        "peux",
        "puis",
        "maintenant",
        "comment",
    }
)


def _fold(s: str) -> str:
    return "".join(
        c for c in unicodedata.normalize("NFD", s or "") if unicodedata.category(c) != "Mn"
    ).lower()


def _float_env(name: str, default: float) -> float:
    try:
        return float(os.environ.get(name, str(default)))
    except ValueError:
        return default


def _hit_score(h: dict[str, Any]) -> float:
    return float(h.get("rerank_score", h.get("score", 0.0)) or 0.0)


def _combined_top_text(hits: list[dict[str, Any]], n: int = 3) -> str:
    parts: list[str] = []
    for h in hits[:n]:
        meta = h.get("metadata") or {}
        parts.append(str(h.get("text") or ""))
        parts.append(str(meta.get("label") or ""))
        parts.append(str(meta.get("title") or ""))
    return _fold(" ".join(parts))


def _discriminative_terms(question: str) -> list[str]:
    q = _fold(question)
    terms = [
        t
        for t in re.findall(r"[a-z0-9]{3,}", q)
        if t not in _STOP and t not in _WEAK
    ]
    try:
        from app.phrase_context import analyze_phrase

        pctx = analyze_phrase(question)
        if pctx.primary_subject:
            for noise in ("cnie", "cine", "passeport", "watiqa"):
                if noise != pctx.primary_subject and noise in pctx.negated_subjects:
                    terms = [t for t in terms if t != noise]
    except ImportError:
        pass
    if "salaire minimum" in q and "salaire" not in terms:
        terms.append("salaire")
    if "code du travail" in q:
        for extra in ("travail", "code"):
            if extra not in terms:
                terms.append(extra)
    if "passeport" in q and any(x in q for x in ("perdu", "perdue", "perte", "vol")):
        for extra in ("passeport", "perte", "vol", "declaration", "declarer"):
            if extra not in terms:
                terms.append(extra)
    seen: set[str] = set()
    out: list[str] = []
    for t in terms:
        if t not in seen:
            seen.add(t)
            out.append(t)
    return out


def _required_phrases_ok(
    question: str,
    top_text: str,
    hits: list[dict[str, Any]] | None = None,
) -> bool:
    q = _fold(question)
    subs_needed: set[str] = set()
    for needle, sub in _REQUIRED_PHRASES:
        if _fold(needle) in q:
            subs_needed.add(_fold(sub))
    if not subs_needed:
        return True

    try:
        from app.phrase_context import analyze_phrase

        pctx = analyze_phrase(question)
        if pctx.primary_subject and pctx.subject_scores.get(pctx.primary_subject, 0) >= 5.0:
            return True
    except ImportError:
        pass

    # e5-small : retrieval sémantique — score fort ou sous-chaîne dans un top hit suffit
    if hits:
        if _hit_score(hits[0]) >= 0.55:
            try:
                from app.portal_cases import cnie_case_from_question, top_hits_match_cnie_case

                if cnie_case_from_question(question) and not top_hits_match_cnie_case(
                    question, hits
                ):
                    pass
                else:
                    return True
            except ImportError:
                return True
        for h in hits[:5]:
            blob = _fold(str(h.get("text") or ""))
            if any(s in blob for s in subs_needed):
                return True

    return all(s in top_text for s in subs_needed)


def _term_overlap_ok(question: str, top_text: str) -> bool:
    terms = _discriminative_terms(question)
    if not terms:
        return len(top_text) >= 100
    found = sum(1 for t in terms if t in top_text)
    ratio = found / max(1, len(terms))
    if len(terms) == 1:
        return found >= 1
    if len(terms) == 2:
        return found >= 2
    min_ratio = _float_env("CORPUS_MIN_TERM_RATIO", 0.5)
    return found >= 2 and ratio >= min_ratio


def _strong_portal_match(
    question: str,
    top_sc: float,
    top_text: str,
    hits: list[dict[str, Any]] | None = None,
) -> bool:
    """Score élevé + sujet portail présent dans les extraits (questions génériques « obtenir »)."""
    try:
        from app.phrase_context import analyze_phrase

        pctx = analyze_phrase(question)
        if not (
            pctx.primary_subject in ("cnie", "passeport", "watiqa")
            and top_sc >= 0.55
            and pctx.primary_subject in top_text
        ):
            return False
        if hits:
            from app.portal_cases import cnie_case_from_question, top_hits_match_cnie_case

            if cnie_case_from_question(question) and not top_hits_match_cnie_case(
                question, hits
            ):
                return False
        return True
    except ImportError:
        return False


def _bo_food_nomenclature_mismatch(question: str, hits: list[dict[str, Any]]) -> bool:
    """Recette / cuisine grand public vs extrait BO nomenclature alimentaire."""
    try:
        from app.question_type import is_general_knowledge_question

        if not is_general_knowledge_question(question):
            return False
    except ImportError:
        return False
    top_text = _combined_top_text(hits, 3)
    bo_food = (
        "produit de la minoterie",
        "semoule",
        "sans fermentation",
        "nomenclature",
        "denree alimentaire",
        "denrée alimentaire",
        "couscous est le produit",
    )
    return any(m in top_text for m in bo_food)


def corpus_covers_question(question: str, hits: list[dict[str, Any]]) -> bool:
    """
    True = les meilleurs extraits du dataset permettent de répondre (pas de web).
    False = sujet absent ou hors sujet → activer fallback web.
    """
    if not hits:
        return False

    try:
        from app.question_type import is_general_knowledge_question

        if is_general_knowledge_question(question):
            return False
    except ImportError:
        pass

    if _bo_food_nomenclature_mismatch(question, hits):
        return False

    # Décision intelligente basée sur le type de question
    try:
        from app.question_type import corpus_should_suffice, is_current_data_question
        from app.web_fallback import (
            _is_passeport_fee_question,
            passeport_fee_hits_substantive,
        )

        top_sc_check = _hit_score(hits[0])

        # Timbre / e-timbre passeport : pas couvert par un BO « loi de finances »
        if _is_passeport_fee_question(question):
            return passeport_fee_hits_substantive(question, hits)

        # Question sur données actuelles chiffrées
        if is_current_data_question(question):
            # Vérifier si le corpus contient vraiment la donnée
            top_text_check = _combined_top_text(hits, 3)
            has_numbers = bool(
                re.search(
                    r"\d+[,.]?\d*\s*(dh|dirham|%|pour\s*cent)",
                    top_text_check,
                )
            )
            if has_numbers and top_sc_check >= 0.45:
                return True  # corpus a la donnée chiffrée
            return False  # laisser web fallback chercher le montant exact

        # Question de cadre légal avec bon score → corpus suffit
        if corpus_should_suffice(question, hits, top_sc_check):
            return True

    except ImportError:
        pass

    top_sc = _hit_score(hits[0])
    top_text = _combined_top_text(hits, 5)

    try:
        from app.labor_corpus import (
            is_labor_code_question,
            labor_hits_substantively_answer,
            merge_labor_hits,
        )

        if is_labor_code_question(question):
            hits = merge_labor_hits(question, hits, top_k=max(8, len(hits)))
            top_sc = _hit_score(hits[0])
            top_text = _combined_top_text(hits, 5)
            if labor_hits_substantively_answer(question, hits):
                return True
    except ImportError:
        pass

    # Construction : si FAISS ne retourne pas de chunk urbanisme → web fallback OK
    try:
        from app.query_understanding import analyze_query

        qa = analyze_query(question)
        if "construction" in qa.topics:
            has_urban_hit = any(
                any(
                    m in (h.get("text") or "").lower()
                    for m in (
                        "permis de construire",
                        "autorisation de construire",
                        "urbanisme",
                        "agence urbaine",
                        "rokhas",
                    )
                )
                for h in hits[:5]
            )
            if not has_urban_hit:
                return False
            if top_sc >= 0.22:
                return True
    except ImportError:
        pass

    # Registre global dataset (BO SGG, admin, tous domaines) — avant filtres lexicaux stricts
    try:
        from app.dataset_registry import dataset_covers_via_registry

        reg = dataset_covers_via_registry(question, hits)
        if reg is True:
            try:
                from app.portal_cases import cnie_case_from_question, top_hits_match_cnie_case

                if cnie_case_from_question(question) and not top_hits_match_cnie_case(
                    question, hits
                ):
                    pass
                else:
                    return True
            except ImportError:
                return True
        if reg is False and not _strong_portal_match(question, top_sc, top_text, hits):
            return False
    except ImportError:
        pass

    min_top = _float_env("CORPUS_MIN_TOP_SCORE", 0.35)

    if top_sc >= 0.65:
        try:
            from app.phrase_context import analyze_phrase
            from app.portal_cases import cnie_case_from_question, top_hits_match_cnie_case

            pctx = analyze_phrase(question)
            if pctx.primary_subject and pctx.primary_subject not in top_text:
                pass
            elif cnie_case_from_question(question) and not top_hits_match_cnie_case(
                question, hits
            ):
                pass
            else:
                return True
        except ImportError:
            return True

    if top_sc < min_top:
        return False

    if not _required_phrases_ok(question, top_text, hits):
        return False

    if not _term_overlap_ok(question, top_text) and not _strong_portal_match(
        question, top_sc, top_text, hits
    ):
        return False

    # SMIG : éviter les mentions isolées hors code du travail
    q = _fold(question)
    if "smig" in q or "salaire minimum" in q:
        if not any(
            x in top_text
            for x in (
                "smig",
                "salaire minimum interprofessionnel",
                "salaire minimum legal",
                "salaire minimum légal",
                "article 356",
                "code du travail",
                "loi 65-99",
                "loi n 65-99",
            )
        ):
            return False

    # Passeport perdu / vol : extraits consulat avec déclaration suffisent
    try:
        from app.passeport_cases import passeport_case_from_question, passeport_case_aligned

        if passeport_case_from_question(question) == "perte_vol":
            if any(passeport_case_aligned(question, h) for h in hits[:2]):
                if _term_overlap_ok(question, top_text):
                    return True
    except ImportError:
        pass

    # Sujet principal de la phrase (passeport vs CNIE vs Watiqa) doit correspondre aux extraits
    try:
        from app.phrase_context import analyze_phrase, top_hits_match_primary_subject
        from app.portal_cases import cnie_case_from_question, top_hits_match_cnie_case
        from app.dataset_registry import top_hits_match_domains

        pctx = analyze_phrase(question)

        if pctx.primary_subject and not top_hits_match_primary_subject(question, hits):
            if top_sc >= 0.55 and pctx.primary_subject in top_text:
                pass
            else:
                return False

        cnie_case = cnie_case_from_question(question)
        if cnie_case and not top_hits_match_cnie_case(question, hits):
            if pctx.primary_subject == "cnie" and top_sc >= 0.50:
                pass
            else:
                return False

        if not pctx.primary_subject and not top_hits_match_domains(question, hits):
            return False
    except ImportError:
        pass

    # Chunk curated / loi explicite = couverture forte (si cas CNIE aligné)
    for h in hits[:3]:
        cid = str((h.get("metadata") or {}).get("chunk_id") or "")
        doc = str((h.get("metadata") or {}).get("doc_id") or "")
        if doc == "LABOR65":
            return True
        if cid.startswith("curated::labor"):
            return True
        if cid.startswith("curated::watiqa"):
            return True
        if cid.startswith("curated::cnie"):
            try:
                from app.portal_cases import cnie_case_aligned

                if cnie_case_aligned(question, h):
                    return True
            except ImportError:
                return True
        if cid.startswith("curated::passeport"):
            try:
                from app.passeport_cases import passeport_case_aligned, passeport_case_from_question
                from app.phrase_context import analyze_phrase

                if analyze_phrase(question).primary_subject == "passeport":
                    if passeport_case_from_question(question):
                        return passeport_case_aligned(question, h)
                    return True
            except ImportError:
                return True

    # Admin avec URL = procédure souvent suffisante
    meta0 = hits[0].get("metadata") or {}
    if meta0.get("source_type") == "admin" and meta0.get("source_url"):
        if _term_overlap_ok(question, top_text) and top_sc >= min_top * 0.85:
            return True

    return top_sc >= min_top and (
        _term_overlap_ok(question, top_text)
        or _strong_portal_match(question, top_sc, top_text, hits)
    )


def explain_coverage(question: str, hits: list[dict[str, Any]]) -> str:
    """Courte explication pour logs / debug."""
    if not hits:
        return "aucun extrait"
    if corpus_covers_question(question, hits):
        return "corpus_ok"
    top_text = _combined_top_text(hits, 2)[:120]
    sc = _hit_score(hits[0])
    return f"corpus_insuffisant score={sc:.2f} extrait={top_text!r}..."
