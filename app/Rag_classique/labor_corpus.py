"""
Code du travail (loi n° 65-99) : détection de sujet, anti-faux-positifs BO (SMIG/fiscal),
chunks de secours et fusion des hits retrieval.
"""

from __future__ import annotations

import re
import unicodedata
from typing import Any

# Mots-clés heures supplémentaires (question)
_OVERTIME_Q = (
    "heures supplementaires",
    "heures supplémentaires",
    "heure supplementaire",
    "heure supplémentaire",
    "majoration heures",
)

# Passages BO hors sujet souvent remontés pour « rémunération » / « labor »
_OFF_TOPIC_PATTERNS = (
    "revenu forfaitaire",
    "revenus forfaitaires",
    "article 15 ter",
    "article 184",
    "code general des impots",
    "code général des impôts",
    "circulaire n° ps/",
    "prevoyance sociale",
    "prévoyance sociale",
    "assurance maladie obligatoire",
    "cnss",
    "produits de location",
    "sanctions pour defaut",
    "sanctions pour défaut",
)

# Indices que le corpus répond vraiment au droit du travail demandé
_LABOR_ON_TOPIC = (
    "code du travail",
    "loi n° 65-99",
    "loi n 65-99",
    "loi 65-99",
    "article 201",
    "article 202",
    "article 197",
    "article 231",
    "article 356",
    "article 357",
    "article 43",
    "article 41",
    "heures supplementaires",
    "heures supplémentaires",
    "majoration de salaire",
    "conge annuel paye",
    "congé annuel payé",
    "delai de preavis",
    "délai de préavis",
    "licenciement",
    "contrat de travail a duree determinee",
    "contrat de travail à durée déterminée",
    "smig",
    "salaire minimum interprofessionnel",
    "salaire minimum legal",
    "salaire minimum légal",
    "accident de travail",
    "accidents de travail",
    "obligations de l employeur",
    "obligation de l employeur",
    "declaration d accident",
    "déclaration d accident",
)


def _fold(s: str) -> str:
    return "".join(
        c for c in unicodedata.normalize("NFD", s or "") if unicodedata.category(c) != "Mn"
    ).lower()


def is_smig_question(question: str) -> bool:
    q = _fold(question)
    return "smig" in q or "salaire minimum" in q


def is_accident_travail_question(question: str) -> bool:
    q = _fold(question)
    if "accident" in q and "travail" in q:
        return True
    return "accident" in q and "employeur" in q


def is_overtime_question(question: str) -> bool:
    q = _fold(question)
    if any(m in q for m in _OVERTIME_Q):
        return True
    return "heure" in q and "suppl" in q


def is_labor_code_question(question: str) -> bool:
    q = _fold(question)
    if is_overtime_question(question):
        return True
    labor_kw = (
        "code du travail",
        "smig",
        "salaire minimum",
        "licenciement",
        "preavis",
        "préavis",
        "demission",
        "démission",
        "conge annuel",
        "congé annuel",
        "heures supplement",
        "heures supplément",
        "contrat de travail",
        "accident de travail",
        "representants du personnel",
        "représentants du personnel",
        "indemnite de licenciement",
        "indemnité de licenciement",
        "periode d'essai",
        "période d'essai",
        "cdd",
        "cdi",
        "obligations employeur",
        "obligation employeur",
        "obligations de l employeur",
    )
    return any(k in q for k in labor_kw)


def is_targeted_labor_topic(question: str) -> bool:
    """SMIG, accident, heures sup. : injection curated + filtrage BO bruit."""
    return (
        is_smig_question(question)
        or is_accident_travail_question(question)
        or is_overtime_question(question)
    )


def primary_hit_for_answer(
    question: str, hits: list[dict[str, Any]]
) -> dict[str, Any] | None:
    """Premier extrait à citer (curated / LABOR65 / pertinent), pas un BO fiscal."""
    for h in hits:
        cid = str((h.get("metadata") or {}).get("chunk_id") or "")
        if cid.startswith("curated::labor"):
            return h
        if (h.get("metadata") or {}).get("doc_id") == "LABOR65":
            return h
        if _is_strong_labor_hit(h, question):
            return h
    for h in hits:
        if not _is_bo_admin_noise_for_labor(_hit_blob(h)):
            return h
    return hits[0] if hits else None


def filter_hits_for_labor_prompt(
    question: str,
    hits: list[dict[str, Any]],
    *,
    top_k: int,
) -> list[dict[str, Any]]:
    """
    Extraits envoyés au LLM : curated en tête, chunks BO fiscal/nomenclature retirés
    pour SMIG / accident / heures sup.
    """
    if not is_targeted_labor_topic(question):
        return hits[:top_k]
    curated = curated_labor_hits_for(question)
    cids = {c["metadata"]["chunk_id"] for c in curated} if curated else set()
    kept: list[dict[str, Any]] = []
    if curated:
        kept.extend(curated)
    for h in hits:
        cid = (h.get("metadata") or {}).get("chunk_id")
        if cid in cids:
            continue
        if _is_bo_admin_noise_for_labor(_hit_blob(h)):
            continue
        kept.append(h)
    return kept[:top_k]


def is_off_topic_labor_hit(text: str, *, overtime: bool) -> bool:
    t = _fold(text)
    if any(p in t for p in _OFF_TOPIC_PATTERNS):
        return True
    if overtime:
        # SMIG / forfaitaire seul sans heures supp.
        if ("salaire minimum" in t or "revenu forfaitaire" in t) and "heures suppl" not in t:
            if "article 201" not in t and "majoration de salaire" not in t:
                return True
    return False


def _hit_blob(h: dict[str, Any]) -> str:
    meta = h.get("metadata") or {}
    return _fold(
        " ".join(
            (
                str(h.get("text") or ""),
                str(meta.get("label") or ""),
                str(meta.get("chunk_id") or ""),
            )
        )
    )


def _is_bo_admin_noise_for_labor(blob: str) -> bool:
    """Faux positifs fréquents du BO (fiscal, nomenclature dépenses État, etc.)."""
    noise = (
        "nomenclature des pieces justificatives",
        "propositions d engagement",
        "paiement des depenses",
        "biens et services de l etat",
        "revenu forfaitaire",
        "prevoyance sociale",
        "circulaire n ps",
        "bank al-maghrib",
    )
    return any(n in blob for n in noise)


def _is_strong_labor_hit(hit: dict[str, Any], question: str) -> bool:
    """Extrait réellement pertinent pour la question travail (pas un BO générique)."""
    cid = str((hit.get("metadata") or {}).get("chunk_id") or "")
    if cid.startswith("curated::labor") or (hit.get("metadata") or {}).get("doc_id") == "LABOR65":
        return True
    blob = _hit_blob(hit)
    if _is_bo_admin_noise_for_labor(blob):
        return False
    if is_smig_question(question):
        return any(
            x in blob
            for x in (
                "smig",
                "salaire minimum interprofessionnel",
                "article 356",
                "articles 356",
            )
        )
    if is_accident_travail_question(question):
        return "accident de travail" in blob or "accidents de travail" in blob
    if is_overtime_question(question):
        return any(
            x in blob
            for x in ("heures suppl", "article 201", "majoration de salaire", "article 197")
        )
    return any(m in blob for m in _LABOR_ON_TOPIC) and not is_off_topic_labor_hit(
        str(hit.get("text") or ""), overtime=is_overtime_question(question)
    )


def labor_hits_substantively_answer(question: str, hits: list[dict[str, Any]]) -> bool:
    if not hits:
        return False

    for h in hits[:5]:
        cid = str((h.get("metadata") or {}).get("chunk_id") or "")
        if cid.startswith("curated::labor") or (h.get("metadata") or {}).get("doc_id") == "LABOR65":
            return True

    overtime = is_overtime_question(question)
    top = hits[:5]
    texts = [_hit_blob(h) for h in top]

    if is_smig_question(question) or is_accident_travail_question(question) or is_overtime_question(
        question
    ):
        return any(_is_strong_labor_hit(h, question) for h in top)

    on_topic = 0
    off_topic = 0
    for t in texts:
        if any(m in t for m in _LABOR_ON_TOPIC):
            if not is_off_topic_labor_hit(t, overtime=overtime):
                on_topic += 1
        if is_off_topic_labor_hit(t, overtime=overtime):
            off_topic += 1

    if on_topic >= 1 and off_topic == 0:
        return True
    if on_topic >= 2:
        return True

    top_sc = float(hits[0].get("rerank_score", hits[0].get("score", 0.0)) or 0.0)
    if top_sc >= 0.45:
        combined = " ".join(texts)
        if any(m in combined for m in _LABOR_ON_TOPIC) and off_topic <= on_topic:
            return True

    if overtime:
        combined = " ".join(texts)
        return (
            "heures suppl" in combined
            and ("article 201" in combined or "majoration" in combined or "25 %" in combined)
        )

    combined = " ".join(texts)
    return any(m in combined for m in _LABOR_ON_TOPIC) and off_topic < on_topic


def _curated_overtime_hits() -> list[dict[str, Any]]:
    text = (
        "Loi n° 65-99 — Code du travail marocain. Article 197 : sont considérées comme heures "
        "supplémentaires les heures de travail accomplies au-delà de la durée normale de travail "
        "du salarié. Article 198 : les heures supplémentaires sont payées en un seul versement "
        "en même temps que le salaire dû. Article 201 : quel que soit le mode de rémunération du "
        "salarié, les heures supplémentaires donnent lieu à une majoration de salaire de 25 % si "
        "elles sont effectuées entre 6 heures et 21 heures pour les activités non agricoles et "
        "entre 5 heures et 20 heures pour les activités agricoles, et de 50 % si elles sont "
        "effectuées entre 21 heures et 6 heures pour les activités non agricoles et entre 20 "
        "heures et 5 heures pour les activités agricoles. La majoration est portée respectivement "
        "à 50 % et à 100 % si les heures supplémentaires sont effectuées le jour du repos "
        "hebdomadaire du salarié, même si un repos compensateur lui est accordé. Article 202 : la "
        "rémunération des heures supplémentaires est calculée tant sur le salaire que sur ses "
        "accessoires, à l'exclusion des allocations familiales, des pourboires (sauf personnel "
        "exclusivement au pourboire), et des indemnités de remboursement de frais."
    )
    return [
        {
            "index": -1,
            "score": 0.99,
            "rerank_score": 0.99,
            "metadata": {
                "chunk_id": "curated::labor_art_201",
                "doc_id": "LABOR65",
                "title": "Code du travail — Loi n° 65-99",
                "label": "Articles 197-202 (heures supplémentaires)",
                "source_type": "loi",
                "source_url": "https://www.emploi.gov.ma/",
                "category": "code_travail",
                "source_org": "Législation",
                "document_type": "loi",
            },
            "text": text,
        }
    ]


def _curated_smig_hits() -> list[dict[str, Any]]:
    text = (
        "Loi n° 65-99 — Code du travail marocain. SMIG (salaire minimum interprofessionnel garanti) : "
        "les articles 356 à 361 fixent le salaire minimum légal pour les activités non agricoles "
        "(durée normale de travail, montant horaire et mensuel). Le montant en vigueur est revalorisé "
        "par décret ou arrêté publié au Bulletin officiel (sgg.gov.ma) et par l'accord social "
        "(ministère de l'Emploi — emploi.gov.ma, miepeec.gov.ma). Le SMIG s'applique aux salariés "
        "du secteur privé ; des minima conventionnels peuvent exister par branche. "
        "Ne pas confondre avec des mentions isolées de « salaire minimum » dans des textes fiscaux "
        "ou de prévoyance qui ne fixent pas le SMIG national."
    )
    return [
        {
            "index": -1,
            "score": 0.99,
            "rerank_score": 0.99,
            "metadata": {
                "chunk_id": "curated::labor_smig",
                "doc_id": "LABOR65",
                "title": "Code du travail — Loi n° 65-99",
                "label": "Articles 356-361 (SMIG)",
                "source_type": "loi",
                "source_url": "https://www.emploi.gov.ma/",
                "category": "code_travail",
                "source_org": "Législation",
                "document_type": "loi",
            },
            "text": text,
        }
    ]


def _curated_accident_hits() -> list[dict[str, Any]]:
    text = (
        "Loi n° 65-99 — Code du travail marocain. Accident de travail : tout accident survenu "
        "par le fait ou à l'occasion du travail. Obligations de l'employeur : assurer les soins "
        "immédiats, déclarer l'accident à la CNSS dans les délais prévus, conserver les éléments "
        "utiles à l'enquête, coopérer avec la caisse et respecter les mesures de réinsertion. "
        "L'employeur ne peut mettre fin au contrat du salarié victime pendant la période de "
        "consolidation ni pendant un délai après la consolidation selon la gravité. "
        "Indemnisation et prestations relèvent du régime de sécurité sociale (CNSS) et des "
        "dispositions du Code du travail sur la responsabilité et la réparation du préjudice."
    )
    return [
        {
            "index": -1,
            "score": 0.99,
            "rerank_score": 0.99,
            "metadata": {
                "chunk_id": "curated::labor_accident",
                "doc_id": "LABOR65",
                "title": "Code du travail — Loi n° 65-99",
                "label": "Accidents de travail — obligations employeur",
                "source_type": "loi",
                "source_url": "https://www.emploi.gov.ma/",
                "category": "code_travail",
                "source_org": "Législation",
                "document_type": "loi",
            },
            "text": text,
        }
    ]


def curated_labor_hits_for(question: str) -> list[dict[str, Any]]:
    if is_overtime_question(question):
        return _curated_overtime_hits()
    if is_smig_question(question):
        return _curated_smig_hits()
    if is_accident_travail_question(question):
        return _curated_accident_hits()
    return []


def merge_labor_hits(
    question: str,
    hits: list[dict[str, Any]],
    *,
    top_k: int,
) -> list[dict[str, Any]]:
    """
    Réordonne / injecte des sources Code du travail si le retrieval BO est hors sujet.
    """
    if not is_labor_code_question(question):
        return hits[:top_k]

    try:
        from app.Rag_classique.text_sanitize import sanitize_hit, text_is_usable_for_llm

        hits = [sanitize_hit(h) for h in hits if text_is_usable_for_llm(str(h.get("text") or ""))]
    except ImportError:
        pass

    overtime = is_overtime_question(question)
    cleaned: list[dict[str, Any]] = []
    for h in hits:
        txt = h.get("text") or ""
        meta = h.get("metadata") or {}
        if (meta.get("doc_id") == "LABOR65" or str(meta.get("chunk_id", "")).startswith("LABOR65")):
            cleaned.append(h)
            continue
        if is_off_topic_labor_hit(txt, overtime=overtime):
            hh = dict(h)
            hh["rerank_score"] = float(h.get("rerank_score", h.get("score", 0.0)) or 0.0) - 0.45
            cleaned.append(hh)
        else:
            cleaned.append(h)

    cleaned.sort(
        key=lambda x: float(x.get("rerank_score", x.get("score", 0.0)) or 0.0),
        reverse=True,
    )

    curated = curated_labor_hits_for(question)
    if curated and is_targeted_labor_topic(question):
        cids = {c["metadata"]["chunk_id"] for c in curated}
        rest = [
            h
            for h in cleaned
            if (h.get("metadata") or {}).get("chunk_id") not in cids
            and not _is_bo_admin_noise_for_labor(_hit_blob(h))
        ]
        return (curated + rest)[:top_k]

    if labor_hits_substantively_answer(question, cleaned):
        return cleaned[:top_k]

    if curated:
        cids = {c["metadata"]["chunk_id"] for c in curated}
        rest = [h for h in cleaned if (h.get("metadata") or {}).get("chunk_id") not in cids]
        return (curated + rest)[:top_k]

    # Prioriser tout chunk LABOR65 déjà dans la liste
    labor_only = [
        h
        for h in cleaned
        if (h.get("metadata") or {}).get("doc_id") == "LABOR65"
        or str((h.get("metadata") or {}).get("chunk_id", "")).startswith("LABOR65")
    ]
    if labor_only:
        rest = [h for h in cleaned if h not in labor_only]
        return (labor_only + rest)[:top_k]

    return cleaned[:top_k]
