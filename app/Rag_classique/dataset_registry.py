"""
Registre des domaines du dataset (final_chunks.jsonl) — couverture globale.

Aligne question ↔ catégories / sources : juridique (SGG), état civil, CNIE,
passeport, code du travail, urbanisme, fiscalité, santé, éducation, etc.
"""

from __future__ import annotations

import re
import unicodedata
from typing import Any

# topic query_understanding → domaine dataset
_TOPIC_TO_DOMAIN: dict[str, str] = {
    "cnie": "cnie",
    "passeport": "passeport",
    "watiqa": "watiqa",
    "labor": "labor",
    "construction": "construction",
    "education_doctorate": "education",
    "education_master": "education",
    "education_master_generic": "education",
    "bulletin": "bulletin",
    "fiscalite": "fiscalite",
    "sante": "sante",
    "marches_publics": "marches_publics",
}

# domaine → critères de correspondance dans les chunks
_DOMAIN_SPECS: dict[str, dict[str, Any]] = {
    "cnie": {
        "categories": ("cnie", "identite"),
        "orgs": ("cnie maroc",),
        "url_parts": ("cnie.ma",),
        "chunk_prefixes": ("cnie_procedure_", "curated::cnie"),
        "text_markers": ("cnie", "carte nationale", "identite electronique"),
    },
    "passeport": {
        "categories": ("passeport", "voyage", "services_publics"),
        "orgs": ("consulat.ma",),
        "url_parts": ("consulat.ma", "passeport.ma"),
        "chunk_prefixes": ("consulat_passeport_", "curated::passeport"),
        "text_markers": ("passeport biometrique", "passeport biométrique", "immatriculation consulaire"),
    },
    "watiqa": {
        "categories": ("etat_civil",),
        "orgs": ("watiqa",),
        "url_parts": ("watiqa.ma",),
        "chunk_prefixes": ("watiqa_procedure_", "curated::watiqa"),
        "text_markers": (
            "watiqa",
            "acte de naissance",
            "etat civil",
            "guichet electronique",
            "commencer la demarche",
        ),
    },
    "labor": {
        "categories": ("code_travail", "travail"),
        "doc_ids": ("LABOR65",),
        "chunk_prefixes": ("LABOR65", "curated::labor"),
        "text_markers": (
            "code du travail",
            "loi 65-99",
            "loi n 65-99",
            "heures suppl",
            "licenciement",
            "preavis",
            "préavis",
            "smig",
            "conge annuel",
        ),
    },
    "construction": {
        "categories": ("urbanisme",),
        "text_markers": (
            "permis de construire",
            "autorisation de construire",
            "urbanisme",
            "lotissement",
            "rokhas",
            "agence urbaine",
            "coefficient d occupation",
            "plan d amenagement",
        ),
    },
    "education": {
        "categories": ("education", "juridique"),
        "text_markers": (
            "normes pedagog",
            "normes pédagog",
            "cycle de master",
            "cycle master",
            "doctorat",
            "these",
            "thèse",
            "enseignement superieur",
            "universite",
            "université",
            "credits master",
        ),
    },
    "fiscalite": {
        "categories": ("fiscalite", "juridique"),
        "text_markers": (
            "fiscal",
            "impot",
            "impôt",
            "taxe",
            "tva",
            "code general des impots",
            "revenu",
        ),
    },
    "sante": {
        "categories": ("sante", "juridique"),
        "text_markers": ("sante", "santé", "hopital", "hôpital", "medicament", "médicament"),
    },
    "marches_publics": {
        "categories": ("marches_publics", "juridique"),
        "text_markers": ("marche public", "marché public", "appel d'offres", "soumissionnaire"),
    },
    "bulletin": {
        "categories": ("juridique",),
        "source_types": ("bulletin_officiel",),
        "doc_id_prefixes": ("SGG",),
        "orgs": ("sgg",),
        "text_markers": (
            "bulletin officiel",
            "journal officiel",
            "dahir",
            "decret",
            "décret",
            "arrêté",
            "arrete",
        ),
    },
    "juridique": {
        "categories": ("juridique", "administratif"),
        "source_types": ("bulletin_officiel", "loi", "admin"),
        "doc_id_prefixes": ("SGG",),
    },
    "administratif": {
        "categories": ("administratif", "etat_civil", "cnie", "passeport"),
        "source_types": ("admin",),
    },
}

# Mots-clés question → domaine (si analyze_query ne renvoie rien)
_KEYWORD_DOMAINS: tuple[tuple[str, str], ...] = (
    ("cnie", "cnie"),
    ("carte nationale", "cnie"),
    ("passeport", "passeport"),
    ("watiqa", "watiqa"),
    ("acte de naissance", "watiqa"),
    ("code du travail", "labor"),
    ("heures suppl", "labor"),
    ("licenciement", "labor"),
    ("smig", "labor"),
    ("autorisation de construire", "construction"),
    ("permis de construire", "construction"),
    ("urbanisme", "construction"),
    ("rokhas", "construction"),
    ("agence urbaine", "construction"),
    ("doctorat", "education"),
    ("cycle de master", "education"),
    ("normes pedagog", "education"),
    ("bulletin officiel", "bulletin"),
    ("journal officiel", "bulletin"),
    ("dahir", "bulletin"),
    ("decret", "bulletin"),
    ("décret", "bulletin"),
    ("arrêté", "bulletin"),
    ("arrete", "bulletin"),
    ("impot", "fiscalite"),
    ("impôt", "fiscalite"),
    ("fiscal", "fiscalite"),
    ("marche public", "marches_publics"),
)


def _fold(s: str) -> str:
    return "".join(
        c for c in unicodedata.normalize("NFD", s or "") if unicodedata.category(c) != "Mn"
    ).lower()


def _hit_blob(hit: dict[str, Any]) -> str:
    meta = hit.get("metadata") or {}
    return _fold(
        " ".join(
            (
                str(hit.get("text") or ""),
                str(meta.get("label") or ""),
                str(meta.get("title") or ""),
                str(meta.get("category") or ""),
                str(meta.get("source_org") or ""),
                str(meta.get("doc_id") or ""),
                str(meta.get("chunk_id") or ""),
            )
        )
    )


def infer_domains_from_question(question: str) -> list[str]:
    q = _fold(question)
    found: list[str] = []
    for needle, domain in _KEYWORD_DOMAINS:
        if _fold(needle) in q and domain not in found:
            found.append(domain)
    return found


def active_domains(question: str) -> list[str]:
    """Domaines actifs pour cette question (phrase + topics + mots-clés)."""
    domains: list[str] = []

    try:
        from app.Rag_classique.phrase_context import analyze_phrase

        ctx = analyze_phrase(question)
        if ctx.primary_subject and ctx.primary_subject in _DOMAIN_SPECS:
            domains.append(ctx.primary_subject)
        elif ctx.primary_subject == "labor":
            domains.append("labor")
    except ImportError:
        pass

    try:
        from app.Rag_classique.query_understanding import analyze_query

        qa = analyze_query(question)
        for topic in qa.topics:
            dom = _TOPIC_TO_DOMAIN.get(topic, topic)
            if dom in _DOMAIN_SPECS and dom not in domains:
                domains.append(dom)
        if qa.bo_intent and "bulletin" not in domains:
            domains.append("bulletin")
        if qa.labor_intent and "labor" not in domains:
            domains.append("labor")
        if qa.education_doctorate_intent or qa.education_master_intent:
            if "education" not in domains:
                domains.append("education")
    except ImportError:
        pass

    for dom in infer_domains_from_question(question):
        if dom not in domains:
            domains.append(dom)

    # Corpus majoritairement BO : défaut juridique / bulletin
    if not domains:
        domains = ["bulletin", "juridique"]

    return domains


def hit_matches_domain(hit: dict[str, Any], domain: str) -> bool:
    spec = _DOMAIN_SPECS.get(domain)
    if not spec:
        return True

    meta = hit.get("metadata") or {}
    blob = _hit_blob(hit)
    cat = _fold(str(meta.get("category") or ""))
    st = _fold(str(meta.get("source_type") or ""))
    doc = str(meta.get("doc_id") or "")
    cid = str(meta.get("chunk_id") or "")
    org = _fold(str(meta.get("source_org") or ""))
    url = _fold(str(meta.get("source_url") or ""))

    for c in spec.get("categories", ()):
        if c in cat or cat == c:
            return True
    for st_spec in spec.get("source_types", ()):
        if st_spec in st:
            return True
    for d in spec.get("doc_ids", ()):
        if doc == d:
            return True
    for prefix in spec.get("doc_id_prefixes", ()):
        if doc.startswith(prefix):
            return True
    for prefix in spec.get("chunk_prefixes", ()):
        if cid.startswith(prefix):
            return True
    for o in spec.get("orgs", ()):
        if o in org:
            return True
    for u in spec.get("url_parts", ()):
        if u in url:
            return True
    for marker in spec.get("text_markers", ()):
        if _fold(marker) in blob:
            return True
    return False


def hit_matches_question_domains(question: str, hit: dict[str, Any]) -> bool:
    domains = active_domains(question)
    if not domains:
        return True
    return any(hit_matches_domain(hit, d) for d in domains)


def filter_hits_by_domains(
    question: str, hits: list[dict[str, Any]]
) -> list[dict[str, Any]]:
    """
    Garde les extraits cohérents avec le domaine de la question.
    Ne vide jamais la liste (fallback sur hits d'origine).
    """
    if not hits:
        return hits

    domains = active_domains(question)
    # Bulletin / juridique seul : ne pas exclure le BO (97 % du corpus)
    if domains == ["bulletin"] or domains == ["juridique"] or set(domains) == {"bulletin", "juridique"}:
        return hits

    kept = [h for h in hits if hit_matches_question_domains(question, h)]
    return kept if kept else hits


def top_hits_match_domains(
    question: str, hits: list[dict[str, Any]], n: int = 3
) -> bool:
    if not hits:
        return False
    return any(hit_matches_question_domains(question, h) for h in hits[:n])


def is_bulletin_hit(hit: dict[str, Any]) -> bool:
    meta = hit.get("metadata") or {}
    doc = str(meta.get("doc_id") or "")
    st = str(meta.get("source_type") or "")
    return doc.startswith("SGG") or st == "bulletin_officiel"


def dataset_covers_via_registry(
    question: str, hits: list[dict[str, Any]], *, min_score: float = 0.22
) -> bool | None:
    """
    True/False si le registre tranche ; None = laisser corpus_coverage décider.
    """
    if not hits:
        return False

    from app.Rag_classique.corpus_coverage import _combined_top_text, _hit_score, _term_overlap_ok

    domains = active_domains(question)
    if not top_hits_match_domains(question, hits, 3):
        return False

    top_sc = _hit_score(hits[0])
    top_text = _combined_top_text(hits, 3)

    if top_sc < min_score:
        return False

    # Portails / lois structurées : registre suffit si overlap OK
    if any(
        d in domains
        for d in (
            "cnie",
            "passeport",
            "watiqa",
            "labor",
        )
    ):
        try:
            from app.Rag_classique.passeport_cases import passeport_case_from_question

            if passeport_case_from_question(question) == "perte_vol" and "passeport" in domains:
                if "perte" in top_text or "vol" in top_text:
                    return _term_overlap_ok(question, top_text)
        except ImportError:
            pass
        return _term_overlap_ok(question, top_text)

    # Bulletin officiel (bulk du dataset)
    if "bulletin" in domains or "juridique" in domains:
        if is_bulletin_hit(hits[0]) or any(is_bulletin_hit(h) for h in hits[:3]):
            return _term_overlap_ok(question, top_text)

    if any(d in domains for d in ("construction", "education", "fiscalite", "marches_publics", "sante")):
        if top_hits_match_domains(question, hits, 3):
            return _term_overlap_ok(question, top_text)
        return None

    return None
