"""
Compréhension contextuelle des phrases (pas seulement des mots-clés isolés).

Détecte : document visé, négations (« sans CNIE »), action demandée (demander mon passeport),
âge (mineur), et le sujet principal de la question.
"""

from __future__ import annotations

import re
import unicodedata
from dataclasses import dataclass, field
from typing import Any

# Sujets administratifs courants
_SUBJECT_ALIASES: dict[str, tuple[str, ...]] = {
    "cnie": (
        "cnie",
        "cine",
        "carte nationale",
        "carte d identite",
        "carte d'identite",
        "identite electronique",
        "identité électronique",
    ),
    "passeport": ("passeport", "passeport biometrique", "passeport biométrique"),
    "watiqa": (
        "watiqa",
        "acte de naissance",
        "acte de mariage",
        "acte de deces",
        "acte de décès",
        "etat civil",
        "état civil",
    ),
    "labor": (
        "code du travail",
        "licenciement",
        "preavis",
        "préavis",
        "demission",
        "démission",
        "conge annuel",
        "congé annuel",
        "heures supplementaires",
        "heures supplémentaires",
        "smig",
        "salaire minimum",
    ),
    "construction": (
        "permis de construire",
        "autorisation de construire",
        "urbanisme",
        "construire",
        "construction",
    ),
}

_REQUEST_RE = re.compile(
    r"\b(?:je\s+)?(?:veux|voudrais|souhaite|aimerais|besoin|souhaiterai|"
    r"souhaiterais|comment|puis[- ]je)\b",
    re.I,
)
_ACTION_RE = re.compile(
    r"\b(?:demander|demande|obtenir|faire|delivrer|délivrer|renouveler|renouvellement|"
    r"passer|deposer|déposer|constituer|commander|declarer|déclarer|refaire|perdre|perdu|perdue)\b",
    re.I,
)
_NEGATION_TEMPLATES = (
    r"sans\s+(?:la\s+|le\s+|une?\s+)?{alias}",
    r"pas\s+(?:de\s+|d')?{alias}",
    r"n['\s]?ai\s+pas\s+(?:de\s+|d')?{alias}",
    r"je\s+n['\s]?ai\s+pas\s+(?:de\s+|d')?{alias}",
    r"ne\s+possede\s+pas\s+(?:de\s+|d')?{alias}",
    r"aucune?\s+{alias}",
    r"jamais\s+eu\s+(?:de\s+|d')?{alias}",
    r"pas\s+encore\s+(?:de\s+|d')?{alias}",
)
_TARGET_RE = re.compile(
    r"(?:demander|demande|obtenir|faire|renouveler|passer|deposer|déposer|"
    r"constituer|commander|delivrer|délivrer|declarer|déclarer|refaire|perdre)"
    r".{0,55}?"
    r"(passeport|cnie|cine|carte\s+nationale|acte\s+de\s+naissance|watiqa)",
    re.I,
)
_TARGET_PERDU_RE = re.compile(
    r"(?:perdu|perdue|perte|vol|vole)\s+(?:mon\s+|ma\s+|le\s+|la\s+)?(passeport|cnie)",
    re.I,
)
_TARGET_RE_REV = re.compile(
    r"(passeport|cnie|cine|carte\s+nationale|acte\s+de\s+naissance|watiqa)"
    r".{0,40}?"
    r"(?:demander|demande|obtenir|faire|renouveler|pieces|pièces|procedure|procédure)",
    re.I,
)


@dataclass
class PhraseContext:
    """Résumé interprétatif d'une question utilisateur."""

    question_folded: str = ""
    primary_subject: str | None = None
    subject_scores: dict[str, float] = field(default_factory=dict)
    negated_subjects: set[str] = field(default_factory=set)
    requested_subjects: set[str] = field(default_factory=set)
    is_procedure_request: bool = False
    age_hint: str | None = None  # mineur | majeur


def _fold(s: str) -> str:
    t = unicodedata.normalize("NFD", s or "")
    t = "".join(c for c in t if unicodedata.category(c) != "Mn")
    return t.lower().replace("'", " ").replace("’", " ")


def _alias_to_subject(alias: str) -> str | None:
    a = _fold(alias)
    for subj, aliases in _SUBJECT_ALIASES.items():
        for al in aliases:
            if al in a or a in al:
                return subj
    return None


def _subject_negated(q: str, subject: str) -> bool:
    aliases = _SUBJECT_ALIASES.get(subject, ())
    for alias in aliases:
        al = re.escape(_fold(alias))
        for tmpl in _NEGATION_TEMPLATES:
            if re.search(tmpl.format(alias=al), q):
                return True
    return False


def _detect_requested_subjects(q: str) -> set[str]:
    found: set[str] = set()
    for m in _TARGET_PERDU_RE.finditer(q):
        subj = _alias_to_subject(m.group(1))
        if subj:
            found.add(subj)
    for m in _TARGET_RE.finditer(q):
        subj = _alias_to_subject(m.group(1))
        if subj:
            found.add(subj)
    for m in _TARGET_RE_REV.finditer(q):
        subj = _alias_to_subject(m.group(1))
        if subj:
            found.add(subj)
    if _REQUEST_RE.search(q) and _ACTION_RE.search(q):
        for subj, aliases in _SUBJECT_ALIASES.items():
            if any(a in q for a in (_fold(x) for x in aliases)):
                if not _subject_negated(q, subj):
                    found.add(subj)
    return found


def _score_subjects(q: str) -> dict[str, float]:
    scores: dict[str, float] = {k: 0.0 for k in _SUBJECT_ALIASES}
    requested = _detect_requested_subjects(q)
    has_request = bool(_REQUEST_RE.search(q) or _ACTION_RE.search(q))

    for subj, aliases in _SUBJECT_ALIASES.items():
        if _subject_negated(q, subj):
            scores[subj] = -8.0
            continue
        mention = 0.0
        for al in aliases:
            if _fold(al) in q:
                mention += 2.0 if " " in al else 1.0
        if subj in requested:
            mention += 6.0
        if has_request and subj in requested:
            mention += 4.0
        # Pénaliser une simple mention sans action vers ce document
        if mention > 0 and subj not in requested and has_request:
            other_requested = requested - {subj}
            if other_requested:
                mention *= 0.25
        scores[subj] = mention

    return scores


def _age_hint(q: str) -> str | None:
    if re.search(r"\bmineur(?:e)?s?\b", q):
        return "mineur"
    if re.search(r"\bmajeur(?:e)?s?\b", q):
        return "majeur"
    if re.search(r"\bmoins\s+de\s+18\b", q):
        return "mineur"
    return None


def analyze_phrase(question: str) -> PhraseContext:
    q = _fold(question)
    scores = _score_subjects(q)
    negated = {s for s in _SUBJECT_ALIASES if _subject_negated(q, s)}
    requested = _detect_requested_subjects(q)

    ranked = sorted(scores.items(), key=lambda x: x[1], reverse=True)
    primary: str | None = None
    if ranked and ranked[0][1] > 0.5:
        top_score = ranked[0][1]
        second = ranked[1][1] if len(ranked) > 1 else 0.0
        if top_score >= 2.0 and (top_score - second) >= 1.0:
            primary = ranked[0][0]

    return PhraseContext(
        question_folded=q,
        primary_subject=primary,
        subject_scores=scores,
        negated_subjects=negated,
        requested_subjects=requested,
        is_procedure_request=bool(
            _ACTION_RE.search(q)
            or re.search(r"\b(?:procedure|procédure|demarche|démarche|pieces|pièces)\b", q)
        ),
        age_hint=_age_hint(q),
    )


def primary_portal_intent(question: str) -> str | None:
    """Portail cible (cnie / passeport / watiqa) selon le sens de la phrase."""
    ctx = analyze_phrase(question)
    if ctx.primary_subject in ("cnie", "passeport", "watiqa"):
        return ctx.primary_subject
    if ctx.requested_subjects:
        for subj in ("passeport", "cnie", "watiqa"):
            if subj in ctx.requested_subjects and subj not in ctx.negated_subjects:
                return subj
    return None


def subject_is_primary(question: str, subject: str) -> bool:
    ctx = analyze_phrase(question)
    return ctx.primary_subject == subject


def subject_mentioned_but_not_target(question: str, subject: str) -> bool:
    """Le sujet est cité mais ce n'est pas la demande (ex. « sans CNIE » + passeport)."""
    ctx = analyze_phrase(question)
    if subject in ctx.negated_subjects:
        return True
    if ctx.primary_subject and ctx.primary_subject != subject:
        q = ctx.question_folded
        if any(_fold(a) in q for a in _SUBJECT_ALIASES.get(subject, ())):
            return True
    return False


def hit_matches_primary_subject(question: str, hit: dict[str, Any]) -> bool:
    ctx = analyze_phrase(question)
    if not ctx.primary_subject:
        try:
            from app.dataset_registry import hit_matches_question_domains

            return hit_matches_question_domains(question, hit)
        except ImportError:
            return True
    meta = hit.get("metadata") or {}
    text = _fold(
        " ".join(
            (
                str(hit.get("text") or ""),
                str(meta.get("label") or ""),
                str(meta.get("title") or ""),
                str(meta.get("category") or ""),
                str(meta.get("chunk_id") or ""),
            )
        )
    )
    subj = ctx.primary_subject
    if subj == "passeport":
        if "passeport" in text or "consulat" in text:
            return True
        cid = str(meta.get("chunk_id") or "")
        url = str(meta.get("source_url") or "").lower()
        if meta.get("category") == "cnie" or "cnie.ma" in url:
            if cid.startswith("curated::cnie") or cid.startswith("cnie_procedure_"):
                return False
            if "premiere demande" in text and "passeport" not in text:
                return False
        return False
    if subj == "cnie":
        return "cnie" in text or "carte nationale" in text
    if subj == "watiqa":
        return "watiqa" in text or "acte de naissance" in text
    if subj == "labor":
        return any(
            x in text
            for x in (
                "code du travail",
                "loi 65-99",
                "heures suppl",
                "licenciement",
                "preavis",
                "préavis",
                "smig",
                "conge annuel",
            )
        )
    if subj == "construction":
        return any(
            x in text
            for x in (
                "permis de construire",
                "autorisation de construire",
                "urbanisme",
                "construction",
            )
        )
    return True


def top_hits_match_primary_subject(
    question: str, hits: list[dict[str, Any]], n: int = 3
) -> bool:
    ctx = analyze_phrase(question)
    if not ctx.primary_subject or not hits:
        return True
    return any(hit_matches_primary_subject(question, h) for h in hits[:n])
