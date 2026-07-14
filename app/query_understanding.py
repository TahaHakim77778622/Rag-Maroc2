"""
Compréhension de requête pour le RAG : objectif utilisateur, sujets, relances, transitions.

Centralise la logique auparavant dispersée (master vs doctorat, CNIE, etc.)
pour mieux interpréter les formulations naturelles en français.
"""

from __future__ import annotations

import re
import unicodedata
from dataclasses import dataclass, field
from typing import Any

_TOKEN_RE = re.compile(r"[a-z0-9àâäéèêëïîôùûœç]{2,}", re.IGNORECASE)

_STOP = frozenset(
    """
    le la les un une des du de d et ou mais pour sur sous dans avec sans au aux
    ce cet cette ces mon ma mes ton ta tes son sa ses leur leurs notre nos votre vos
    est sont etre été ete avoir fait faire plus moins tres bien non oui pas ne ni
    comment quoi quel quelle quels quelles ou qui que dont est-ce estce svp stp moi
    je tu il elle on nous vous ils elles me te se
    """.split()
)

# Objectifs utilisateur (formulations courantes)
_GOAL_PROCEDURE = re.compile(
    r"\b(comment|procedure|procédure|démarche|demarche|étapes?|etapes?|"
    r"pi[eè]ces?|documents?|fournir|déposer|deposer|obtenir|faire|passer|passage)\b",
    re.I,
)
_GOAL_INFO = re.compile(
    r"\b(quelle?s?|combien|montant|délai|delai|durée|duree|conditions?|"
    r"définition|definition|signifie|expliquer)\b",
    re.I,
)
_TRANSITION_RE = re.compile(
    r"(?:passage|passer|après|apres|suite|ensuite|vers|vers le|vers la|"
    r"du .+ vers|de .+ vers|de .+ au|de .+ à)\b",
    re.I,
)
_PRONOUN_FOLLOWUP_RE = re.compile(
    r"\b(le passage|la procédure|ce sujet|ça|cela|celui|celle|les mêmes|pareil)\b",
    re.I,
)

_TOPIC_SPECS: list[dict[str, Any]] = [
    {
        "id": "cnie",
        "weight": 1.0,
        "keywords": (
            "cnie",
            "cine",
            "carte nationale",
            "identite electronique",
            "identité électronique",
            "dgsn",
        ),
        "hints": (
            "carte nationale identite electronique cnie dgsn delivrance renouvellement "
            "documents pieces premiere demande"
        ),
    },
    {
        "id": "passeport",
        "weight": 1.0,
        "keywords": ("passeport", "passeport.ma", "consulat", "biometrique", "biométrique"),
        "hints": (
            "passeport biométrique maroc délivrance renouvellement immatriculation consulaire "
            "cnie pieces formulaire"
        ),
    },
    {
        "id": "watiqa",
        "weight": 1.0,
        "keywords": ("watiqa", "acte de naissance", "etat civil", "état civil", "extrait", "copie integrale"),
        "hints": "watiqa guichet etat civil acte naissance commande en ligne",
    },
    {
        "id": "labor",
        "weight": 1.0,
        "keywords": (
            "smig",
            "salaire minimum",
            "code du travail",
            "licenciement",
            "préavis",
            "preavis",
            "démission",
            "demission",
            "congé",
            "conge",
            "heures supplémentaires",
            "heures supplementaires",
        ),
        "hints": (
            "code du travail loi 65-99 heures supplementaires majoration article 201 202 "
            "smig salaire minimum legal conge annuel preavis licenciement maroc"
        ),
    },
    {
        "id": "construction",
        "weight": 1.0,
        "keywords": (
            "construire",
            "construction",
            "permis de construire",
            "autorisation de construire",
            "urbanisme",
            "rokhas",
            "lotissement",
            "copropriété",
            "copropriete",
        ),
        "hints": (
            "autorisation de construire permis construire maroc procedure dossier architecte "
            "rokhas plateforme urbanisme dahir decret loi bulletin officiel pieces requises"
        ),
    },
    {
        "id": "education_doctorate",
        "weight": 1.2,
        "keywords": (
            "doctorat",
            "doctoral",
            "doctorants",
            "these",
            "thèse",
            "phd",
            "3e cycle",
            "troisieme cycle",
            "troisième cycle",
            "cycle doctoral",
        ),
        "hints": (
            "cycle doctoral inscription these loi 01-00 enseignement superieur recherche "
            "scientifique doctorants conditions acces"
        ),
        "suppress_topics": ("education_master",),
    },
    {
        "id": "education_master",
        "weight": 1.0,
        "keywords": (
            "cycle de master",
            "cycle master",
            "normes pedagogiques",
            "normes pédagogiques",
            "cahier des normes",
            "credits master",
            "semestre master",
        ),
        "hints": (
            "normes pedagogiques nationales cycle master credits semestres filiere "
            "enseignement superieur"
        ),
        "suppress_if_topics": ("education_doctorate",),
    },
    {
        "id": "education_master_generic",
        "weight": 0.35,
        "keywords": ("master", "licence", "universite", "université", "enseignement superieur"),
        "hints": "enseignement superieur universite maroc",
        "suppress_if_topics": ("education_doctorate",),
    },
    {
        "id": "bulletin",
        "weight": 0.5,
        "keywords": (
            "bulletin officiel",
            "bulletin",
            "buletin",
            "journal officiel",
            "dahir",
            "decret",
            "décret",
            "arrêté",
            "arrete",
        ),
        "hints": "bulletin officiel maroc dahir decret arrete loi article sgg",
    },
    {
        "id": "fiscalite",
        "weight": 0.9,
        "keywords": (
            "impot",
            "impôt",
            "fiscal",
            "fiscalite",
            "fiscalité",
            "taxe",
            "tva",
            "ir",
            "isf",
        ),
        "hints": "fiscalite impot taxe code general impots maroc bulletin officiel",
    },
    {
        "id": "sante",
        "weight": 0.8,
        "keywords": (
            "sante",
            "santé",
            "hopital",
            "hôpital",
            "medicament",
            "médicament",
            "cnops",
            "amo",
        ),
        "hints": "sante publique hopital medicament maroc loi decret",
    },
    {
        "id": "marches_publics",
        "weight": 0.85,
        "keywords": (
            "marche public",
            "marché public",
            "marches publics",
            "appel d'offres",
            "soumissionnaire",
            "adjudication",
        ),
        "hints": "marches publics appel offres adjudication bulletin officiel maroc",
    },
]


@dataclass
class QueryAnalysis:
    """Résultat d'analyse pour retrieval + rerank + prompts."""

    original: str
    resolved: str
    goal: str  # procedure | info | transition | general
    topics: list[str] = field(default_factory=list)
    retrieval_hints: list[str] = field(default_factory=list)
    use_conversation_history: bool = False
    # Flags rerank (compatibles avec retrieval_rerank existant)
    cnie_intent: bool = False
    passeport_intent: bool = False
    watiqa_intent: bool = False
    primary_subject: str | None = None
    labor_intent: bool = False
    bo_intent: bool = False
    education_master_intent: bool = False
    education_doctorate_intent: bool = False
    # corpus = règles seules ; corpus+llm = réécriture LLM avant FAISS (~20 % cible)
    retrieval_path: str = "corpus"


def _strip_accents(text: str) -> str:
    return "".join(
        ch for ch in unicodedata.normalize("NFD", text or "") if unicodedata.category(ch) != "Mn"
    )


def _normalize(text: str) -> str:
    return _strip_accents((text or "").lower())


def _tokens(text: str) -> list[str]:
    out: list[str] = []
    for t in _TOKEN_RE.findall(_normalize(text)):
        if t not in _STOP and len(t) >= 2:
            out.append(t)
    return out


def _history_user_text(history: list[dict[str, Any]] | None, max_turns: int = 5) -> str:
    if not history:
        return ""
    parts: list[str] = []
    for m in history[-14:]:
        if not isinstance(m, dict):
            continue
        if str(m.get("role", "")).strip().lower() != "user":
            continue
        c = str(m.get("content", "")).strip()
        if c:
            parts.append(c)
    return "\n".join(parts[-max_turns:])


def _detect_goal(text: str) -> str:
    low = _normalize(text)
    if _TRANSITION_RE.search(low):
        return "transition"
    if _GOAL_PROCEDURE.search(low):
        return "procedure"
    if _GOAL_INFO.search(low):
        return "info"
    return "general"


def _score_topics(text: str) -> dict[str, float]:
    low = _normalize(text)
    scores: dict[str, float] = {}
    for spec in _TOPIC_SPECS:
        tid = spec["id"]
        w = float(spec.get("weight", 1.0))
        hit = 0.0
        for kw in spec.get("keywords", ()):
            kn = _normalize(kw)
            if kn in low:
                hit += w * (2.0 if " " in kn else 1.0)
        if hit:
            scores[tid] = scores.get(tid, 0.0) + hit
    return scores


def _apply_topic_suppression(scores: dict[str, float]) -> dict[str, float]:
    out = dict(scores)
    active = {k for k, v in out.items() if v > 0}
    for spec in _TOPIC_SPECS:
        tid = spec["id"]
        if tid not in active:
            continue
        for other in spec.get("suppress_topics", ()):
            if other in active:
                out[other] = out.get(other, 0.0) * 0.15
        for other in spec.get("suppress_if_topics", ()):
            if other in active and out.get(other, 0) >= out.get(tid, 0):
                out[tid] = out.get(tid, 0.0) * 0.12
    # master générique affaibli si doctorat présent avec objectif procédure/transition
    if out.get("education_doctorate", 0) > 0 and out.get("education_master_generic", 0) > 0:
        out["education_master_generic"] *= 0.1
        out["education_master"] = out.get("education_master", 0) * 0.2
    return out


def _ordered_topics(scores: dict[str, float], min_score: float = 0.8) -> list[str]:
    items = [(k, v) for k, v in scores.items() if v >= min_score]
    items.sort(key=lambda x: x[1], reverse=True)
    return [k for k, _ in items]


def _is_followup_ellipsis(question: str) -> bool:
    q = _normalize(question.strip())
    if not q:
        return False
    starters = (
        "et pour",
        "et sinon",
        "et le",
        "et la",
        "et les",
        "pour le renouvellement",
        "pour renouvellement",
        "moi je veux",
        "je veux le",
        "je veux la",
    )
    if any(q.startswith(s) for s in starters):
        return True
    if _PRONOUN_FOLLOWUP_RE.search(q):
        return True
    words = _tokens(q)
    return len(words) <= 6 and any(
        w in q for w in ("renouvellement", "perte", "vol", "delai", "passage", "doctorat", "master")
    )


def _is_ambiguous_short_query(question: str) -> bool:
    toks = _tokens(question)
    if len(toks) <= 4:
        return True
    low = _normalize(question)
    vague = (
        "comment renouveler",
        "quelles pieces",
        "quelle procedure",
        "comment faire",
        "je veux",
    )
    return any(v in low for v in vague)


def needs_conversation_context(question: str, history: list[dict[str, Any]] | None) -> bool:
    """Historique utile pour embedding / résolution (au-delà des seules relances « et pour ? »)."""
    if not history:
        return False
    q = (question or "").strip()
    if not q:
        return False
    if _is_followup_ellipsis(q):
        return True
    if _is_ambiguous_short_query(q):
        return True
    low = _normalize(q)
    toks = _tokens(q)
    if len(toks) <= 10 and _PRONOUN_FOLLOWUP_RE.search(low):
        return True
    if len(toks) <= 8 and re.search(r"\b(le|la|les|ce|cette|cet|ça|ca)\b", low):
        return True
    # Suite explicite du tour précédent
    if len(toks) <= 12 and any(
        p in low
        for p in (
            "je veux",
            "moi je",
            "pas ça",
            "pas ca",
            "autre chose",
            "plutot",
            "plutôt",
            "precisement",
            "précisément",
        )
    ):
        return True
    return False


def _last_topic_phrase(history: list[dict[str, Any]]) -> str:
    """Dernière formulation utilisateur substantielle."""
    for m in reversed(history):
        if str(m.get("role", "")).strip().lower() != "user":
            continue
        content = str(m.get("content", "")).strip()
        if content and len(_tokens(content)) >= 3:
            return content
    return ""


def resolve_question(question: str, history: list[dict[str, Any]] | None) -> str:
    """
    Reformule la question avec le contexte conversationnel si nécessaire.
    Ex. tour 1 : master + doctorat vague ; tour 2 : « je veux le passage » → fusion.
    """
    q = (question or "").strip()
    if not q or not history:
        return q

    hist_text = _history_user_text(history, max_turns=3)
    if not hist_text:
        return q

    if not needs_conversation_context(q, history):
        return q

    prev = _last_topic_phrase(history)
    if not prev:
        return q

    qn = _normalize(q)
    pn = _normalize(prev)
    # Déjà suffisamment explicite
    if len(_tokens(prev)) >= 4 and all(t in qn or t in pn for t in _tokens(q)[:6] if len(t) > 4):
        if _tokens(q) and max(len(_tokens(q)), 1) / max(len(_tokens(prev)), 1) > 0.5:
            pass
        elif pn in qn or qn in pn:
            return q

    # Relance courte : concaténer l'intention précédente
    if _is_followup_ellipsis(q) or len(_tokens(q)) <= 8:
        return f"{q}\n(Contexte demande précédente: {prev})"

    if _PRONOUN_FOLLOWUP_RE.search(qn) or qn.startswith("je veux"):
        return f"{q}\n(Précision par rapport à: {prev})"

    return f"{q}\n(Contexte: {prev})"


def analyze_query(
    question: str,
    *,
    history: list[dict[str, Any]] | None = None,
    profile_hint: str = "",
    retrieval_query: str = "",
) -> QueryAnalysis:
    """
    Analyse complète : question résolue, sujets, hints retrieval, flags rerank.
    """
    original = (question or "").strip()
    resolved = resolve_question(original, history)
    blob = resolved
    if profile_hint:
        blob = f"{blob}\n{profile_hint}"
    if retrieval_query:
        blob = f"{blob}\n{retrieval_query}"
    if history and needs_conversation_context(original, history):
        blob = f"{blob}\n{_history_user_text(history, max_turns=4)}"

    goal = _detect_goal(blob)
    scores = _apply_topic_suppression(_score_topics(blob))
    pctx = None
    try:
        from app.phrase_context import analyze_phrase

        pctx = analyze_phrase(blob)
        for subj, sc in pctx.subject_scores.items():
            tid = subj
            if tid in ("cnie", "passeport", "watiqa"):
                if sc < 0:
                    scores[tid] = scores.get(tid, 0.0) * 0.05
                elif sc > 0:
                    scores[tid] = scores.get(tid, 0.0) + sc * 1.5
        if pctx.primary_subject and pctx.primary_subject in scores:
            for other in ("cnie", "passeport", "watiqa"):
                if other != pctx.primary_subject:
                    scores[other] = scores.get(other, 0.0) * 0.12
            scores[pctx.primary_subject] = scores.get(pctx.primary_subject, 0.0) + 4.0
    except ImportError:
        pctx = None
    topics = _ordered_topics(scores)

    hints: list[str] = []
    seen_hints: set[str] = set()
    for tid in topics:
        for spec in _TOPIC_SPECS:
            if spec["id"] == tid:
                h = str(spec.get("hints", "")).strip()
                if h and h not in seen_hints:
                    seen_hints.add(h)
                    hints.append(h)

    # Transition explicite : renforcer le sujet cible (2e terme souvent après master)
    low = _normalize(blob)
    if goal == "transition" and "education_doctorate" in topics:
        hints.insert(
            0,
            "passage master vers doctorat inscription these conditions acces 3e cycle universite maroc",
        )
    if goal == "procedure" and topics and topics[0] not in ("bulletin",):
        hints.append(f"procedure demarche maroc {topics[0].replace('_', ' ')}")

    qlow = _normalize(blob)
    primary_subject = pctx.primary_subject if pctx else None
    passeport_intent = primary_subject == "passeport" or (
        "passeport" in topics and primary_subject != "cnie"
    )
    cnie_intent = primary_subject == "cnie" or (
        "cnie" in topics and primary_subject not in ("passeport", "watiqa")
    )
    if pctx and pctx.negated_subjects:
        if "cnie" in pctx.negated_subjects:
            cnie_intent = False
        if "passeport" in pctx.negated_subjects:
            passeport_intent = False
    if cnie_intent and not passeport_intent:
        cnie_intent = any(
            k in qlow
            for k in (
                "piece",
                "pièce",
                "pieces",
                "pièces",
                "document",
                "demande",
                "premiere",
                "première",
                "renouvel",
                "fournir",
                "photo",
                "delai",
                "délai",
                "procedure",
                "procédure",
            )
        ) or "cnie" in topics
    if passeport_intent:
        cnie_intent = False

    analysis = QueryAnalysis(
        original=original,
        resolved=resolved,
        goal=goal,
        topics=topics,
        retrieval_hints=hints,
        use_conversation_history=needs_conversation_context(original, history),
        cnie_intent=cnie_intent,
        passeport_intent=passeport_intent,
        primary_subject=primary_subject,
        watiqa_intent="watiqa" in topics
        or ("watiqa" in qlow)
        or ("état civil" in qlow and any(x in qlow for x in ("guichet", "commande", "acte"))),
        labor_intent="labor" in topics,
        bo_intent=any(
            x in qlow
            for x in (
                "bulletin officiel",
                "bulletin",
                "journal officiel",
                "dahir",
                "decret",
                "décret",
                "arrêté",
                "arrete",
            )
        )
        or ("article" in qlow and "loi" in qlow),
        education_doctorate_intent="education_doctorate" in topics,
        education_master_intent=(
            ("education_master" in topics or scores.get("education_master_generic", 0) >= 0.35)
            and "education_doctorate" not in topics
        ),
    )
    # master générique seul (sans doctorat dominant)
    if (
        not analysis.education_doctorate_intent
        and scores.get("education_master_generic", 0) >= 0.35
        and "master" in qlow
    ):
        analysis.education_master_intent = True

    return analysis


def build_retrieval_query(
    analysis: QueryAnalysis,
    *,
    embed_prefix: str = "",
) -> str:
    """Construit la requête enrichie pour FAISS à partir d'une analyse déjà calculée."""
    q = (embed_prefix or "").strip() or analysis.resolved or analysis.original
    if not q or not analysis.retrieval_hints:
        return q
    dedup: list[str] = []
    seen: set[str] = set()
    for h in analysis.retrieval_hints:
        hh = " ".join(h.split())
        if hh and hh not in seen:
            seen.add(hh)
            dedup.append(hh)
    return q + "\n\n" + " ".join(dedup)


def expand_query_for_retrieval(
    question: str,
    *,
    history: list[dict[str, Any]] | None = None,
    profile_hint: str = "",
) -> str:
    """Construit la requête enrichie pour FAISS (analyse + hints)."""
    return build_retrieval_query(
        analyze_query(question, history=history, profile_hint=profile_hint)
    )
