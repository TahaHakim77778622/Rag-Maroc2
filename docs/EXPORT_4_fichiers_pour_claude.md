# Export pour Claude — modules corpus / requête

---
## FICHIER: app/corpus_coverage.py
```python
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


def _required_phrases_ok(question: str, top_text: str) -> bool:
    q = _fold(question)
    subs_needed: set[str] = set()
    for needle, sub in _REQUIRED_PHRASES:
        if _fold(needle) in q:
            subs_needed.add(_fold(sub))
    if not subs_needed:
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


def corpus_covers_question(question: str, hits: list[dict[str, Any]]) -> bool:
    """
    True = les meilleurs extraits du dataset permettent de répondre (pas de web).
    False = sujet absent ou hors sujet → activer fallback web.
    """
    if not hits:
        return False

    try:
        from app.labor_corpus import is_labor_code_question, labor_hits_substantively_answer

        if is_labor_code_question(question):
            return labor_hits_substantively_answer(question, hits)
    except ImportError:
        pass

    # Registre global dataset (BO SGG, admin, tous domaines) — avant filtres lexicaux stricts
    try:
        from app.dataset_registry import dataset_covers_via_registry

        reg = dataset_covers_via_registry(question, hits)
        if reg is True:
            return True
        if reg is False:
            return False
    except ImportError:
        pass

    top_text = _combined_top_text(hits, 3)
    top_sc = _hit_score(hits[0])
    min_top = _float_env("CORPUS_MIN_TOP_SCORE", 0.26)

    if top_sc < min_top:
        return False

    if not _required_phrases_ok(question, top_text):
        return False

    if not _term_overlap_ok(question, top_text):
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
            return False
        if cnie_case_from_question(question) and not top_hits_match_cnie_case(question, hits):
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
        if cid.startswith("curated::labor") or cid.startswith("curated::watiqa"):
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

    return top_sc >= min_top and _term_overlap_ok(question, top_text)


def explain_coverage(question: str, hits: list[dict[str, Any]]) -> str:
    """Courte explication pour logs / debug."""
    if not hits:
        return "aucun extrait"
    if corpus_covers_question(question, hits):
        return "corpus_ok"
    top_text = _combined_top_text(hits, 2)[:120]
    sc = _hit_score(hits[0])
    return f"corpus_insuffisant score={sc:.2f} extrait={top_text!r}..."
```

---
## FICHIER: app/corpus_first.py
```python
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

    from app.corpus_coverage import corpus_covers_question, explain_coverage

    if corpus_covers_question(question, prepared):
        logger.info("Dataset suffisant (corpus_first) — %s", explain_coverage(question, prepared))
        return False, prepared

    logger.info(
        "Dataset insuffisant → web — %s",
        explain_coverage(question, prepared),
    )
    return True, prepared
```

---
## FICHIER: app/query_understanding.py
```python
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
        "hints": "urbanisme permis construire autorisation maroc bulletin officiel",
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
```

---
## FICHIER: app/dataset_registry.py
```python
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
        "categories": ("urbanisme", "juridique"),
        "text_markers": (
            "permis de construire",
            "autorisation de construire",
            "urbanisme",
            "lotissement",
            "construction",
            "construire",
            "rokhas",
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
    ("permis de construire", "construction"),
    ("urbanisme", "construction"),
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
        from app.phrase_context import analyze_phrase

        ctx = analyze_phrase(question)
        if ctx.primary_subject and ctx.primary_subject in _DOMAIN_SPECS:
            domains.append(ctx.primary_subject)
        elif ctx.primary_subject == "labor":
            domains.append("labor")
    except ImportError:
        pass

    try:
        from app.query_understanding import analyze_query

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

    from app.corpus_coverage import _combined_top_text, _hit_score, _term_overlap_ok

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
            from app.passeport_cases import passeport_case_from_question

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

    if "construction" in domains or "education" in domains or "fiscalite" in domains:
        return _term_overlap_ok(question, top_text)

    return None
```

