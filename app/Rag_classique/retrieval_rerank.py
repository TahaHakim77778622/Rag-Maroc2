"""
Reranking hybride :
- score vectoriel FAISS
- score lexical BM25-like (mot-clés)
- signaux métiers (bonus/malus domaine)
- option cross-encoder (activable en env)
"""

from __future__ import annotations

import logging
import math
import os
import re
import unicodedata
from functools import lru_cache
from typing import Any

logger = logging.getLogger(__name__)

# Mots trop courts / peu discriminants pour le scoring lexical
_STOP = frozenset(
    """
    pour les des une un une la le et ou mais avec sans son sa ses est sont été être
    avoir fait faire plus moins très tout tous toute dans sur par que qui dont où
    quel quels quelle quelles comme tel tels lors ainsi chez aux ces ses leur leurs
    faut peut doit même aussi bien non oui pas ne ni maroc marocaine marocain
    """.split()
)

_TOKEN_RE = re.compile(r"[a-z0-9àâäéèêëïîôùûœç]{2,}", re.IGNORECASE)


def _float_env(name: str, default: float) -> float:
    try:
        return float(os.environ.get(name, str(default)))
    except ValueError:
        return default


def _int_env(name: str, default: int) -> int:
    try:
        return int(os.environ.get(name, str(default)))
    except ValueError:
        return default


def _bool_env(name: str, default: bool = False) -> bool:
    raw = os.environ.get(name)
    if raw is None:
        return default
    return raw.strip().lower() in {"1", "true", "yes", "on"}


def _strip_accents(text: str) -> str:
    return "".join(
        ch for ch in unicodedata.normalize("NFD", text or "") if unicodedata.category(ch) != "Mn"
    )


def _tokenize(text: str) -> list[str]:
    folded = _strip_accents((text or "").lower())
    toks = _TOKEN_RE.findall(folded)
    out: list[str] = []
    for t in toks:
        if len(t) < 2 or t in _STOP:
            continue
        out.append(t)
    return out


def _minmax_norm(values: list[float]) -> list[float]:
    if not values:
        return []
    vmin = min(values)
    vmax = max(values)
    if math.isclose(vmin, vmax):
        return [1.0 for _ in values]
    span = vmax - vmin
    return [(v - vmin) / span for v in values]


def _terms_from_texts(question: str, retrieval_query: str) -> set[str]:
    raw = f"{question} {retrieval_query}".lower()
    terms = set(_tokenize(raw))
    # Acronymes / courts utiles
    for token in ("cnie", "dgsn", "cine", "watiqa", "master", "bo"):
        if token in raw:
            terms.add(token)
    return terms


def _profile_penalty(question: str, text: str) -> float:
    """Pénalité si le passage contredit explicitement le profil utilisateur."""
    q = question.lower()
    t = text.lower()
    penalty = 0.0

    if "celibataire" in q or "célibataire" in q:
        contradictory = (
            "acte de mariage",
            "mariage",
            "conjoint",
            "epoux",
            "époux",
            "epouse",
            "épouse",
            "veuf",
            "veuve",
            "divorce",
            "divorcé",
            "divorcee",
        )
        if any(k in t for k in contradictory):
            penalty += 0.45

    if "marie" in q or "marié" in q or "mariée" in q:
        if "celibataire" in t or "célibataire" in t:
            penalty += 0.25

    return penalty


def _education_doctorate_intent(qlow: str) -> bool:
    """Passage master → doctorat, inscription en thèse / 3e cycle (pas les normes pédagogiques du master)."""
    if not any(
        x in qlow
        for x in (
            "doctorat",
            "doctoral",
            "these",
            "thèse",
            "phd",
            "3e cycle",
            "3ème cycle",
            "troisieme cycle",
            "troisième cycle",
        )
    ):
        return False
    if "master" in qlow and any(x in qlow for x in ("doctorat", "doctoral", "these", "thèse")):
        return True
    return any(
        x in qlow
        for x in (
            "passage",
            "passer",
            "inscription",
            "inscrire",
            "accès",
            "acces",
            "candidature",
            "candidater",
            "après le master",
            "apres le master",
            "vers le doctorat",
            "faire un doctorat",
            "faire la doctorat",
            "conditions d'accès",
            "conditions d acces",
        )
    )


def _education_master_intent(qlow: str) -> bool:
    """Question sur le cycle master / normes pédagogiques (enseignement supérieur), pas sur tout le BO."""
    if _education_doctorate_intent(qlow):
        return False
    return any(
        x in qlow
        for x in (
            "cycle de master",
            "cycle master",
            "norme pédagogique",
            "normes pédagogiques",
            "norme pedagogique",
            "normes pedagogiques",
            "cahier des normes",
            "diplôme d'études universitaires",
            "diplome d'etudes universitaires",
            "projet de fin d'études",
            "projet de fin d'etudes",
            "filière du cycle de master",
            "filiere du cycle de master",
        )
    ) or (
        "master" in qlow
        and any(
            x in qlow
            for x in (
                "pedagogique",
                "pédagogique",
                "enseignement superieur",
                "enseignement supérieur",
                "universit",
                "credits",
                "crédits",
                "semestre",
                "filiere",
                "filière",
            )
        )
    )


def _procurement_public_chunk(text: str) -> bool:
    """Passage manifestement lié aux marchés publics (hors sujet pour master / pédagogie)."""
    t = text.lower()
    if "marchés publics" in t or "marches publics" in t:
        return True
    if "marché public" in t or "marche public" in t:
        if any(x in t for x in ("soumissionnaire", "consultation", "allotissement", "dossier de candidature")):
            return True
    if "soumissionnaire" in t and "règlement de consultation" in t:
        return True
    if "règlement de consultation" in t and "pièces justificatives" in t:
        return True
    return False


def _education_master_chunk_bonus(text: str) -> float:
    t = text.lower()
    bonus = 0.0
    phrases = (
        ("cycle de master", 5.0),
        ("normes pédagogiques", 6.0),
        ("normes pedagogiques", 6.0),
        ("cahier des normes", 6.0),
        ("cycle master", 4.0),
        ("enseignement supérieur", 3.0),
        ("enseignement superieur", 3.0),
        ("ministère de l'enseignement supérieur", 4.0),
        ("ministere de l'enseignement superieur", 4.0),
        ("recherche scientifique et de l'innovation", 3.0),
        ("diplôme national", 2.5),
        ("projet de fin d'études", 3.0),
        ("fin d'études", 2.0),
        ("filière du cycle de master", 4.0),
        ("modules disciplinaires", 2.5),
        ("nombre de crédits", 2.5),
        ("nombre de credits", 2.5),
        ("accréditation", 2.0),
        ("article premier", 2.0),
    )
    for needle, add in phrases:
        if needle in t:
            bonus += add
    if " master" in t or t.startswith("master ") or " master." in t:
        bonus += 2.0
    return bonus


def _education_doctorate_chunk_bonus(text: str) -> float:
    t = text.lower()
    bonus = 0.0
    phrases = (
        ("cycle doctoral", 8.0),
        ("inscriptions en cycle doctoral", 10.0),
        ("inscription en cycle doctoral", 10.0),
        ("achèvement des thèses", 8.0),
        ("achevement des theses", 8.0),
        ("recherches doctorales", 6.0),
        ("diplôme de doctorat", 5.0),
        ("diplome de doctorat", 5.0),
        ("loi 01-00", 4.0),
        ("enseignement supérieur", 3.0),
        ("enseignement superieur", 3.0),
        ("thèses", 4.0),
        ("theses", 4.0),
        ("doctorants", 6.0),
        ("financement des doctorants", 7.0),
    )
    for needle, add in phrases:
        if needle in t:
            bonus += add
    return bonus


def _education_master_norms_chunk(text: str) -> bool:
    t = text.lower()
    return (
        "normes pédagogiques" in t or "normes pedagogiques" in t
    ) and ("cycle de master" in t or "cycle master" in t)


def _build_bm25_scores(hits: list[dict[str, Any]], query_tokens: list[str]) -> list[float]:
    if not hits:
        return []
    if not query_tokens:
        return [0.0 for _ in hits]

    docs_tokens = [_tokenize(str(h.get("text") or "")) for h in hits]
    n_docs = len(docs_tokens)
    avgdl = (sum(len(d) for d in docs_tokens) / n_docs) if n_docs else 1.0
    if avgdl <= 0:
        avgdl = 1.0

    # fréquence documentaire
    df: dict[str, int] = {}
    for d in docs_tokens:
        for tok in set(d):
            df[tok] = df.get(tok, 0) + 1

    k1 = _float_env("RERANK_BM25_K1", 1.2)
    b = _float_env("RERANK_BM25_B", 0.75)

    out: list[float] = []
    for d in docs_tokens:
        dl = len(d)
        tf: dict[str, int] = {}
        for t in d:
            tf[t] = tf.get(t, 0) + 1

        score = 0.0
        for t in query_tokens:
            f = tf.get(t, 0)
            if f <= 0:
                continue
            dfi = df.get(t, 0)
            idf = math.log(1.0 + ((n_docs - dfi + 0.5) / (dfi + 0.5)))
            denom = f + k1 * (1.0 - b + b * (dl / avgdl))
            score += idf * ((f * (k1 + 1.0)) / max(1e-9, denom))
        out.append(score)
    return out


def _labor_law_intent(qlow: str) -> bool:
    return any(
        k in qlow
        for k in (
            "smig",
            "salaire minimum",
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
            "contrat de travail",
            "accident de travail",
        )
    )


def _cnie_procedure_intent(qlow: str) -> bool:
    if not any(k in qlow for k in ("cnie", "carte nationale", "identite electronique", "identité électronique", "cine")):
        return False
    if "passeport" in qlow and "cnie" not in qlow:
        return False
    return any(
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
    )


def _build_keyword_scores(
    hits: list[dict[str, Any]],
    terms: set[str],
    *,
    question: str,
    retrieval_query: str,
    watiqa_intent: bool,
    bo_intent: bool,
    ed_master_intent: bool,
    ed_doctorate_intent: bool,
    cnie_intent: bool,
    passeport_intent: bool,
    labor_intent: bool,
) -> list[float]:
    intent_phrases = (
        "document",
        "pièce",
        "pièces",
        "délivrance",
        "demande",
        "demander",
        "formulaire",
        "copie",
        "original",
        "joindre",
        "fournir",
        "requis",
        "condition",
        "modalité",
        "carte nationale",
        "identité électronique",
    )

    out: list[float] = []
    for h in hits:
        text = (h.get("text") or "").lower()
        meta = h.get("metadata") or {}
        url = (meta.get("source_url") or "").lower()
        label = ((meta.get("label") or "") + " " + (meta.get("title") or "")).lower()
        category = str(meta.get("category") or "").lower()
        org = str(meta.get("source_org") or "").lower()
        st = str(meta.get("source_type") or "")
        ministry = _strip_accents(str(meta.get("ministry") or "").lower())
        document_type = _strip_accents(str(meta.get("document_type") or "").lower())
        language = str(meta.get("language") or "").lower()

        n = sum(1.0 for t in terms if t in text)
        n += sum(2.0 for p in intent_phrases if p in text)

        if "cnie" in url or "dgsn" in url:
            n += 3.0
        if category == "cnie" or "cnie maroc" in org:
            n += 8.0

        if cnie_intent:
            if category == "cnie":
                n += 12.0
            qlow_cnie = f"{question} {retrieval_query}".lower()
            try:
                from app.Rag_classique.portal_cases import cnie_case_from_question, chunk_cnie_case

                case = cnie_case_from_question(question) or cnie_case_from_question(retrieval_query)
                if case:
                    hit_case = chunk_cnie_case(h)
                    if hit_case == case:
                        n += 16.0
                    elif hit_case and hit_case != case:
                        n -= 14.0
                    if case == "perte_vol" and any(
                        x in text for x in ("perte", "vol", "declaration de perte", "carte perdue")
                    ):
                        n += 10.0
                    if case == "perte_vol" and (
                        "premiere demande" in text or "première demande" in text
                    ) and "perte" not in text and " vol " not in f" {text} ":
                        n -= 18.0
            except ImportError:
                pass
            if any(
                x in text
                for x in (
                    "première demande",
                    "premiere demande",
                    "pièces à fournir",
                    "pieces a fournir",
                    "justificatif de domicile",
                    "acte de naissance",
                    "normes des photographies",
                )
            ):
                n += 6.0
            # Éviter les pages passeport consulat quand la question vise la CNIE seule.
            if "passeport" in url and "cnie" not in url:
                n -= 22.0
            if "consulat.ma" in url and category != "cnie":
                n -= 18.0
            if "passeport biométrique" in text or "passeport biometrique" in text:
                if "première demande" not in text and "premiere demande" not in text:
                    if category != "cnie":
                        n -= 22.0
                    elif "passeport" in label and "cnie" not in label:
                        n -= 20.0
            if st == "bulletin_officiel" and "cnie" in text and "première demande" not in text and "premiere demande" not in text:
                if "pièces à fournir" not in text and "pieces a fournir" not in text:
                    n -= 4.0

        if passeport_intent:
            if "passeport" in text or "consulat" in url or category == "passeport":
                n += 14.0
            if "mineur" in f"{question} {retrieval_query}".lower() and "mineur" in text:
                n += 8.0
            if "sans cnie" in text or "moins de 12 ans sans cnie" in text:
                n += 6.0
            cid = str(meta.get("chunk_id") or "")
            if cid.startswith("curated::cnie") or cid.startswith("cnie_procedure_"):
                n -= 22.0
            if category == "cnie" and "passeport" not in text:
                n -= 20.0
            try:
                from app.Rag_classique.phrase_context import subject_mentioned_but_not_target

                if subject_mentioned_but_not_target(question, "cnie"):
                    if category == "cnie" or "cnie.ma" in url:
                        n -= 18.0
            except ImportError:
                pass

        if watiqa_intent:
            if "watiqa" in url or "watiqa" in text:
                n += 6.0
            if (meta.get("source_org") or "").lower() == "watiqa":
                n += 8.0
            if meta.get("source_type") == "admin" and meta.get("category") == "etat_civil":
                n += 3.0
            if any(
                x in text
                for x in (
                    "commencer la démarche",
                    "commencer la demarche",
                    "nouvelle demande",
                    "suivre une commande",
                    "créer une demande",
                )
            ):
                n += 8.0
            if category == "cnie" or "cnie.ma" in url:
                n -= 22.0

        if labor_intent:
            qlow_full = f"{question} {retrieval_query}".lower()
            overtime_q = "heure" in qlow_full and "suppl" in qlow_full
            try:
                from app.Rag_classique.labor_corpus import is_off_topic_labor_hit

                if is_off_topic_labor_hit(text, overtime=overtime_q):
                    n -= 28.0
            except ImportError:
                pass
            if any(
                x in text
                for x in (
                    "nomenclature des pièces justificatives",
                    "nomenclature des pieces justificatives",
                    "propositions d'engagement et de paiement",
                    "depenses des biens et services",
                )
            ):
                n -= 35.0
            if (h.get("metadata") or {}).get("doc_id") == "LABOR65" or str(
                (h.get("metadata") or {}).get("chunk_id", "")
            ).startswith("LABOR65"):
                n += 22.0
            if overtime_q:
                if any(
                    x in text
                    for x in (
                        "heures supplementaires",
                        "heures supplémentaires",
                        "article 201",
                        "majoration de salaire",
                    )
                ):
                    n += 18.0
            elif any(
                x in text
                for x in ("smig", "salaire minimum", "salaire minimum légal", "salaire minimum legal")
            ):
                n += 12.0
            if st == "bulletin_officiel" and (h.get("metadata") or {}).get("doc_id") != "LABOR65":
                n += 4.0
            if any(x in text for x in ("code du travail", "loi n° 65-99", "loi n 65-99", "licenciement", "préavis", "preavis")):
                n += 6.0
            if category in ("cnie", "passeport") or "cnie.ma" in url or "consulat.ma" in url:
                n -= 24.0

        if bo_intent:
            bo_boost = 1.0 if ed_master_intent else 5.0
            if st == "bulletin_officiel":
                n += bo_boost
            if "bulletin" in text or "dahir" in text or "décret" in text or "decret" in text:
                n += 1.0
            if "bulletin" in label:
                n += 1.0 if ed_master_intent else 2.0

        if ed_master_intent:
            n += _education_master_chunk_bonus(h.get("text") or "")
            if _procurement_public_chunk(h.get("text") or ""):
                n -= 18.0

        if ed_doctorate_intent:
            n += _education_doctorate_chunk_bonus(h.get("text") or "")
            if _education_master_norms_chunk(h.get("text") or ""):
                n -= 28.0
            if "normes pédagogiques" in text and "master" in text and "doctorat" not in text:
                n -= 12.0

        if meta.get("source_type") == "admin" and "cnie" in text:
            n += 2.0

        # Bonus metadata : améliore le retrieval hybride en exploitant les champs enrichis
        qn = _strip_accents(" ".join(sorted(terms)))
        if "fiscal" in qn and "fiscal" in category:
            n += 3.0
        if any(x in qn for x in ("master", "enseignement", "universite")) and "education" in category:
            n += 2.5
        if any(x in qn for x in ("urbanisme", "construction", "permis")) and "urbanisme" in category:
            n += 2.5
        if "etat civil" in qn and "etat_civil" in category:
            n += 2.0
        if any(x in qn for x in ("decret", "dahir", "arrete", "loi")) and any(
            x in document_type for x in ("decret", "dahir", "arrete", "loi")
        ):
            n += 1.8
        if "interieur" in qn and "interieur" in ministry:
            n += 1.5
        if "justice" in qn and "justice" in ministry:
            n += 1.5
        if "francais" in qn and language == "fr":
            n += 0.8

        out.append(max(0.0, n))
    return out


@lru_cache(maxsize=1)
def _load_cross_encoder(model_name: str):
    from sentence_transformers.cross_encoder import CrossEncoder  # noqa: PLC0415

    return CrossEncoder(model_name)


def _build_cross_encoder_scores(
    hits: list[dict[str, Any]],
    *,
    question: str,
    retrieval_query: str,
) -> list[float]:
    if not hits:
        return []
    if not _bool_env("RERANK_ENABLE_CROSS_ENCODER", False):
        return [0.0 for _ in hits]

    model_name = os.environ.get("RERANK_CE_MODEL", "").strip() or "cross-encoder/ms-marco-MiniLM-L-6-v2"
    max_chars = _int_env("RERANK_CE_TEXT_MAX_CHARS", 1600)
    q = (question or "").strip() or (retrieval_query or "").strip()
    pairs = [(q, str(h.get("text") or "")[:max_chars]) for h in hits]
    try:
        model = _load_cross_encoder(model_name)
        preds = model.predict(pairs)
        return [float(x) for x in preds]
    except Exception:
        logger.exception("Cross-encoder indisponible (%s), fallback hybride sans CE.", model_name)
        return [0.0 for _ in hits]


def rerank_hits(
    hits: list[dict[str, Any]],
    *,
    question: str,
    retrieval_query: str,
    top_k: int,
    semantic_weight: float = 0.5,
    lexical_weight: float = 0.5,
) -> list[dict[str, Any]]:
    if not hits:
        return hits

    terms = _terms_from_texts(question, retrieval_query)
    if not terms:
        return hits[:top_k]

    from app.Rag_classique.query_understanding import analyze_query

    qa = analyze_query(question, retrieval_query=retrieval_query)
    ed_doctorate_intent = qa.education_doctorate_intent
    ed_master_intent = qa.education_master_intent
    watiqa_intent = qa.watiqa_intent
    cnie_intent = qa.cnie_intent
    passeport_intent = getattr(qa, "passeport_intent", False)
    labor_intent = qa.labor_intent
    bo_intent = qa.bo_intent
    qlow = f"{question} {retrieval_query}".lower()

    sem_raw = [float(h.get("score", 0.0)) for h in hits]
    bm25_raw = _build_bm25_scores(hits, _tokenize(f"{question} {retrieval_query}"))
    kw_raw = _build_keyword_scores(
        hits,
        terms,
        question=question,
        retrieval_query=retrieval_query,
        watiqa_intent=watiqa_intent,
        bo_intent=bo_intent,
        ed_master_intent=ed_master_intent,
        ed_doctorate_intent=ed_doctorate_intent,
        cnie_intent=cnie_intent,
        passeport_intent=passeport_intent,
        labor_intent=labor_intent,
    )
    sem_norm = _minmax_norm(sem_raw)
    bm25_norm = _minmax_norm(bm25_raw)
    kw_norm = _minmax_norm(kw_raw)

    ce_enabled = _bool_env("RERANK_ENABLE_CROSS_ENCODER", False)

    # Poids par défaut : mode hybride recommandé (FAISS + BM25 + lexical).
    sem_w = _float_env("RERANK_SEMANTIC_WEIGHT", semantic_weight)
    bm25_w = _float_env("RERANK_BM25_WEIGHT", 0.35)
    kw_w = _float_env("RERANK_KEYWORD_WEIGHT", lexical_weight)
    ce_w = _float_env("RERANK_CE_WEIGHT", 0.0)
    if ce_enabled:
        ce_w = _float_env("RERANK_CE_WEIGHT", 0.35)
        # réduire légèrement le poids lexical si CE actif (somme plus stable)
        kw_w = _float_env("RERANK_KEYWORD_WEIGHT", 0.25)
        bm25_w = _float_env("RERANK_BM25_WEIGHT", 0.25)
        sem_w = _float_env("RERANK_SEMANTIC_WEIGHT", 0.30)

    # Cas spécial déjà observé : master/pédagogie est souvent mal classé par proximité vectorielle.
    if ed_master_intent:
        sem_w = min(sem_w, 0.30)
        bm25_w = max(bm25_w, 0.30)
        kw_w = max(kw_w, 0.35)
    if ed_doctorate_intent:
        sem_w = min(sem_w, 0.32)
        bm25_w = max(bm25_w, 0.32)
        kw_w = max(kw_w, 0.38)

    ce_raw = [0.0 for _ in hits]
    if ce_enabled:
        ce_max = max(1, _int_env("RERANK_CE_MAX_CANDIDATES", 24))
        if len(hits) <= ce_max:
            ce_raw = _build_cross_encoder_scores(
                hits, question=question, retrieval_query=retrieval_query
            )
        else:
            # 2 passes : pré-score FAISS+BM25+mots-clés, CE seulement sur le top-N (latence ×5–10 plus faible).
            pre_w_sum = sem_w + bm25_w + kw_w
            if pre_w_sum <= 0:
                pre_w_sum = 1.0
            pre_scored: list[tuple[float, int]] = []
            for i in range(len(hits)):
                pre = (
                    (sem_w / pre_w_sum) * sem_norm[i]
                    + (bm25_w / pre_w_sum) * bm25_norm[i]
                    + (kw_w / pre_w_sum) * kw_norm[i]
                )
                pre_scored.append((pre, i))
            pre_scored.sort(key=lambda x: x[0], reverse=True)
            top_idx = [idx for _, idx in pre_scored[:ce_max]]
            subset = [hits[i] for i in top_idx]
            ce_subset = _build_cross_encoder_scores(
                subset, question=question, retrieval_query=retrieval_query
            )
            for j, idx in enumerate(top_idx):
                ce_raw[idx] = ce_subset[j]

    ce_norm = _minmax_norm(ce_raw) if ce_enabled else [0.0 for _ in hits]

    weight_sum = sem_w + bm25_w + kw_w + ce_w
    if weight_sum <= 0:
        sem_w, bm25_w, kw_w, ce_w = 0.4, 0.3, 0.3, 0.0
        weight_sum = 1.0
    sem_w, bm25_w, kw_w, ce_w = (
        sem_w / weight_sum,
        bm25_w / weight_sum,
        kw_w / weight_sum,
        ce_w / weight_sum,
    )

    scored: list[tuple[float, dict[str, Any]]] = []
    for i, h in enumerate(hits):
        combined = (
            sem_w * sem_norm[i]
            + bm25_w * bm25_norm[i]
            + kw_w * kw_norm[i]
            + ce_w * ce_norm[i]
        )
        combined -= _profile_penalty(question, (h.get("text") or ""))
        hh = dict(h)
        hh["rerank_score"] = float(combined)
        hh["rerank_components"] = {
            "semantic": round(sem_norm[i], 4),
            "bm25": round(bm25_norm[i], 4),
            "keywords": round(kw_norm[i], 4),
            "cross_encoder": round(ce_norm[i], 4),
        }
        scored.append((float(combined), hh))

    scored.sort(key=lambda x: x[0], reverse=True)
    out = [h for _, h in scored[:top_k]]
    return out
