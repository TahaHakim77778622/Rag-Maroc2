"""Orchestration : retrieval → prompt → génération."""

from __future__ import annotations

import logging
import os
import re
import unicodedata
from urllib.parse import urlparse
from typing import Any

from app.llm_client import LLMClient, MockLLMClient
from app.prompt_builder import build_no_corpus_conversation_prompt, build_rag_prompt, hits_to_source_lines
from app.query_rewrite_llm import enrich_analysis_with_llm_rewrite
from app.query_understanding import analyze_query, build_retrieval_query
from app.retrieval_rerank import rerank_hits
from app.retriever import Retriever
from app.labor_corpus import (
    filter_hits_for_labor_prompt,
    is_labor_code_question,
    merge_labor_hits,
)
from app.web_fallback import (
    _curated_cnie_hits,
    _curated_passeport_hits,
    _curated_smig_hits,
    _curated_watiqa_hits,
    _is_passeport_fee_question,
    passeport_fee_hits_substantive,
    _labor_topic_question,
    _portal_intent,
    curated_fallback_hits,
    _prioritize_watiqa_hits,
    _web_hit_is_cnie_not_watiqa,
    _web_hit_is_passeport_not_cnie,
    fetch_web_hits,
    portal_local_hits_sufficient,
    save_web_hits_to_queue,
    should_use_web_fallback,
)

logger = logging.getLogger(__name__)


def _apply_retrieval_pool_cap(pool: int, nvec: int) -> int:
    """
    Limite le nombre de candidats FAISS (latence rerank / cross-encoder).
    RAG_RETRIEVAL_POOL_CAP>0 : plafond explicite.
    Sinon, si cross-encoder actif : plafond automatique 64 (au lieu de 128).
    """
    raw = os.environ.get("RAG_RETRIEVAL_POOL_CAP", "").strip()
    if raw:
        try:
            cap = int(raw)
            if cap > 0:
                return min(pool, cap, nvec)
        except ValueError:
            pass
        return min(pool, nvec)
    ce_on = os.environ.get("RERANK_ENABLE_CROSS_ENCODER", "").strip().lower() in (
        "1",
        "true",
        "yes",
        "on",
    )
    if ce_on:
        return min(pool, 64, nvec)
    return min(pool, nvec)


def _safe_llm_complete(llm: LLMClient, prompt: str) -> str:
    """
    Ne propage pas d’exception vers l’API / JSON : l’utilisateur reçoit un message utile, pas un HTTP 500.
    """
    try:
        return llm.complete(prompt)
    except Exception as exc:
        logger.exception("Échec de l’appel LLM (timeout ou erreur API)")
        err = str(exc).lower()
        if any(
            x in err
            for x in (
                "nodename nor servname",
                "name or service not known",
                "connecterror",
                "connection refused",
                "network is unreachable",
                "failed to resolve",
                "getaddrinfo",
            )
        ):
            return (
                "Impossible de joindre l’API du modèle (Cohere/OpenAI) : problème réseau ou DNS. "
                "Vérifiez votre connexion Internet, le VPN/proxy, et les variables `.env` "
                "(`COHERE_API_KEY`, `COHERE_BASE_URL` si renseigné). "
                "Pour tester sans API : `LLM_MOCK=1` puis redémarrer uvicorn."
            )
        if "timeout" in err or "timed out" in err:
            return (
                "Le modèle n’a pas répondu à temps. Réessayez dans quelques instants.\n\n"
                "Conseil : augmentez `LLM_TIMEOUT` dans `.env` (ex. 300) si les réponses sont longues."
            )
        return (
            "Le service de génération de texte est momentanément indisponible. "
            "Réessayez dans quelques instants ou vérifiez la clé API dans `.env`."
        )


def _fold_accents(s: str) -> str:
    return "".join(c for c in unicodedata.normalize("NFD", s) if unicodedata.category(c) != "Mn")


def _scrub_apology_when_sourced(answer: str, *, has_corpus_hits: bool) -> str:
    """
    Le modèle ignore parfois les consignes : si des extraits ont été fournis, on retire les ouvertures
    « désolé / je ne peux pas » pour garder un ton professionnel.
    """
    if not has_corpus_hits or not (answer or "").strip():
        return answer
    t = answer.strip()
    head = _fold_accents(t[:360].lower())
    markers = (
        "desole",
        "je ne peux pas",
        "ne peux pas repondre",
        "impossible de repondre",
        "je suis desole",
        "je ne peux vous aider",
        "je suis incapable",
    )
    if not any(m in head for m in markers):
        return answer

    parts = re.split(r"\n\s*\n+", t)
    kept: list[str] = []
    removed_apology_block = False
    skip_first_bad = True
    for para in parts:
        p = para.strip()
        if not p:
            continue
        pf = _fold_accents(p.lower())
        has_cite = bool(re.search(r"\[\s*\d+\s*\]", p))
        is_short_refusal = (
            len(p) < 520 and any(m in pf for m in markers) and not has_cite
        )
        if skip_first_bad and is_short_refusal:
            skip_first_bad = False
            removed_apology_block = True
            continue
        kept.append(p)
    if kept:
        out = "\n\n".join(kept)
        if removed_apology_block and len(out) < 450:
            out = (
                "Les passages fournis comme sources ne correspondent pas au texte précis de votre question ; "
                "indiquez si possible le n° d’arrêté ou la parution du B.O.\n\n" + out
            )
        return _strip_leading_apology_sentences_if_needed(out, markers)

    return (
        "Les extraits fournis ne correspondent pas au texte précis que vous ciblez. "
        "Indiquez la référence de l’arrêté ou la date du Bulletin officiel pour retrouver le bon passage dans l’index. "
        "Les documents sélectionnés concernent un autre acte : je peux vous en résumer le contenu factuel sur demande."
    )


def _strip_leading_apology_sentences_if_needed(text: str, markers: tuple[str, ...]) -> str:
    """Une seule ligne ou un bloc sans saut : retire les premières phrases d’excuse/refus."""
    if not text or not any(m in _fold_accents(text[:400].lower()) for m in markers):
        return text
    chunks = re.split(r"(?<=[.!?…])\s+", text.strip(), maxsplit=8)
    out_parts: list[str] = []
    for i, chunk in enumerate(chunks):
        cf = _fold_accents(chunk.lower())
        if i < 3 and any(m in cf for m in markers) and len(chunk) < 550:
            continue
        out_parts.append(chunk)
    tail = " ".join(out_parts).strip()
    return tail if tail else text


def _is_doctorate_pathway_question(question: str, history: list[dict[str, Any]] | None = None) -> bool:
    return analyze_query(question, history=history).education_doctorate_intent


def _is_master_pedagogic_question(question: str) -> bool:
    if _is_doctorate_pathway_question(question):
        return False
    q = _strip_accents((question or "").lower())
    return any(
        x in q
        for x in (
            "cycle de master",
            "cycle master",
            "master",
            "normes pedagogiques",
            "cahier des normes",
            "enseignement superieur",
        )
    )


def _hits_look_master_aligned(hits: list[dict[str, Any]]) -> bool:
    if not hits:
        return False
    probes = []
    for h in hits[:3]:
        meta = h.get("metadata") or {}
        probes.append(str(meta.get("label") or ""))
        probes.append(str(meta.get("title") or ""))
        probes.append(str(h.get("text") or "")[:800])
    blob = _strip_accents(" ".join(probes).lower())
    master_markers = (
        "cycle de master",
        "cycle master",
        "master",
        "normes pedagogiques",
        "cahier des normes",
        "enseignement superieur",
        "modules disciplinaires",
        "credits",
        "semestre",
    )
    return any(m in blob for m in master_markers)


def _remove_bo_reference_request_if_aligned(answer: str) -> str:
    if not answer:
        return answer
    lines = answer.splitlines()
    out: list[str] = []
    for line in lines:
        low = _strip_accents(line.lower())
        if (
            "j'aurais besoin" in low
            or "j aurais besoin" in low
            or "avez-vous une reference" in low
            or "avez vous une reference" in low
            or "reference exacte du bulletin officiel" in low
            or "reference precise" in low
        ) and ("b.o" in low or "bulletin officiel" in low or "reference" in low):
            continue
        out.append(line)
    cleaned = "\n".join(out)
    cleaned = re.sub(r"\n{3,}", "\n\n", cleaned).strip()
    return cleaned if cleaned else answer


def _enforce_master_consistency(answer: str) -> str:
    """Empêche les réponses qui mélangent licence/master quand la question cible master."""
    if not answer:
        return answer
    parts = re.split(r"(?<=[.!?…])\s+", answer.strip())
    kept: list[str] = []
    for p in parts:
        low = _strip_accents(p.lower())
        if "licence" in low and "master" not in low:
            continue
        kept.append(p)
    out = " ".join(kept).strip()
    return out if out else answer


def _extract_bo_number(text: str) -> str | None:
    t = text or ""
    m = re.search(r"(?:N[º°o]\s*|n[º°o]\s*)(\d{3,5}(?:\s*bis)?)", t)
    if not m:
        m = re.search(r"\b(\d{3,5}(?:\s*bis)?)\s*[-–]\s*\d{1,2}\s*\w+\s*\d{4}", t)
    if not m:
        return None
    return " ".join(m.group(1).split())


def _extract_article(text: str, label: str) -> str | None:
    l = (label or "").strip()
    if l:
        return l
    m = re.search(
        r"\b(?:ART(?:ICLE)?\.?\s*)(?:N[º°o]\s*)?([0-9]{1,4}|PREMIER|UNIQUE)\b",
        text or "",
        flags=re.IGNORECASE,
    )
    if not m:
        return None
    g = m.group(1)
    if g.isdigit():
        return f"Article {g}"
    return f"Article {g.capitalize()}"


def _host_from_url(url: str) -> str:
    if not url:
        return ""
    try:
        h = urlparse(url).netloc.strip().lower()
    except ValueError:
        return ""
    if h.startswith("www."):
        h = h[4:]
    return h


def _reference_line_for_hit(hit: dict[str, Any], idx: int) -> str | None:
    meta = hit.get("metadata") or {}
    text = str(hit.get("text") or "")
    st = str(meta.get("source_type") or "").lower()
    label = str(meta.get("label") or "")
    title = str(meta.get("title") or "") or "Source"
    page = meta.get("page_start")
    url = str(meta.get("source_url") or "")
    host = _host_from_url(url)

    if st == "bulletin_officiel":
        bo = _extract_bo_number(text)
        art = _extract_article(text, label)
        bits = []
        if bo:
            bits.append(f"BO n°{bo}")
        else:
            bits.append("Bulletin Officiel")
        if art:
            bits.append(art)
        if isinstance(page, int):
            bits.append(f"p.{page}")
        return f"- [{idx}] " + " — ".join(bits)

    if "watiqa" in host or "watiqa" in title.lower():
        suffix = f" — {label}" if label else ""
        return f"- [{idx}] Source: watiqa.ma{suffix}"

    if host:
        suffix = f" — {label}" if label else f" — {title}"
        return f"- [{idx}] Source: {host}{suffix}"

    return None


def _append_precise_references(
    answer: str, hits: list[dict[str, Any]], *, question: str = ""
) -> str:
    if not hits:
        return answer
    cite_hits = hits
    if question:
        try:
            from app.labor_corpus import filter_hits_for_labor_prompt, is_targeted_labor_topic

            if is_targeted_labor_topic(question):
                cite_hits = filter_hits_for_labor_prompt(question, hits, top_k=len(hits))
        except ImportError:
            pass
    # Évite de dupliquer si un bloc existe déjà.
    low = _strip_accents((answer or "").lower())
    if "references precises" in low or "références précises" in low:
        return answer

    lines: list[str] = []
    for i, h in enumerate(cite_hits[:4], start=1):
        ln = _reference_line_for_hit(h, i)
        if ln:
            lines.append(ln)
    if not lines:
        return answer

    block = "Références précises\n" + "\n".join(lines)
    base = (answer or "").rstrip()
    if not base:
        return block
    return f"{base}\n\n{block}"


def _mandatory_first_sentence_prefix(
    hits: list[dict[str, Any]],
    *,
    question: str = "",
) -> str:
    """Construit un préfixe obligatoire pour la première phrase (BO/article ou Watiqa)."""
    if not hits:
        return ""
    top = hits[0]
    if question:
        try:
            from app.labor_corpus import is_labor_code_question, primary_hit_for_answer

            if is_labor_code_question(question):
                picked = primary_hit_for_answer(question, hits)
                if picked is not None:
                    top = picked
        except ImportError:
            pass
    meta = top.get("metadata") or {}
    text = str(top.get("text") or "")
    st = str(meta.get("source_type") or "").lower()
    label = str(meta.get("label") or "").strip()
    url = str(meta.get("source_url") or "")
    host = _host_from_url(url)
    page = meta.get("page_start")

    if st == "bulletin_officiel":
        bo = _extract_bo_number(text)
        art = _extract_article(text, label)
        if bo and art:
            prefix = f"Selon BO n°{bo}, {art}"
        elif bo:
            prefix = f"Selon BO n°{bo}"
        elif art:
            prefix = f"Selon le Bulletin Officiel, {art}"
        else:
            prefix = "Selon le Bulletin Officiel"
        if isinstance(page, int):
            prefix += f" (p.{page})"
        return prefix + ", "

    if "watiqa" in host or "watiqa" in str(meta.get("title") or "").lower():
        suffix = f" — {label}" if label else ""
        return f"Source : watiqa.ma{suffix}. "

    if host:
        suffix = f" — {label}" if label else ""
        return f"Source : {host}{suffix}. "
    return ""


def _prepend_mandatory_lead_citation(
    answer: str, hits: list[dict[str, Any]], *, question: str = ""
) -> str:
    if not hits:
        return answer
    base = (answer or "").strip()
    if not base:
        return answer
    low_head = _strip_accents(base[:140].lower())
    if question:
        try:
            from app.labor_corpus import is_targeted_labor_topic

            if is_targeted_labor_topic(question):
                if low_head.startswith("selon bo n") or low_head.startswith("selon le bulletin officiel"):
                    base = re.sub(
                        r"^(Selon\s+BO[^.]*\.\s*|Selon le Bulletin Officiel[^.]*\.\s*)",
                        "",
                        base,
                        count=1,
                        flags=re.IGNORECASE,
                    ).strip()
                    low_head = _strip_accents(base[:140].lower())
        except ImportError:
            pass
    if low_head.startswith("selon bo n") or low_head.startswith("selon le bulletin officiel") or low_head.startswith("source : watiqa.ma"):
        return answer
    prefix = _mandatory_first_sentence_prefix(hits, question=question)
    if not prefix:
        return answer
    return prefix + base


_DOMAIN_HINTS = (
    "cnie",
    "passeport",
    "carte",
    "identite",
    "identité",
    "document",
    "documents",
    "piece",
    "pièce",
    "demande",
    "renouvellement",
    "perte",
    "vol",
    "dgsn",
    "consulat",
    "loi",
    "lois",
    "article",
    "procedure",
    "procédure",
    "delai",
    "délai",
    "juridique",
    "urbanisme",
    "construction",
    "construire",
    "maison",
    "terrain",
    "immeuble",
    "logement",
    "cuisine",
    "couscous",
)


def _clarification_reply(question: str) -> str:
    q = _strip_accents(question.strip().lower())
    if any(
        k in q
        for k in (
            "construire",
            "construction",
            "maison",
            "immeuble",
            "logement",
            "terrain",
            "urbanisme",
            "batir",
            "batiment",
            "permis de construire",
            "autorisation de construire",
        )
    ):
        return (
            "Pour cadrer le droit/les demarches de construction au Maroc, pouvez-vous preciser ?\n"
            "- Ville/commune et zone (urbain/periurbain) ?\n"
            "- Projet: maison individuelle, R+..., fonds de commerce, promotion ?\n"
            "- Vous avez deja un terrain (titre foncier) ou c’est encore a l’etude ?"
        )
    if any(k in q for k in ("route", "conduite", "conduire", "permis")):
        return (
            "Pour bien vous aider, pouvez-vous preciser votre besoin sur la conduite au Maroc ?\n"
            "- Vous cherchez le code de la route (infractions/amendes) ?\n"
            "- Les conditions du permis (inscription, examen, renouvellement) ?\n"
            "- Une situation precise (exces de vitesse, alcoolemie, retrait de permis) ?"
        )
    if any(k in q for k in ("cnie", "carte", "identite", "passeport", "document", "piece")):
        return (
            "Pour vous donner une reponse utile, pouvez-vous preciser votre cas ?\n"
            "- Type de document (CNIE, passeport, acte d'etat civil) ?\n"
            "- Nature de la demande (premiere demande, renouvellement, perte/vol) ?\n"
            "- Lieu de la demande (au Maroc ou au consulat) ?"
        )
    return (
        "Pouvez-vous preciser votre question pour que je cible la bonne procedure marocaine ?\n"
        "- Domaine: documents, etat civil, conduite, fiscalite, travail, etc.\n"
        "- Situation exacte: premiere demande, renouvellement, recours, delais, pieces.\n"
        "- Ville ou organisme concerne, si applicable."
    )


def _strip_accents(text: str) -> str:
    return "".join(
        ch for ch in unicodedata.normalize("NFD", text) if unicodedata.category(ch) != "Mn"
    )


def _has_specific_article_ref(question: str) -> bool:
    """Référence juridique suffisamment précise pour tenter le RAG (pas « l’article du bulletin » seul)."""
    qn = _strip_accents((question or "").lower())
    if re.search(r"\barticle\s*(?:n[o°]\s*)?[0-9]{1,4}\b", qn):
        return True
    if re.search(r"\bart\.?\s*[0-9]{1,4}\b", qn):
        return True
    if re.search(r"\barticles?\s+[0-9]{1,4}\s*(?:a|à|et|-)\s*[0-9]{1,4}\b", qn):
        return True
    if re.search(r"\barticle\s+premier\b|\bpremier\s+article\b|\b1er\s+article\b|\barticle\s+unique\b", qn):
        return True
    return False


def _is_vague_bulletin_article_question(question: str) -> bool:
    """
    « Que dit l’article du bulletin » sans numéro d’article ni référence de texte : impossible à citer proprement.
    On répond par une clarification structurée plutôt que d’inventer ou de renvoyer un extrait hors sujet.
    """
    raw = (question or "").strip()
    if len(raw) > 550:
        return False
    qn = _strip_accents(raw.lower())
    bulletin_markers = (
        "bulletin officiel",
        "bulletin officie",
        "bulletin",
        "journal officiel",
        "b.o.",
        "b. o.",
    )
    if any(typo in qn for typo in ("buletin", "blutein", "bulutein")):
        has_bulletin = True
    else:
        has_bulletin = any(m in qn for m in bulletin_markers)
    if not has_bulletin:
        return False
    if _has_specific_article_ref(raw):
        return False
    asks_content = any(
        x in qn
        for x in (
            "article",
            "que dit",
            "que dis",
            "qu'il dit",
            "qu'il dis",
            "quest ce que",
            "qu'est ce que",
            "explique",
            "resum",
            "résum",
            "c'est quoi",
            "signifie",
            "dis quoi",
            "dit quoi",
        )
    )
    return asks_content


def _bulletin_reference_clarification() -> str:
    return (
        "Votre question vise un texte paru au Bulletin officiel, mais sans les références "
        "nécessaires pour identifier **quel** article et **quel** texte parmi les parutions.\n\n"
        "Pour une réponse fiable (teneur et citation), merci d’indiquer au minimum :\n"
        "• le **numéro d’article** (ex. « article 12 ») ou un **extrait** à copier-coller ;\n"
        "• sachez-vous le **type de texte** (dahir, décret, arrêté, circulaire…) ou la **matière** "
        "(ex. CNIE, état civil, fiscalité, urbanisme) ?\n"
        "• si vous les connaissez : **date de publication** au B.O. ou **numéro / cote** utile à votre recherche.\n\n"
        "Chaque parution regroupe de nombreux actes ; sans ces éléments je ne peux pas rattacher votre question "
        "à un passage précis du corpus. Dès qu’une référence ou un extrait est fourni, je peux m’appuyer sur "
        "les sources indexées pour résumer et citer."
    )


def _smalltalk_reply(question: str) -> str | None:
    q = _strip_accents(question.strip().lower())
    q = re.sub(r"\s+", " ", q)

    if not q:
        return "Je suis la. Posez votre question sur les procedures/documents (ex: CNIE, passeport, delais)."

    greetings = {"bonjour", "bonsoir", "salut", "salam", "hello", "hi", "coucou"}
    thanks = {"merci", "merci beaucoup", "thank you"}
    bye = {"au revoir", "bye", "a bientot", "bonne journee", "bonne soiree"}

    if q in greetings:
        return (
            "Bonjour. Je peux vous aider sur le Maroc : demarches, droit et services publics, et aussi "
            "repondre a des questions pratiques en restant clair sur ce qui est sur et ce qui manque d’infos."
        )
    if q in thanks:
        return "Avec plaisir. Si vous voulez, je peux aussi vous donner la liste des pieces requises selon votre cas."
    if q in bye:
        return "A bientot. Revenez quand vous voulez pour une nouvelle question."
    return None


def _is_too_generic(question: str) -> bool:
    q = _strip_accents(question.strip().lower())
    words = [w for w in re.findall(r"[a-z0-9]+", q) if len(w) >= 2]
    if not words:
        return True
    if any(h in q for h in _DOMAIN_HINTS):
        return False
    return len(words) <= 2


def _smart_clarification(question: str) -> str:
    q = _strip_accents((question or "").lower())
    if any(k in q for k in ("loi", "lois", "article", "juridique")):
        return "Vous voulez une loi marocaine dans quel domaine exactement (construction, travail, fiscalite, famille, route) ?"
    if any(k in q for k in ("document", "piece", "cnie", "passeport", "etat civil")):
        return "Pour quel document et quel cas precis (premiere demande, renouvellement, perte/vol) souhaitez-vous la liste exacte ?"
    if any(k in q for k in ("construction", "construire", "maison", "terrain", "urbanisme")):
        return "Votre besoin concerne plutot l'achat du terrain, l'autorisation de construire, ou les deux ?"
    if any(k in q for k in ("aide", "info", "renseignement", "question")):
        return "Je peux vous aider, quel est le domaine exact au Maroc (documents, urbanisme, travail, fiscalite, transport) ?"
    return "Pouvez-vous formuler une question plus precise (domaine + situation + ville/organisme) pour que je vous reponde utilement ?"


def _is_followup_ellipsis(question: str) -> bool:
    q = _strip_accents((question or "").strip().lower())
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
        "et renouvellement",
        "et pour renouveler",
        "sinon",
    )
    if any(q.startswith(s) for s in starters):
        return True
    words = [w for w in re.findall(r"[a-z0-9]+", q) if len(w) >= 2]
    return len(words) <= 5 and any(w in q for w in ("renouvellement", "perte", "vol", "delai", "délais"))


def _last_user_topic(history: list[dict[str, str]]) -> str:
    topic_keys = (
        "cnie",
        "carte nationale",
        "passeport",
        "etat civil",
        "acte de naissance",
        "watiqa",
        "permis de conduire",
        "route",
        "construction",
        "urbanisme",
        "terrain",
        "master",
        "bulletin officiel",
        "decret",
        "dahir",
        "arrete",
    )
    for m in reversed(history):
        if m.get("role") != "user":
            continue
        txt = _strip_accents(m.get("content", "").lower())
        if not txt:
            continue
        for k in topic_keys:
            if _strip_accents(k) in txt:
                return k
        # fallback: dernière question utilisateur non triviale
        words = re.findall(r"[a-z0-9]+", txt)
        if len(words) >= 4:
            return m.get("content", "").strip()
    return ""


def _resolve_followup_question(question: str, history: list[dict[str, str]]) -> str:
    """
    Injecte le sujet précédent pour les questions elliptiques:
    « Et pour le renouvellement ? » -> « Et pour le renouvellement (concernant passeport) ? »
    """
    q = (question or "").strip()
    if not q or not history:
        return q
    if not _is_followup_ellipsis(q):
        return q
    topic = _last_user_topic(history)
    if not topic:
        return q
    qn = _strip_accents(q.lower())
    tn = _strip_accents(topic.lower())
    if tn in qn:
        return q
    return f"{q} (contexte: {topic})"


def _normalize_history(raw: list[dict[str, Any]] | None) -> list[dict[str, str]]:
    if not raw:
        return []
    out: list[dict[str, str]] = []
    for m in raw[-24:]:
        if not isinstance(m, dict):
            continue
        role = str(m.get("role", "")).strip().lower()
        content = str(m.get("content", "")).strip()
        if role not in ("user", "assistant") or not content:
            continue
        out.append({"role": role, "content": content[:8000]})
    return out


def _retrieval_blob(history: list[dict[str, str]], question: str, *, max_chars: int = 2800) -> str:
    """Texte pour embedding : derniers tours + question (tronqué par la tête si trop long)."""
    chunks: list[str] = []
    for m in history[-10:]:
        label = "Utilisateur" if m["role"] == "user" else "Assistant"
        body = m["content"]
        if len(body) > 1200:
            body = body[:1197] + "…"
        chunks.append(f"{label}: {body}")
    chunks.append(f"Utilisateur: {question.strip()}")
    text = "\n".join(chunks)
    if len(text) <= max_chars:
        return text
    return text[-max_chars:]


def _lexical_anchor(history: list[dict[str, str]], question: str) -> str:
    tail = "\n".join(m["content"] for m in history[-4:])
    return f"{question.strip()}\n{tail}".strip()


def _extract_user_profile(history: list[dict[str, str]], question: str) -> dict[str, str]:
    text = " ".join(m["content"] for m in history if m["role"] == "user")
    text = f"{text}\n{question}".lower()
    text = _strip_accents(text)

    profile: dict[str, str] = {}

    if "celibataire" in text:
        profile["statut_familial"] = "celibataire"
    elif any(k in text for k in ("marie", "mariee", "epoux", "epouse", "conjoint")):
        profile["statut_familial"] = "marie"

    if any(k in text for k in ("mineur", "mineure", "moins de 18")):
        profile["age_statut"] = "mineur"
    elif any(k in text for k in ("majeur", "plus de 18", "18 ans")):
        profile["age_statut"] = "majeur"

    if any(k in text for k in ("consulat", "etranger", "réside a l'etranger", "reside a l'etranger")):
        profile["lieu_demande"] = "consulat_etranger"
    elif any(k in text for k in ("au maroc", "au maroc.", "maroc")):
        profile["lieu_demande"] = "maroc"

    if "premiere demande" in text or "premiere fois" in text:
        profile["type_demande"] = "premiere_demande"
    elif "renouvellement" in text:
        profile["type_demande"] = "renouvellement"
    elif any(k in text for k in ("perte", "vole", "vol")):
        profile["type_demande"] = "perte_ou_vol"

    return profile


def _profile_to_hint(profile: dict[str, str]) -> str:
    if not profile:
        return ""
    parts = [f"{k}:{v}" for k, v in sorted(profile.items())]
    return " ".join(parts)


class RAGPipeline:
    def __init__(
        self,
        retriever: Retriever | None = None,
        llm: LLMClient | None = None,
        top_k: int = 5,
    ) -> None:
        self.retriever = retriever or Retriever()
        self.llm = llm or MockLLMClient()
        self.top_k = top_k

    def answer(
        self,
        question: str,
        *,
        history: list[dict[str, Any]] | None = None,
    ) -> dict[str, Any]:
        q_clean = question.strip()
        hist = _normalize_history(history)
        from app.query_understanding import resolve_question

        q_resolved = resolve_question(q_clean, hist)
        # Salutations / messages de politesse : on repond directement, sans retrieval.
        smalltalk = _smalltalk_reply(q_clean)
        if smalltalk is not None:
            return {
                "answer": smalltalk,
                "hits": [],
                "prompt": "",
                "sources_display": [],
            }

        if _is_vague_bulletin_article_question(q_clean):
            return {
                "answer": _bulletin_reference_clarification(),
                "hits": [],
                "prompt": "",
                "sources_display": [],
            }

        # Question trop courte/generique : demander une precision plutot que renvoyer un extrait hors sujet.
        if _is_too_generic(q_clean) and not _is_followup_ellipsis(q_clean):
            return {
                "answer": _smart_clarification(q_clean),
                "hits": [],
                "prompt": "",
                "sources_display": [],
            }

        try:
            from app.question_type import is_general_knowledge_question, out_of_scope_reply

            if is_general_knowledge_question(q_clean):
                return {
                    "answer": out_of_scope_reply(q_clean),
                    "hits": [],
                    "prompt": "",
                    "sources_display": [],
                    "corpus_sufficient": False,
                    "answer_source": "out_of_scope",
                }
        except ImportError:
            pass

        profile = _extract_user_profile(hist, q_clean)
        profile_hint = _profile_to_hint(profile)
        query_analysis = analyze_query(
            q_clean,
            history=hist,
            profile_hint=profile_hint,
        )
        query_analysis = enrich_analysis_with_llm_rewrite(
            self.llm,
            query_analysis,
            q_clean,
            history=hist,
        )
        q_resolved = query_analysis.resolved
        use_history_for_retrieval = query_analysis.use_conversation_history
        embed_prefix = _retrieval_blob(hist, q_resolved) if use_history_for_retrieval else ""
        retrieval_q = build_retrieval_query(
            query_analysis,
            embed_prefix=embed_prefix,
        )
        anchor_for_rerank = (
            _lexical_anchor(hist, q_resolved) if use_history_for_retrieval else q_resolved
        )
        anchor_for_rerank = f"{anchor_for_rerank}\n{profile_hint}".strip()
        # Pool large FAISS puis re-classement lexical (évite les mentions « CNIE » hors sujet).
        nvec = self.retriever.n_vectors
        if nvec <= 0:
            raw_hits: list[dict[str, Any]] = []
        else:
            pool = min(max(self.top_k * 6, 24), 48, nvec)
            pool = max(pool, self.top_k)
            q_low = q_clean.lower()
            if "watiqa" in q_low or (
                "guichet" in q_low
                and ("état civil" in q_low or "etat civil" in q_low)
            ):
                pool = min(max(pool, 128), nvec)
            elif any(
                k in q_low
                for k in (
                    "cnie",
                    "carte nationale",
                    "identité électronique",
                    "identite electronique",
                    "cine",
                )
            ) and "passeport" not in q_low:
                pool = min(max(pool, 128), nvec)
            elif "passeport" in q_low:
                try:
                    from app.phrase_context import analyze_phrase

                    if analyze_phrase(q_clean).primary_subject == "passeport":
                        pool = min(max(pool, 128), nvec)
                except ImportError:
                    pool = min(max(pool, 128), nvec)
            elif any(
                k in q_low
                for k in (
                    "smig",
                    "salaire minimum",
                    "code du travail",
                    "licenciement",
                    "préavis",
                    "preavis",
                    "démission",
                    "demission",
                    "congé annuel",
                    "conge annuel",
                    "heures supplémentaires",
                    "heures supplementaires",
                )
            ):
                pool = min(max(pool, 128), nvec)
            elif _is_doctorate_pathway_question(q_clean, hist):
                pool = min(max(pool, 128), nvec)
            elif any(
                x in q_low
                for x in (
                    "cycle de master",
                    "cycle master",
                    "normes pédagogiques",
                    "normes pedagogiques",
                    "cahier des normes",
                )
            ) or (
                "master" in q_low
                and any(
                    x in q_low
                    for x in (
                        "pédagogique",
                        "pedagogique",
                        "enseignement supérieur",
                        "enseignement superieur",
                        "universit",
                    )
                )
                and "doctorat" not in q_low
                and "doctoral" not in q_low
                and "thèse" not in q_low
                and "these" not in q_low
            ):
                pool = min(max(pool, 128), nvec)
            elif _has_specific_article_ref(q_clean) or any(
                x in q_low
                for x in (
                    "bulletin",
                    "buletin",
                    "journal officiel",
                    "decret",
                    "décret",
                    "dahir",
                    "arrete du",
                    "arrêté",
                )
            ):
                pool = min(max(pool, 96), nvec)
            pool = _apply_retrieval_pool_cap(pool, nvec)
            raw_hits = self.retriever.search(retrieval_q, pool)
        hits = rerank_hits(
            raw_hits,
            question=anchor_for_rerank,
            retrieval_query=retrieval_q,
            top_k=self.top_k,
        )
        from app.corpus_first import prepare_local_hits, should_use_web_after_corpus

        portal = _portal_intent(q_clean)
        web_fallback_used = False
        from app.corpus_coverage import corpus_covers_question, explain_coverage

        need_web, hits = should_use_web_after_corpus(q_clean, hits, top_k=self.top_k)

        # Timbre passeport : ne pas confondre avec les montants d'un BO budget
        if _is_passeport_fee_question(q_clean) and not passeport_fee_hits_substantive(
            q_clean, hits
        ):
            need_web = True
            logger.info(
                "Web fallback (timbre passeport) — extraits corpus non pertinents"
            )

        # Décision intelligente : données actuelles → forcer web si corpus
        # ne contient pas les chiffres exacts
        try:
            from app.corpus_coverage import _combined_top_text
            from app.question_type import is_current_data_question

            if (
                is_current_data_question(q_clean)
                and not need_web
                and hits
                and not _is_passeport_fee_question(q_clean)
            ):
                top_text_check = _combined_top_text(hits, 3)
                has_numbers = bool(
                    re.search(
                        r"\d+[,.]?\d*\s*(dh|dirham|%|pour\s*cent)",
                        top_text_check,
                    )
                )
                if not has_numbers:
                    logger.info(
                        "Données chiffrées absentes du corpus → web fallback"
                    )
                    need_web = True
        except Exception:
            pass

        corpus_ok = corpus_covers_question(q_clean, hits)
        logger.info(
            "Retrieval %s | corpus_ok=%s | web_fallback=%s | %s",
            query_analysis.retrieval_path,
            corpus_ok,
            need_web,
            explain_coverage(q_clean, hits),
        )

        if need_web and _labor_topic_question(q_clean):
            from app.labor_corpus import merge_labor_hits

            labor_prepared = merge_labor_hits(q_clean, hits, top_k=self.top_k)
            if corpus_covers_question(q_clean, labor_prepared):
                hits = labor_prepared
                need_web = False
                corpus_ok = True
                logger.info(
                    "Corpus travail suffisant après merge_labor — pas de web pour : %s",
                    q_clean[:80],
                )

        if need_web:
            web_hits = fetch_web_hits(q_clean, top_k=self.top_k)
            if web_hits:
                save_web_hits_to_queue(q_clean, web_hits)
                if portal == "cnie":
                    web_clean = [
                        h for h in web_hits if not _web_hit_is_passeport_not_cnie(h)
                    ]
                    if web_clean:
                        hits = web_clean[: self.top_k]
                    else:
                        hits = _curated_cnie_hits(q_clean)[: self.top_k]
                elif portal == "watiqa":
                    web_clean = [
                        h for h in web_hits if not _web_hit_is_cnie_not_watiqa(h)
                    ]
                    if web_clean:
                        hits = _prioritize_watiqa_hits(web_clean)[: self.top_k]
                    else:
                        hits = _curated_watiqa_hits()[: self.top_k]
                else:
                    web_prepared = prepare_local_hits(q_clean, web_hits, top_k=self.top_k)
                    if corpus_covers_question(q_clean, web_prepared):
                        hits = web_prepared
                    else:
                        hits = web_hits[: self.top_k]
                web_fallback_used = True
            elif portal == "cnie":
                curated = _curated_cnie_hits(q_clean)
                cid = curated[0]["metadata"]["chunk_id"]
                hits = [h for h in hits if (h.get("metadata") or {}).get("chunk_id") != cid]
                hits = (curated + hits)[: self.top_k]
            elif portal == "watiqa":
                curated = _curated_watiqa_hits()
                cid = curated[0]["metadata"]["chunk_id"]
                hits = [h for h in hits if (h.get("metadata") or {}).get("chunk_id") != cid]
                hits = (curated + hits)[: self.top_k]
            elif portal == "passeport" or _is_passeport_fee_question(q_clean):
                curated = curated_fallback_hits(q_clean, top_k=self.top_k)
                if curated:
                    hits = curated[: self.top_k]
                    web_fallback_used = True
                    corpus_ok = True
                    logger.info(
                        "Fallback curated passeport (web vide) — %s",
                        q_clean[:80],
                    )
                else:
                    hits = _curated_passeport_hits(q_clean)[: self.top_k]
                    web_fallback_used = True
            elif _labor_topic_question(q_clean):
                from app.labor_corpus import curated_labor_hits_for, merge_labor_hits

                labor_prepared = merge_labor_hits(q_clean, hits, top_k=self.top_k)
                if corpus_covers_question(q_clean, labor_prepared):
                    hits = labor_prepared
                    web_fallback_used = False
                    corpus_ok = True
                else:
                    curated_labor = curated_labor_hits_for(q_clean)
                    if curated_labor:
                        hits = curated_labor[: self.top_k]
                        web_fallback_used = False
                        corpus_ok = True
                        logger.warning(
                            "Corpus travail curated — %s",
                            q_clean[:100],
                        )
                    else:
                        hits = labor_prepared[: self.top_k] or hits[: self.top_k]
                        web_fallback_used = True
            else:
                rescue = curated_fallback_hits(q_clean, top_k=self.top_k)
                if rescue:
                    hits = rescue[: self.top_k]
                    web_fallback_used = True
                    corpus_ok = True
                    logger.info(
                        "Fallback curated (web vide) — %s",
                        q_clean[:80],
                    )
                else:
                    logger.warning("Fallback web sans résultat pour : %s", q_clean[:100])
                    return {
                        "answer": (
                            "Le corpus local ne contient pas de passage sur ce sujet. "
                            "La recherche web n’a pas renvoyé de source exploitable "
                            "(DuckDuckGo bloqué, sites .ma en 403, ou réseau).\n\n"
                            "Réessayez plus tard ou consultez un portail officiel .ma adapté au sujet."
                        ),
                        "hits": [],
                        "prompt": "",
                        "sources_display": [],
                        "web_fallback_used": True,
                        "corpus_sufficient": False,
                        "answer_source": "web",
                        "retrieval_path": query_analysis.retrieval_path,
                    }
        elif portal:
            hits = hits[: self.top_k]
        else:
            hits = hits[: self.top_k]

        try:
            from app.labor_corpus import is_labor_code_question, merge_labor_hits

            if is_labor_code_question(q_clean):
                hits = merge_labor_hits(q_clean, hits, top_k=self.top_k)
                hits = filter_hits_for_labor_prompt(q_clean, hits, top_k=self.top_k)
                corpus_ok = corpus_covers_question(q_clean, hits)
        except ImportError:
            pass

        if not hits:
            prompt = build_no_corpus_conversation_prompt(
                q_clean, history=hist, user_profile=profile
            )
            text = _safe_llm_complete(self.llm, prompt)
            return {
                "answer": text,
                "hits": [],
                "prompt": prompt,
                "sources_display": [],
                "web_fallback_used": web_fallback_used,
                "corpus_sufficient": False,
                "answer_source": "web" if web_fallback_used else "corpus",
                "retrieval_path": query_analysis.retrieval_path,
            }

        try:
            from app.question_type import is_general_knowledge_question, out_of_scope_reply

            if is_general_knowledge_question(q_clean):
                return {
                    "answer": out_of_scope_reply(q_clean),
                    "hits": [],
                    "prompt": "",
                    "sources_display": [],
                    "web_fallback_used": web_fallback_used,
                    "corpus_sufficient": False,
                    "answer_source": "out_of_scope",
                    "retrieval_path": query_analysis.retrieval_path,
                }
        except ImportError:
            pass

        prompt = build_rag_prompt(q_clean, hits, history=hist, user_profile=profile)
        text = _safe_llm_complete(self.llm, prompt)
        text = _scrub_apology_when_sourced(text, has_corpus_hits=True)
        if _is_master_pedagogic_question(q_clean):
            text = _enforce_master_consistency(text)
            if _hits_look_master_aligned(hits):
                text = _remove_bo_reference_request_if_aligned(text)
        text = _prepend_mandatory_lead_citation(text, hits, question=q_clean)
        text = _append_precise_references(text, hits, question=q_clean)
        return {
            "answer": text,
            "hits": hits,
            "prompt": prompt,
            "sources_display": hits_to_source_lines(hits),
            "web_fallback_used": web_fallback_used,
            "corpus_sufficient": corpus_ok and not web_fallback_used,
            "answer_source": "web" if web_fallback_used else "corpus",
            "retrieval_path": query_analysis.retrieval_path,
        }
