"""Assemble le prompt envoyé au LLM (contexte + sources + question)."""

from __future__ import annotations

import re
from typing import Any


SYSTEM_INSTRUCTIONS = """Tu es un assistant spécialisé dans le droit, l'administration et les services publics au Maroc.
Réponds en français, de façon claire, directe et utile.

## RÈGLE PRINCIPALE
Quand des extraits sont fournis, UTILISE-LES pour répondre directement à la question.
Ne dis JAMAIS que les extraits sont insuffisants si tu peux en tirer une réponse utile,
même partielle. Synthétise, déduis, explique à partir de ce que les extraits contiennent.

## FORMAT DE RÉPONSE
- Commence directement par la réponse, sans introduction inutile.
- Cite [1], [2], etc. quand tu t'appuies sur un extrait.
- Si les extraits couvrent partiellement le sujet : donne ce qu'ils permettent d'affirmer,
  puis complète avec le cadre général marocain connu (sans inventer de loi).
- Termine par un mini-bloc "Références précises" avec 1-3 sources courtes.

## INTERDICTIONS ABSOLUES
- Ne commence JAMAIS par "Je suis désolé", "Désolé", "Je ne peux pas".
- Ne dis JAMAIS "les extraits ne fournissent pas d'informations spécifiques" si les
  extraits parlent du même domaine que la question (travail, urbanisme, CNIE, etc.)
- Ne demande PAS de référence précise (BO, article) si les extraits te donnent déjà
  assez d'informations pour répondre.
- N'invente pas de lois ou d'articles non présents dans les extraits.

## COMPORTEMENT PAR DOMAINE

### Code du travail (SMIG, heures sup, CDD, accident, licenciement)
Si les extraits mentionnent le Code du travail ou la loi 65-99 :
- Réponds directement avec les informations des extraits [1] (Code du travail / emploi.gov.ma).
- Ne cite PAS en [1] un Bulletin Officiel sur la nomenclature des dépenses de l'État ou le fiscal
  si la question porte sur le SMIG, un accident de travail ou les heures supplémentaires.
- Complète avec le cadre général du droit du travail marocain si nécessaire.
- Ne dis pas que les extraits sont hors sujet si [1] est déjà un extrait Code du travail pertinent.

### Urbanisme / Construction
Si les extraits mentionnent architecte, commune, urbanisme, autorisation :
- Explique la procédure à partir des extraits.
- Mentionne Rokhas.ma comme plateforme nationale si pertinent.
- Ne demande pas de référence BO si la procédure est claire.

### CNIE / Passeport / Watiqa
Si les extraits parlent de pièces, procédure, demande :
- Donne la liste structurée directement.
- Cite la source pour chaque élément.

### Bulletin Officiel / Lois
Si les extraits contiennent du texte juridique :
- Résume le contenu de façon claire et accessible.
- Cite le BO et l'article précis si disponibles dans les extraits.
- Si le texte est en arabe ou illisible, dis-le en une phrase et donne ce qui est lisible.

## QUAND LES EXTRAITS SONT VRAIMENT INSUFFISANTS
Seulement si les extraits ne parlent PAS du même domaine que la question :
- Une phrase courte pour expliquer l'écart.
- Puis donne le cadre général marocain connu sur le sujet.
- Pose 2-3 questions de clarification si besoin.

## CITATIONS
Format : "Selon [1], ..." ou "D'après le Code du travail [2], ..."
Mini-bloc final obligatoire :
"Références précises :
- Source [1] : ...
- Source [2] : ..."

## PROFIL UTILISATEUR
Si un profil est fourni (célibataire, marié, mineur, lieu), adapte la réponse à ce profil.

## CONVERSATION
Si une conversation précédente est fournie, tiens compte des échanges pour interpréter la question actuelle
(pronoms, sujets implicites). Avec des extraits, n'invente pas de faits qui les contrediraient."""


def build_no_corpus_conversation_prompt(
    question: str,
    *,
    history: list[dict[str, Any]] | None = None,
    user_profile: dict[str, str] | None = None,
) -> str:
    """
    Quand le retrieval + web n’ont rien donné : garde un ton conversationnel et oriente vers le Maroc
    (questions ciblées) au lieu d’un refus sec.
    """
    instructions = (
        "Aucun extrait n’a été trouvé pour l’instant. Tu ne cites pas de sources [n].\n"
        "Réponds en 4 à 8 phrases maximum. Pas de formule d’excuse lourde.\n"
        "Pour un projet (construction, achat, entreprise, etc.) au Maroc, structure ta réponse ainsi :\n"
        "- 1-2 phrases sur le principe (autorisations / acteurs : commune, autorisations territoriales, professionnels habilités) sans inventer des numéros d’articles.\n"
        "- 2 à 4 questions de précision (ville, commune, type de fonds de commerce ou terrain, surface, logement/activité, etc.).\n"
        "Si la question est hors sujet par rapport au Maroc, réponds brièvement puis recentre poliment sur ce que tu peux aider côté Maroc."
    )
    hist_block = format_history_block(history or [])
    convo_section = ""
    if hist_block:
        convo_section = f"### Conversation précédente\n\n{hist_block}\n\n"
    profile_section = ""
    if user_profile:
        pairs = [f"- {k}: {v}" for k, v in user_profile.items()]
        profile_section = "### Profil utilisateur infere\n\n" + "\n".join(pairs) + "\n\n"
    return (
        f"{SYSTEM_INSTRUCTIONS}\n\n"
       f"{convo_section}"
        f"{profile_section}"
        f"### Directives (pas d’extraits)\n\n{instructions}\n\n"
        f"### Question actuelle\n\n{question.strip()}\n"
    )


def format_history_block(history: list[dict[str, Any]], *, max_turns: int = 14) -> str:
    """Réduit l’historique pour le prompt (ordre chronologique)."""
    if not history:
        return ""
    lines: list[str] = []
    for m in history[-max_turns:]:
        role = (m.get("role") or "").strip().lower()
        content = (m.get("content") or "").strip()
        if not content or role not in ("user", "assistant"):
            continue
        label = "Utilisateur" if role == "user" else "Assistant"
        # Limite par message pour ne pas exploser le contexte LLM
        if len(content) > 3500:
            content = content[:3497] + "…"
        lines.append(f"{label}: {content}")
    return "\n\n".join(lines)


def _extract_bo_number(text: str) -> str | None:
    t = text or ""
    m = re.search(r"(?:N[º°o]\s*|n[º°o]\s*)(\d{3,5}(?:\s*bis)?)", t)
    if not m:
        m = re.search(r"\b(\d{3,5}(?:\s*bis)?)\s*[-–]\s*\d{1,2}\s*\w+\s*\d{4}", t)
    if not m:
        return None
    return " ".join(m.group(1).split())


def _extract_article_hint(label: str, body: str) -> str | None:
    l = (label or "").strip()
    if l:
        return l
    m = re.search(r"\b(?:ART(?:ICLE)?\.?\s*)(?:N[º°o]\s*)?([0-9]{1,4}|PREMIER|UNIQUE)\b", body or "", flags=re.IGNORECASE)
    if not m:
        return None
    raw = m.group(1)
    if raw.isdigit():
        return f"Article {raw}"
    return f"Article {raw.capitalize()}"


def _build_precise_citation(meta: dict[str, Any], body: str) -> str:
    source_type = str(meta.get("source_type") or "").strip().lower()
    title = str(meta.get("title") or "").strip() or "Source"
    label = str(meta.get("label") or "").strip()
    url = str(meta.get("source_url") or "").strip()
    page_start = meta.get("page_start")

    if source_type == "bulletin_officiel":
        bo = _extract_bo_number(body)
        article = _extract_article_hint(label, body)
        bits: list[str] = ["Bulletin Officiel"]
        if bo:
            bits.append(f"n°{bo}")
        if article:
            bits.append(article)
        if isinstance(page_start, int):
            bits.append(f"p.{page_start}")
        return " — ".join(bits)

    host_hint = ""
    if url:
        host = re.sub(r"^https?://", "", url).split("/", 1)[0].strip()
        if host:
            host_hint = host
    if host_hint:
        if label:
            return f"{host_hint} — {label}"
        return f"{host_hint} — {title}"
    if label:
        return f"{title} — {label}"
    return title


def build_rag_prompt(
    question: str,
    hits: list[dict[str, Any]],
    *,
    history: list[dict[str, Any]] | None = None,
    user_profile: dict[str, str] | None = None,
) -> str:
    """
    Construit un unique prompt texte prêt pour un LLM (API ou local).
    `hits` : liste telle que retournée par Retriever.search (clés metadata, text, score).
    """
    blocks: list[str] = []
    try:
        from app.text_sanitize import sanitize_text_for_llm, text_is_usable_for_llm
    except ImportError:

        def sanitize_text_for_llm(t: str, **_: object) -> str:
            return (t or "").strip()

        def text_is_usable_for_llm(t: str, **_: object) -> bool:
            return bool((t or "").strip())

    for i, h in enumerate(hits, start=1):
        meta = h.get("metadata") or {}
        raw = (h.get("text") or "").strip()
        if not text_is_usable_for_llm(raw):
            continue
        body = sanitize_text_for_llm(raw)
        precise_citation = _build_precise_citation(meta, body)
        header = (
            f"[{i}] chunk_id={meta.get('chunk_id', '')!r} | "
            f"title={meta.get('title', '')!r} | label={meta.get('label', '')!r} | "
            f"source_type={meta.get('source_type', '')!r}"
        )
        if meta.get("source_url"):
            header += f"\n    url={meta['source_url']}"
        if meta.get("page_start") is not None:
            header += f"\n    page_start={meta.get('page_start')!r}"
        header += f"\n    citation_suggeree={precise_citation!r}"
        blocks.append(f"{header}\n{body}")

    context = "\n\n---\n\n".join(blocks) if blocks else "(Aucun extrait pertinent.)"

    hist_block = format_history_block(history or [])
    convo_section = ""
    if hist_block:
        convo_section = f"### Conversation précédente\n\n{hist_block}\n\n"

    profile_section = ""
    if user_profile:
        pairs = [f"- {k}: {v}" for k, v in user_profile.items()]
        profile_section = "### Profil utilisateur infere\n\n" + "\n".join(pairs) + "\n\n"

    labor_hint = ""
    try:
        from app.labor_corpus import is_targeted_labor_topic

        if is_targeted_labor_topic(question):
            labor_hint = (
                "### Priorité des sources (droit du travail)\n\n"
                "Pour cette question, base-toi sur [1] (Code du travail / loi 65-99). "
                "N'utilise pas un BO fiscal ou « nomenclature des pièces justificatives » comme source principale.\n\n"
            )
    except ImportError:
        pass

    return (
        f"{SYSTEM_INSTRUCTIONS}\n\n"
        f"{convo_section}"
        f"{profile_section}"
        f"{labor_hint}"
        f"### Extraits du corpus\n\n{context}\n\n"
        f"### Question actuelle\n\n{question.strip()}\n\n"
        f"### Réponse attendue\n\n"
        "Réponds directement à partir des extraits ci-dessus (règle principale). "
        "Pas d'excuse ni de phrase sur l'insuffisance des extraits si le même domaine y figure.\n\n"
    )


def hits_to_source_lines(hits: list[dict[str, Any]]) -> list[str]:
    """Résumé court des sources pour affichage terminal / UI."""
    lines: list[str] = []
    for i, h in enumerate(hits, start=1):
        m = h.get("metadata") or {}
        disp = h.get("rerank_score")
        if disp is None:
            disp = h.get("score", 0)
        lines.append(
            f"  [{i}] score={float(disp):.4f} | "
            f"{m.get('chunk_id', '')} | {m.get('title', '')} — {m.get('label', '')}"
        )
    return lines
