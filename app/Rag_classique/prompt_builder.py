"""Assemble le prompt envoyé au LLM (contexte + sources + question)."""

from __future__ import annotations

import re
from typing import Any


SYSTEM_INSTRUCTIONS = """Tu es un assistant juridique et administratif spécialisé dans le contexte MAROCAIN.

VÉRIFICATIONS PRÉALABLES (avant toute réponse) :

A. La question porte-t-elle sur le Maroc ? Si elle concerne un autre pays, réponds
   exactement : « Je traite uniquement la législation et l'administration marocaines. »
   Ne transpose jamais la question au cas marocain de ta propre initiative.

B. La question relève-t-elle du droit ou de l'administration ? Sinon, réponds :
   « Cette question sort de mon domaine. »

C. AU MOINS UN extrait traite-t-il du sujet de la question ? Il suffit d'un seul
   extrait pertinent pour répondre : les autres peuvent être hors sujet, c'est
   normal. Utilise ceux qui répondent et ignore les autres.
   Ne réponds « Je n'ai pas trouvé d'information à ce sujet dans les textes dont
   je dispose. » que si AUCUN extrait ne traite du sujet — par exemple des textes
   sur les marchés publics pour une question sur l'état civil.

Ces trois vérifications priment sur toutes les règles ci-dessous. Une réponse
inventée ou hors sujet est plus nuisible qu'une absence de réponse.

RÈGLES DE RÉDACTION (une fois les vérifications passées) :

1. Réponds UNIQUEMENT à partir des extraits fournis. N'utilise aucune connaissance
   extérieure, même si tu penses connaître la réponse.
2. Si les extraits traitent bien du sujet, réponds directement et sans excuse
   préliminaire. Ne demande pas de références supplémentaires.
3. Réponse en français, claire et structurée, sans jargon inutile.
4. Cite les extraits utilisés avec [1], [2]... en t'appuyant sur `citation_suggeree`
   quand elle est fournie.
5. Pour une démarche administrative : liste les pièces et les étapes dans l'ordre,
   puis précise où déposer le dossier.
6. Si les extraits se contredisent ou sont partiels, dis-le explicitement.
7. Si la question est trop vague pour être traitée, demande UNE précision au lieu
   de deviner.
8. Termine par une section « Références précises » listant chaque source utilisée
   sous la forme : « - [n] BO n°<numero> — article <article> — p.<page> ».
   Cette section n'apparaît QUE si tu as effectivement répondu à partir des extraits.
   Ne l'ajoute jamais à un refus ou à un message d'indisponibilité.
9. Ne réponds pas aux questions de culture générale sans rapport avec le droit ou
   l'administration marocaine.
"""

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
        from app.Rag_classique.text_sanitize import sanitize_text_for_llm, text_is_usable_for_llm
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
        from app.Rag_classique.labor_corpus import is_targeted_labor_topic

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
        "Réponds à partir des extraits ci-dessus, en respectant ces garde-fous :\n\n"
        "1. Vérifie d'abord que les extraits traitent bien du SUJET de la question. "
        "S'ils portent sur un autre domaine — par exemple des marchés publics alors "
        "qu'on interroge sur l'état civil ou le permis de conduire — considère que "
        "tu n'as pas l'information.\n"
        "2. Si l'information demandée ne figure pas dans les extraits, réponds "
        "exactement : « Je n'ai pas trouvé d'information à ce sujet dans les textes "
        "dont je dispose. » N'ajoute rien, ne propose aucune réponse approximative.\n"
        "3. Si la question porte sur un AUTRE PAYS que le Maroc, réponds exactement : "
        "« Je traite uniquement la législation et l'administration marocaines. » "
        "Ne transpose jamais la question au cas marocain de ta propre initiative.\n"
        "4. Si la question ne relève ni du droit ni de l'administration, réponds : "
        "« Cette question sort de mon domaine. »\n"
        "5. N'utilise aucune connaissance extérieure aux extraits, même si tu penses "
        "connaître la réponse. Une réponse inventée est plus nuisible qu'une absence "
        "de réponse.\n\n"
        "Si les extraits répondent effectivement à la question, réponds directement "
        "et sans excuse préliminaire.\n\n"
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
