"""
Chargement paresseux du pipeline RAG (index + corpus lourds au premier usage).
"""

from __future__ import annotations

import hashlib
import logging
import re
import time
from typing import Any

logger = logging.getLogger(__name__)

def _top_source_subject_mismatch(question: str, top_preview: str) -> bool:
    """
    Détecte le cas observé en prod : question « master / normes pédagogiques » mais premier extrait = marchés publics.
    Sert à ne pas afficher « fiabilité forte » quand le retrieval s’est trompé de domaine.
    """
    if not question or not top_preview:
        return False
    try:
        from app.retrieval_rerank import _education_master_intent, _procurement_public_chunk
    except ImportError:
        return False
    qlow = question.lower()
    if not _education_master_intent(qlow):
        return False
    if not _procurement_public_chunk(top_preview):
        return False
    pl = top_preview.lower()
    if any(
        x in pl
        for x in (
            "normes pédagogiques",
            "normes pedagogiques",
            "cycle de master",
            "cahier des normes",
            "enseignement supérieur",
            "enseignement superieur",
            "projet de fin d'études",
            "projet de fin d'etudes",
        )
    ):
        return False
    return True


_pipeline = None
_ASK_CACHE: dict[str, tuple[float, dict[str, Any]]] = {}
_ASK_CACHE_TTL_SEC = 120.0
_ASK_CACHE_MAX_ITEMS = 200


def get_pipeline():
    global _pipeline
    if _pipeline is None:
        # Recharge .env au moment du 1er RAG (uvicorn peut avoir démarré avant que .env soit lu).
        try:
            from dotenv import load_dotenv  # noqa: PLC0415
            from pathlib import Path  # noqa: PLC0415

            env_path = Path(__file__).resolve().parent.parent / ".env"
            if env_path.is_file():
                load_dotenv(env_path, override=False)
        except ImportError:
            pass

        logger.info("Initialisation du pipeline RAG (premier appel, peut prendre du temps)…")
        # Évite souvent des soucis natifs (tokenizers) sur macOS.
        import os as _os

        _os.environ.setdefault("TOKENIZERS_PARALLELISM", "false")

        from app.llm_factory import get_llm_client  # noqa: PLC0415
        from app.rag_pipeline import RAGPipeline  # noqa: PLC0415

        _pipeline = RAGPipeline(llm=get_llm_client())
    return _pipeline


def ask(
    question: str,
    top_k: int,
    history: list[dict[str, Any]] | None = None,
    *,
    user_id: int | None = None,
) -> dict[str, Any]:
    """Réponse JSON-sérialisable pour l’API web."""
    hist = history or []
    # Cache court pour accélérer les répétitions (double-click, reformulation identique, refresh).
    hist_tail = hist[-12:]
    cache_key_raw = f"{question.strip()}|{int(top_k)}|{hist_tail!r}"
    cache_key = hashlib.sha256(cache_key_raw.encode("utf-8")).hexdigest()
    now = time.time()
    cached = _ASK_CACHE.get(cache_key)
    if cached and now - cached[0] <= _ASK_CACHE_TTL_SEC:
        out_cached = dict(cached[1])
        out_cached["cached"] = True
        return out_cached

    t0 = time.time()
    pipeline = get_pipeline()
    pipeline.top_k = max(1, min(top_k, 20))
    llm_error = False
    try:
        out = pipeline.answer(question.strip(), history=hist or None)
    except Exception:
        llm_error = True
        raise

    def _term_set(text: str) -> set[str]:
        return {
            t
            for t in re.findall(r"[a-zA-Z0-9àâäéèêëïîôùûüçœ]{4,}", (text or "").lower())
            if t not in {"avec", "dans", "pour", "maroc", "question", "votre", "vous"}
        }

    def _source_confidence(score: float, source_type: str | None) -> tuple[str, str]:
        boost = 0.06 if source_type in ("admin", "bulletin_officiel") else 0.0
        s = score + boost
        if s >= 0.58:
            return ("Fort", "high")
        if s >= 0.34:
            return ("Moyen", "medium")
        return ("Faible", "low")

    def _explain_source(question_text: str, preview: str, meta: dict[str, Any], score: float) -> str:
        q_terms = _term_set(question_text)
        p_terms = _term_set(preview)
        overlap = sorted(q_terms.intersection(p_terms))
        overlap_txt = ", ".join(overlap[:4]) if overlap else "votre intention générale"
        st = str(meta.get("source_type") or "")
        if st == "admin":
            origin = "source administrative"
        elif st == "bulletin_officiel":
            origin = "publication officielle (BO)"
        elif st == "web_fallback":
            origin = "source web de secours"
        else:
            origin = "source du corpus"
        return (
            f"Choisie car elle recoupe {overlap_txt}; "
            f"type: {origin}; score de pertinence {score:.2f}."
        )

    sources = []
    confidence_points: list[float] = []
    for h in out.get("hits") or []:
        meta = h.get("metadata") or {}
        raw_text = h.get("text") or ""
        text = str(raw_text)[:600]
        sc = round(
            float(
                h["rerank_score"]
                if "rerank_score" in h
                else h.get("score", 0.0)
            ),
            4,
        )
        confidence_points.append(float(sc))
        conf_label, conf_code = _source_confidence(float(sc), meta.get("source_type"))
        sources.append(
            {
                "chunk_id": meta.get("chunk_id"),
                "title": meta.get("title"),
                "label": meta.get("label"),
                "source_type": meta.get("source_type"),
                "source_url": meta.get("source_url"),
                "score": sc,
                "confidence_label": conf_label,
                "confidence_code": conf_code,
                "preview": text + ("…" if len(str(raw_text)) > 600 else ""),
                "explain": _explain_source(question, text, meta, float(sc)),
            }
        )

    top_mismatch = _top_source_subject_mismatch(question, sources[0]["preview"]) if sources else False

    logger.info("RAG /api/ask : %s passages renvoyés au client", len(sources))

    reliability = {
        "label": "Faible",
        "code": "low",
        "score": 0.0,
        "detail": "Aucune source locale solide n'a ete retournee.",
    }
    if confidence_points:
        avg = sum(confidence_points) / max(1, len(confidence_points))
        admin_share = sum(
            1 for s in sources if s.get("source_type") in ("admin", "bulletin_officiel")
        ) / max(1, len(sources))
        final = avg + (0.08 * admin_share)
        if final >= 0.58:
            reliability = {
                "label": "Fort",
                "code": "high",
                "score": round(final, 3),
                "detail": "Extraits bien alignes avec la question et majoritairement officiels.",
            }
        elif final >= 0.34:
            reliability = {
                "label": "Moyen",
                "code": "medium",
                "score": round(final, 3),
                "detail": "Reponse plausible, mais certains passages demandent verification.",
            }
        else:
            reliability = {
                "label": "Faible",
                "code": "low",
                "score": round(final, 3),
                "detail": "Peu de recouvrement lexical ou sources trop generales.",
            }

    if top_mismatch:
        prev_sc = float(reliability.get("score") or 0.0)
        reliability = {
            "label": "Faible",
            "code": "low",
            "score": round(min(prev_sc, 0.40), 3),
            "detail": (
                "Le premier extrait semble concerner un autre domaine du Bulletin officiel que votre question "
                "(ex. marches publics vs enseignement superieur). Fiez-vous aux sources listees et reformulez "
                "si la reponse reste hors sujet."
            ),
        }

    # Suggestions rapides pour garder l'échange interactif.
    ql = (question or "").lower()
    quick_suggestions = [
        "Pouvez-vous me donner la liste exacte des pieces ?",
        "Quelles etapes dois-je suivre en premier ?",
        "A qui dois-je m'adresser dans ma ville ?",
    ]
    if "construction" in ql or "construire" in ql or "terrain" in ql:
        quick_suggestions = [
            "Je veux la checklist terrain + autorisation.",
            "Quels documents pour demander l'autorisation de construire ?",
            "Quels delais moyens et points de blocage ?",
        ]
    elif "cnie" in ql or "passeport" in ql or "etat civil" in ql:
        quick_suggestions = [
            "Donnez-moi la version premiere demande.",
            "Et pour renouvellement ou perte/vol ?",
            "Quelles differences Maroc vs consulat ?",
        ]
    elif any(
        x in ql
        for x in (
            "cycle de master",
            "cycle master",
            "normes pédagogiques",
            "normes pedagogiques",
            "cahier des normes",
            "enseignement superieur",
            "enseignement supérieur",
        )
    ) or ("master" in ql and ("pedagogique" in ql or "pédagogique" in ql or "universit" in ql)):
        quick_suggestions = [
            "Citez le numero de l'arrêté ou la date de publication au B.O.",
            "Resume l'article premier des normes pédagogiques du master selon les extraits.",
            "Quels credits et semestres pour une filiere master ?",
        ]

    web_fb = bool(out.get("web_fallback_used")) or any(
        s.get("source_type") == "web_fallback" for s in sources
    )
    latency = time.time() - t0
    try:
        from webapp.metrics_store import record_ask_event  # noqa: PLC0415

        record_ask_event(
            user_id=user_id,
            question=question,
            latency_sec=latency,
            n_sources=len(sources),
            web_fallback=web_fb,
            llm_error=llm_error,
        )
    except Exception:
        pass

    payload = {
        "answer": out.get("answer", ""),
        "sources": sources,
        "sources_lines": out.get("sources_display") or [],
        "reliability": reliability,
        "quick_suggestions": quick_suggestions,
        "web_fallback_used": web_fb,
        "corpus_sufficient": bool(out.get("corpus_sufficient", not web_fb)),
        "answer_source": out.get("answer_source", "web" if web_fb else "corpus"),
        "retrieval_path": out.get("retrieval_path", "corpus"),
        "latency_sec": round(latency, 3),
        "cached": False,
    }
    _ASK_CACHE[cache_key] = (now, payload)
    if len(_ASK_CACHE) > _ASK_CACHE_MAX_ITEMS:
        # Eviction simple: supprimer les entrées les plus anciennes.
        keys_by_oldest = sorted(_ASK_CACHE.items(), key=lambda kv: kv[1][0])
        for k, _ in keys_by_oldest[: max(1, len(_ASK_CACHE) - _ASK_CACHE_MAX_ITEMS)]:
            _ASK_CACHE.pop(k, None)
    return payload
