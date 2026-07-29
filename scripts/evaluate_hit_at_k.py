#!/usr/bin/env python3
"""
Évaluation hit@k (article IEEE, 50 questions) — retrieval seul vs + reranking.

Usage:
    python scripts/evaluate_hit_at_k.py
    python scripts/evaluate_hit_at_k.py --k 3 --limit 10
"""

from __future__ import annotations

import argparse
import json
import sys
from collections import defaultdict
from pathlib import Path

_ROOT = Path(__file__).resolve().parent.parent
if str(_ROOT) not in sys.path:
    sys.path.insert(0, str(_ROOT))

QUESTIONS_PATH = _ROOT / "data" / "eval" / "questions_50.json"


def _keyword_hit(question: str, text: str, theme: str) -> bool:
    q = (question + " " + theme).lower()
    t = (text or "").lower()
    theme_kw = {
        "cnie": ("cnie", "carte nationale", "identité", "identite", "dgsn"),
        "passeport": ("passeport", "biométrique", "biometrique", "consulat"),
        "watiqa": ("watiqa", "état civil", "etat civil", "acte", "guichet"),
        "droit_travail": ("travail", "employeur", "salarié", "licenciement", "contrat"),
        "construction": ("construction", "urbanisme", "permis", "bâtiment", "batiment"),
    }
    kws = theme_kw.get(theme, ())
    if any(k in t for k in kws):
        return True
    # fallback : au moins 2 mots significatifs de la question
    tokens = [w for w in q.split() if len(w) > 4][:6]
    return sum(1 for w in tokens if w in t) >= 2


def main() -> int:
    p = argparse.ArgumentParser()
    p.add_argument("--k", type=int, default=3)
    p.add_argument("--limit", type=int, default=0, help="0 = toutes les questions")
    p.add_argument("--questions", type=Path, default=QUESTIONS_PATH)
    args = p.parse_args()

    if not args.questions.is_file():
        print(f"Fichier introuvable : {args.questions}", file=sys.stderr)
        return 1

    from app.Rag_classique.query_expand import expand_query_for_retrieval  # noqa: PLC0415
    from app.Rag_classique.retrieval_rerank import rerank_hits  # noqa: PLC0415
    from app.Rag_classique.retriever import Retriever  # noqa: PLC0415

    items = json.loads(args.questions.read_text(encoding="utf-8"))
    if args.limit > 0:
        items = items[: args.limit]

    retriever = Retriever()
    stats_faiss = defaultdict(lambda: {"hit": 0, "n": 0})
    stats_rerank = defaultdict(lambda: {"hit": 0, "n": 0})

    for item in items:
        theme = item.get("theme", "other")
        q = item["question"]
        rq = expand_query_for_retrieval(q, history=None)
        raw = retriever.search(rq, k=max(10, args.k * 3))
        reranked = rerank_hits(raw, question=q, retrieval_query=rq, top_k=args.k)

        def top_hit(hits) -> bool:
            for h in hits[: args.k]:
                if _keyword_hit(q, h.get("text", ""), theme):
                    return True
            return False

        stats_faiss[theme]["n"] += 1
        stats_rerank[theme]["n"] += 1
        if top_hit(raw):
            stats_faiss[theme]["hit"] += 1
        if top_hit(reranked):
            stats_rerank[theme]["hit"] += 1

    print(f"\n=== hit@{args.k} par thématique ===")
    total_f, total_r = 0, 0
    total_n = 0
    for theme in sorted(set(list(stats_faiss.keys()) + list(stats_rerank.keys()))):
        n = stats_faiss[theme]["n"]
        hf = 100.0 * stats_faiss[theme]["hit"] / max(1, n)
        hr = 100.0 * stats_rerank[theme]["hit"] / max(1, n)
        print(f"  {theme:16}  FAISS {hf:5.1f}%   +rerank {hr:5.1f}%   (n={n})")
        total_f += stats_faiss[theme]["hit"]
        total_r += stats_rerank[theme]["hit"]
        total_n += n
    if total_n:
        print(
            f"\n  GLOBAL            FAISS {100*total_f/total_n:5.1f}%   +rerank {100*total_r/total_n:5.1f}%"
        )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
