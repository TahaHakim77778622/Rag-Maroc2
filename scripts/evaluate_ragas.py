#!/usr/bin/env python3
"""
Évaluation de la qualité RAG avec métriques simples.
Ne nécessite pas la librairie ragas — calcul manuel.
"""

from __future__ import annotations

import json
import os
import sys
import time
from pathlib import Path

import requests

BASE = os.environ.get("RAG_TEST_BASE", "http://127.0.0.1:8000").rstrip("/")
DEMO_USER = os.environ.get("WEBAPP_DEMO_USER", "demo")
DEMO_PASS = os.environ.get("WEBAPP_DEMO_PASSWORD", "demo123")

PROJECT_ROOT = Path(__file__).resolve().parent.parent

QUESTIONS = [
    {
        "question": "comment obtenir la CNIE au Maroc ?",
        "keywords": ["cnie", "carte nationale", "commissariat", "pièces", "formulaire"],
        "domain": "cnie",
    },
    {
        "question": "renouvellement passeport marocain pièces",
        "keywords": ["passeport", "acte de naissance", "cnie", "photo", "timbre"],
        "domain": "passeport",
    },
    {
        "question": "heures supplémentaires majoration taux",
        "keywords": ["heures", "majoration", "25%", "50%", "code du travail"],
        "domain": "labor",
    },
    {
        "question": "acte de naissance Watiqa commander",
        "keywords": ["watiqa", "état civil", "naissance", "commande", "en ligne"],
        "domain": "watiqa",
    },
    {
        "question": "autorisation de construire procédure Maroc",
        "keywords": ["construire", "architecte", "commune", "rokhas", "dossier"],
        "domain": "urbanisme",
    },
    {
        "question": "licenciement préavis délai Maroc",
        "keywords": ["préavis", "licenciement", "délai", "code du travail"],
        "domain": "labor",
    },
    {
        "question": "perte CNIE que faire",
        "keywords": ["perte", "vol", "déclaration", "cnie"],
        "domain": "cnie",
    },
    {
        "question": "congé annuel payé jours Maroc",
        "keywords": ["congé", "annuel", "jours", "travail"],
        "domain": "labor",
    },
    {
        "question": "SMIG salaire minimum Maroc",
        "keywords": ["smig", "salaire", "minimum", "code du travail"],
        "domain": "labor",
    },
    {
        "question": "passeport mineur pièces requises",
        "keywords": ["mineur", "passeport", "tuteur", "naissance"],
        "domain": "passeport",
    },
    {
        "question": "normes pédagogiques cycle master",
        "keywords": ["master", "crédits", "semestre", "normes"],
        "domain": "education",
    },
    {
        "question": "accident de travail obligations employeur",
        "keywords": ["accident", "cnss", "déclaration", "employeur"],
        "domain": "labor",
    },
    {
        "question": "contrat CDD conditions durée",
        "keywords": ["cdd", "durée", "déterminée", "renouvellement"],
        "domain": "labor",
    },
    {
        "question": "pièces requises permis de construire",
        "keywords": ["titre foncier", "plans", "architecte", "formulaire"],
        "domain": "urbanisme",
    },
    {
        "question": "inscription doctorat conditions Maroc",
        "keywords": ["doctorat", "thèse", "inscription", "conditions"],
        "domain": "education",
    },
]


def get_token() -> str:
    r = requests.post(
        f"{BASE}/api/auth/login",
        json={"username": DEMO_USER, "password": DEMO_PASS},
        timeout=10,
    )
    if r.status_code == 200:
        return r.json().get("access_token", "")
    return ""


def evaluate() -> int:
    print("=" * 60)
    print("ÉVALUATION RAG-MAROC2")
    print("=" * 60)

    token = get_token()
    if not token:
        print("❌ Impossible d'obtenir le token. Serveur démarré ?")
        return 1

    results: list[dict] = []
    total = len(QUESTIONS)

    for i, q in enumerate(QUESTIONS, 1):
        print(f"\n[{i}/{total}] {q['question'][:50]}...")
        start = time.time()

        try:
            r = requests.post(
                f"{BASE}/api/ask",
                json={"question": q["question"]},
                headers={"Authorization": f"Bearer {token}"},
                timeout=60,
            )
            elapsed = time.time() - start

            if r.status_code != 200:
                results.append(
                    {
                        "question": q["question"],
                        "domain": q["domain"],
                        "status": "ERREUR",
                        "latency": round(elapsed, 2),
                        "relevance": 0,
                        "faithfulness": 0,
                    }
                )
                print(f"   ❌ HTTP {r.status_code}")
                continue

            data = r.json()
            answer = data.get("answer", "").lower()
            source = data.get("answer_source", "unknown")

            found = sum(1 for k in q["keywords"] if k.lower() in answer)
            relevance = found / len(q["keywords"]) if q["keywords"] else 0.0

            faithfulness = 1.0 if source == "corpus" else 0.5

            completeness = min(len(answer) / 200, 1.0)

            score = (relevance + faithfulness + completeness) / 3

            results.append(
                {
                    "question": q["question"],
                    "domain": q["domain"],
                    "status": "OK",
                    "latency": round(elapsed, 2),
                    "relevance": round(relevance, 2),
                    "faithfulness": round(faithfulness, 2),
                    "completeness": round(completeness, 2),
                    "score": round(score, 2),
                    "source": source,
                    "answer_preview": answer[:100],
                }
            )

            emoji = "✅" if score >= 0.6 else "⚠️"
            print(
                f"   {emoji} score={score:.2f} | "
                f"relevance={relevance:.2f} | "
                f"source={source} | "
                f"latency={elapsed:.1f}s"
            )

        except Exception as e:
            results.append(
                {
                    "question": q["question"],
                    "domain": q["domain"],
                    "status": "EXCEPTION",
                    "error": str(e),
                }
            )
            print(f"   ❌ Exception: {e}")

    ok = [r for r in results if r.get("status") == "OK"]
    print("\n" + "=" * 60)
    print("RÉSULTATS FINAUX")
    print("=" * 60)

    avg_score = avg_relevance = avg_latency = corpus_rate = 0.0

    if ok:
        avg_score = sum(r["score"] for r in ok) / len(ok)
        avg_relevance = sum(r["relevance"] for r in ok) / len(ok)
        avg_latency = sum(r["latency"] for r in ok) / len(ok)
        corpus_rate = sum(1 for r in ok if r.get("source") == "corpus") / len(ok)

        print(f"Questions testées    : {total}")
        print(f"Réponses OK          : {len(ok)}/{total}")
        print(f"Score moyen          : {avg_score:.2%}")
        print(f"Pertinence moyenne   : {avg_relevance:.2%}")
        print(f"Latence moyenne      : {avg_latency:.1f}s")
        print(f"Taux corpus          : {corpus_rate:.2%}")
        print(f"Taux fallback web    : {1 - corpus_rate:.2%}")

        print("\nPar domaine :")
        domains = sorted({r["domain"] for r in ok})
        for domain in domains:
            domain_results = [r for r in ok if r["domain"] == domain]
            domain_score = sum(r["score"] for r in domain_results) / len(domain_results)
            print(f"  {domain:15} : {domain_score:.2%} ({len(domain_results)} questions)")
    else:
        print("Aucune réponse OK.")

    output = PROJECT_ROOT / "metrics" / "ragas_results.json"
    output.parent.mkdir(parents=True, exist_ok=True)
    with output.open("w", encoding="utf-8") as f:
        json.dump(
            {
                "summary": {
                    "total": total,
                    "ok": len(ok),
                    "avg_score": round(avg_score, 3),
                    "avg_relevance": round(avg_relevance, 3),
                    "corpus_rate": round(corpus_rate, 3),
                    "avg_latency": round(avg_latency, 2),
                },
                "details": results,
            },
            f,
            ensure_ascii=False,
            indent=2,
        )

    print(f"\n💾 Résultats sauvegardés dans {output}")
    print("=" * 60)
    return 0 if ok else 1


if __name__ == "__main__":
    sys.exit(evaluate())
