"""Alignement cas CNIE (perte/vol vs première demande)."""

from app.corpus_coverage import corpus_covers_question
from app.portal_cases import (
    cnie_case_from_question,
    cnie_case_aligned,
    top_hits_match_cnie_case,
)
from app.web_fallback import portal_local_hits_sufficient, should_use_web_fallback


def _cnie_hit(text: str, *, chunk_id: str, label: str) -> dict:
    return {
        "score": 0.85,
        "rerank_score": 0.85,
        "text": text,
        "metadata": {
            "chunk_id": chunk_id,
            "category": "cnie",
            "source_type": "admin",
            "source_url": "https://www.cnie.ma/static/procedure",
            "source_org": "CNIE Maroc",
            "label": label,
        },
    }


def test_cnie_case_perte_vol_detected():
    q = "Perte ou vol de la CNIE : procédure"
    assert cnie_case_from_question(q) == "perte_vol"


def test_premiere_hit_not_sufficient_for_perte_vol_question():
    from app.corpus_first import should_use_web_after_corpus

    q = "Perte ou vol de la CNIE : procédure"
    hits = [
        _cnie_hit(
            "Première demande de CNIE — pièces à fournir formulaire acte de naissance",
            chunk_id="curated::cnie_premiere_demande",
            label="Premiere demande - pieces a fournir",
        )
    ]
    assert not cnie_case_aligned(q, hits[0])
    assert not portal_local_hits_sufficient(q, "cnie", hits)
    assert not corpus_covers_question(q, hits)
    need_web, prepared = should_use_web_after_corpus(q + " Maroc", hits, top_k=5)
    assert not need_web
    assert corpus_covers_question(q, prepared)
    assert "perte" in prepared[0]["text"].lower() or "vol" in prepared[0]["text"].lower()


def test_perte_vol_curated_covers_question():
    q = "Perte ou vol de la CNIE : procédure"
    from app.web_fallback import _curated_cnie_hits

    hits = _curated_cnie_hits(q)
    assert top_hits_match_cnie_case(q, hits)
    assert corpus_covers_question(q, hits)
    assert portal_local_hits_sufficient(q, "cnie", hits)
    assert not should_use_web_fallback(q + " Maroc", hits)
