"""Politique dataset d'abord, web ensuite."""

from app.Rag_classique.corpus_coverage import corpus_covers_question
from app.Rag_classique.corpus_first import prepare_local_hits, should_use_web_after_corpus
from app.Rag_classique.web_fallback import should_use_web_fallback


def _bad_cnie_hit():
    return {
        "score": 0.7,
        "rerank_score": 0.7,
        "text": "Premiere demande CNIE pieces a fournir",
        "metadata": {
            "chunk_id": "curated::cnie_premiere_demande",
            "category": "cnie",
            "source_url": "https://cnie.ma",
        },
    }


def test_mineur_passeport_dataset_before_web():
    q = "je suis un mineur sans CNIE et je veux demander mon passeport Maroc"
    need_web, hits = should_use_web_after_corpus(q, [_bad_cnie_hit()], top_k=5)
    assert not need_web
    assert corpus_covers_question(q, hits)
    assert not should_use_web_fallback(q, [_bad_cnie_hit()])
    assert "passeport" in hits[0]["text"].lower()


def test_perte_vol_cnie_dataset_before_web():
    q = "Perte ou vol de la CNIE : procédure Maroc"
    need_web, hits = should_use_web_after_corpus(q, [_bad_cnie_hit()], top_k=5)
    assert not need_web
    assert "perte" in hits[0]["text"].lower() or "vol" in hits[0]["text"].lower()
