"""Couverture corpus vs fallback web."""

from app.Rag_classique.corpus_coverage import corpus_covers_question
from app.Rag_classique.web_fallback import should_use_web_fallback


def _hit(text: str, *, chunk_id: str = "x", doc_id: str = "SGG", score: float = 0.7):
    return {
        "score": score,
        "rerank_score": score,
        "text": text,
        "metadata": {
            "chunk_id": chunk_id,
            "doc_id": doc_id,
            "source_type": "bulletin_officiel",
            "label": "Article 201",
        },
    }


def test_overtime_covered_by_labor_chunk():
    hits = [_hit("Article 201 heures supplementaires majoration 25 pour cent", doc_id="LABOR65")]
    assert corpus_covers_question("rémunération heures supplémentaires", hits)
    assert not should_use_web_fallback("rémunération heures supplémentaires Maroc", hits)


def test_fiscal_smig_not_covering_overtime():
    from app.Rag_classique.corpus_first import prepare_local_hits, should_use_web_after_corpus
    from app.Rag_classique.labor_corpus import labor_hits_substantively_answer

    hits = [
        _hit(
            "revenu forfaitaire salaire minimum legal prevoyance sociale article 14",
            doc_id="SGG0043",
        )
    ]
    q = "heures supplémentaires rémunération Maroc"
    assert not labor_hits_substantively_answer(q, hits)
    prepared = prepare_local_hits(q, hits, top_k=5)
    assert corpus_covers_question(q, prepared)
    need_web, _ = should_use_web_after_corpus(q, hits, top_k=5)
    assert not need_web


def test_empty_hits_triggers_web():
    assert should_use_web_fallback("permis de construire Maroc", [])
