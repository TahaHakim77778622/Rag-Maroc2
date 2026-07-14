"""Couverture de tout le dataset via le registre des domaines."""

from app.corpus_coverage import corpus_covers_question
from app.corpus_first import should_use_web_after_corpus
from app.dataset_registry import active_domains, hit_matches_domain, is_bulletin_hit


def _bo_hit(text: str, doc_id: str = "SGG0042"):
    return {
        "score": 0.65,
        "rerank_score": 0.65,
        "text": text,
        "metadata": {
            "chunk_id": f"{doc_id}_unit_1",
            "doc_id": doc_id,
            "source_type": "bulletin_officiel",
            "category": "juridique",
            "source_org": "SGG",
            "label": "Article 12",
        },
    }


def test_active_domains_defaults_to_bulletin():
    domains = active_domains("Quelles sont les règles générales ?")
    assert "bulletin" in domains or "juridique" in domains


def test_bo_hit_recognized():
    h = _bo_hit("Décret relatif aux normes pédagogiques du cycle de master")
    assert is_bulletin_hit(h)
    assert hit_matches_domain(h, "bulletin")
    assert hit_matches_domain(h, "education")


def test_master_question_covered_by_bo_without_web():
    q = "normes pédagogiques cycle de master au Maroc"
    hits = [_bo_hit("Normes pédagogiques nationales du cycle de master filière crédits semestres")]
    assert corpus_covers_question(q, hits)
    need_web, _ = should_use_web_after_corpus(q, hits, top_k=5)
    assert not need_web


def test_passeport_domain_in_registry():
    domains = active_domains("je suis mineur sans cnie et je veux mon passeport")
    assert "passeport" in domains
