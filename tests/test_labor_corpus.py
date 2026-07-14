"""Tests détection travail / heures supplémentaires et anti-faux-positifs BO."""

from app.corpus_coverage import corpus_covers_question
from app.labor_corpus import (
    filter_hits_for_labor_prompt,
    is_accident_travail_question,
    is_labor_code_question,
    is_off_topic_labor_hit,
    is_overtime_question,
    is_smig_question,
    labor_hits_substantively_answer,
    merge_labor_hits,
    primary_hit_for_answer,
)


def test_overtime_question_detected():
    assert is_overtime_question("Heures supplémentaires rémunération")
    assert is_labor_code_question("Heures supplémentaires rémunération")


def test_off_topic_fiscal_chunk():
    text = (
        "article 14 de la circulaire prevoyance sociale revenu forfaitaire "
        "salaire minimum legal activites non agricoles"
    )
    assert is_off_topic_labor_hit(text, overtime=True)


def test_on_topic_labor_chunk():
    text = "Article 201 heures supplementaires majoration de salaire 25 pour cent"
    assert not is_off_topic_labor_hit(text, overtime=True)


def test_labor_hits_not_satisfied_by_fiscal_only():
    hits = [
        {
            "text": "revenu forfaitaire salaire minimum legal article 14 prevoyance sociale",
            "metadata": {"label": "article 14"},
            "score": 0.8,
        }
    ]
    assert not labor_hits_substantively_answer("Heures supplémentaires rémunération", hits)


def test_smig_question_uses_corpus_not_empty_curated():
    q = "quelle est le SMIG salaire minimum Maroc arrêté ?"
    assert is_smig_question(q)
    hits = [
        {
            "text": "revenu forfaitaire fiscal article 14 prevoyance",
            "metadata": {"chunk_id": "SGG_fiscal", "label": "article 14"},
            "score": 0.82,
            "rerank_score": 0.82,
        }
    ]
    out = merge_labor_hits(q, hits, top_k=3)
    assert labor_hits_substantively_answer(q, out)
    assert corpus_covers_question(q, out)


def test_accident_travail_curated_covers():
    q = "Accident de travail obligations employeur"
    assert is_accident_travail_question(q)
    hits = [
        {
            "text": "licence telecom itissalat",
            "metadata": {"chunk_id": "SGG_tel", "label": "licence"},
            "score": 0.75,
        }
    ]
    out = merge_labor_hits(q, hits, top_k=3)
    assert any("accident" in (h.get("text") or "").lower() for h in out)
    assert corpus_covers_question(q, out)


def test_bo_nomenclature_excluded_from_accident_prompt():
    q = "Accident de travail obligations employeur"
    bo = {
        "text": "ART. 11 arrêté 3528 nomenclature des pièces justificatives engagement paiement dépenses Etat",
        "metadata": {
            "chunk_id": "SGG0067_unit_48",
            "doc_id": "SGG0067",
            "label": "ART. 11",
            "source_type": "bulletin_officiel",
        },
        "score": 0.9,
        "rerank_score": 0.9,
    }
    merged = merge_labor_hits(q, [bo], top_k=5)
    filtered = filter_hits_for_labor_prompt(q, merged, top_k=5)
    assert filtered[0]["metadata"]["chunk_id"] == "curated::labor_accident"
    assert not any(
        (h.get("metadata") or {}).get("chunk_id") == "SGG0067_unit_48" for h in filtered
    )
    assert primary_hit_for_answer(q, filtered)["metadata"]["chunk_id"] == "curated::labor_accident"


def test_merge_injects_curated_for_overtime():
    hits = [
        {
            "text": "revenu forfaitaire CNSS",
            "metadata": {"chunk_id": "SGG0043_x", "label": "article 14"},
            "score": 0.9,
            "rerank_score": 0.9,
        }
    ]
    out = merge_labor_hits("Heures supplémentaires rémunération", hits, top_k=3)
    assert any("201" in (h.get("text") or "") for h in out)
    assert any(
        (h.get("metadata") or {}).get("chunk_id", "").startswith("curated::labor")
        for h in out
    )
