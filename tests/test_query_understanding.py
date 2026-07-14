"""Tests compréhension de requête (objectifs, relances, transitions)."""

from __future__ import annotations

from app.query_understanding import (
    analyze_query,
    needs_conversation_context,
    resolve_question,
)


def test_master_to_doctorate_not_master_norms_intent():
    q = "moi je suis en master et je veux faire le doctorat comment faire ?"
    a = analyze_query(q)
    assert a.education_doctorate_intent is True
    assert a.education_master_intent is False
    assert "education_doctorate" in a.topics
    assert any("doctoral" in h or "doctorat" in h for h in a.retrieval_hints)


def test_followup_passage_uses_history():
    hist = [
        {
            "role": "user",
            "content": "je suis en master et je veux faire un doctorat comment faire ?",
        },
        {"role": "assistant", "content": "Les extraits parlent du cycle master..."},
    ]
    q2 = "moi je veux le passage du master vers doctorat"
    assert needs_conversation_context(q2, hist) is True
    resolved = resolve_question(q2, hist)
    assert "doctorat" in resolved.lower() or "master" in resolved.lower()
    a = analyze_query(q2, history=hist)
    assert a.education_doctorate_intent is True
    assert a.education_master_intent is False


def test_cnie_pieces_intent():
    a = analyze_query("Quelles pièces pour une première demande de CNIE ?")
    assert a.cnie_intent is True
    assert "cnie" in a.topics


def test_watiqa_procedure_goal():
    a = analyze_query("Comment commander un acte de naissance sur Watiqa ?")
    assert a.watiqa_intent is True
    assert a.goal == "procedure"


def test_transition_goal_detected():
    a = analyze_query("passage de la licence au master au Maroc")
    assert a.goal == "transition"
