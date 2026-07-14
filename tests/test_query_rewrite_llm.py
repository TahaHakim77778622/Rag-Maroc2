"""Tests hybride 80 % corpus / 20 % LLM (seuils, sans appel API)."""

from __future__ import annotations

import os

from app.query_rewrite_llm import ambiguity_score, should_use_llm_query_rewrite
from app.query_understanding import analyze_query


def test_clear_cnie_stays_corpus_only(monkeypatch):
    monkeypatch.setenv("QUERY_REWRITE_LLM", "1")
    monkeypatch.delenv("LLM_MOCK", raising=False)
    q = "Quelles pièces pour une première demande de CNIE ?"
    a = analyze_query(q)
    assert ambiguity_score(a, q) < 0.52
    assert should_use_llm_query_rewrite(a, q) is False


def test_master_doctorat_followup_triggers_llm_path(monkeypatch):
    monkeypatch.setenv("QUERY_REWRITE_LLM", "1")
    monkeypatch.delenv("LLM_MOCK", raising=False)
    hist = [
        {"role": "user", "content": "je suis en master et je veux le doctorat"},
        {"role": "assistant", "content": "réponse"},
    ]
    q = "moi je veux le passage du master vers doctorat"
    a = analyze_query(q, history=hist)
    assert ambiguity_score(a, q) >= 0.52
    assert should_use_llm_query_rewrite(a, q) is True


def test_llm_disabled_when_mock(monkeypatch):
    monkeypatch.setenv("QUERY_REWRITE_LLM", "1")
    monkeypatch.setenv("LLM_MOCK", "1")
    q = "moi je veux le passage master doctorat"
    a = analyze_query(q)
    assert should_use_llm_query_rewrite(a, q) is False


def test_llm_off_by_default(monkeypatch):
    monkeypatch.delenv("QUERY_REWRITE_LLM", raising=False)
    q = "moi je veux le passage master doctorat"
    a = analyze_query(q)
    assert should_use_llm_query_rewrite(a, q) is False
