"""Tests W01–W04 — fallback web / champs answer_source."""

from __future__ import annotations

import pytest

from tests.http_config import get_token, post_ask

pytestmark = [pytest.mark.integration, pytest.mark.slow]


def test_no_web_fallback_for_covered_question():
    token = get_token()
    if not token:
        pytest.skip("Token non disponible")
    r = post_ask(token, "comment obtenir la CNIE ?")
    data = r.json()
    assert r.status_code == 200
    corpus_sufficient = data.get("corpus_sufficient", True)
    assert corpus_sufficient is not False or data.get("answer_source") == "corpus"


def test_answer_always_provided():
    token = get_token()
    if not token:
        pytest.skip("Token non disponible")
    r = post_ask(token, "quel est le montant exact SMIG 2025")
    assert r.status_code == 200
    assert len(r.json().get("answer", "")) > 0


def test_answer_source_field():
    token = get_token()
    if not token:
        pytest.skip("Token non disponible")
    r = post_ask(token, "CNIE renouvellement")
    assert r.status_code == 200
    data = r.json()
    assert "answer" in data
    assert "answer_source" in data or "corpus_sufficient" in data


def test_no_500_on_fallback_failure():
    token = get_token()
    if not token:
        pytest.skip("Token non disponible")
    r = post_ask(token, "xyzabcdef123 question inexistante")
    assert r.status_code != 500
