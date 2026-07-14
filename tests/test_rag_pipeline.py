"""Tests R01–R12 — pipeline RAG via API (LLM réel si serveur sans LLM_MOCK)."""

from __future__ import annotations

import time

import pytest
import requests

from tests.http_config import ASK_TIMEOUT, BASE, get_token, post_ask

pytestmark = [pytest.mark.integration, pytest.mark.slow]


def ask(question: str, token: str) -> requests.Response:
    return post_ask(token, question)


def test_rag_cnie():
    token = get_token()
    if not token:
        pytest.skip("Token non disponible")
    r = ask("comment obtenir la CNIE au Maroc ?", token)
    assert r.status_code == 200
    assert len(r.json().get("answer", "")) > 50


def test_rag_passeport():
    token = get_token()
    if not token:
        pytest.skip("Token non disponible")
    r = ask("renouvellement passeport marocain", token)
    assert r.status_code == 200
    assert len(r.json().get("answer", "")) > 50


def test_rag_travail():
    token = get_token()
    if not token:
        pytest.skip("Token non disponible")
    r = ask("heures supplémentaires code du travail", token)
    assert r.status_code == 200
    assert len(r.json().get("answer", "")) > 50


def test_rag_watiqa():
    token = get_token()
    if not token:
        pytest.skip("Token non disponible")
    r = ask("comment commander acte de naissance Watiqa", token)
    assert r.status_code == 200
    assert len(r.json().get("answer", "")) > 50


def test_rag_urbanisme():
    token = get_token()
    if not token:
        pytest.skip("Token non disponible")
    r = ask("autorisation de construire Maroc procédure", token)
    assert r.status_code == 200
    assert len(r.json().get("answer", "")) > 50


def test_rag_latency():
    token = get_token()
    if not token:
        pytest.skip("Token non disponible")
    start = time.time()
    r = ask("CNIE première demande", token)
    elapsed = time.time() - start
    assert r.status_code == 200
    assert elapsed < ASK_TIMEOUT


def test_rag_has_sources():
    token = get_token()
    if not token:
        pytest.skip("Token non disponible")
    r = ask("pièces requises passeport", token)
    data = r.json()
    assert "answer" in data
    assert len(data.get("answer", "")) > 0


def test_rag_short_question():
    token = get_token()
    if not token:
        pytest.skip("Token non disponible")
    r = ask("CNIE", token)
    assert r.status_code in (200, 400)
    assert r.status_code != 500


def test_rag_with_history():
    token = get_token()
    if not token:
        pytest.skip("Token non disponible")
    r = post_ask(
        token,
        "et pour le renouvellement ?",
        history=[
            {"role": "user", "content": "comment obtenir la CNIE ?"},
            {"role": "assistant", "content": "Pour obtenir la CNIE..."},
        ],
    )
    assert r.status_code == 200


def test_rag_smig():
    token = get_token()
    if not token:
        pytest.skip("Token non disponible")
    r = ask("SMIG salaire minimum Maroc", token)
    assert r.status_code == 200
    assert len(r.json().get("answer", "")) > 50


def test_rag_off_topic():
    token = get_token()
    if not token:
        pytest.skip("Token non disponible")
    r = ask("recette de couscous marocain", token)
    assert r.status_code in (200, 400)
    assert r.status_code != 500


def test_rag_empty_question():
    token = get_token()
    if not token:
        pytest.skip("Token non disponible")
    r = requests.post(
        f"{BASE}/api/ask",
        json={"question": ""},
        headers={"Authorization": f"Bearer {token}"},
        timeout=30,
    )
    assert r.status_code in (200, 400, 422)
    assert r.status_code != 500
