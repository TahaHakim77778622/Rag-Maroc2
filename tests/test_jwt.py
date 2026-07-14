"""Tests J01–J06 — API JWT /api/auth/login et /api/ask."""

from __future__ import annotations

import pytest
import requests

from tests.http_config import BASE, get_token, post_ask

pytestmark = pytest.mark.integration


def test_get_jwt_token():
    token = get_token()
    assert token != ""


def test_ask_without_token():
    r = requests.post(f"{BASE}/api/ask", json={"question": "test"}, timeout=10)
    assert r.status_code == 401


def test_ask_with_valid_token():
    token = get_token()
    if not token:
        pytest.skip("Token non disponible")
    r = post_ask(token, "comment obtenir la CNIE ?")
    assert r.status_code == 200


def test_ask_with_invalid_token():
    r = requests.post(
        f"{BASE}/api/ask",
        json={"question": "test"},
        headers={"Authorization": "Bearer fake_token_xyz"},
        timeout=10,
    )
    assert r.status_code == 401


def test_ask_response_structure():
    token = get_token()
    if not token:
        pytest.skip("Token non disponible")
    r = post_ask(token, "renouvellement passeport")
    assert r.status_code == 200
    data = r.json()
    assert "answer" in data
    assert data["answer"] != ""


def test_ask_get_method_not_allowed():
    r = requests.get(f"{BASE}/api/ask", timeout=10)
    assert r.status_code in (401, 405)
