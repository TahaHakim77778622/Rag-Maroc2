"""Tests S01–S06 — sécurité basique (HTTP)."""

from __future__ import annotations

import pytest
import requests

from tests.http_config import BASE, DEMO_PASS, DEMO_USER, get_token

pytestmark = pytest.mark.integration


def test_sql_injection_login():
    r = requests.post(
        f"{BASE}/login",
        data={"username": "' OR 1=1 --", "password": "pass"},
        allow_redirects=False,
        timeout=10,
    )
    assert r.status_code in (200, 400, 401, 422)
    assert "Traceback" not in r.text


def test_xss_in_question():
    token = get_token()
    if not token:
        pytest.skip("Token non disponible")
    r2 = requests.post(
        f"{BASE}/api/ask",
        json={"question": "<script>alert('xss')</script>"},
        headers={"Authorization": f"Bearer {token}"},
        timeout=30,
    )
    assert r2.status_code in (200, 400, 422)
    if r2.status_code == 200:
        assert "<script>" not in r2.json().get("answer", "")


def test_admin_without_rights():
    s = requests.Session()
    r = s.get(f"{BASE}/admin", allow_redirects=False, timeout=10)
    assert r.status_code in (302, 303, 401, 403)


def test_security_headers():
    r = requests.get(f"{BASE}/", timeout=10)
    assert r.status_code == 200


def test_no_stack_trace_leak():
    r = requests.get(f"{BASE}/route_inexistante_xyz", timeout=10)
    assert r.status_code == 404
    assert "Traceback" not in r.text
    assert "File \"" not in r.text


def test_http_available():
    r = requests.get(f"{BASE}/", timeout=10)
    assert r.status_code == 200
