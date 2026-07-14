"""Tests P01–P06 — pages publiques et healthcheck."""

from __future__ import annotations

import pytest
import requests

from tests.http_config import BASE

pytestmark = pytest.mark.integration


def test_home_page():
    r = requests.get(f"{BASE}/", timeout=10)
    assert r.status_code == 200
    assert "RAG" in r.text or "maroc" in r.text.lower()


def test_faq_page():
    r = requests.get(f"{BASE}/faq", timeout=10)
    assert r.status_code == 200


def test_about_page():
    r = requests.get(f"{BASE}/about", timeout=10)
    assert r.status_code in (200, 404)


def test_contact_page():
    r = requests.get(f"{BASE}/contact", timeout=10)
    assert r.status_code in (200, 404)


def test_healthcheck():
    r = requests.get(f"{BASE}/health", timeout=10)
    assert r.status_code == 200
    assert "ok" in r.json().get("status", "").lower()


def test_static_css():
    r = requests.get(f"{BASE}/static/app.css", allow_redirects=True, timeout=10)
    assert r.status_code in (200, 404)
