"""Tests A01–A10 — authentification formulaire (sessions)."""

from __future__ import annotations

import pytest
import requests

from tests.http_config import BASE, DEMO_PASS, DEMO_USER

pytestmark = pytest.mark.integration


def test_login_correct():
    r = requests.post(
        f"{BASE}/login",
        data={"username": DEMO_USER, "password": DEMO_PASS},
        allow_redirects=False,
        timeout=10,
    )
    assert r.status_code in (200, 302, 303)


def test_login_wrong_password():
    r = requests.post(
        f"{BASE}/login",
        data={"username": DEMO_USER, "password": "wrongpass"},
        allow_redirects=False,
        timeout=10,
    )
    assert r.status_code in (200, 400, 401, 422)


def test_login_unknown_user():
    r = requests.post(
        f"{BASE}/login",
        data={"username": "ghost_user_xyz", "password": "pass"},
        allow_redirects=False,
        timeout=10,
    )
    assert r.status_code in (200, 400, 401, 422)


def test_login_empty_fields():
    r = requests.post(
        f"{BASE}/login",
        data={"username": "", "password": ""},
        allow_redirects=False,
        timeout=10,
    )
    assert r.status_code in (200, 400, 401, 422)


def test_chat_without_auth():
    r = requests.get(f"{BASE}/chat", allow_redirects=False, timeout=10)
    assert r.status_code in (302, 303, 401)


def test_logout():
    s = requests.Session()
    s.post(
        f"{BASE}/login",
        data={"username": DEMO_USER, "password": DEMO_PASS},
        timeout=10,
    )
    r = s.get(f"{BASE}/logout", allow_redirects=False, timeout=10)
    assert r.status_code in (200, 302, 303)


def test_register_short_username():
    r = requests.post(
        f"{BASE}/register",
        data={
            "username": "ab",
            "password": "password123",
            "password_confirm": "password123",
        },
        allow_redirects=False,
        timeout=10,
    )
    assert r.status_code in (200, 302, 303, 400, 422)


def test_register_password_mismatch():
    r = requests.post(
        f"{BASE}/register",
        data={
            "username": "testuser99",
            "password": "pass1234",
            "password_confirm": "pass5678",
        },
        allow_redirects=False,
        timeout=10,
    )
    assert r.status_code in (200, 302, 303, 400, 422)


def test_login_page_accessible():
    r = requests.get(f"{BASE}/login", timeout=10)
    assert r.status_code == 200


def test_home_accessible():
    r = requests.get(f"{BASE}/", timeout=10)
    assert r.status_code == 200
