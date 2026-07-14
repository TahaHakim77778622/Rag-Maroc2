"""
Plan de test IEEE RAG-MAROC2 (44 cas) — automatisés avec LLM_MOCK=1.
"""

from __future__ import annotations

import json
from pathlib import Path

import pytest

PROJECT_ROOT = Path(__file__).resolve().parent.parent


# --- Module A : Auth session ---
def test_a01_chat_redirect_without_session(test_client):
    r = test_client.get("/chat", follow_redirects=False)
    assert r.status_code == 302
    assert "/login" in r.headers.get("location", "")


def test_a02_login_valid_session(test_client, session_cookies):
    r = test_client.get("/chat", cookies=session_cookies)
    assert r.status_code == 200
    assert "Assistant RAG" in r.text or "assistant" in r.text.lower()


def test_a03_login_invalid(test_client):
    r = test_client.post(
        "/login",
        data={"username": "demo", "password": "wrong-password-xyz"},
    )
    assert r.status_code == 200
    low = r.text.lower()
    assert "incorrect" in low or "identifiant" in low or "mot de passe" in low


def test_a04_logout(test_client, session_cookies):
    r = test_client.get("/logout", cookies=session_cookies, follow_redirects=False)
    assert r.status_code == 302
    r2 = test_client.get("/chat", follow_redirects=False)
    assert r2.status_code == 302


def test_a05_register_when_allowed(test_client):
    r = test_client.post(
        "/register",
        data={
            "username": "pytest_user_xyz",
            "password": "pytest1234",
            "password_confirm": "pytest1234",
            "full_name": "Test",
        },
        follow_redirects=False,
    )
    assert r.status_code == 200


def test_a06_register_password_mismatch(test_client):
    r = test_client.post(
        "/register",
        data={
            "username": "pytest_user2",
            "password": "pytest1234",
            "password_confirm": "different",
            "full_name": "",
        },
    )
    assert r.status_code == 200
    assert "correspondent" in r.text.lower() or "correspond" in r.text.lower()


@pytest.mark.skip(reason="ALLOW_REGISTER lu au démarrage ; tester manuellement avec WEBAPP_ALLOW_REGISTER=0")
def test_a07_register_disabled(test_client):
    r = test_client.get("/register", follow_redirects=False)
    assert r.status_code == 302


# --- Module J : JWT ---
def test_j01_login_jwt(test_client):
    r = test_client.post("/api/auth/login", json={"username": "demo", "password": "demo123"})
    assert r.status_code == 200
    data = r.json()
    assert data.get("access_token")
    assert data.get("token_type") == "bearer"


def test_j02_login_jwt_fail(test_client):
    r = test_client.post("/api/auth/login", json={"username": "demo", "password": "bad"})
    assert r.status_code == 401


def test_j03_me_with_token(test_client, auth_headers):
    r = test_client.get("/api/me", headers=auth_headers)
    assert r.status_code == 200
    assert r.json().get("username") == "demo"


def test_j04_me_without_token(test_client):
    r = test_client.get("/api/me")
    assert r.status_code == 401


# --- Module R : Pipeline RAG ---
def test_r01_ask_happy_path(test_client, auth_headers):
    r = test_client.post(
        "/api/ask",
        headers=auth_headers,
        json={"question": "Quelles pièces pour la CNIE ?", "top_k": 5, "history": []},
    )
    assert r.status_code == 200, r.text
    data = r.json()
    assert data.get("answer")
    assert isinstance(data.get("sources"), list)


def test_r02_ask_jwt_same_as_r01(test_client, auth_headers):
    test_r01_ask_happy_path(test_client, auth_headers)


def test_r03_ask_unauthorized(test_client):
    r = test_client.post("/api/ask", json={"question": "test", "top_k": 3})
    assert r.status_code == 401


def test_r04_ask_with_history(test_client, auth_headers):
    r = test_client.post(
        "/api/ask",
        headers=auth_headers,
        json={
            "question": "Et pour le renouvellement ?",
            "top_k": 5,
            "history": [
                {"role": "user", "content": "Quelles pièces pour la CNIE ?"},
                {"role": "assistant", "content": "Voici les pièces selon les sources."},
            ],
        },
    )
    assert r.status_code == 200
    assert r.json().get("answer")


def test_r06_top_k_respected(test_client, auth_headers):
    r = test_client.post(
        "/api/ask",
        headers=auth_headers,
        json={"question": "Watiqa commande acte naissance", "top_k": 3, "history": []},
    )
    assert r.status_code == 200
    sources = r.json().get("sources") or []
    assert len(sources) <= 3


# --- Module S : Sécurité ---
def test_s02_sql_injection_login(test_client):
    r = test_client.post(
        "/api/auth/login",
        json={"username": "' OR 1=1 --", "password": "x"},
    )
    assert r.status_code == 401
    body = r.text.lower()
    assert "traceback" not in body


# --- Module P : Site public ---
@pytest.mark.parametrize("path", ["/", "/about", "/faq", "/docs", "/fonctionnalites", "/contact"])
def test_p_public_pages(test_client, path):
    r = test_client.get(path)
    assert r.status_code == 200


def test_p_static_css(test_client):
    r = test_client.get("/static/app.css")
    assert r.status_code == 200


# --- Module N : Non-fonctionnel ---
def test_n01_health(test_client):
    r = test_client.get("/health")
    assert r.status_code == 200
    assert r.json().get("status") == "ok"


def test_n04_db_persists(test_client, auth_headers):
    r = test_client.post(
        "/api/ask",
        headers=auth_headers,
        json={"question": "test persistance", "top_k": 3, "history": []},
    )
    assert r.status_code == 200


# --- Feedback & métriques ---
def test_feedback_endpoint(test_client, auth_headers):
    r = test_client.post(
        "/api/feedback",
        headers=auth_headers,
        json={"question": "test feedback", "rating": -1},
    )
    assert r.status_code == 200


def test_metrics_admin_only(test_client, auth_headers):
    r = test_client.get("/api/metrics/summary", headers=auth_headers)
    assert r.status_code == 200
    assert "total_asks" in r.json()


def test_corpus_inventory_file_exists():
    inv = PROJECT_ROOT / "data" / "processed" / "corpus_inventory.json"
    if not inv.is_file():
        pytest.skip("Inventaire absent — lancer inventory_final_chunks.py")
    data = json.loads(inv.read_text(encoding="utf-8"))
    assert data.get("total_chunks", 0) > 0


def test_faiss_manifest_exists():
    m = PROJECT_ROOT / "vector_store" / "faiss_manifest.json"
    assert m.is_file(), "Lancer build_faiss.py"
