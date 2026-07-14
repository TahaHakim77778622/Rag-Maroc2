"""Configuration partagée pour les tests HTTP d'intégration (serveur live).

Pour des tests rapides et stables, lancez le serveur avec mock LLM :
  LLM_MOCK=1 ./scripts/run_webapp.sh

Délai /api/ask (LLM réel souvent > 60 s) : variable RAG_TEST_ASK_TIMEOUT (défaut 120).
"""

from __future__ import annotations

import os
from typing import Any

import pytest
import requests

BASE = os.environ.get("RAG_TEST_BASE", "http://127.0.0.1:8000").rstrip("/")
DEMO_USER = os.environ.get("WEBAPP_DEMO_USER", "demo")
DEMO_PASS = os.environ.get("WEBAPP_DEMO_PASSWORD", "demo123")
ASK_TIMEOUT = int(os.environ.get("RAG_TEST_ASK_TIMEOUT", "120"))


def server_available() -> bool:
    try:
        r = requests.get(f"{BASE}/health", timeout=3)
        return r.status_code == 200
    except (requests.RequestException, OSError):
        return False


def get_token() -> str:
    r = requests.post(
        f"{BASE}/api/auth/login",
        json={"username": DEMO_USER, "password": DEMO_PASS},
        timeout=10,
    )
    if r.status_code == 200:
        return r.json().get("access_token", "")
    return ""


def post_ask(
    token: str,
    question: str,
    *,
    history: list[dict[str, str]] | None = None,
    timeout: int | None = None,
) -> requests.Response:
    """POST /api/ask ; skip si timeout (pipeline RAG trop lent)."""
    body: dict[str, Any] = {"question": question}
    if history is not None:
        body["history"] = history
    try:
        return requests.post(
            f"{BASE}/api/ask",
            json=body,
            headers={"Authorization": f"Bearer {token}"},
            timeout=timeout if timeout is not None else ASK_TIMEOUT,
        )
    except requests.ReadTimeout:
        pytest.skip(
            f"/api/ask a dépassé {timeout or ASK_TIMEOUT}s — "
            "relancez avec LLM_MOCK=1 ./scripts/run_webapp.sh "
            "ou augmentez RAG_TEST_ASK_TIMEOUT."
        )
