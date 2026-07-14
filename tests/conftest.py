"""Fixtures communes — LLM_MOCK, SQLite de test, client FastAPI."""

from __future__ import annotations

import os
import tempfile
from pathlib import Path

import pytest
from fastapi.testclient import TestClient

PROJECT_ROOT = Path(__file__).resolve().parent.parent


def pytest_configure(config):
    config.addinivalue_line("markers", "integration: tests HTTP (serveur sur 127.0.0.1:8000)")
    config.addinivalue_line("markers", "slow: tests lents (LLM réel)")


def pytest_runtest_setup(item):
    if item.get_closest_marker("integration"):
        from tests.http_config import BASE, server_available

        if not server_available():
            pytest.skip(f"Serveur indisponible sur {BASE} — lancez ./scripts/run_webapp.sh")


@pytest.fixture(scope="session")
def test_client() -> TestClient:
    os.environ["WEBAPP_SECRET_KEY"] = "pytest-secret-key-fixed"
    os.environ["LLM_MOCK"] = "1"
    os.environ["WEBAPP_ALLOW_REGISTER"] = "1"
    os.environ["WEBAPP_SHOW_DEMO_LOGIN"] = "0"
    os.environ.pop("WEBAPP_DATABASE_URL", None)
    os.environ.pop("DATABASE_URL", None)
    os.environ.setdefault("TOKENIZERS_PARALLELISM", "false")

    import webapp.db as db_mod  # noqa: PLC0415

    tmp_db = Path(tempfile.mkdtemp(prefix="rag_maroc2_test_")) / "users.db"
    db_mod.DB_PATH = tmp_db
    db_mod.USE_POSTGRES = False
    db_mod.DATABASE_URL = ""

    from webapp.main import app  # noqa: PLC0415

    db_mod.init_db()
    with TestClient(app) as client:
        yield client


@pytest.fixture
def auth_headers(test_client: TestClient) -> dict[str, str]:
    r = test_client.post(
        "/api/auth/login",
        json={"username": "demo", "password": "demo123"},
    )
    assert r.status_code == 200, r.text
    token = r.json()["access_token"]
    return {"Authorization": f"Bearer {token}"}


@pytest.fixture
def session_cookies(test_client: TestClient) -> dict[str, str]:
    r = test_client.post(
        "/login",
        data={"username": "demo", "password": "demo123"},
        follow_redirects=False,
    )
    assert r.status_code in (302, 303), r.text
    return dict(r.cookies)
