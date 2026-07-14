"""Paramètres partagés (clé JWT / sessions, flags)."""

from __future__ import annotations

import os
import secrets
from pathlib import Path

try:
    from dotenv import load_dotenv  # noqa: PLC0415

    load_dotenv(Path(__file__).resolve().parent.parent / ".env")
except ImportError:
    pass

SECRET_KEY = os.environ.get("WEBAPP_SECRET_KEY") or secrets.token_hex(32)

ALLOW_REGISTER = os.environ.get("WEBAPP_ALLOW_REGISTER", "").strip().lower() in (
    "1",
    "true",
    "yes",
)

# Affiche l’indice compte démo sur /login (désactivé par défaut, plus pro en prod).
SHOW_DEMO_LOGIN_HINT = os.environ.get("WEBAPP_SHOW_DEMO_LOGIN", "").strip().lower() in (
    "1",
    "true",
    "yes",
)

# Utilisateurs autorisés sur /admin et /api/metrics/summary (séparés par des virgules).
_ADMIN_RAW = os.environ.get("WEBAPP_ADMIN_USERS", "demo").strip()
ADMIN_USERNAMES = {u.strip().lower() for u in _ADMIN_RAW.split(",") if u.strip()}


def is_admin_user(username: str | None, user: dict | None = None) -> bool:
    if user and user.get("is_admin"):
        return True
    if user and user.get("username"):
        username = user.get("username")
    if username:
        try:
            from webapp import db as db_mod

            row = db_mod.get_user_by_username(username.strip().lower())
            if row and row.get("is_admin"):
                return True
        except Exception:
            pass
    return bool(username) and username.strip().lower() in ADMIN_USERNAMES
