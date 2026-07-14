"""
Authentification pour les routes API : cookie de session (UI Jinja) OU Bearer JWT (Streamlit / clients).
"""

from __future__ import annotations

from typing import Annotated

from fastapi import Depends, HTTPException, Request
from fastapi.security import HTTPAuthorizationCredentials, HTTPBearer

from webapp import db as db_mod
from webapp.auth_jwt import decode_access_token
from webapp.settings import SECRET_KEY

_bearer = HTTPBearer(auto_error=False)


def get_current_user_for_api(
    request: Request,
    creds: Annotated[HTTPAuthorizationCredentials | None, Depends(_bearer)],
) -> dict:
    """Utilisateur connecté : session navigateur ou en-tête Authorization: Bearer <jwt>."""
    uid = request.session.get("uid")
    if uid is not None:
        user = db_mod.get_user_by_id(int(uid))
        if user:
            return user

    if creds is not None and creds.scheme.lower() == "bearer":
        try:
            payload = decode_access_token(creds.credentials, SECRET_KEY)
            user = db_mod.get_user_by_id(int(payload["sub"]))
            if user:
                return user
        except Exception:
            pass

    raise HTTPException(status_code=401, detail="Non authentifié")
