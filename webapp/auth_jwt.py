"""Jetons JWT pour les clients API (ex. Streamlit) — même SECRET_KEY que les sessions."""

from __future__ import annotations

import time

import jwt

ALGORITHM = "HS256"
# Durée longue pour démo / PFE ; en prod raccourcir + refresh token.
ACCESS_TOKEN_EXPIRE_SECONDS = 7 * 24 * 3600


def create_access_token(user_id: int, username: str, secret: str) -> str:
    now = int(time.time())
    payload = {
        "sub": str(user_id),
        "username": username,
        "iat": now,
        "exp": now + ACCESS_TOKEN_EXPIRE_SECONDS,
    }
    return jwt.encode(payload, secret, algorithm=ALGORITHM)


def decode_access_token(token: str, secret: str) -> dict:
    return jwt.decode(token, secret, algorithms=[ALGORITHM])
