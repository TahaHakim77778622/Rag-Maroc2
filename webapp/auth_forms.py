"""Validation des identifiants pour les formulaires web (inscription / connexion)."""

from __future__ import annotations

import re

_USERNAME_RE = re.compile(r"^[a-z0-9]([a-z0-9._-]{1,30}[a-z0-9])?$")
_MIN_USER = 3
_MAX_USER = 32
_MIN_PASSWORD = 8
_MAX_PASSWORD = 72  # limite pratique bcrypt


def validate_username(username: str) -> str:
    """
    Retourne le nom d’utilisateur normalisé (minuscules) ou lève ValueError.
    3–32 caractères : lettres minuscules, chiffres, . _ - (pas d’espace).
    """
    u = (username or "").strip().lower()
    if len(u) < _MIN_USER or len(u) > _MAX_USER:
        raise ValueError(
            f"L’identifiant doit comporter entre {_MIN_USER} et {_MAX_USER} caractères."
        )
    if not u.isascii():
        raise ValueError("L’identifiant ne doit contenir que des caractères ASCII (a–z, 0–9, . _ -).")
    if ".." in u or u[0] in "._-" or u[-1] in "._-":
        raise ValueError("L’identifiant ne peut pas commencer/finir par « . » « _ » ou « - », ni contenir « .. ».")
    if not _USERNAME_RE.match(u):
        raise ValueError(
            "Utilisez uniquement des lettres minuscules, chiffres, point, tiret ou souligné (a–z, 0–9, . _ -)."
        )
    return u


def validate_new_password(password: str) -> None:
    if not password:
        raise ValueError("Le mot de passe est obligatoire.")
    if len(password) < _MIN_PASSWORD or len(password) > _MAX_PASSWORD:
        raise ValueError(
            f"Le mot de passe doit comporter entre {_MIN_PASSWORD} et {_MAX_PASSWORD} caractères."
        )
    if password.strip() != password:
        raise ValueError("Le mot de passe ne doit pas commencer ou se terminer par un espace.")


def passwords_match(a: str, b: str) -> bool:
    return a == b
