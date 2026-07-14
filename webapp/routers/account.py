"""
Routes pages compte utilisateur (profil, mot de passe, paramètres).
"""

from __future__ import annotations

from pathlib import Path
from urllib.parse import quote_plus

from fastapi import APIRouter, Form, Request
from fastapi.responses import HTMLResponse, RedirectResponse
from jinja2 import Environment, FileSystemLoader, select_autoescape

from webapp import auth_forms
from webapp import db as db_mod
from webapp.settings import ALLOW_REGISTER, SHOW_DEMO_LOGIN_HINT

router = APIRouter(prefix="/account", tags=["account"])

_WEBAPP_DIR = Path(__file__).resolve().parent.parent
_TEMPLATES_DIR = _WEBAPP_DIR / "templates"


def _current_user(request: Request) -> dict | None:
    uid = request.session.get("uid")
    if not uid:
        return None
    return db_mod.get_user_by_id(int(uid))


def _render(name: str, request: Request, **ctx) -> HTMLResponse:
    ctx.setdefault("current_user", _current_user(request))
    ctx.setdefault("allow_register", ALLOW_REGISTER)
    ctx.setdefault("db_label", db_mod.database_label())
    ctx.setdefault("show_demo_login", SHOW_DEMO_LOGIN_HINT)
    env = Environment(
        loader=FileSystemLoader(str(_TEMPLATES_DIR)),
        autoescape=select_autoescape(["html", "xml"]),
    )
    tpl = env.get_template(name)
    return HTMLResponse(tpl.render(request=request, **ctx))


def _user_initials(user: dict) -> str:
    display = (user.get("full_name") or user.get("username") or "").strip()
    parts = display.split()
    if len(parts) >= 2:
        return (parts[0][:1] + parts[-1][:1]).upper()
    return display[:2].upper() if display else "??"


def _require_login(request: Request) -> dict | RedirectResponse:
    user = _current_user(request)
    if not user:
        return RedirectResponse("/login", status_code=302)
    return user


@router.get("/profile", response_class=HTMLResponse)
def account_profile_page(request: Request):
    user = _require_login(request)
    if isinstance(user, RedirectResponse):
        return user
    return _render(
        "account_profile.html",
        request,
        user=user,
        initials=_user_initials(user),
        success=request.query_params.get("success") == "1",
        error=request.query_params.get("error"),
    )


@router.post("/profile", response_class=HTMLResponse)
def account_profile_submit(
    request: Request,
    full_name: str = Form(""),
):
    user = _require_login(request)
    if isinstance(user, RedirectResponse):
        return user
    name = (full_name or "").strip() or None
    if name and len(name) > 120:
        return RedirectResponse(
            "/account/profile?error=" + quote_plus("Le nom affiché est trop long (120 caractères max)."),
            status_code=302,
        )
    db_mod.update_user_full_name(int(user["id"]), name)
    return RedirectResponse("/account/profile?success=1", status_code=302)


@router.get("/password", response_class=HTMLResponse)
def account_password_page(request: Request):
    user = _require_login(request)
    if isinstance(user, RedirectResponse):
        return user
    return _render(
        "account_password.html",
        request,
        user=user,
        success=request.query_params.get("success") == "1",
        error=request.query_params.get("error"),
    )


@router.post("/password", response_class=HTMLResponse)
def account_password_submit(
    request: Request,
    current_password: str = Form(...),
    new_password: str = Form(...),
    new_password_confirm: str = Form(""),
):
    user = _require_login(request)
    if isinstance(user, RedirectResponse):
        return user

    row = db_mod.get_user_by_id_with_hash(int(user["id"]))
    if not row or not db_mod.verify_password(current_password, row["password_hash"]):
        return RedirectResponse(
            "/account/password?error=" + quote_plus("Mot de passe actuel incorrect."),
            status_code=302,
        )

    try:
        auth_forms.validate_new_password(new_password)
    except ValueError as e:
        return RedirectResponse(
            "/account/password?error=" + quote_plus(str(e)),
            status_code=302,
        )

    if not auth_forms.passwords_match(new_password, new_password_confirm):
        return RedirectResponse(
            "/account/password?error=" + quote_plus("Les mots de passe ne correspondent pas."),
            status_code=302,
        )

    db_mod.update_user_password(int(user["id"]), new_password)
    return RedirectResponse("/account/password?success=1", status_code=302)


@router.get("/settings", response_class=HTMLResponse)
def account_settings_page(request: Request):
    user = _require_login(request)
    if isinstance(user, RedirectResponse):
        return user
    return _render("account_settings.html", request, user=user)
