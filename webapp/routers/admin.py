"""
Panneau d'administration RAG-MAROC2.
"""

from __future__ import annotations

import json
import subprocess
import sys
from pathlib import Path
from urllib.parse import quote_plus

from fastapi import APIRouter, Form, Request
from fastapi.responses import HTMLResponse, PlainTextResponse, RedirectResponse
from jinja2 import Environment, FileSystemLoader, select_autoescape

from webapp import auth_forms
from webapp import db as db_mod
from webapp import admin_db
from webapp.admin_format import fmt_date, fmt_datetime, fmt_number, fmt_relative
from webapp.settings import ALLOW_REGISTER, SHOW_DEMO_LOGIN_HINT

router = APIRouter(prefix="/admin", tags=["admin"])

_jinja_env: Environment | None = None

_WEBAPP_DIR = Path(__file__).resolve().parent.parent
_PROJECT_ROOT = _WEBAPP_DIR.parent
_TEMPLATES_DIR = _WEBAPP_DIR / "templates"
_MANIFEST_PATH = _PROJECT_ROOT / "vector_store" / "faiss_manifest.json"
_INVENTORY_PATH = _PROJECT_ROOT / "data" / "processed" / "corpus_inventory.json"
_REBUILD_LOG = _PROJECT_ROOT / "data" / "admin" / "rebuild.log"
_rebuild_proc: subprocess.Popen | None = None


def _jinja() -> Environment:
    global _jinja_env
    if _jinja_env is None:
        _jinja_env = Environment(
            loader=FileSystemLoader(str(_TEMPLATES_DIR)),
            autoescape=select_autoescape(["html", "xml"]),
        )
        _jinja_env.filters["fmt_date"] = fmt_date
        _jinja_env.filters["fmt_datetime"] = fmt_datetime
        _jinja_env.filters["fmt_relative"] = fmt_relative
        _jinja_env.filters["fmt_number"] = fmt_number
    return _jinja_env


def _render(name: str, request: Request, **ctx) -> HTMLResponse:
    ctx.setdefault("allow_register", ALLOW_REGISTER)
    ctx.setdefault("db_label", db_mod.database_label())
    ctx.setdefault("show_demo_login", SHOW_DEMO_LOGIN_HINT)
    return HTMLResponse(_jinja().get_template(name).render(request=request, **ctx))


def _current_user(request: Request) -> dict | None:
    uid = request.session.get("uid")
    if not uid:
        return None
    return db_mod.get_user_by_id(int(uid))


def _require_admin(request: Request):
    user = _current_user(request)
    if not user:
        return RedirectResponse("/login", status_code=302)
    if not admin_db.user_is_admin(user):
        return RedirectResponse("/chat?admin_denied=1", status_code=302)
    return user


def _corpus_stats() -> dict | None:
    try:
        if _MANIFEST_PATH.is_file():
            data = json.loads(_MANIFEST_PATH.read_text(encoding="utf-8"))
            ntotal = int(data.get("ntotal", 0))
            return {
                "ntotal": ntotal,
                "ntotal_display": f"{ntotal:,}".replace(",", "\u202f"),
                "embedding_model": str(data.get("embedding_model") or ""),
                "updated": _MANIFEST_PATH.stat().st_mtime,
            }
    except (OSError, ValueError, TypeError):
        pass
    return None


def _corpus_domains() -> list[dict]:
    inv = None
    try:
        if _INVENTORY_PATH.is_file():
            inv = json.loads(_INVENTORY_PATH.read_text(encoding="utf-8"))
    except (OSError, ValueError):
        inv = None
    by_cat = (inv or {}).get("by_category") or {}
    by_org = (inv or {}).get("by_source_org") or {}
    mapping = [
        ("juridique", "Juridique", int(by_cat.get("juridique", 0))),
        ("labor", "Droit du travail", int(by_cat.get("code_travail", 0))),
        ("cnie", "CNIE", int(by_cat.get("cnie", 0)) + int(by_org.get("CNIE Maroc", 0))),
        ("passeport", "Passeport", int(by_org.get("Maroc.ma", 0))),
        ("urbanisme", "Urbanisme", 0),
        ("education", "Éducation", 0),
        ("watiqa", "Watiqa", int(by_org.get("Watiqa", 0)) + int(by_cat.get("etat_civil", 0))),
    ]
    total = sum(m[2] for m in mapping) or 1
    out = []
    for key, label, count in mapping:
        out.append(
            {
                "key": key,
                "label": label,
                "count": count,
                "pct": round(100.0 * count / total, 1),
            }
        )
    return out


@router.get("", response_class=HTMLResponse)
@router.get("/", response_class=HTMLResponse)
def admin_dashboard(request: Request):
    user = _require_admin(request)
    if isinstance(user, RedirectResponse):
        return user
    from webapp.metrics_store import avg_latency_sec

    stats = {
        "total_users": admin_db.count_users(),
        "conversations_today": admin_db.count_user_messages_today(),
        "total_questions": admin_db.count_total_user_questions(),
        "avg_latency": avg_latency_sec(),
    }
    activity = admin_db.activity_last_n_days(7)
    recent = admin_db.list_recent_questions(10)
    return _render(
        "admin_dashboard.html",
        request,
        admin_user=user,
        active_page="dashboard",
        stats=stats,
        activity=activity,
        recent=recent,
    )


@router.get("/users", response_class=HTMLResponse)
def admin_users_list(request: Request):
    user = _require_admin(request)
    if isinstance(user, RedirectResponse):
        return user
    q = request.query_params.get("q", "")
    page = max(1, int(request.query_params.get("page", 1) or 1))
    rows, total = admin_db.list_users_admin(q=q, page=page, page_size=20)
    total_pages = max(1, (total + 19) // 20)
    for r in rows:
        r["initials"] = admin_db.initials_for_user(r)
        r["is_admin_flag"] = admin_db.user_is_admin(r)
    return _render(
        "admin_users.html",
        request,
        admin_user=user,
        active_page="users",
        users=rows,
        search_q=q,
        page=page,
        total_pages=total_pages,
        total_users=total,
        created_msg=request.query_params.get("created"),
        reset_pw=request.query_params.get("reset_pw"),
    )


@router.post("/users/create", response_class=HTMLResponse)
def admin_users_create(
    request: Request,
    username: str = Form(...),
    password: str = Form(...),
    full_name: str = Form(""),
):
    user = _require_admin(request)
    if isinstance(user, RedirectResponse):
        return user
    try:
        u_norm = auth_forms.validate_username(username)
        auth_forms.validate_new_password(password)
        db_mod.create_user(u_norm, password, full_name=(full_name or "").strip() or None)
    except (ValueError, db_mod.DuplicateUsernameError) as e:
        return RedirectResponse(
            "/admin/users?error=" + quote_plus(str(e)),
            status_code=302,
        )
    return RedirectResponse("/admin/users?created=1", status_code=302)


@router.post("/users/{user_id}/disable", response_class=HTMLResponse)
def admin_users_disable(user_id: int, request: Request):
    user = _require_admin(request)
    if isinstance(user, RedirectResponse):
        return user
    target = admin_db.get_user_admin_detail(user_id)
    if target and not admin_db.user_is_admin(target):
        admin_db.set_user_active(user_id, False)
    ref = request.headers.get("referer") or f"/admin/users/{user_id}"
    return RedirectResponse(ref, status_code=302)


@router.post("/users/{user_id}/enable", response_class=HTMLResponse)
def admin_users_enable(user_id: int, request: Request):
    user = _require_admin(request)
    if isinstance(user, RedirectResponse):
        return user
    admin_db.set_user_active(user_id, True)
    return RedirectResponse(f"/admin/users/{user_id}", status_code=302)


@router.post("/users/{user_id}/reset-password", response_class=HTMLResponse)
def admin_users_reset_password(user_id: int, request: Request):
    user = _require_admin(request)
    if isinstance(user, RedirectResponse):
        return user
    new_pw = admin_db.reset_user_password(user_id)
    return RedirectResponse(
        f"/admin/users/{user_id}?reset_pw={quote_plus(new_pw)}",
        status_code=302,
    )


@router.post("/users/{user_id}/delete", response_class=HTMLResponse)
def admin_users_delete(user_id: int, request: Request):
    user = _require_admin(request)
    if isinstance(user, RedirectResponse):
        return user
    if int(user_id) == int(user["id"]):
        return RedirectResponse("/admin/users?error=cannot_delete_self", status_code=302)
    target = admin_db.get_user_admin_detail(user_id)
    if target and not admin_db.user_is_admin(target):
        admin_db.delete_user(user_id)
    return RedirectResponse("/admin/users", status_code=302)


@router.get("/users/{user_id}", response_class=HTMLResponse)
def admin_user_detail(user_id: int, request: Request):
    user = _require_admin(request)
    if isinstance(user, RedirectResponse):
        return user
    target = admin_db.get_user_admin_detail(user_id)
    if not target:
        return RedirectResponse("/admin/users", status_code=302)
    target["initials"] = admin_db.initials_for_user(target)
    if admin_db.user_is_admin(target):
        target["is_admin"] = True
    convos = admin_db.list_user_conversations(user_id, 20)
    domains = admin_db.user_top_domains(user_id)
    return _render(
        "admin_user_detail.html",
        request,
        admin_user=user,
        active_page="users",
        target=target,
        conversations=convos,
        domains=domains,
        reset_pw=request.query_params.get("reset_pw"),
    )


@router.get("/corpus", response_class=HTMLResponse)
def admin_corpus(request: Request):
    user = _require_admin(request)
    if isinstance(user, RedirectResponse):
        return user
    import datetime

    cs = _corpus_stats()
    domains = _corpus_domains()
    faiss_updated = "—"
    if cs and cs.get("updated"):
        faiss_updated = datetime.datetime.fromtimestamp(cs["updated"]).strftime(
            "%Y-%m-%d %H:%M"
        )
    log_tail = ""
    if _REBUILD_LOG.is_file():
        try:
            log_tail = _REBUILD_LOG.read_text(encoding="utf-8", errors="replace")[-8000:]
        except OSError:
            pass
    return _render(
        "admin_corpus.html",
        request,
        admin_user=user,
        active_page="corpus",
        corpus_stats=cs,
        domains=domains,
        faiss_updated=faiss_updated,
        rebuild_log=log_tail,
    )


@router.post("/corpus/rebuild")
def admin_corpus_rebuild_start(request: Request):
    user = _require_admin(request)
    if isinstance(user, RedirectResponse):
        return user
    global _rebuild_proc
    if _rebuild_proc and _rebuild_proc.poll() is None:
        return RedirectResponse("/admin/corpus?rebuild=running", status_code=302)
    _REBUILD_LOG.parent.mkdir(parents=True, exist_ok=True)
    _REBUILD_LOG.write_text("Démarrage du rebuild…\n", encoding="utf-8")
    script = _PROJECT_ROOT / "scripts" / "rebuild_corpus_and_index.py"
    with _REBUILD_LOG.open("a", encoding="utf-8") as logf:
        _rebuild_proc = subprocess.Popen(
            [sys.executable, str(script)],
            cwd=str(_PROJECT_ROOT),
            stdout=logf,
            stderr=subprocess.STDOUT,
        )
    return RedirectResponse("/admin/corpus?rebuild=started", status_code=302)


@router.get("/corpus/rebuild/status")
def admin_corpus_rebuild_status(request: Request):
    user = _require_admin(request)
    if isinstance(user, RedirectResponse):
        return user
    text = ""
    if _REBUILD_LOG.is_file():
        try:
            text = _REBUILD_LOG.read_text(encoding="utf-8", errors="replace")[-12000:]
        except OSError:
            text = ""
    global _rebuild_proc
    running = _rebuild_proc is not None and _rebuild_proc.poll() is None
    if running:
        text += "\n[En cours…]"
    elif _rebuild_proc is not None:
        code = _rebuild_proc.returncode
        text += f"\n[Terminé — code {code}]"
    return PlainTextResponse(text or "(aucun log)")


@router.get("/logs", response_class=HTMLResponse)
def admin_logs(request: Request):
    user = _require_admin(request)
    if isinstance(user, RedirectResponse):
        return user
    source = request.query_params.get("source", "all")
    from webapp.metrics_store import list_ask_events_filtered

    events = list_ask_events_filtered(limit=100, source=source)
    return _render(
        "admin_logs.html",
        request,
        admin_user=user,
        active_page="logs",
        events=events,
        source_filter=source,
    )


@router.get("/logs/export.csv")
def admin_logs_export(request: Request):
    user = _require_admin(request)
    if isinstance(user, RedirectResponse):
        return user
    from webapp.metrics_store import export_events_csv

    csv_data = export_events_csv()
    return PlainTextResponse(
        csv_data,
        media_type="text/csv",
        headers={"Content-Disposition": "attachment; filename=rag_ask_logs.csv"},
    )


@router.get("/settings", response_class=HTMLResponse)
def admin_settings(request: Request):
    user = _require_admin(request)
    if isinstance(user, RedirectResponse):
        return user
    return _render(
        "admin_settings.html",
        request,
        admin_user=user,
        active_page="settings",
    )
