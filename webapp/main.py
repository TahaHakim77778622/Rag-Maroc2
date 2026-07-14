"""
Plateforme web RAG-MAROC2 — FastAPI (API + UI Jinja) + JWT pour Streamlit.

API backend (à lancer en premier pour Streamlit) :
    ./scripts/run_webapp.sh
    # ou : uvicorn webapp.main:app --host 127.0.0.1 --port 8000 --http h11 --loop asyncio
    # Sur Mac : éviter --reload si segfault ; ne pas mélanger conda + pip sur torch/faiss si possible.

Streamlit (client de l’API) :
    export RAG_API_BASE="http://127.0.0.1:8000"
    streamlit run app/streamlit_app.py
"""

from __future__ import annotations

import json
import os
import secrets
from pathlib import Path

# Avant tout import « lourd » : limite les crashs segfault sur macOS (OpenMP / MKL / torch / FAISS).
for _k, _v in (
    ("TOKENIZERS_PARALLELISM", "false"),
    ("OMP_NUM_THREADS", "1"),
    ("MKL_NUM_THREADS", "1"),
    ("VECLIB_MAXIMUM_THREADS", "1"),
    ("NUMEXPR_NUM_THREADS", "1"),
    ("KMP_DUPLICATE_LIB_OK", "TRUE"),
):
    os.environ.setdefault(_k, _v)

# Charger .env avant lecture des variables (SECRET_KEY, COHERE_API_KEY, etc.)
try:
    from dotenv import load_dotenv  # noqa: PLC0415

    load_dotenv(Path(__file__).resolve().parent.parent / ".env", override=False)
except ImportError:
    pass

from fastapi import Depends, FastAPI, Form, Request
from fastapi.encoders import jsonable_encoder
from fastapi.responses import HTMLResponse, JSONResponse, PlainTextResponse, RedirectResponse
from fastapi.staticfiles import StaticFiles
from typing import Literal

from pydantic import BaseModel, Field
from starlette.middleware.sessions import SessionMiddleware

from webapp import db as db_mod
from webapp.auth_jwt import create_access_token
from webapp.rag_service import ask as rag_ask
from webapp.security import get_current_user_for_api
from webapp import auth_forms
from webapp.settings import ALLOW_REGISTER, SECRET_KEY, SHOW_DEMO_LOGIN_HINT, is_admin_user
from webapp.routers.account import router as account_router
from webapp.routers.admin import router as admin_router

BASE_DIR = Path(__file__).resolve().parent
PROJECT_ROOT = BASE_DIR.parent
TEMPLATES_DIR = BASE_DIR / "templates"
STATIC_DIR = BASE_DIR / "static"
MANIFEST_PATH = PROJECT_ROOT / "vector_store" / "faiss_manifest.json"

if not os.environ.get("WEBAPP_SECRET_KEY"):
    print(
        "[webapp] AVIS : WEBAPP_SECRET_KEY non défini — une clé aléatoire a été générée "
        "(sessions + JWT invalidés au redémarrage). Définissez WEBAPP_SECRET_KEY en prod."
    )

app = FastAPI(title="RAG-MAROC2", version="0.1.0")
app.add_middleware(SessionMiddleware, secret_key=SECRET_KEY, max_age=14 * 24 * 3600)
app.include_router(account_router)
app.include_router(admin_router)

if STATIC_DIR.is_dir():
    app.mount("/static", StaticFiles(directory=str(STATIC_DIR)), name="static")


class LoginJSON(BaseModel):
    username: str
    password: str


class ChatTurn(BaseModel):
    role: Literal["user", "assistant"]
    content: str = Field(..., min_length=1, max_length=8000)


class AskBody(BaseModel):
    question: str = Field(..., min_length=1, max_length=8000)
    top_k: int = Field(5, ge=1, le=20)
    conversation_id: int | None = Field(
        None,
        description="Conversation active ; créée automatiquement si absente.",
    )
    history: list[ChatTurn] = Field(
        default_factory=list,
        max_length=40,
        description="Échanges précédents (sans la question actuelle).",
    )


class FeedbackBody(BaseModel):
    question: str = Field(..., min_length=1, max_length=8000)
    rating: Literal[-1, 1] = Field(..., description="-1 = négatif, 1 = positif")


@app.on_event("startup")
def _startup() -> None:
    db_mod.init_db()
    if os.environ.get("RAG_PRELOAD", "1").strip().lower() in ("1", "true", "yes", "on"):
        import logging
        import threading

        def _preload_rag() -> None:
            try:
                from webapp.rag_service import get_pipeline  # noqa: PLC0415

                get_pipeline()
                logging.getLogger(__name__).info("Pipeline RAG préchargé (index + modèles).")
            except Exception:
                logging.getLogger(__name__).exception("Échec du préchargement RAG au démarrage")

        threading.Thread(target=_preload_rag, daemon=True, name="rag-preload").start()


def _render(name: str, request: Request, **ctx) -> HTMLResponse:
    from jinja2 import Environment, FileSystemLoader, select_autoescape  # noqa: PLC0415

    ctx.setdefault("current_user", _current_user(request))
    ctx.setdefault("allow_register", ALLOW_REGISTER)
    ctx.setdefault("db_label", db_mod.database_label())
    ctx.setdefault("show_demo_login", SHOW_DEMO_LOGIN_HINT)

    env = Environment(
        loader=FileSystemLoader(str(TEMPLATES_DIR)),
        autoescape=select_autoescape(["html", "xml"]),
    )
    tpl = env.get_template(name)
    html = tpl.render(request=request, **ctx)
    return HTMLResponse(html)


def _current_user(request: Request) -> dict | None:
    uid = request.session.get("uid")
    if not uid:
        return None
    return db_mod.get_user_by_id(int(uid))


def _corpus_stats() -> dict | None:
    """Lecture légère du manifeste FAISS pour affichage marketing (accueil)."""
    try:
        if not MANIFEST_PATH.is_file():
            return None
        data = json.loads(MANIFEST_PATH.read_text(encoding="utf-8"))
        ntotal = int(data.get("ntotal", 0))
        return {
            "ntotal": ntotal,
            "ntotal_display": f"{ntotal:,}".replace(",", "\u202f"),
            "dimension": int(data.get("dimension", 0)),
            "embedding_model": str(data.get("embedding_model") or ""),
            "index_type": str(data.get("index_type") or ""),
            "metric": str(data.get("metric") or ""),
        }
    except (OSError, ValueError, TypeError):
        return None


# --------------------------------------------------------------------------- API (Streamlit / mobile / etc.)
@app.post("/api/auth/login")
def api_auth_login(body: LoginJSON):
    """Connexion JSON → JWT Bearer pour clients hors navigateur (ex. Streamlit)."""
    row = db_mod.get_user_by_username(body.username)
    if not row or not db_mod.verify_password(body.password, row["password_hash"]):
        return JSONResponse(
            {"detail": "Identifiant ou mot de passe incorrect."},
            status_code=401,
        )
    token = create_access_token(row["id"], row["username"], SECRET_KEY)
    return {
        "access_token": token,
        "token_type": "bearer",
        "username": row["username"],
    }


@app.get("/api/me")
def api_me(user: dict = Depends(get_current_user_for_api)):
    return {"id": user["id"], "username": user["username"], "full_name": user.get("full_name")}


@app.get("/api/chat/conversations")
def api_chat_conversations(
    user: dict = Depends(get_current_user_for_api),
    limit: int = 25,
):
    items = db_mod.list_conversations(int(user["id"]), limit=limit)
    out = []
    for c in items:
        title = (c.get("title") or "").strip()
        first = (c.get("first_question") or "").strip()
        if not title or title == "Nouvelle conversation":
            title = first[:80] + ("…" if len(first) > 80 else "") if first else title or "Nouvelle conversation"
        out.append(
            {
                "id": int(c["id"]),
                "title": title,
                "message_count": int(c.get("message_count") or 0),
                "updated_at": c.get("updated_at"),
                "created_at": c.get("created_at"),
            }
        )
    return {"conversations": out}


@app.post("/api/chat/conversations")
def api_chat_conversations_create(user: dict = Depends(get_current_user_for_api)):
    cid = db_mod.create_conversation(int(user["id"]))
    return {"id": cid, "title": "Nouvelle conversation"}


@app.get("/api/chat/history")
def api_chat_history(
    user: dict = Depends(get_current_user_for_api),
    conversation_id: int | None = None,
    page: int = 1,
    page_size: int = 40,
):
    if conversation_id is None:
        return JSONResponse(
            {"detail": "conversation_id requis."},
            status_code=400,
        )
    uid = int(user["id"])
    cid = int(conversation_id)
    if not db_mod.get_conversation(uid, cid):
        return JSONResponse({"detail": "Conversation introuvable."}, status_code=404)
    p = max(1, int(page))
    sz = max(1, min(int(page_size), 100))
    total = db_mod.count_chat_messages(uid, conversation_id=cid)
    offset = (p - 1) * sz
    page_items = db_mod.list_chat_history(
        uid, conversation_id=cid, limit=sz, offset=offset
    )
    total_pages = max(1, (total + sz - 1) // sz)
    return {
        "conversation_id": cid,
        "history": page_items,
        "page": p,
        "page_size": sz,
        "total": total,
        "total_pages": total_pages,
        "has_prev": p > 1,
        "has_next": p < total_pages,
    }


@app.delete("/api/chat/history")
def api_chat_history_clear(user: dict = Depends(get_current_user_for_api)):
    deleted = db_mod.clear_chat_history(int(user["id"]))
    return {"deleted": deleted}


@app.post("/api/ask")
async def api_ask(
    body: AskBody,
    user: dict = Depends(get_current_user_for_api),
):
    uid = int(user["id"])
    try:
        hist = [t.model_dump() for t in body.history]
        out = rag_ask(
            body.question,
            body.top_k,
            history=hist or None,
            user_id=uid,
        )
    except ValueError as e:
        msg = str(e)
        if "chunks dans le JSONL" in msg and "vecteurs" in msg:
            return JSONResponse(
                {
                    "error": (
                        "Index désynchronisé : le corpus (JSONL) et l’index FAISS "
                        "n’ont pas le même nombre de segments."
                    ),
                    "hint": (
                        "À la racine du projet : python scripts/build_embeddings.py "
                        "puis python scripts/build_faiss.py, puis redémarrer le serveur."
                    ),
                    "detail": msg,
                },
                status_code=503,
            )
        return JSONResponse({"error": msg}, status_code=500)
    except Exception as e:  # noqa: BLE001
        return JSONResponse({"error": str(e)}, status_code=500)
    try:
        cid = body.conversation_id
        if cid is None:
            cid = db_mod.create_conversation(uid)
        else:
            if not db_mod.get_conversation(uid, int(cid)):
                cid = db_mod.create_conversation(uid)
        cid = int(cid)
        db_mod.add_chat_message(uid, "user", body.question, conversation_id=cid)
        db_mod.add_chat_message(
            uid, "assistant", str(out.get("answer", "")), conversation_id=cid
        )
        out["conversation_id"] = cid
    except Exception:  # noqa: BLE001
        # Ne pas casser la réponse RAG si l'écriture d'historique échoue ponctuellement.
        pass
    # numpy / types exotiques dans les métadonnées → JSON strict pour le navigateur
    return JSONResponse(content=jsonable_encoder(out))


# --------------------------------------------------------------------------- UI Jinja (site public + app)
@app.get("/", response_class=HTMLResponse)
def home_page(request: Request):
    return _render("home.html", request, corpus_stats=_corpus_stats())


@app.get("/about", response_class=HTMLResponse)
def about_page(request: Request):
    return _render("about.html", request)


@app.get("/fonctionnalites", response_class=HTMLResponse)
def features_page(request: Request):
    return _render("features.html", request)


@app.get("/contact", response_class=HTMLResponse)
def contact_page(request: Request):
    return _render("contact.html", request)


@app.get("/docs", response_class=HTMLResponse)
def docs_page(request: Request):
    return _render("docs.html", request, corpus_stats=_corpus_stats())


@app.get("/faq", response_class=HTMLResponse)
def faq_page(request: Request):
    return _render("faq.html", request)


@app.get("/confidentialite", response_class=HTMLResponse)
def privacy_page(request: Request):
    return _render("confidentialite.html", request)


@app.get("/robots.txt", response_class=PlainTextResponse)
def robots_txt():
    return PlainTextResponse(
        "User-agent: *\nAllow: /\nDisallow: /api/\nDisallow: /chat\n",
        media_type="text/plain; charset=utf-8",
    )


@app.get("/login", response_class=HTMLResponse)
def login_page(request: Request):
    if _current_user(request):
        return RedirectResponse("/chat", status_code=302)
    return _render(
        "login.html",
        request,
        error=None,
        success=None,
        allow_register=ALLOW_REGISTER,
    )


@app.post("/login", response_class=HTMLResponse)
def login_submit(
    request: Request,
    username: str = Form(...),
    password: str = Form(...),
):
    if _current_user(request):
        return RedirectResponse("/chat", status_code=302)
    username = (username or "").strip()
    row = db_mod.get_user_by_username(username)
    err = "Identifiant ou mot de passe incorrect."
    if row and db_mod.verify_password(password, row["password_hash"]):
        request.session["uid"] = row["id"]
        try:
            from webapp import admin_db  # noqa: PLC0415

            admin_db.touch_last_login(int(row["id"]))
        except Exception:
            pass
        return RedirectResponse("/chat", status_code=302)
    return _render(
        "login.html",
        request,
        error=err,
        success=None,
        allow_register=ALLOW_REGISTER,
    )


@app.get("/register", response_class=HTMLResponse)
def register_page(request: Request):
    if not ALLOW_REGISTER:
        return RedirectResponse("/login", status_code=302)
    if _current_user(request):
        return RedirectResponse("/chat", status_code=302)
    return _render("register.html", request, error=None)


@app.post("/register", response_class=HTMLResponse)
def register_submit(
    request: Request,
    username: str = Form(...),
    password: str = Form(...),
    password_confirm: str = Form(""),
    full_name: str = Form(""),
):
    if not ALLOW_REGISTER:
        return RedirectResponse("/login", status_code=302)
    if _current_user(request):
        return RedirectResponse("/chat", status_code=302)
    full_name = (full_name or "").strip() or None
    err: str | None = None
    u_norm: str | None = None
    try:
        u_norm = auth_forms.validate_username(username)
    except ValueError as e:
        err = str(e)
    if err is None:
        try:
            auth_forms.validate_new_password(password)
        except ValueError as e:
            err = str(e)
    if err is None and not auth_forms.passwords_match(password, password_confirm):
        err = "Les mots de passe ne correspondent pas."
    if err is not None:
        return _render("register.html", request, error=err)
    assert u_norm is not None
    try:
        db_mod.create_user(u_norm, password, full_name=full_name)
    except db_mod.DuplicateUsernameError:
        return _render(
            "register.html",
            request,
            error="Ce nom d’utilisateur est déjà pris. Choisissez-en un autre.",
        )
    return _render(
        "login.html",
        request,
        error=None,
        success="Inscription enregistrée. Connectez-vous avec votre identifiant.",
        allow_register=True,
    )


@app.get("/logout")
def logout(request: Request):
    request.session.clear()
    return RedirectResponse("/", status_code=302)


@app.get("/chat", response_class=HTMLResponse)
def chat_page(request: Request):
    user = _current_user(request)
    if not user:
        return RedirectResponse("/login", status_code=302)
    return _render(
        "chat.html",
        request,
        user=user,
        is_admin=is_admin_user(user.get("username"), user),
    )


@app.post("/api/feedback")
def api_feedback(
    body: FeedbackBody,
    user: dict = Depends(get_current_user_for_api),
):
    try:
        fid = db_mod.add_ask_feedback(int(user["id"]), body.question, int(body.rating))
    except ValueError as e:
        return JSONResponse({"detail": str(e)}, status_code=400)
    return {"id": fid, "status": "ok"}


@app.get("/api/metrics/summary")
def api_metrics_summary(user: dict = Depends(get_current_user_for_api)):
    if not is_admin_user(user.get("username"), user):
        return JSONResponse({"detail": "Accès réservé aux administrateurs."}, status_code=403)
    from webapp.metrics_store import summary  # noqa: PLC0415

    return summary()


@app.get("/health")
def health():
    return {"status": "ok"}
