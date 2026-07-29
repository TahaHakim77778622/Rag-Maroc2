"""
Interface Streamlit — consomme uniquement l’API FastAPI (RAG côté serveur).

1) Terminal A — backend :
    uvicorn webapp.main:app --host 127.0.0.1 --port 8000

2) Terminal B — Streamlit :
    export RAG_API_BASE="http://127.0.0.1:8000"
    streamlit run app/streamlit_app.py

Même base utilisateurs que l’UI web (SQLite) : compte démo demo / demo123 par défaut.

Conversation : l’historique est envoyé à `/api/ask` pour des suites (« et les délais ? »).
"""

from __future__ import annotations

import os

import httpx
import streamlit as st

DEFAULT_API = os.environ.get("RAG_API_BASE", "http://127.0.0.1:8000").rstrip("/")


def _format_api_error(payload: dict | None, fallback: str) -> str:
    if not payload:
        return fallback
    if payload.get("error"):
        return str(payload["error"])
    d = payload.get("detail")
    if isinstance(d, str):
        return d
    if isinstance(d, list):
        parts: list[str] = []
        for x in d:
            if isinstance(x, str):
                parts.append(x)
            elif isinstance(x, dict) and x.get("msg"):
                parts.append(str(x["msg"]))
            else:
                parts.append(str(x))
        return " ".join(parts) if parts else fallback
    if isinstance(d, dict):
        return str(d)
    return fallback


def _show_login(base: str) -> None:
    st.subheader("Connexion")
    u = st.text_input("Identifiant", key="sl_user")
    p = st.text_input("Mot de passe", type="password", key="sl_pass")
    if st.button("Se connecter", type="primary"):
        try:
            r = httpx.post(
                f"{base}/api/auth/login",
                json={"username": u, "password": p},
                timeout=60.0,
            )
        except httpx.RequestError as e:
            st.error(f"Impossible de joindre l’API : {e}")
            return
        if r.status_code != 200:
            detail = r.json().get("detail", r.text) if "application/json" in r.headers.get(
                "content-type", ""
            ) else r.text
            st.error(detail)
            return
        data = r.json()
        st.session_state["api_token"] = data["access_token"]
        st.session_state["api_username"] = data.get("username", u)
        st.rerun()


def _logout() -> None:
    st.session_state.pop("api_token", None)
    st.session_state.pop("api_username", None)
    st.session_state.pop("chat_messages", None)
    st.rerun()


def _headers() -> dict[str, str]:
    t = st.session_state.get("api_token")
    if not t:
        return {}
    return {"Authorization": f"Bearer {t}"}


def main() -> None:
    st.set_page_config(page_title="RAG-MAROC2", layout="wide")
    base = st.sidebar.text_input(
        "URL API FastAPI",
        value=DEFAULT_API,
        help="Backend uvicorn (webapp.main:app)",
    ).rstrip("/")

    st.sidebar.markdown("### RAG-MAROC2")
    st.sidebar.caption("Streamlit → FastAPI (conversation)")

    if not st.session_state.get("api_token"):
        _show_login(base)
        st.info(
            "Démarrez le backend : `uvicorn webapp.main:app --host 127.0.0.1 --port 8000`"
        )
        return

    st.sidebar.write(f"**{st.session_state.get('api_username', '')}**")
    if st.sidebar.button("Déconnexion"):
        _logout()

    k = st.sidebar.slider("Passages retrieval (top-k)", 1, 20, 5)
    if st.sidebar.button("Nouvelle conversation"):
        st.session_state.pop("chat_messages", None)
        st.rerun()

    if "chat_messages" not in st.session_state:
        st.session_state.chat_messages = []

    st.title("Assistant juridique & administratif (Maroc)")
    st.caption(
        "Posez une question, puis des précisions : le fil de discussion est pris en compte pour la recherche et la réponse."
    )

    for msg in st.session_state.chat_messages:
        with st.chat_message(msg["role"]):
            st.markdown(msg["content"])

    if prompt := st.chat_input("Votre message…"):
        with st.chat_message("user"):
            st.markdown(prompt)

        history = [
            {"role": m["role"], "content": m["content"]}
            for m in st.session_state.chat_messages
        ]

        with st.spinner("Recherche dans le corpus et génération…"):
            try:
                r = httpx.post(
                    f"{base}/api/ask",
                    json={
                        "question": prompt.strip(),
                        "top_k": k,
                        "history": history,
                    },
                    headers=_headers(),
                    timeout=300.0,
                )
            except httpx.RequestError as e:
                st.error(f"Erreur réseau : {e}")
                return

        if r.status_code == 401:
            st.error("Session expirée. Reconnectez-vous.")
            st.session_state.pop("api_token", None)
            return

        if r.status_code != 200:
            try:
                payload = r.json()
            except Exception:
                payload = None
            st.error(_format_api_error(payload, r.text or "Erreur API"))
            return

        data = r.json()
        answer = data.get("answer", "") or "(Réponse vide)"

        with st.chat_message("assistant"):
            st.markdown(answer)
            src = data.get("sources") or []
            if src:
                st.caption("Sources utilisées pour cette réponse")
                for s in src:
                    with st.expander(
                        f"[score {s.get('score')}] {s.get('title', '')} — {s.get('label', '')}",
                        expanded=False,
                    ):
                        st.caption(s.get("chunk_id", ""))
                        if s.get("source_url"):
                            st.markdown(f"[Lien]({s['source_url']})")
                        st.text(s.get("preview", ""))

        st.session_state.chat_messages.append({"role": "user", "content": prompt.strip()})
        st.session_state.chat_messages.append({"role": "assistant", "content": answer})
        if len(st.session_state.chat_messages) > 40:
            st.session_state.chat_messages = st.session_state.chat_messages[-40:]
        st.rerun()


main()
