"""
Requêtes et agrégations pour le panneau d'administration.
"""

from __future__ import annotations

import os
import secrets
from datetime import datetime, timedelta, timezone
from typing import Any

from webapp import db as db_mod
from webapp.settings import ADMIN_USERNAMES, is_admin_user as _is_admin_env

USE_POSTGRES = db_mod.USE_POSTGRES


def _now_iso() -> str:
    return datetime.now(timezone.utc).isoformat()


def user_is_admin(user: dict | None) -> bool:
    if not user:
        return False
    if user.get("is_admin"):
        return True
    return _is_admin_env(user.get("username"))


def migrate_users_admin_columns() -> None:
    """Ajoute is_admin, is_active, last_login_at si absents."""
    cols_needed = ("is_admin", "is_active", "last_login_at")
    if USE_POSTGRES:
        import psycopg
        from psycopg.rows import dict_row

        with psycopg.connect(db_mod._pg_url(), row_factory=dict_row) as conn:
            with conn.cursor() as cur:
                cur.execute(
                    """
                    SELECT column_name FROM information_schema.columns
                    WHERE table_name = 'users'
                    """
                )
                existing = {r["column_name"] for r in cur.fetchall() or []}
                if "is_admin" not in existing:
                    cur.execute(
                        "ALTER TABLE users ADD COLUMN is_admin BOOLEAN NOT NULL DEFAULT FALSE"
                    )
                if "is_active" not in existing:
                    cur.execute(
                        "ALTER TABLE users ADD COLUMN is_active BOOLEAN NOT NULL DEFAULT TRUE"
                    )
                if "last_login_at" not in existing:
                    cur.execute(
                        "ALTER TABLE users ADD COLUMN last_login_at TIMESTAMPTZ"
                    )
            conn.commit()
    else:
        with db_mod._connect() as conn:
            info = conn.execute("PRAGMA table_info(users)").fetchall()
            existing = {row[1] for row in info}
            if "is_admin" not in existing:
                conn.execute(
                    "ALTER TABLE users ADD COLUMN is_admin INTEGER NOT NULL DEFAULT 0"
                )
            if "is_active" not in existing:
                conn.execute(
                    "ALTER TABLE users ADD COLUMN is_active INTEGER NOT NULL DEFAULT 1"
                )
            if "last_login_at" not in existing:
                conn.execute("ALTER TABLE users ADD COLUMN last_login_at TEXT")
            conn.commit()

    _sync_admin_flags_from_env()


def _sync_admin_flags_from_env() -> None:
    for name in ADMIN_USERNAMES:
        if name:
            set_user_admin_by_username(name, True)


def set_user_admin_by_username(username: str, is_admin: bool = True) -> None:
    username = username.strip().lower()
    if USE_POSTGRES:
        import psycopg

        with psycopg.connect(db_mod._pg_url()) as conn:
            with conn.cursor() as cur:
                cur.execute(
                    "UPDATE users SET is_admin = %s WHERE username = %s",
                    (bool(is_admin), username),
                )
            conn.commit()
        return
    with db_mod._connect() as conn:
        conn.execute(
            "UPDATE users SET is_admin = ? WHERE username = ?",
            (1 if is_admin else 0, username),
        )
        conn.commit()


def touch_last_login(user_id: int) -> None:
    ts = _now_iso()
    if USE_POSTGRES:
        import psycopg

        with psycopg.connect(db_mod._pg_url()) as conn:
            with conn.cursor() as cur:
                cur.execute(
                    "UPDATE users SET last_login_at = NOW() WHERE id = %s",
                    (int(user_id),),
                )
            conn.commit()
        return
    with db_mod._connect() as conn:
        conn.execute(
            "UPDATE users SET last_login_at = ? WHERE id = ?",
            (ts, int(user_id)),
        )
        conn.commit()


def count_users(*, active_only: bool = False) -> int:
    where = " WHERE is_active = 1" if active_only else ""
    if active_only and not USE_POSTGRES:
        where = " WHERE is_active = 1"
    if USE_POSTGRES:
        import psycopg
        from psycopg.rows import dict_row

        q = "SELECT COUNT(*) AS c FROM users"
        if active_only:
            q += " WHERE is_active = TRUE"
        with psycopg.connect(db_mod._pg_url(), row_factory=dict_row) as conn:
            with conn.cursor() as cur:
                cur.execute(q)
                row = cur.fetchone()
        return int(row["c"]) if row else 0
    q = "SELECT COUNT(*) AS c FROM users"
    if active_only:
        q += " WHERE is_active = 1"
    with db_mod._connect() as conn:
        row = conn.execute(q).fetchone()
    return int(row["c"]) if row else 0


def count_user_messages_today() -> int:
    if USE_POSTGRES:
        import psycopg
        from psycopg.rows import dict_row

        with psycopg.connect(db_mod._pg_url(), row_factory=dict_row) as conn:
            with conn.cursor() as cur:
                cur.execute(
                    """
                    SELECT COUNT(*) AS c FROM chat_messages
                    WHERE role = 'user' AND created_at::date = CURRENT_DATE
                    """
                )
                row = cur.fetchone()
        return int(row["c"]) if row else 0
    with db_mod._connect() as conn:
        row = conn.execute(
            """
            SELECT COUNT(*) AS c FROM chat_messages
            WHERE role = 'user' AND date(created_at) = date('now')
            """
        ).fetchone()
    return int(row["c"]) if row else 0


def count_total_user_questions() -> int:
    if USE_POSTGRES:
        import psycopg
        from psycopg.rows import dict_row

        with psycopg.connect(db_mod._pg_url(), row_factory=dict_row) as conn:
            with conn.cursor() as cur:
                cur.execute(
                    "SELECT COUNT(*) AS c FROM chat_messages WHERE role = 'user'"
                )
                row = cur.fetchone()
        return int(row["c"]) if row else 0
    with db_mod._connect() as conn:
        row = conn.execute(
            "SELECT COUNT(*) AS c FROM chat_messages WHERE role = 'user'"
        ).fetchone()
    return int(row["c"]) if row else 0


def activity_last_n_days(n: int = 7) -> list[dict[str, Any]]:
    """Nombre de messages utilisateur par jour (7 derniers jours)."""
    days: list[dict[str, Any]] = []
    today = datetime.now(timezone.utc).date()
    for i in range(n - 1, -1, -1):
        d = today - timedelta(days=i)
        days.append({"date": d.isoformat(), "label": d.strftime("%d/%m"), "count": 0})

    if USE_POSTGRES:
        import psycopg
        from psycopg.rows import dict_row

        with psycopg.connect(db_mod._pg_url(), row_factory=dict_row) as conn:
            with conn.cursor() as cur:
                cur.execute(
                    """
                    SELECT created_at::date AS d, COUNT(*) AS c
                    FROM chat_messages
                    WHERE role = 'user'
                      AND created_at >= CURRENT_DATE - %s
                    GROUP BY 1
                    """,
                    (n - 1,),
                )
                rows = cur.fetchall() or []
    else:
        with db_mod._connect() as conn:
            rows = conn.execute(
                """
                SELECT date(created_at) AS d, COUNT(*) AS c
                FROM chat_messages
                WHERE role = 'user'
                  AND date(created_at) >= date('now', ?)
                GROUP BY date(created_at)
                """,
                (f"-{n - 1} day",),
            ).fetchall()
        rows = [{"d": r["d"], "c": r["c"]} for r in rows]

    by_date = {}
    for r in rows:
        key = str(r["d"])[:10]
        by_date[key] = int(r["c"])
    for item in days:
        item["count"] = by_date.get(item["date"], 0)
    max_c = max((d["count"] for d in days), default=1) or 1
    for item in days:
        item["pct"] = round(100.0 * item["count"] / max_c, 1)
    return days


def list_recent_questions(limit: int = 10) -> list[dict[str, Any]]:
    """Dernières questions (chat + métriques si dispo)."""
    out: list[dict[str, Any]] = []
    try:
        from webapp.metrics_store import list_recent_ask_events

        events = list_recent_ask_events(limit=limit)
        for ev in events:
            uid = ev.get("user_id")
            uname = "—"
            if uid:
                u = db_mod.get_user_by_id(int(uid))
                if u:
                    uname = u.get("username") or uname
            out.append(
                {
                    "username": uname,
                    "question": (ev.get("question") or "")[:50],
                    "created_at": ev.get("created_at_display") or "—",
                    "source": "web" if ev.get("web_fallback") else "corpus",
                }
            )
        if out:
            return out
    except Exception:
        pass

    if USE_POSTGRES:
        import psycopg
        from psycopg.rows import dict_row

        with psycopg.connect(db_mod._pg_url(), row_factory=dict_row) as conn:
            with conn.cursor() as cur:
                cur.execute(
                    """
                    SELECT u.username, cm.content, cm.created_at
                    FROM chat_messages cm
                    JOIN users u ON u.id = cm.user_id
                    WHERE cm.role = 'user'
                    ORDER BY cm.id DESC
                    LIMIT %s
                    """,
                    (int(limit),),
                )
                rows = cur.fetchall() or []
    else:
        with db_mod._connect() as conn:
            rows = conn.execute(
                """
                SELECT u.username, cm.content, cm.created_at
                FROM chat_messages cm
                JOIN users u ON u.id = cm.user_id
                WHERE cm.role = 'user'
                ORDER BY cm.id DESC
                LIMIT ?
                """,
                (int(limit),),
            ).fetchall()
        rows = [dict(r) for r in rows]

    for r in rows:
        out.append(
            {
                "username": r.get("username") or "—",
                "question": (r.get("content") or "")[:50],
                "created_at": str(r.get("created_at") or "—"),
                "source": "corpus",
            }
        )
    return out


def list_users_admin(
    *,
    q: str = "",
    page: int = 1,
    page_size: int = 20,
) -> tuple[list[dict[str, Any]], int]:
    page = max(1, int(page))
    page_size = max(1, min(int(page_size), 100))
    offset = (page - 1) * page_size
    q = (q or "").strip().lower()
    params: list[Any] = []
    where = ""
    if q:
        where = " WHERE (LOWER(u.username) LIKE %s OR LOWER(COALESCE(u.full_name,'')) LIKE %s)"
        if not USE_POSTGRES:
            where = " WHERE (LOWER(u.username) LIKE ? OR LOWER(COALESCE(u.full_name,'')) LIKE ?)"
        like = f"%{q}%"
        params = [like, like]

    if USE_POSTGRES:
        import psycopg
        from psycopg.rows import dict_row

        count_sql = f"SELECT COUNT(*) AS c FROM users u{where}"
        list_sql = f"""
            SELECT u.id, u.username, u.full_name, u.created_at, u.last_login_at,
                   u.is_admin, u.is_active,
                   (SELECT COUNT(*) FROM chat_messages cm
                    WHERE cm.user_id = u.id AND cm.role = 'user') AS conv_count
            FROM users u{where}
            ORDER BY u.id DESC
            LIMIT %s OFFSET %s
        """
        with psycopg.connect(db_mod._pg_url(), row_factory=dict_row) as conn:
            with conn.cursor() as cur:
                cur.execute(count_sql, params)
                total = int(cur.fetchone()["c"])
                cur.execute(list_sql, params + [page_size, offset])
                rows = [dict(r) for r in cur.fetchall() or []]
    else:
        count_sql = f"SELECT COUNT(*) AS c FROM users u{where}"
        list_sql = f"""
            SELECT u.id, u.username, u.full_name, u.created_at, u.last_login_at,
                   u.is_admin, u.is_active,
                   (SELECT COUNT(*) FROM chat_messages cm
                    WHERE cm.user_id = u.id AND cm.role = 'user') AS conv_count
            FROM users u{where}
            ORDER BY u.id DESC
            LIMIT ? OFFSET ?
        """
        with db_mod._connect() as conn:
            total = int(conn.execute(count_sql, params).fetchone()["c"])
            rows = conn.execute(list_sql, params + [page_size, offset]).fetchall()
            rows = [dict(r) for r in rows]

    return rows, total


def get_user_admin_detail(user_id: int) -> dict[str, Any] | None:
    if USE_POSTGRES:
        import psycopg
        from psycopg.rows import dict_row

        with psycopg.connect(db_mod._pg_url(), row_factory=dict_row) as conn:
            with conn.cursor() as cur:
                cur.execute(
                    """
                    SELECT u.*,
                      (SELECT COUNT(*) FROM chat_messages cm
                       WHERE cm.user_id = u.id AND cm.role = 'user') AS question_count
                    FROM users u WHERE u.id = %s
                    """,
                    (int(user_id),),
                )
                row = cur.fetchone()
    else:
        with db_mod._connect() as conn:
            row = conn.execute(
                """
                SELECT u.id, u.username, u.full_name, u.created_at, u.last_login_at,
                       u.is_admin, u.is_active,
                       (SELECT COUNT(*) FROM chat_messages cm
                        WHERE cm.user_id = u.id AND cm.role = 'user') AS question_count
                FROM users u WHERE u.id = ?
                """,
                (int(user_id),),
            ).fetchone()
    if not row:
        return None
    return dict(row)


def list_user_conversations(user_id: int, limit: int = 20) -> list[dict[str, Any]]:
    if USE_POSTGRES:
        import psycopg
        from psycopg.rows import dict_row

        with psycopg.connect(db_mod._pg_url(), row_factory=dict_row) as conn:
            with conn.cursor() as cur:
                cur.execute(
                    """
                    SELECT content, created_at FROM chat_messages
                    WHERE user_id = %s AND role = 'user'
                    ORDER BY id DESC LIMIT %s
                    """,
                    (int(user_id), int(limit)),
                )
                rows = [dict(r) for r in cur.fetchall() or []]
    else:
        with db_mod._connect() as conn:
            rows = conn.execute(
                """
                SELECT content, created_at FROM chat_messages
                WHERE user_id = ? AND role = 'user'
                ORDER BY id DESC LIMIT ?
                """,
                (int(user_id), int(limit)),
            ).fetchall()
            rows = [dict(r) for r in rows]
    return rows


def set_user_active(user_id: int, active: bool) -> None:
    if USE_POSTGRES:
        import psycopg

        with psycopg.connect(db_mod._pg_url()) as conn:
            with conn.cursor() as cur:
                cur.execute(
                    "UPDATE users SET is_active = %s WHERE id = %s",
                    (bool(active), int(user_id)),
                )
            conn.commit()
        return
    with db_mod._connect() as conn:
        conn.execute(
            "UPDATE users SET is_active = ? WHERE id = ?",
            (1 if active else 0, int(user_id)),
        )
        conn.commit()


def delete_user(user_id: int) -> None:
    if USE_POSTGRES:
        import psycopg

        with psycopg.connect(db_mod._pg_url()) as conn:
            with conn.cursor() as cur:
                cur.execute("DELETE FROM users WHERE id = %s", (int(user_id),))
            conn.commit()
        return
    with db_mod._connect() as conn:
        conn.execute("DELETE FROM users WHERE id = ?", (int(user_id),))
        conn.commit()


def reset_user_password(user_id: int) -> str:
    new_pw = secrets.token_urlsafe(10)
    db_mod.update_user_password(int(user_id), new_pw)
    return new_pw


def user_top_domains(user_id: int, limit: int = 5) -> list[dict[str, Any]]:
    """Heuristique simple sur le texte des questions."""
    rows = list_user_conversations(user_id, limit=200)
    keys = (
        ("travail", "Droit du travail"),
        ("smig", "SMIG / Travail"),
        ("cnie", "CNIE"),
        ("passeport", "Passeport"),
        ("urbanisme", "Urbanisme"),
        ("construire", "Construction"),
        ("watiqa", "Watiqa"),
        ("master", "Éducation"),
        ("bulletin", "Bulletin officiel"),
    )
    counts: dict[str, int] = {}
    for r in rows:
        t = (r.get("content") or "").lower()
        for kw, label in keys:
            if kw in t:
                counts[label] = counts.get(label, 0) + 1
    ranked = sorted(counts.items(), key=lambda x: -x[1])[:limit]
    return [{"domain": k, "count": v} for k, v in ranked]


def initials_for_user(user: dict) -> str:
    display = (user.get("full_name") or user.get("username") or "").strip()
    parts = display.split()
    if len(parts) >= 2:
        return (parts[0][:1] + parts[-1][:1]).upper()
    return display[:2].upper() if display else "??"
