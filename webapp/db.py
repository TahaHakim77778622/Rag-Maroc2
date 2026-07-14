"""
Stockage utilisateurs (PostgreSQL si WEBAPP_DATABASE_URL/DATABASE_URL est défini,
sinon fallback SQLite local).
"""

from __future__ import annotations

import os
import sqlite3
from pathlib import Path

import bcrypt

PROJECT_ROOT = Path(__file__).resolve().parent.parent
DB_PATH = PROJECT_ROOT / "data" / "webapp_users.db"

# Permet d'utiliser WEBAPP_DATABASE_URL depuis .env meme hors webapp.main.
try:
    from dotenv import load_dotenv  # noqa: PLC0415

    load_dotenv(PROJECT_ROOT / ".env", override=False)
except ImportError:
    pass

DATABASE_URL = (
    os.environ.get("WEBAPP_DATABASE_URL", "").strip()
    or os.environ.get("DATABASE_URL", "").strip()
)
USE_POSTGRES = DATABASE_URL.startswith(("postgresql://", "postgres://"))


def database_label() -> str:
    """Libellé court pour l’UI (connexion / pied de page)."""
    if USE_POSTGRES:
        return "PostgreSQL"
    try:
        rel = DB_PATH.relative_to(PROJECT_ROOT)
    except ValueError:
        rel = DB_PATH
    return f"SQLite ({rel})"


class DuplicateUsernameError(ValueError):
    """Nom d'utilisateur deja existant."""


if USE_POSTGRES:
    try:
        import psycopg
        from psycopg.rows import dict_row
    except ImportError as exc:  # pragma: no cover - error depends on env
        raise RuntimeError(
            "PostgreSQL configure mais psycopg n'est pas installe. "
            "Installez requirements.txt puis relancez l'application."
        ) from exc


def _pg_url() -> str:
    # psycopg prefere postgresql://
    if DATABASE_URL.startswith("postgres://"):
        return "postgresql://" + DATABASE_URL[len("postgres://") :]
    return DATABASE_URL


def _connect() -> sqlite3.Connection:
    DB_PATH.parent.mkdir(parents=True, exist_ok=True)
    conn = sqlite3.connect(DB_PATH)
    conn.row_factory = sqlite3.Row
    return conn


def init_db() -> None:
    if USE_POSTGRES:
        with psycopg.connect(_pg_url(), row_factory=dict_row) as conn:
            with conn.cursor() as cur:
                cur.execute(
                    """
                    CREATE TABLE IF NOT EXISTS users (
                        id BIGSERIAL PRIMARY KEY,
                        username TEXT UNIQUE NOT NULL,
                        password_hash TEXT NOT NULL,
                        full_name TEXT,
                        created_at TIMESTAMPTZ NOT NULL DEFAULT NOW()
                    )
                    """
                )
                cur.execute(
                    """
                    CREATE TABLE IF NOT EXISTS chat_messages (
                        id BIGSERIAL PRIMARY KEY,
                        user_id BIGINT NOT NULL REFERENCES users(id) ON DELETE CASCADE,
                        role TEXT NOT NULL CHECK (role IN ('user', 'assistant')),
                        content TEXT NOT NULL,
                        created_at TIMESTAMPTZ NOT NULL DEFAULT NOW()
                    )
                    """
                )
                cur.execute(
                    """
                    CREATE TABLE IF NOT EXISTS ask_feedback (
                        id BIGSERIAL PRIMARY KEY,
                        user_id BIGINT NOT NULL REFERENCES users(id) ON DELETE CASCADE,
                        question TEXT NOT NULL,
                        rating SMALLINT NOT NULL CHECK (rating IN (-1, 1)),
                        created_at TIMESTAMPTZ NOT NULL DEFAULT NOW()
                    )
                    """
                )
            conn.commit()
    else:
        with _connect() as conn:
            conn.execute(
                """
                CREATE TABLE IF NOT EXISTS users (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    username TEXT UNIQUE NOT NULL,
                    password_hash TEXT NOT NULL,
                    full_name TEXT,
                    created_at TEXT DEFAULT (datetime('now'))
                )
                """
            )
            conn.execute(
                """
                CREATE TABLE IF NOT EXISTS chat_messages (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    user_id INTEGER NOT NULL,
                    role TEXT NOT NULL CHECK (role IN ('user', 'assistant')),
                    content TEXT NOT NULL,
                    created_at TEXT DEFAULT (datetime('now')),
                    FOREIGN KEY (user_id) REFERENCES users(id) ON DELETE CASCADE
                )
                """
            )
            conn.execute(
                """
                CREATE TABLE IF NOT EXISTS ask_feedback (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    user_id INTEGER NOT NULL,
                    question TEXT NOT NULL,
                    rating INTEGER NOT NULL CHECK (rating IN (-1, 1)),
                    created_at TEXT DEFAULT (datetime('now')),
                    FOREIGN KEY (user_id) REFERENCES users(id) ON DELETE CASCADE
                )
                """
            )
            conn.commit()

    try:
        from webapp.admin_db import migrate_users_admin_columns

        migrate_users_admin_columns()
    except Exception:
        pass

    try:
        migrate_conversations_schema()
    except Exception:
        pass

    if _user_count() == 0:
        user = os.environ.get("WEBAPP_DEMO_USER", "demo").strip() or "demo"
        password = os.environ.get("WEBAPP_DEMO_PASSWORD", "demo123").strip() or "demo123"
        uid = create_user(user, password, full_name="Compte démo")
        try:
            from webapp.admin_db import set_user_admin_by_username

            set_user_admin_by_username(user, True)
        except Exception:
            pass
        print(
            f"[webapp] Aucun utilisateur : création du compte démo "
            f"login={user!r} (changez via WEBAPP_DEMO_USER / WEBAPP_DEMO_PASSWORD)"
        )


def _user_count() -> int:
    if USE_POSTGRES:
        with psycopg.connect(_pg_url(), row_factory=dict_row) as conn:
            with conn.cursor() as cur:
                cur.execute("SELECT COUNT(*) AS c FROM users")
                row = cur.fetchone()
        return int(row["c"]) if row else 0
    with _connect() as conn:
        row = conn.execute("SELECT COUNT(*) AS c FROM users").fetchone()
    return int(row["c"]) if row else 0


def hash_password(plain: str) -> str:
    data = plain.encode("utf-8")
    if len(data) > 72:
        data = data[:72]
    salt = bcrypt.gensalt(rounds=12)
    return bcrypt.hashpw(data, salt).decode("ascii")


def verify_password(plain: str, password_hash: str) -> bool:
    if not password_hash:
        return False
    try:
        data = plain.encode("utf-8")
        if len(data) > 72:
            data = data[:72]
        return bcrypt.checkpw(data, password_hash.encode("ascii"))
    except (ValueError, TypeError):
        return False


def create_user(username: str, password: str, full_name: str | None = None) -> int:
    username = username.strip().lower()
    if not username or not password:
        raise ValueError("Identifiant ou mot de passe vide.")
    ph = hash_password(password)
    if USE_POSTGRES:
        try:
            with psycopg.connect(_pg_url(), row_factory=dict_row) as conn:
                with conn.cursor() as cur:
                    cur.execute(
                        "INSERT INTO users (username, password_hash, full_name) "
                        "VALUES (%s, %s, %s) RETURNING id",
                        (username, ph, full_name),
                    )
                    row = cur.fetchone()
                conn.commit()
            return int(row["id"])
        except psycopg.errors.UniqueViolation as exc:
            raise DuplicateUsernameError("Ce nom d'utilisateur est deja pris.") from exc

    try:
        with _connect() as conn:
            cur = conn.execute(
                "INSERT INTO users (username, password_hash, full_name) VALUES (?, ?, ?)",
                (username, ph, full_name),
            )
            conn.commit()
            return int(cur.lastrowid)
    except sqlite3.IntegrityError as exc:
        raise DuplicateUsernameError("Ce nom d'utilisateur est deja pris.") from exc


def get_user_by_username(username: str) -> dict | None:
    username = username.strip().lower()
    if USE_POSTGRES:
        with psycopg.connect(_pg_url(), row_factory=dict_row) as conn:
            with conn.cursor() as cur:
                cur.execute(
                    "SELECT id, username, password_hash, full_name, "
                    "COALESCE(is_active, TRUE) AS is_active, "
                    "COALESCE(is_admin, FALSE) AS is_admin, last_login_at "
                    "FROM users WHERE username = %s",
                    (username,),
                )
                row = cur.fetchone()
    else:
        with _connect() as conn:
            row = conn.execute(
                """
                SELECT id, username, password_hash, full_name,
                       COALESCE(is_active, 1) AS is_active,
                       COALESCE(is_admin, 0) AS is_admin,
                       last_login_at
                FROM users WHERE username = ?
                """,
                (username,),
            ).fetchone()
    if row is None:
        return None
    data = dict(row)
    if not data.get("is_active", 1):
        return None
    return data


def get_user_by_id(user_id: int) -> dict | None:
    if USE_POSTGRES:
        with psycopg.connect(_pg_url(), row_factory=dict_row) as conn:
            with conn.cursor() as cur:
                cur.execute(
                    """
                    SELECT id, username, full_name, created_at, last_login_at,
                           COALESCE(is_active, TRUE) AS is_active,
                           COALESCE(is_admin, FALSE) AS is_admin
                    FROM users WHERE id = %s
                    """,
                    (user_id,),
                )
                row = cur.fetchone()
    else:
        with _connect() as conn:
            row = conn.execute(
                """
                SELECT id, username, full_name, created_at, last_login_at,
                       COALESCE(is_active, 1) AS is_active,
                       COALESCE(is_admin, 0) AS is_admin
                FROM users WHERE id = ?
                """,
                (user_id,),
            ).fetchone()
    return dict(row) if row else None


def get_user_by_id_with_hash(user_id: int) -> dict | None:
    """Profil + hash mot de passe (changement de mot de passe)."""
    if USE_POSTGRES:
        with psycopg.connect(_pg_url(), row_factory=dict_row) as conn:
            with conn.cursor() as cur:
                cur.execute(
                    "SELECT id, username, password_hash, full_name, created_at "
                    "FROM users WHERE id = %s",
                    (user_id,),
                )
                row = cur.fetchone()
    else:
        with _connect() as conn:
            row = conn.execute(
                "SELECT id, username, password_hash, full_name, created_at "
                "FROM users WHERE id = ?",
                (user_id,),
            ).fetchone()
    return dict(row) if row else None


def update_user_password(user_id: int, new_password: str) -> None:
    ph = hash_password(new_password)
    if USE_POSTGRES:
        with psycopg.connect(_pg_url(), row_factory=dict_row) as conn:
            with conn.cursor() as cur:
                cur.execute(
                    "UPDATE users SET password_hash = %s WHERE id = %s",
                    (ph, int(user_id)),
                )
            conn.commit()
        return
    with _connect() as conn:
        conn.execute(
            "UPDATE users SET password_hash = ? WHERE id = ?",
            (ph, int(user_id)),
        )
        conn.commit()


def update_user_full_name(user_id: int, full_name: str | None) -> None:
    if USE_POSTGRES:
        with psycopg.connect(_pg_url(), row_factory=dict_row) as conn:
            with conn.cursor() as cur:
                cur.execute(
                    "UPDATE users SET full_name = %s WHERE id = %s",
                    (full_name, int(user_id)),
                )
            conn.commit()
        return
    with _connect() as conn:
        conn.execute(
            "UPDATE users SET full_name = ? WHERE id = ?",
            (full_name, int(user_id)),
        )
        conn.commit()


def username_exists(username: str) -> bool:
    return get_user_by_username(username) is not None


def migrate_conversations_schema() -> None:
    """Table conversations + lien conversation_id sur chat_messages."""
    if USE_POSTGRES:
        import psycopg
        from psycopg.rows import dict_row

        with psycopg.connect(_pg_url(), row_factory=dict_row) as conn:
            with conn.cursor() as cur:
                cur.execute(
                    """
                    CREATE TABLE IF NOT EXISTS conversations (
                        id BIGSERIAL PRIMARY KEY,
                        user_id BIGINT NOT NULL REFERENCES users(id) ON DELETE CASCADE,
                        title TEXT,
                        created_at TIMESTAMPTZ NOT NULL DEFAULT NOW(),
                        updated_at TIMESTAMPTZ NOT NULL DEFAULT NOW()
                    )
                    """
                )
                cur.execute(
                    """
                    SELECT column_name FROM information_schema.columns
                    WHERE table_name = 'chat_messages' AND column_name = 'conversation_id'
                    """
                )
                if not cur.fetchone():
                    cur.execute(
                        """
                        ALTER TABLE chat_messages
                        ADD COLUMN conversation_id BIGINT
                        REFERENCES conversations(id) ON DELETE CASCADE
                        """
                    )
            conn.commit()
        _backfill_conversations_postgres()
        return

    with _connect() as conn:
        conn.execute(
            """
            CREATE TABLE IF NOT EXISTS conversations (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                user_id INTEGER NOT NULL,
                title TEXT,
                created_at TEXT DEFAULT (datetime('now')),
                updated_at TEXT DEFAULT (datetime('now')),
                FOREIGN KEY (user_id) REFERENCES users(id) ON DELETE CASCADE
            )
            """
        )
        info = conn.execute("PRAGMA table_info(chat_messages)").fetchall()
        cols = {row[1] for row in info}
        if "conversation_id" not in cols:
            conn.execute(
                """
                ALTER TABLE chat_messages
                ADD COLUMN conversation_id INTEGER
                REFERENCES conversations(id) ON DELETE CASCADE
                """
            )
        conn.commit()
    _backfill_conversations_sqlite()


def _backfill_conversations_postgres() -> None:
    import psycopg
    from psycopg.rows import dict_row

    with psycopg.connect(_pg_url(), row_factory=dict_row) as conn:
        with conn.cursor() as cur:
            cur.execute(
                """
                SELECT DISTINCT user_id FROM chat_messages
                WHERE conversation_id IS NULL
                """
            )
            users = [int(r["user_id"]) for r in cur.fetchall() or []]
            for uid in users:
                cur.execute(
                    """
                    INSERT INTO conversations (user_id, title)
                    VALUES (%s, %s) RETURNING id
                    """,
                    (uid, "Conversation précédente"),
                )
                cid = int(cur.fetchone()["id"])
                cur.execute(
                    """
                    UPDATE chat_messages SET conversation_id = %s
                    WHERE user_id = %s AND conversation_id IS NULL
                    """,
                    (cid, uid),
                )
        conn.commit()


def _backfill_conversations_sqlite() -> None:
    with _connect() as conn:
        rows = conn.execute(
            "SELECT DISTINCT user_id FROM chat_messages WHERE conversation_id IS NULL"
        ).fetchall()
        for row in rows:
            uid = int(row["user_id"])
            cur = conn.execute(
                "INSERT INTO conversations (user_id, title) VALUES (?, ?)",
                (uid, "Conversation précédente"),
            )
            cid = int(cur.lastrowid)
            conn.execute(
                """
                UPDATE chat_messages SET conversation_id = ?
                WHERE user_id = ? AND conversation_id IS NULL
                """,
                (cid, uid),
            )
        conn.commit()


def create_conversation(user_id: int, *, title: str | None = None) -> int:
    title = (title or "").strip() or "Nouvelle conversation"
    if USE_POSTGRES:
        import psycopg
        from psycopg.rows import dict_row

        with psycopg.connect(_pg_url(), row_factory=dict_row) as conn:
            with conn.cursor() as cur:
                cur.execute(
                    """
                    INSERT INTO conversations (user_id, title)
                    VALUES (%s, %s) RETURNING id
                    """,
                    (int(user_id), title),
                )
                row = cur.fetchone()
            conn.commit()
        return int(row["id"]) if row else 0

    with _connect() as conn:
        cur = conn.execute(
            "INSERT INTO conversations (user_id, title) VALUES (?, ?)",
            (int(user_id), title),
        )
        conn.commit()
        return int(cur.lastrowid)


def list_conversations(user_id: int, *, limit: int = 25) -> list[dict]:
    lim = max(1, min(int(limit), 50))
    if USE_POSTGRES:
        import psycopg
        from psycopg.rows import dict_row

        with psycopg.connect(_pg_url(), row_factory=dict_row) as conn:
            with conn.cursor() as cur:
                cur.execute(
                    """
                    SELECT c.id, c.title, c.created_at, c.updated_at,
                      (SELECT COUNT(*) FROM chat_messages m
                       WHERE m.conversation_id = c.id) AS message_count,
                      (SELECT m.content FROM chat_messages m
                       WHERE m.conversation_id = c.id AND m.role = 'user'
                       ORDER BY m.id ASC LIMIT 1) AS first_question
                    FROM conversations c
                    WHERE c.user_id = %s
                    ORDER BY c.updated_at DESC
                    LIMIT %s
                    """,
                    (int(user_id), lim),
                )
                rows = [dict(r) for r in cur.fetchall() or []]
        return rows

    with _connect() as conn:
        rows = conn.execute(
            """
            SELECT c.id, c.title, c.created_at, c.updated_at,
              (SELECT COUNT(*) FROM chat_messages m
               WHERE m.conversation_id = c.id) AS message_count,
              (SELECT m.content FROM chat_messages m
               WHERE m.conversation_id = c.id AND m.role = 'user'
               ORDER BY m.id ASC LIMIT 1) AS first_question
            FROM conversations c
            WHERE c.user_id = ?
            ORDER BY c.updated_at DESC
            LIMIT ?
            """,
            (int(user_id), lim),
        ).fetchall()
    return [dict(r) for r in rows]


def get_conversation(user_id: int, conversation_id: int) -> dict | None:
    if USE_POSTGRES:
        import psycopg
        from psycopg.rows import dict_row

        with psycopg.connect(_pg_url(), row_factory=dict_row) as conn:
            with conn.cursor() as cur:
                cur.execute(
                    """
                    SELECT id, user_id, title, created_at, updated_at
                    FROM conversations
                    WHERE id = %s AND user_id = %s
                    """,
                    (int(conversation_id), int(user_id)),
                )
                row = cur.fetchone()
        return dict(row) if row else None

    with _connect() as conn:
        row = conn.execute(
            """
            SELECT id, user_id, title, created_at, updated_at
            FROM conversations WHERE id = ? AND user_id = ?
            """,
            (int(conversation_id), int(user_id)),
        ).fetchone()
    return dict(row) if row else None


def touch_conversation(conversation_id: int) -> None:
    if USE_POSTGRES:
        import psycopg

        with psycopg.connect(_pg_url()) as conn:
            with conn.cursor() as cur:
                cur.execute(
                    "UPDATE conversations SET updated_at = NOW() WHERE id = %s",
                    (int(conversation_id),),
                )
            conn.commit()
        return
    with _connect() as conn:
        conn.execute(
            "UPDATE conversations SET updated_at = datetime('now') WHERE id = ?",
            (int(conversation_id),),
        )
        conn.commit()


def update_conversation_title(conversation_id: int, title: str) -> None:
    title = (title or "").strip()[:120]
    if not title:
        return
    if USE_POSTGRES:
        import psycopg

        with psycopg.connect(_pg_url()) as conn:
            with conn.cursor() as cur:
                cur.execute(
                    "UPDATE conversations SET title = %s, updated_at = NOW() WHERE id = %s",
                    (title, int(conversation_id)),
                )
            conn.commit()
        return
    with _connect() as conn:
        conn.execute(
            """
            UPDATE conversations SET title = ?, updated_at = datetime('now')
            WHERE id = ?
            """,
            (title, int(conversation_id)),
        )
        conn.commit()


def add_chat_message(
    user_id: int,
    role: str,
    content: str,
    *,
    conversation_id: int,
) -> int:
    role = (role or "").strip().lower()
    if role not in ("user", "assistant"):
        raise ValueError("Role invalide pour chat message.")
    content = (content or "").strip()
    if not content:
        raise ValueError("Message vide.")
    cid = int(conversation_id)
    if not get_conversation(user_id, cid):
        raise ValueError("Conversation introuvable.")

    if USE_POSTGRES:
        import psycopg
        from psycopg.rows import dict_row

        with psycopg.connect(_pg_url(), row_factory=dict_row) as conn:
            with conn.cursor() as cur:
                cur.execute(
                    """
                    INSERT INTO chat_messages (user_id, conversation_id, role, content)
                    VALUES (%s, %s, %s, %s) RETURNING id
                    """,
                    (int(user_id), cid, role, content),
                )
                row = cur.fetchone()
            conn.commit()
        touch_conversation(cid)
        if role == "user":
            conv = get_conversation(user_id, cid)
            if conv and (not conv.get("title") or conv.get("title") == "Nouvelle conversation"):
                short = content[:80] + ("…" if len(content) > 80 else "")
                update_conversation_title(cid, short)
        return int(row["id"]) if row else 0

    with _connect() as conn:
        cur = conn.execute(
            """
            INSERT INTO chat_messages (user_id, conversation_id, role, content)
            VALUES (?, ?, ?, ?)
            """,
            (int(user_id), cid, role, content),
        )
        conn.commit()
        mid = int(cur.lastrowid)
    touch_conversation(cid)
    if role == "user":
        conv = get_conversation(user_id, cid)
        if conv and (not conv.get("title") or conv.get("title") == "Nouvelle conversation"):
            short = content[:80] + ("…" if len(content) > 80 else "")
            update_conversation_title(cid, short)
    return mid


def list_chat_history(
    user_id: int,
    *,
    conversation_id: int,
    limit: int = 40,
    offset: int = 0,
) -> list[dict]:
    lim = max(1, min(int(limit), 200))
    off = max(0, int(offset))
    cid = int(conversation_id)
    if not get_conversation(user_id, cid):
        return []
    if USE_POSTGRES:
        import psycopg
        from psycopg.rows import dict_row

        with psycopg.connect(_pg_url(), row_factory=dict_row) as conn:
            with conn.cursor() as cur:
                cur.execute(
                    """
                    SELECT role, content, created_at
                    FROM chat_messages
                    WHERE user_id = %s AND conversation_id = %s
                    ORDER BY id DESC
                    LIMIT %s OFFSET %s
                    """,
                    (int(user_id), cid, lim, off),
                )
                rows = cur.fetchall() or []
        rows = list(reversed(rows))
        return [dict(r) for r in rows]

    with _connect() as conn:
        rows = conn.execute(
            """
            SELECT role, content, created_at
            FROM chat_messages
            WHERE user_id = ? AND conversation_id = ?
            ORDER BY id DESC
            LIMIT ? OFFSET ?
            """,
            (int(user_id), cid, lim, off),
        ).fetchall()
    rows = list(reversed(rows))
    return [dict(r) for r in rows]


def clear_chat_history(user_id: int) -> int:
    """Supprime toutes les conversations et messages de l'utilisateur."""
    if USE_POSTGRES:
        import psycopg
        from psycopg.rows import dict_row

        with psycopg.connect(_pg_url(), row_factory=dict_row) as conn:
            with conn.cursor() as cur:
                cur.execute("DELETE FROM chat_messages WHERE user_id = %s", (int(user_id),))
                deleted = cur.rowcount or 0
                cur.execute("DELETE FROM conversations WHERE user_id = %s", (int(user_id),))
            conn.commit()
        return int(deleted)

    with _connect() as conn:
        cur = conn.execute("DELETE FROM chat_messages WHERE user_id = ?", (int(user_id),))
        deleted = int(cur.rowcount or 0)
        conn.execute("DELETE FROM conversations WHERE user_id = ?", (int(user_id),))
        conn.commit()
    return deleted


def add_ask_feedback(user_id: int, question: str, rating: int) -> int:
    rating = int(rating)
    if rating not in (-1, 1):
        raise ValueError("rating doit être -1 ou 1")
    question = (question or "").strip()
    if not question:
        raise ValueError("question vide")
    if USE_POSTGRES:
        with psycopg.connect(_pg_url(), row_factory=dict_row) as conn:
            with conn.cursor() as cur:
                cur.execute(
                    "INSERT INTO ask_feedback (user_id, question, rating) VALUES (%s, %s, %s) RETURNING id",
                    (int(user_id), question[:8000], rating),
                )
                row = cur.fetchone()
            conn.commit()
        return int(row["id"]) if row else 0
    with _connect() as conn:
        cur = conn.execute(
            "INSERT INTO ask_feedback (user_id, question, rating) VALUES (?, ?, ?)",
            (int(user_id), question[:8000], rating),
        )
        conn.commit()
        return int(cur.lastrowid)


def count_negative_feedback() -> int:
    if USE_POSTGRES:
        with psycopg.connect(_pg_url(), row_factory=dict_row) as conn:
            with conn.cursor() as cur:
                cur.execute("SELECT COUNT(*) AS c FROM ask_feedback WHERE rating = -1")
                row = cur.fetchone()
        return int(row["c"]) if row else 0
    with _connect() as conn:
        row = conn.execute("SELECT COUNT(*) AS c FROM ask_feedback WHERE rating = -1").fetchone()
    return int(row["c"]) if row else 0


def count_chat_messages(user_id: int, *, conversation_id: int | None = None) -> int:
    if USE_POSTGRES:
        import psycopg
        from psycopg.rows import dict_row

        with psycopg.connect(_pg_url(), row_factory=dict_row) as conn:
            with conn.cursor() as cur:
                if conversation_id is not None:
                    cur.execute(
                        """
                        SELECT COUNT(*) AS c FROM chat_messages
                        WHERE user_id = %s AND conversation_id = %s
                        """,
                        (int(user_id), int(conversation_id)),
                    )
                else:
                    cur.execute(
                        "SELECT COUNT(*) AS c FROM chat_messages WHERE user_id = %s",
                        (int(user_id),),
                    )
                row = cur.fetchone()
        return int(row["c"]) if row else 0

    with _connect() as conn:
        if conversation_id is not None:
            row = conn.execute(
                """
                SELECT COUNT(*) AS c FROM chat_messages
                WHERE user_id = ? AND conversation_id = ?
                """,
                (int(user_id), int(conversation_id)),
            ).fetchone()
        else:
            row = conn.execute(
                "SELECT COUNT(*) AS c FROM chat_messages WHERE user_id = ?",
                (int(user_id),),
            ).fetchone()
    return int(row["c"]) if row else 0
