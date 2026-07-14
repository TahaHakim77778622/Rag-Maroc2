"""
KPI de production (article IEEE) : événements /api/ask persistés en JSONL.
"""

from __future__ import annotations

import json
import os
import threading
import time
from pathlib import Path
from typing import Any

PROJECT_ROOT = Path(__file__).resolve().parent.parent
METRICS_DIR = PROJECT_ROOT / "data" / "metrics"
EVENTS_PATH = METRICS_DIR / "ask_events.jsonl"
_lock = threading.Lock()


def record_ask_event(
    *,
    user_id: int | None,
    question: str,
    latency_sec: float,
    n_sources: int,
    web_fallback: bool,
    llm_error: bool = False,
) -> None:
    METRICS_DIR.mkdir(parents=True, exist_ok=True)
    q = (question or "").strip()
    row = {
        "ts": time.time(),
        "user_id": user_id,
        "question": q[:500],
        "question_len": len(q),
        "latency_sec": round(latency_sec, 3),
        "n_sources": int(n_sources),
        "web_fallback": bool(web_fallback),
        "llm_error": bool(llm_error),
    }
    with _lock:
        with EVENTS_PATH.open("a", encoding="utf-8") as f:
            f.write(json.dumps(row, ensure_ascii=False) + "\n")


def summary(*, max_events: int = 5000) -> dict[str, Any]:
    if not EVENTS_PATH.is_file():
        return {
            "total_asks": 0,
            "sourced_rate_pct": 0.0,
            "llm_error_rate_pct": 0.0,
            "fallback_rate_pct": 0.0,
            "latency_p50_sec": 0.0,
            "latency_p95_sec": 0.0,
            "targets": {
                "sourced_rate_min_pct": 85.0,
                "llm_error_rate_max_pct": 2.0,
                "latency_p50_max_sec": 8.0,
                "latency_p95_max_sec": 25.0,
                "fallback_rate_max_pct": 15.0,
            },
        }

    rows: list[dict[str, Any]] = []
    with EVENTS_PATH.open(encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            try:
                rows.append(json.loads(line))
            except json.JSONDecodeError:
                continue
    if len(rows) > max_events:
        rows = rows[-max_events:]

    n = len(rows)
    if n == 0:
        return summary()

    sourced = sum(1 for r in rows if int(r.get("n_sources", 0)) >= 1)
    llm_err = sum(1 for r in rows if r.get("llm_error"))
    fallback = sum(1 for r in rows if r.get("web_fallback"))
    lats = sorted(float(r.get("latency_sec", 0)) for r in rows)

    def pctile(p: float) -> float:
        if not lats:
            return 0.0
        idx = min(len(lats) - 1, max(0, int(round(p * (len(lats) - 1)))))
        return lats[idx]

    return {
        "total_asks": n,
        "sourced_rate_pct": round(100.0 * sourced / n, 1),
        "llm_error_rate_pct": round(100.0 * llm_err / n, 1),
        "fallback_rate_pct": round(100.0 * fallback / n, 1),
        "latency_p50_sec": round(pctile(0.50), 2),
        "latency_p95_sec": round(pctile(0.95), 2),
        "negative_feedback_count": _feedback_negative_count(),
        "targets": {
            "sourced_rate_min_pct": float(os.environ.get("KPI_SOURCED_MIN_PCT", "85")),
            "llm_error_rate_max_pct": float(os.environ.get("KPI_LLM_ERROR_MAX_PCT", "2")),
            "latency_p50_max_sec": float(os.environ.get("KPI_P50_MAX_SEC", "8")),
            "latency_p95_max_sec": float(os.environ.get("KPI_P95_MAX_SEC", "25")),
            "fallback_rate_max_pct": float(os.environ.get("KPI_FALLBACK_MAX_PCT", "15")),
        },
    }


def avg_latency_sec(*, max_events: int = 500) -> float:
    rows = _load_events(max_events)
    if not rows:
        return 0.0
    lats = [float(r.get("latency_sec", 0)) for r in rows]
    return round(sum(lats) / len(lats), 2)


def _load_events(max_events: int) -> list[dict[str, Any]]:
    if not EVENTS_PATH.is_file():
        return []
    rows: list[dict[str, Any]] = []
    with EVENTS_PATH.open(encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            try:
                rows.append(json.loads(line))
            except json.JSONDecodeError:
                continue
    if len(rows) > max_events:
        rows = rows[-max_events:]
    return rows


def list_recent_ask_events(*, limit: int = 50) -> list[dict[str, Any]]:
    rows = _load_events(max(1, limit * 2))[-limit:]
    rows.reverse()
    out: list[dict[str, Any]] = []
    for r in reversed(rows):
        ts = float(r.get("ts", 0))
        out.append(
            {
                "user_id": r.get("user_id"),
                "question": r.get("question") or "",
                "latency_sec": r.get("latency_sec", 0),
                "web_fallback": bool(r.get("web_fallback")),
                "n_sources": int(r.get("n_sources", 0)),
                "created_at_display": time.strftime(
                    "%Y-%m-%d %H:%M", time.localtime(ts)
                )
                if ts
                else "—",
            }
        )
    return out[:limit]


def list_ask_events_filtered(
    *,
    limit: int = 100,
    source: str = "all",
) -> list[dict[str, Any]]:
    rows = _load_events(5000)
    rows.reverse()
    out: list[dict[str, Any]] = []
    for r in rows:
        wf = bool(r.get("web_fallback"))
        if source == "corpus" and wf:
            continue
        if source == "web" and not wf:
            continue
        uid = r.get("user_id")
        uname = "—"
        if uid:
            try:
                from webapp import db as db_mod

                u = db_mod.get_user_by_id(int(uid))
                if u:
                    uname = u.get("username") or uname
            except Exception:
                pass
        ts = float(r.get("ts", 0))
        out.append(
            {
                "username": uname,
                "question": (r.get("question") or "")[:120],
                "source": "web" if wf else "corpus",
                "score": int(r.get("n_sources", 0)),
                "latency_sec": r.get("latency_sec", 0),
                "created_at": time.strftime("%Y-%m-%d %H:%M", time.localtime(ts))
                if ts
                else "—",
            }
        )
        if len(out) >= limit:
            break
    return out


def export_events_csv() -> str:
    import csv
    import io

    rows = _load_events(10000)
    buf = io.StringIO()
    w = csv.writer(buf)
    w.writerow(
        [
            "ts",
            "user_id",
            "question",
            "latency_sec",
            "n_sources",
            "web_fallback",
            "llm_error",
        ]
    )
    for r in rows:
        w.writerow(
            [
                r.get("ts"),
                r.get("user_id"),
                r.get("question", ""),
                r.get("latency_sec"),
                r.get("n_sources"),
                r.get("web_fallback"),
                r.get("llm_error"),
            ]
        )
    return buf.getvalue()


def _feedback_negative_count() -> int:
    try:
        from webapp import db as db_mod  # noqa: PLC0415

        return db_mod.count_negative_feedback()
    except Exception:
        return 0
