"""Formatage dates et nombres pour le panneau admin."""

from __future__ import annotations

import re
from datetime import datetime, timezone
from typing import Any

_FR_MONTHS = (
    "jan.",
    "fév.",
    "mars",
    "avr.",
    "mai",
    "juin",
    "juil.",
    "août",
    "sept.",
    "oct.",
    "nov.",
    "déc.",
)


def _parse_dt(value: Any) -> datetime | None:
    if value is None or value == "":
        return None
    if isinstance(value, datetime):
        dt = value
        if dt.tzinfo is None:
            return dt.replace(tzinfo=timezone.utc)
        return dt
    s = str(value).strip()
    if not s:
        return None
    s = s.replace("Z", "+00:00")
    if " " in s and "T" not in s and "+" not in s[10:] and s[10:11] != "-":
        s = s.replace(" ", "T", 1)
    try:
        dt = datetime.fromisoformat(s)
    except ValueError:
        m = re.match(
            r"(\d{4})-(\d{2})-(\d{2})(?:[ T](\d{2}):(\d{2})(?::(\d{2}))?)?",
            s,
        )
        if not m:
            return None
        y, mo, d = int(m.group(1)), int(m.group(2)), int(m.group(3))
        h = int(m.group(4) or 0)
        mi = int(m.group(5) or 0)
        sec = int(m.group(6) or 0)
        dt = datetime(y, mo, d, h, mi, sec, tzinfo=timezone.utc)
    if dt.tzinfo is None:
        dt = dt.replace(tzinfo=timezone.utc)
    return dt


def _fr_date(dt: datetime) -> str:
    return f"{dt.day} {_FR_MONTHS[dt.month - 1]} {dt.year}"


def fmt_date(value: Any) -> str:
    dt = _parse_dt(value)
    if not dt:
        return "—"
    return _fr_date(dt)


def fmt_datetime(value: Any) -> str:
    dt = _parse_dt(value)
    if not dt:
        return "—"
    return f"{_fr_date(dt)} · {dt.strftime('%H:%M')}"


def fmt_relative(value: Any) -> str:
    dt = _parse_dt(value)
    if not dt:
        return "—"
    now = datetime.now(timezone.utc)
    diff = now - dt.astimezone(timezone.utc)
    sec = int(diff.total_seconds())
    if sec < 60:
        return "À l'instant"
    if sec < 3600:
        m = sec // 60
        return f"Il y a {m} min"
    if sec < 86400:
        h = sec // 3600
        return f"Il y a {h} h"
    if sec < 172800:
        return f"Hier · {dt.strftime('%H:%M')}"
    if sec < 604800:
        d = sec // 86400
        return f"Il y a {d} j"
    return _fr_date(dt)


def fmt_number(value: Any) -> str:
    try:
        n = int(value)
    except (TypeError, ValueError):
        return str(value)
    return f"{n:,}".replace(",", "\u202f")
