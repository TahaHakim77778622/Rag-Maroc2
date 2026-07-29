"""Nettoyage de texte extrait (PDF / web) avant envoi au LLM."""

from __future__ import annotations

import re
import unicodedata


def sanitize_text_for_llm(text: str, *, max_len: int = 6000) -> str:
    """Supprime caractères illisibles (ex. U+FFFD) et contrôles parasites."""
    if not text:
        return ""
    t = unicodedata.normalize("NFC", str(text))
    t = t.replace("\ufffd", " ")
    t = "".join(
        c if (c.isprintable() or c in "\n\t") and unicodedata.category(c) != "Co"
        else " "
        for c in t
    )
    t = re.sub(r"\s+", " ", t).strip()
    if max_len and len(t) > max_len:
        t = t[: max_len - 1] + "…"
    return t


def text_is_usable_for_llm(text: str, *, max_bad_ratio: float = 0.02) -> bool:
    if not (text or "").strip():
        return False
    bad = text.count("\ufffd")
    return (bad / max(1, len(text))) <= max_bad_ratio


def sanitize_hit(hit: dict) -> dict:
    """Retourne une copie du hit avec texte nettoyé."""
    out = dict(hit)
    raw = str(hit.get("text") or "")
    clean = sanitize_text_for_llm(raw)
    if clean:
        out["text"] = clean
    return out
