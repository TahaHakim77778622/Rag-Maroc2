#!/usr/bin/env python3
"""
Filtre les chunks inutilisables dans data/processed/final_chunks.jsonl.

Usage (racine du projet) :
    python scripts/filter_chunks.py
"""

from __future__ import annotations

import json
import re
import shutil
import unicodedata
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parent.parent
CHUNKS_PATH = PROJECT_ROOT / "data" / "processed" / "final_chunks.jsonl"
BACKUP_PATH = PROJECT_ROOT / "data" / "processed" / "final_chunks_backup.jsonl"

MIN_CHARS = 80
PUBLICATION_MAX_CHARS = 150

_PUBLICATION_PATTERNS = (
    "le present arrete sera publie au bulletin officiel",
    "le present decret sera publie au bulletin officiel",
    "sera publie au bulletin officiel",
    "prend effet a compter",
    "entrera en vigueur",
    "sont et demeurent abrogees",
    "sont abrogees",
    "qui sera publie au bulletin officiel",
)


def _fold(s: str) -> str:
    return "".join(
        c for c in unicodedata.normalize("NFD", s or "")
        if unicodedata.category(c) != "Mn"
    ).lower()


def _is_curated(chunk: dict) -> bool:
    cid = str(chunk.get("chunk_id") or "")
    return cid.startswith("curated::")


def _arabic_ratio(text: str) -> float:
    if not text:
        return 0.0
    arabic = 0
    letters = 0
    for c in text:
        if c.isalpha():
            letters += 1
            o = ord(c)
            if 0x0600 <= o <= 0x06FF or 0x0750 <= o <= 0x077F or 0x08A0 <= o <= 0x08FF:
                arabic += 1
    return arabic / max(letters, 1)


def _only_publication_formula(text: str) -> bool:
    if len(text) >= PUBLICATION_MAX_CHARS:
        return False
    folded = _fold(text)
    if not any(p in folded for p in _PUBLICATION_PATTERNS):
        return False
    remainder = folded
    for p in _PUBLICATION_PATTERNS:
        remainder = remainder.replace(p, " ")
    remainder = re.sub(r"art\.?\s*\d+[^a-z]*", " ", remainder)
    remainder = re.sub(r"article\s+(premier|\d+)[^a-z]*", " ", remainder)
    remainder = re.sub(r"[^a-z0-9]+", " ", remainder).strip()
    return len(remainder) < 35


def _bo_page_spam_only(text: str) -> bool:
    bo_matches = re.findall(
        r"(?i)\d*\s*bulletin\s+officiel\s+n[º°o]?\s*\d+",
        text,
    )
    if len(bo_matches) <= 3:
        return False
    stripped = re.sub(
        r"(?i)\d*\s*bulletin\s+officiel\s+n[º°o]?\s*[^.\n]*",
        " ",
        text,
    )
    stripped = re.sub(r"\s+", " ", stripped).strip()
    return len(stripped) < 80


def _should_remove(chunk: dict) -> tuple[bool, str]:
    if _is_curated(chunk):
        return False, ""

    text = (chunk.get("text") or "").strip()
    if not text:
        return True, "empty"

    if len(text) < MIN_CHARS:
        return True, "too_short"

    if _only_publication_formula(text):
        return True, "publication_only"

    if _bo_page_spam_only(text):
        return True, "bo_page_spam"

    if _arabic_ratio(text) > 0.60:
        return True, "arabic_majority"

    return False, ""


def main() -> int:
    if not CHUNKS_PATH.is_file():
        print(f"Fichier introuvable : {CHUNKS_PATH}")
        return 1

    chunks: list[dict] = []
    with CHUNKS_PATH.open(encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if line:
                chunks.append(json.loads(line))

    shutil.copy2(CHUNKS_PATH, BACKUP_PATH)
    print(f"Backup : {BACKUP_PATH}")

    kept: list[dict] = []
    removed_by_reason: dict[str, int] = {}

    for ch in chunks:
        drop, reason = _should_remove(ch)
        if drop:
            removed_by_reason[reason] = removed_by_reason.get(reason, 0) + 1
        else:
            kept.append(ch)

    removed = len(chunks) - len(kept)
    with CHUNKS_PATH.open("w", encoding="utf-8") as f:
        for ch in kept:
            f.write(json.dumps(ch, ensure_ascii=False) + "\n")

    print("===== FILTER CHUNKS =====")
    print(f"Entrée     : {len(chunks)}")
    print(f"Supprimés  : {removed}")
    for reason, n in sorted(removed_by_reason.items(), key=lambda x: -x[1]):
        print(f"  - {reason}: {n}")
    print(f"Conservés  : {len(kept)}")
    print(f"Sauvegardé : {CHUNKS_PATH}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
