"""Cas passeport : perte/vol, renouvellement, première demande."""

from __future__ import annotations

import re
import unicodedata
from typing import Any


def _fold(s: str) -> str:
    return "".join(
        c for c in unicodedata.normalize("NFD", s or "") if unicodedata.category(c) != "Mn"
    ).lower()


def passeport_case_from_question(question: str) -> str | None:
    q = _fold(question)
    if "passeport" not in q:
        return None
    if any(
        k in q
        for k in (
            "perte",
            "perdu",
            "perdue",
            "vol",
            "vole",
            "volee",
            "declare",
            "déclarer",
            "declarer",
            "declaration",
            "déclaration",
        )
    ):
        return "perte_vol"
    if any(k in q for k in ("renouvel", "refaire", "expiration", "expire")):
        return "renouvellement"
    return "demande"


def chunk_passeport_case(hit: dict[str, Any]) -> str | None:
    blob = _fold(
        " ".join(
            (
                str(hit.get("text") or ""),
                str((hit.get("metadata") or {}).get("label") or ""),
                str((hit.get("metadata") or {}).get("chunk_id") or ""),
            )
        )
    )
    if any(
        x in blob
        for x in (
            "perte",
            " vol ",
            "declaration sur l honneur",
            "déclaration sur l'honneur",
            "altération",
        )
    ):
        return "perte_vol"
    if "renouvel" in blob or "ancien passeport" in blob:
        return "renouvellement"
    return "demande"


def passeport_case_aligned(question: str, hit: dict[str, Any]) -> bool:
    wanted = passeport_case_from_question(question)
    if not wanted:
        return True
    got = chunk_passeport_case(hit)
    if wanted == "perte_vol":
        return got == "perte_vol" or "perte" in _fold(hit.get("text") or "")
    if wanted == "renouvellement":
        return got in ("renouvellement", "demande", "perte_vol")
    return True
