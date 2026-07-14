#!/usr/bin/env python3
"""
Fusionne les chunks BO courts consécutifs du même document.

Usage (racine du projet) :
    python scripts/merge_short_chunks.py
"""

from __future__ import annotations

import json
import re
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parent.parent
CHUNKS_PATH = PROJECT_ROOT / "data" / "processed" / "final_chunks.jsonl"

MAX_MERGE_GROUP = 3
SHORT_MAX_CHARS = 300


def _is_curated(chunk: dict) -> bool:
    return str(chunk.get("chunk_id") or "").startswith("curated::")


def _unit_index(chunk_id: str) -> int | None:
    m = re.search(r"_unit_(\d+)(?:_|$)", chunk_id)
    return int(m.group(1)) if m else None


def _can_merge(a: dict, b: dict) -> bool:
    if _is_curated(a) or _is_curated(b):
        return False
    if a.get("doc_id") != b.get("doc_id"):
        return False
    if (a.get("source_type") or "") != "bulletin_officiel":
        return False
    if (b.get("source_type") or "") != "bulletin_officiel":
        return False
    ua, ub = _unit_index(str(a.get("chunk_id") or "")), _unit_index(str(b.get("chunk_id") or ""))
    if ua is None or ub is None:
        return False
    return ub == ua + 1


def _is_short(chunk: dict) -> bool:
    if _is_curated(chunk):
        return False
    return len((chunk.get("text") or "").strip()) < SHORT_MAX_CHARS


def _merge_group(group: list[dict]) -> dict:
    first = dict(group[0])
    texts = [(g.get("text") or "").strip() for g in group]
    labels = [str(g.get("label") or "").strip() for g in group if g.get("label")]
    first["chunk_id"] = f"{first['chunk_id']}_merged"
    first["text"] = "\n\n".join(t for t in texts if t)
    if len(labels) > 1:
        first["label"] = " / ".join(labels[:3])
    return first


def merge_chunks_in_doc(
    doc_chunks: list[dict],
) -> tuple[list[dict], dict[str, dict], set[str], int]:
    """
    Fusionne jusqu'à 3 chunks courts consécutifs.
    Retourne (liste fusionnée par doc, first_id→merged, ids absorbés, nb chunks absorbés).
    """
    out: list[dict] = []
    merged_by_first: dict[str, dict] = {}
    absorbed: set[str] = set()
    absorbed_count = 0
    i = 0
    while i < len(doc_chunks):
        ch = doc_chunks[i]
        if not _is_short(ch) or _is_curated(ch) or (ch.get("source_type") or "") != "bulletin_officiel":
            out.append(ch)
            i += 1
            continue

        group = [ch]
        j = i + 1
        while len(group) < MAX_MERGE_GROUP and j < len(doc_chunks):
            nxt = doc_chunks[j]
            if _is_short(nxt) and _can_merge(group[-1], nxt):
                group.append(nxt)
                j += 1
            else:
                break

        if len(group) > 1:
            first_id = str(group[0].get("chunk_id") or "")
            merged = _merge_group(group)
            merged_by_first[first_id] = merged
            for g in group[1:]:
                absorbed.add(str(g.get("chunk_id") or ""))
            absorbed_count += len(group) - 1
            out.append(merged)
        else:
            out.append(ch)
        i = j
    return out, merged_by_first, absorbed, absorbed_count


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

    before = len(chunks)
    by_doc: dict[str, list[dict]] = {}

    for ch in chunks:
        if _is_curated(ch):
            continue
        doc_id = str(ch.get("doc_id") or "")
        by_doc.setdefault(doc_id, []).append(ch)

    merged_by_first_id: dict[str, dict] = {}
    absorbed_ids: set[str] = set()
    merge_ops = 0

    for doc_list in by_doc.values():
        doc_list.sort(key=lambda c: (_unit_index(str(c.get("chunk_id") or "")) or 0))
        _, mmap, absorbed, n_abs = merge_chunks_in_doc(doc_list)
        merged_by_first_id.update(mmap)
        absorbed_ids |= absorbed
        merge_ops += n_abs

    output: list[dict] = []
    for ch in chunks:
        cid = str(ch.get("chunk_id") or "")
        if cid in absorbed_ids:
            continue
        if cid in merged_by_first_id:
            output.append(merged_by_first_id[cid])
        else:
            output.append(ch)

    with CHUNKS_PATH.open("w", encoding="utf-8") as f:
        for ch in output:
            f.write(json.dumps(ch, ensure_ascii=False) + "\n")

    print("===== MERGE SHORT CHUNKS =====")
    print(f"Entrée        : {before}")
    print(f"Fusions       : {merge_ops} chunks absorbés")
    print(f"Sortie        : {len(output)}")
    print(f"Sauvegardé    : {CHUNKS_PATH}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
