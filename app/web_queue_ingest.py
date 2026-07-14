"""
Ingestion idempotente de la file web fallback → final_chunks.jsonl.

Utilisé par scripts/ingest_web_queue.py (et tests).
"""

from __future__ import annotations

import hashlib
import json
import re
from pathlib import Path
from typing import Any
from urllib.parse import urlparse

from app.config import PROJECT_ROOT

QUEUE_PATH = PROJECT_ROOT / "data" / "processed" / "web_additions_queue.jsonl"
FINAL_CHUNKS_PATH = PROJECT_ROOT / "data" / "processed" / "final_chunks.jsonl"
INGEST_LOG_PATH = PROJECT_ROOT / "data" / "processed" / "web_ingest_log.jsonl"

MIN_TEXT_LEN_DEFAULT = 120
STUB_MAX_LEN = 280


def _norm_url(url: str) -> str:
    u = (url or "").strip().lower().rstrip("/")
    if u.startswith("http://"):
        u = "https://" + u[7:]
    return u


def _text_fingerprint(text: str) -> str:
    t = re.sub(r"\s+", " ", (text or "").strip().lower())[:2000]
    return hashlib.sha256(t.encode("utf-8")).hexdigest()[:16]


def _is_stub(text: str) -> bool:
    t = (text or "").strip()
    if len(t) < MIN_TEXT_LEN_DEFAULT:
        return True
    tl = t.lower()
    if "portail cnie.ma" in tl and len(t) < STUB_MAX_LEN:
        return True
    if "portail watiqa" in tl and len(t) < STUB_MAX_LEN:
        return True
    return False


def _category_from_url(url: str) -> str:
    host = (urlparse(url).netloc or "").lower()
    path = (urlparse(url).path or "").lower()
    if "cnie.ma" in host:
        return "cnie"
    if "passeport.ma" in host or ("consulat.ma" in host and "passeport" in path):
        return "passeport"
    if "watiqa.ma" in host:
        return "etat_civil"
    if "consulat.ma" in host and "cnie" in path:
        return "cnie"
    if host.endswith(".gov.ma") or host in ("maroc.ma", "service-public.ma"):
        return "services_publics"
    return "services_publics"


def _source_org_from_url(url: str) -> str:
    host = (urlparse(url).netloc or "").lower()
    if host.startswith("www."):
        host = host[4:]
    return host or "Web officiel"


def _doc_id_from_url(url: str) -> str:
    host = (urlparse(url).netloc or "web").lower().replace(".", "_")
    h = hashlib.sha256(_norm_url(url).encode()).hexdigest()[:10]
    return f"web_{host}_{h}"


def _chunk_id_from_url(url: str) -> str:
    h = hashlib.sha256(_norm_url(url).encode()).hexdigest()[:12]
    return f"web_ingest_{h}"


def _iter_jsonl(path: Path) -> list[dict[str, Any]]:
    if not path.is_file():
        return []
    rows: list[dict[str, Any]] = []
    with path.open("r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            try:
                rows.append(json.loads(line))
            except json.JSONDecodeError:
                continue
    return rows


def load_corpus_index(path: Path = FINAL_CHUNKS_PATH) -> tuple[set[str], set[str], set[str]]:
    """URLs normalisées, empreintes texte, chunk_id déjà présents."""
    urls: set[str] = set()
    fingerprints: set[str] = set()
    chunk_ids: set[str] = set()
    if not path.is_file():
        return urls, fingerprints, chunk_ids

    from app.corpus_io import _iter_json_objects  # noqa: PLC0415

    raw = path.read_text(encoding="utf-8")
    for record in _iter_json_objects(raw, path):
        if not isinstance(record, dict):
            continue
        cid = str(record.get("chunk_id") or "")
        if cid:
            chunk_ids.add(cid)
        surl = str(record.get("source_url") or "")
        if surl:
            urls.add(_norm_url(surl))
        text = str(record.get("text") or "")
        if text:
            fingerprints.add(_text_fingerprint(text))
    return urls, fingerprints, chunk_ids


def queue_row_to_chunk(
    rec: dict[str, Any],
    *,
    min_text_len: int = MIN_TEXT_LEN_DEFAULT,
) -> dict[str, Any] | None:
    url = str(rec.get("source_url") or "").strip()
    text = str(rec.get("text") or "").strip()
    title = str(rec.get("title") or "Source web officielle").strip()

    if not url or not url.startswith("http"):
        return None
    if len(text) < min_text_len:
        return None
    if _is_stub(text):
        return None

    host = (urlparse(url).netloc or "").lower()
    if not host.endswith(".ma") and ".gov.ma" not in host:
        return None

    category = _category_from_url(url)
    doc_id = _doc_id_from_url(url)
    chunk_id = _chunk_id_from_url(url)
    question = str(rec.get("question") or "").strip()
    label = title[:120] if title else "Web fallback"
    if question and len(question) < 80:
        label = f"{label} — {question[:60]}"

    return {
        "chunk_id": chunk_id,
        "doc_id": doc_id,
        "title": title[:200],
        "source_org": _source_org_from_url(url),
        "source_url": url,
        "filename": doc_id,
        "page_start": 1,
        "source_type": "admin",
        "category": category,
        "label": label[:160],
        "text": text,
        "ingested_from": "web_fallback_queue",
        "ingested_at": rec.get("collected_at"),
    }


def plan_ingest(
    queue_rows: list[dict[str, Any]] | None = None,
    *,
    min_text_len: int = MIN_TEXT_LEN_DEFAULT,
    corpus_path: Path = FINAL_CHUNKS_PATH,
) -> tuple[list[dict[str, Any]], list[dict[str, Any]], dict[str, int]]:
    """
    Retourne (chunks à ajouter, lignes queue rejetées, stats).
    Idempotent : ignore URL / texte / chunk_id déjà dans le corpus.
    """
    if queue_rows is None:
        queue_rows = _iter_jsonl(QUEUE_PATH)

    urls, fingerprints, chunk_ids = load_corpus_index(corpus_path)
    to_add: list[dict[str, Any]] = []
    rejected: list[dict[str, Any]] = []
    stats = {
        "queue_rows": len(queue_rows),
        "added": 0,
        "skipped_duplicate": 0,
        "rejected": 0,
    }
    seen_url_this_run: set[str] = set()
    seen_fp_this_run: set[str] = set()

    for rec in queue_rows:
        chunk = queue_row_to_chunk(rec, min_text_len=min_text_len)
        if chunk is None:
            rejected.append(rec)
            stats["rejected"] += 1
            continue

        url_n = _norm_url(chunk["source_url"])
        fp = _text_fingerprint(chunk["text"])
        cid = chunk["chunk_id"]

        if (
            url_n in urls
            or fp in fingerprints
            or cid in chunk_ids
            or url_n in seen_url_this_run
            or fp in seen_fp_this_run
        ):
            stats["skipped_duplicate"] += 1
            continue

        to_add.append(chunk)
        urls.add(url_n)
        fingerprints.add(fp)
        chunk_ids.add(cid)
        seen_url_this_run.add(url_n)
        seen_fp_this_run.add(fp)
        stats["added"] += 1

    return to_add, rejected, stats


def append_chunks(chunks: list[dict[str, Any]], path: Path = FINAL_CHUNKS_PATH) -> int:
    if not chunks:
        return 0
    path.parent.mkdir(parents=True, exist_ok=True)
    tail = path.read_text(encoding="utf-8") if path.is_file() else ""
    with path.open("a", encoding="utf-8") as f:
        if tail and not tail.endswith("\n"):
            f.write("\n")
        for row in chunks:
            f.write(json.dumps(row, ensure_ascii=False) + "\n")
    return len(chunks)


def log_ingest(chunks: list[dict[str, Any]], stats: dict[str, int]) -> None:
    INGEST_LOG_PATH.parent.mkdir(parents=True, exist_ok=True)
    entry = {
        "stats": stats,
        "chunk_ids": [c["chunk_id"] for c in chunks],
    }
    with INGEST_LOG_PATH.open("a", encoding="utf-8") as f:
        f.write(json.dumps(entry, ensure_ascii=False) + "\n")


def rewrite_queue(keep_rows: list[dict[str, Any]], path: Path = QUEUE_PATH) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as f:
        for rec in keep_rows:
            f.write(json.dumps(rec, ensure_ascii=False) + "\n")
