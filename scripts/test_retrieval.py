"""
Smoke test : une requête, vérifie que le retriever renvoie des hits.

Usage (racine du projet) :
    python scripts/test_retrieval.py
"""

from __future__ import annotations

import logging
import sys
from pathlib import Path

# Permet « python scripts/test_retrieval.py » sans PYTHONPATH
_ROOT = Path(__file__).resolve().parent.parent
if str(_ROOT) not in sys.path:
    sys.path.insert(0, str(_ROOT))

logging.basicConfig(level=logging.INFO, format="%(levelname)s: %(message)s")
logger = logging.getLogger(__name__)


def main() -> int:
    from app.Rag_classique.retriever import Retriever  # noqa: PLC0415

    r = Retriever()
    q = "bulletin officiel arrêté licence"
    hits = r.search(q, k=3)
    assert hits, "Aucun résultat"
    logger.info("OK : %s hits, premier score=%.4f", len(hits), hits[0]["score"])
    for h in hits:
        logger.info("  %s | %s", h["metadata"].get("chunk_id"), h["score"])
    return 0


if __name__ == "__main__":
    try:
        sys.exit(main())
    except Exception as e:  # noqa: BLE001
        logger.exception("%s", e)
        sys.exit(1)
