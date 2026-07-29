"""
Chat minimal en terminal : retrieval + prompt + LLM (mock par défaut).

Usage (depuis la racine du projet) :
    python -m app.chatbot
"""

from __future__ import annotations

import logging
import sys

from app.Rag_classique.llm_factory import get_llm_client
from app.Rag_classique.rag_pipeline import RAGPipeline

logging.basicConfig(level=logging.INFO, format="%(levelname)s: %(message)s")
logger = logging.getLogger(__name__)


def main() -> int:
    logger.info("Initialisation du RAG (chargement index + corpus, peut prendre ~1 min)…")
    pipeline = RAGPipeline(llm=get_llm_client())

    print("RAG-MAROC2 — tapez votre question (quit / exit pour quitter).\n")
    while True:
        try:
            question = input("Vous > ").strip()
        except (EOFError, KeyboardInterrupt):
            print()
            break
        if not question:
            continue
        if question.lower() in ("quit", "exit", "q"):
            break

        out = pipeline.answer(question)
        print("\nAssistant >", out["answer"])
        print("\nSources :")
        for line in out["sources_display"]:
            print(line)
        print()

    return 0


if __name__ == "__main__":
    try:
        sys.exit(main())
    except Exception as e:  # noqa: BLE001 — point d’entrée CLI
        logger.exception("%s", e)
        sys.exit(1)
