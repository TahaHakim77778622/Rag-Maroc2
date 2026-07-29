"""
Query expansion pour retrieval — délégué à app.query_understanding.
"""

from __future__ import annotations

from app.Rag_classique.query_understanding import (
    analyze_query,
    build_retrieval_query,
    expand_query_for_retrieval,
    needs_conversation_context,
    resolve_question,
    _is_ambiguous_short_query,
)

__all__ = [
    "analyze_query",
    "build_retrieval_query",
    "expand_query_for_retrieval",
    "needs_conversation_context",
    "resolve_question",
    "_is_ambiguous_short_query",
]
