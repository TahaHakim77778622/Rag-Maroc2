

from typing import Dict, List
import cohere

from . import config
from .data_loader import load_corpus
from .embeddings import search_faiss
from .bm25_retriever import search_bm25
from .fusion import reciprocal_rank_fusion
from .reranker import rerank

_co_client = None
_chunks_cache = None


def get_cohere_client():
    global _co_client
    if _co_client is None:
        _co_client = cohere.Client(config.COHERE_API_KEY)
    return _co_client


def get_chunks():
    global _chunks_cache
    if _chunks_cache is None:
        _chunks_cache = load_corpus(config.FINAL_CHUNKS_PATH)
    return _chunks_cache


def retrieve(query: str) -> List[Dict]:
    """Étape retrieve + fuse + rerank. Retourne les chunks complets classés avec leur score."""
    chunks = get_chunks()
    chunks_by_id = {c["chunk_id"]: c for c in chunks}

    faiss_results = search_faiss(query, top_k=config.TOP_K_FAISS)
    bm25_results = search_bm25(query, top_k=config.TOP_K_BM25)

    fused = reciprocal_rank_fusion(faiss_results, bm25_results, top_k=config.TOP_K_HYBRID)
    final = rerank(query, fused, chunks_by_id, top_k=config.TOP_K_FINAL)

    return [
        {**chunks_by_id[chunk_id], "score": score}
        for chunk_id, score in final
        if chunk_id in chunks_by_id
    ]


def build_prompt(query: str, retrieved_chunks: List[Dict]) -> str:
    context = "\n\n".join(
        f"[Source {i + 1} - {c.get('title', 'N/A')}]\n{c['text']}"
        for i, c in enumerate(retrieved_chunks)
    )
    return f"""Tu es un assistant juridique spécialisé dans les textes légaux et administratifs marocains.
Réponds à la question en te basant UNIQUEMENT sur le contexte fourni ci-dessous.
Si le contexte ne permet pas de répondre, dis-le clairement.

Contexte :
{context}

Question : {query}

Réponse :"""


def generate_answer(query: str, retrieved_chunks: List[Dict]) -> str:
    co = get_cohere_client()
    prompt = build_prompt(query, retrieved_chunks)

    response = co.chat(
        model=config.COHERE_GEN_MODEL,
        message=prompt,
    )
    return response.text


def answer_query(query: str) -> Dict:
    """Point d'entrée complet : retrieve + generate. Retourne réponse + sources."""
    retrieved = retrieve(query)
    answer = generate_answer(query, retrieved)
    return {
        "query": query,
        "answer": answer,
        "sources": retrieved,
    }