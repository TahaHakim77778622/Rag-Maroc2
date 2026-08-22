"""
Adaptateur : expose l'interface attendue par RAGPipeline, mais fait le retrieval
avec Hybrid RAG + GraphRAG.

RAGPipeline n'utilise que deux choses de son retriever : la propriété n_vectors
et la méthode search(query, k). En fournissant un objet compatible, on remplace
le moteur de recherche sans toucher à la logique conversationnelle de la webapp
(salutations, questions vagues, hors-sujet, reformulation avec l'historique,
références précises...).

Branchement dans webapp/rag_service.py :
    from app.hybrid_graph_rag.retriever_adapter import HybridGraphRetrieverAdapter
    _pipeline = RAGPipeline(llm=get_llm_client(),
                            retriever=HybridGraphRetrieverAdapter())
"""

import os
# Voir hybrid_graph.py : OMP_NUM_THREADS=1 protège FAISS du segfault OpenMP,
# les threads de PyTorch sont réglés séparément.
os.environ.setdefault("KMP_DUPLICATE_LIB_OK", "TRUE")
os.environ.setdefault("OMP_NUM_THREADS", "1")

import torch
torch.set_num_threads(4)

from typing import Any

from app.hybrid_graph_rag.hybrid_graph import (
    retrieve_hybrid,
    retrieve_hybrid_graph,
    get_chunks,
)

# Plafond du nombre de candidats retournés. RAGPipeline peut demander un pool
# large ; le cross-encoder devant scorer chaque candidat, on borne le coût.
MAX_K = 10


class HybridGraphRetrieverAdapter:
    """Retriever compatible RAGPipeline, propulsé par Hybrid RAG + GraphRAG."""

    def __init__(self, use_graph: bool = True):
        """use_graph=False permet de comparer avec la baseline Hybrid seule."""
        self.use_graph = use_graph
        self._n = None

    @property
    def n_vectors(self) -> int:
        """Nombre de chunks indexés — RAGPipeline s'en sert pour plafonner son pool."""
        if self._n is None:
            self._n = len(get_chunks())
        return self._n

    @property
    def embedding_model(self) -> str:
        from app.hybrid_rag import config as hcfg
        return hcfg.EMBED_MODEL

    def search(self, query: str, k: int = 5) -> list[dict[str, Any]]:
        """
        Retourne [{index, score, metadata, text}, ...] — même format que
        app.Rag_classique.retriever.Retriever.search().
        """
        fn = retrieve_hybrid_graph if self.use_graph else retrieve_hybrid
        results = fn(query, top_k=min(max(1, k), MAX_K))

        hits: list[dict[str, Any]] = []
        for i, c in enumerate(results):
            meta = {key: val for key, val in c.items() if key not in ("text", "score")}
            hits.append({
                "index": i,
                "score": float(c.get("score", 0.0)),
                "metadata": meta,
                "text": c.get("text", ""),
            })
        return hits