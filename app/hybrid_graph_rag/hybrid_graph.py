"""
Fusion Hybrid RAG + GraphRAG pour le corpus marocain (SGG / Bulletin Officiel).

Reproduit l'architecture validée sur BSARD (MRR +0.012 à 96 %, F1 +0.005 à 97 %) :
  1. Hybrid RAG produit un pool de candidats (dense + sparse -> fusion RRF)
  2. Le graphe étend ce pool via les renvois entre articles et le voisinage
     dans le même texte juridique
  3. Le cross-encoder tranche sur l'ensemble élargi
  4. Cohere génère la réponse à partir des chunks retenus

Le graphe n'écarte jamais de document : il enrichit le pool et laisse le reranker
décider. C'est ce qui garantit qu'il ne peut pas dégrader le rappel.
"""

import os
# Doit précéder tout import de torch ou faiss : les deux embarquent leur propre
# OpenMP. Passer OMP_NUM_THREADS au-delà de 1 fait crasher FAISS (segfault) ;
# le nombre de threads de PyTorch est donc réglé séparément juste après, car
# c'est lui qui porte le coût du reranking.
os.environ.setdefault("KMP_DUPLICATE_LIB_OK", "TRUE")
os.environ.setdefault("OMP_NUM_THREADS", "1")

import torch
torch.set_num_threads(4)

import sys
from pathlib import Path

BASE_DIR = Path(__file__).resolve().parent
sys.path.insert(0, str(BASE_DIR / "graph_rag"))

from app.hybrid_rag import config as hcfg
from app.hybrid_rag.data_loader import load_corpus
from app.hybrid_rag.embeddings import search_faiss
from app.hybrid_rag.bm25_retriever import search_bm25
from app.hybrid_rag.fusion import reciprocal_rank_fusion
from app.hybrid_rag.reranker import rerank

from graph_retriever import GraphRetriever

# Nombre de candidats du pool servant de seeds à l'expansion graphe.
# Valeur reprise de la configuration validée sur BSARD.
N_SEEDS = 3
TOP_K_FINAL = 10

# Plafond de candidats ajoutés par le graphe. Chaque candidat supplémentaire est
# scoré par le cross-encoder, donc pèse directement sur le temps de réponse.
MAX_EXTRA = 5

_chunks_cache = None
_chunks_by_id = None
_graph = None
_co_client = None


# =====================================================================
# Chargement paresseux
# =====================================================================

def get_chunks():
    global _chunks_cache
    if _chunks_cache is None:
        _chunks_cache = load_corpus(hcfg.FINAL_CHUNKS_PATH)
    return _chunks_cache


def get_chunks_by_id():
    """Index chunk_id -> chunk, construit une seule fois.

    Sans ce cache, le dictionnaire de 18 760 entrées était reconstruit à chaque
    recherche.
    """
    global _chunks_by_id
    if _chunks_by_id is None:
        _chunks_by_id = {c["chunk_id"]: c for c in get_chunks()}
    return _chunks_by_id


def get_graph():
    global _graph
    if _graph is None:
        _graph = GraphRetriever()
    return _graph


def get_cohere_client():
    global _co_client
    if _co_client is None:
        import cohere
        _co_client = cohere.Client(hcfg.COHERE_API_KEY)
    return _co_client


# =====================================================================
# Retrieval
# =====================================================================

def _base_pool(query):
    """Pool de candidats Hybrid : dense + sparse -> fusion RRF."""
    faiss_res = search_faiss(query, top_k=hcfg.TOP_K_FAISS)
    bm25_res = search_bm25(query, top_k=hcfg.TOP_K_BM25)
    return reciprocal_rank_fusion(faiss_res, bm25_res, top_k=hcfg.TOP_K_HYBRID)


def retrieve_hybrid(query, top_k=TOP_K_FINAL):
    """Baseline : dense + sparse -> RRF -> reranking."""
    chunks_by_id = get_chunks_by_id()
    fused = _base_pool(query)
    final = rerank(query, fused, chunks_by_id, top_k=top_k)
    return [{**chunks_by_id[cid], "score": s} for cid, s in final if cid in chunks_by_id]


def retrieve_hybrid_graph(query, top_k=TOP_K_FINAL, n_seeds=N_SEEDS,
                          use_references=True, use_same_texte=True,
                          max_extra=MAX_EXTRA):
    """Hybrid + expansion graphe, puis reranking sur le pool élargi."""
    chunks_by_id = get_chunks_by_id()
    fused = _base_pool(query)
    pool_ids = [cid for cid, _ in fused]

    graph = get_graph()
    expanded = graph.expand(
        pool_ids[:n_seeds],
        use_references=use_references,
        use_same_doc=use_same_texte,
    )

    # On ne garde que les candidats les mieux scorés par le graphe.
    ranked_extra = sorted(
        ((cid, info["score"]) for cid, info in expanded.items()
         if cid not in pool_ids and cid in chunks_by_id),
        key=lambda x: x[1], reverse=True,
    )[:max_extra]

    # Score amont nul pour les candidats du graphe : seul le cross-encoder décide.
    enlarged = fused + [(cid, 0.0) for cid, _ in ranked_extra]

    final = rerank(query, enlarged, chunks_by_id, top_k=top_k)
    return [{**chunks_by_id[cid], "score": s} for cid, s in final if cid in chunks_by_id]


# =====================================================================
# Génération
# =====================================================================

def build_prompt(query, retrieved_chunks):
    context = "\n\n".join(
        f"[Source {i+1} - {c.get('title', 'N/A')}]\n{c['text']}"
        for i, c in enumerate(retrieved_chunks)
    )
    return f"""Tu es un assistant juridique spécialisé dans les textes légaux et
administratifs MAROCAINS.

RÈGLES IMPÉRATIVES :

1. Réponds UNIQUEMENT à partir du contexte ci-dessous. N'utilise aucune
   connaissance extérieure, même si tu penses connaître la réponse.

2. Avant de répondre, vérifie que les extraits traitent bien du SUJET de la
   question. S'ils portent sur un autre domaine — par exemple des marchés
   publics alors qu'on interroge sur l'état civil ou le permis de conduire —
   considère que tu n'as pas l'information.

3. Si le contexte ne contient pas l'information, réponds exactement :
   "Je n'ai pas trouvé d'information à ce sujet dans les textes dont je dispose."
   N'ajoute rien, ne propose aucune réponse approximative.

4. Si la question porte sur un AUTRE PAYS que le Maroc, réponds exactement :
   "Je traite uniquement la législation et l'administration marocaines."
   Ne transpose jamais la question au cas marocain de ta propre initiative.

5. Si la question ne relève ni du droit ni de l'administration, réponds :
   "Cette question sort de mon domaine."

6. Ne devine jamais. Une réponse inventée est plus nuisible qu'une absence de
   réponse.

Contexte :
{context}

Question : {query}

Réponse :"""


def generate_answer(query, retrieved_chunks):
    co = get_cohere_client()
    response = co.chat(
        model=hcfg.COHERE_GEN_MODEL,
        message=build_prompt(query, retrieved_chunks),
    )
    return response.text


def answer_query(query, use_graph=True, top_k=TOP_K_FINAL):
    """
    Point d'entrée complet : retrieval puis génération.

    use_graph=False permet de comparer avec la baseline Hybrid seule.
    Retourne {"query", "answer", "sources"} — même signature que
    app.hybrid_rag.pipeline.answer_query.
    """
    retrieved = (retrieve_hybrid_graph(query, top_k=top_k) if use_graph
                 else retrieve_hybrid(query, top_k=top_k))
    return {
        "query": query,
        "answer": generate_answer(query, retrieved),
        "sources": retrieved,
    }


# =====================================================================

if __name__ == "__main__":
    import time

    questions = [
        "identification du beneficiaire effectif",
        "declaration de soupcon aupres de l unite de traitement",
        "formation du personnel en matiere de vigilance",
    ]

    print("Préchauffage (chargement des modèles, du corpus et du graphe)...")
    retrieve_hybrid_graph(questions[0])

    print("\n=== Temps par requête ===")
    for q in questions:
        t0 = time.time(); retrieve_hybrid(q); th = time.time() - t0
        t0 = time.time(); res = retrieve_hybrid_graph(q); tg = time.time() - t0
        print(f"Hybrid {th:>5.1f}s | +Graphe {tg:>5.1f}s | {q[:45]}")

    print("\n=== Top 5 (dernière question) ===")
    for i, c in enumerate(res[:5], 1):
        print(f"{i}. [{c['score']:.3f}] {c['chunk_id']:<24} {c['text'][:70]}")