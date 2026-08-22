"""
hybrid_rag/config.py
---------------------
Configuration Hybrid RAG sur le corpus marocain (SGG / Bulletin Officiel).

Note : pour les benchmarks BSARD, ce fichier avait été temporairement basculé sur
data/graph_rag/ et indices_bsard/. Ces valeurs sont conservées en commentaire
en bas du fichier.
"""

import os
from pathlib import Path
from dotenv import load_dotenv

BASE_DIR = Path(__file__).resolve().parent          # app/hybrid_rag/
PROJECT_ROOT = BASE_DIR.parent.parent               # rag-maroc2/
DATA_DIR = PROJECT_ROOT / "data"
INDICES_DIR = BASE_DIR / "indices"

load_dotenv(PROJECT_ROOT / ".env", override=False)

# --- Modèles ---
EMBED_MODEL = os.getenv("EMBED_MODEL", "BAAI/bge-m3")
RERANKER_MODEL = os.getenv("RERANKER_MODEL", "BAAI/bge-reranker-v2-m3")
RERANKER_ENABLED = os.getenv("RERANKER_ENABLED", "true").lower() == "true"

COHERE_API_KEY = os.getenv("COHERE_API_KEY", "")
# command-r a été retiré par Cohere le 15 septembre 2025.
COHERE_GEN_MODEL = os.getenv("COHERE_GEN_MODEL", "command-a-03-2025")

# --- Encodage ---
MAX_SEQ_LENGTH = 512
EMBED_BATCH_SIZE = 32

# --- Retrieval ---
RRF_K = int(os.getenv("RRF_K", 60))
TOP_K_FAISS = int(os.getenv("TOP_K_FAISS", 30))
TOP_K_BM25 = int(os.getenv("TOP_K_BM25", 30))

# Pool envoyé au cross-encoder. Le temps de reranking y est directement
# proportionnel : sur cette machine (CPU, 8 Go), 15 candidats coûtent ~12 s avec
# bge-reranker-v2-m3. Réduit à 10 pour ramener le temps de réponse de la webapp
# à un niveau acceptable, au prix d'une marge de réordonnancement plus étroite.
TOP_K_HYBRID = int(os.getenv("TOP_K_HYBRID", 10))
TOP_K_FINAL = int(os.getenv("TOP_K_FINAL", 10))

# --- Corpus marocain ---
FINAL_CHUNKS_PATH = DATA_DIR / "processed" / "final_chunks.jsonl"

FAISS_INDEX_PATH = INDICES_DIR / "faiss_index.bin"
FAISS_IDMAP_PATH = INDICES_DIR / "faiss_idmap.pkl"
BM25_INDEX_PATH = INDICES_DIR / "bm25_index.pkl"

# --- Configuration BSARD (benchmarks, désactivée) ---
# DATA_DIR = PROJECT_ROOT / "data" / "graph_rag"
# INDICES_DIR = BASE_DIR / "indices_bsard"
# ARTICLES_CSV = DATA_DIR / "articles_clean.csv"
# TRAIN_CSV = DATA_DIR / "train_clean.csv"
# TEST_CSV = DATA_DIR / "test_clean.csv"
# FAISS_INDEX_PATH = INDICES_DIR / "faiss_bsard.bin"
# FAISS_IDMAP_PATH = INDICES_DIR / "faiss_idmap_bsard.pkl"
# BM25_INDEX_PATH = INDICES_DIR / "bm25_bsard.pkl"
# TOP_K_HYBRID = 30