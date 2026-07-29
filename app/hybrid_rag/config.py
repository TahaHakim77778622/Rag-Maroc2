"""
hybrid_rag/config.py
---------------------
Configuration de l'architecture Hybrid RAG (isolée du RAG classique).
"""

import os
from pathlib import Path
from dotenv import load_dotenv

BASE_DIR = Path(__file__).resolve().parent #config.py
PROJECT_ROOT = BASE_DIR.parent.parent #RAG-maroc2
DATA_DIR = PROJECT_ROOT / "data"
INDICES_DIR = BASE_DIR / "indices"

load_dotenv(PROJECT_ROOT / ".env", override=False)#pour que le chargement du env soit fiable 

EMBED_MODEL = os.getenv("EMBED_MODEL", "BAAI/bge-m3")
RERANKER_MODEL = os.getenv("RERANKER_MODEL", "BAAI/bge-reranker-v2-m3")
RERANKER_ENABLED = os.getenv("RERANKER_ENABLED", "true").lower() == "true"

COHERE_API_KEY = os.getenv("COHERE_API_KEY", "")
COHERE_GEN_MODEL = os.getenv("COHERE_GEN_MODEL", "command-r")

RRF_K = int(os.getenv("RRF_K", 60))

TOP_K_FAISS = int(os.getenv("TOP_K_FAISS", 30))
TOP_K_BM25 = int(os.getenv("TOP_K_BM25", 30))
TOP_K_HYBRID = int(os.getenv("TOP_K_HYBRID", 15))
TOP_K_FINAL = int(os.getenv("TOP_K_FINAL", 5))

FINAL_CHUNKS_PATH = DATA_DIR / "processed" / "final_chunks.jsonl"

FAISS_INDEX_PATH = INDICES_DIR / "faiss_index.bin"
FAISS_IDMAP_PATH = INDICES_DIR / "faiss_idmap.pkl"#FAISS seul ne sait pas dire "le vecteur n°42 correspond à quel texte
BM25_INDEX_PATH = INDICES_DIR / "bm25_index.pkl"