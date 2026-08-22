"""
Configuration du module GraphRAG pour le corpus marocain (SGG / Bulletin Officiel).

Différences avec la version BSARD :
  - Nœuds = chunks, pas articles entiers (le corpus est déjà découpé)
  - Hiérarchie à 2 niveaux (Document > Chunk) au lieu de 7
  - Un renvoi "article X" peut pointer vers PLUSIEURS chunks du même document
    (un article long est découpé en morceaux)
  - Les visas externes ("Vu la loi n° 43-05...") sont ignorés : ils pointent
    majoritairement vers des textes absents du corpus
"""
from pathlib import Path

BASE_DIR = Path(__file__).resolve().parent          # app/hybrid_graph_rag/graph_rag/
PROJECT_ROOT = BASE_DIR.parent.parent.parent        # rag-maroc2/

CHUNKS_PATH = PROJECT_ROOT / "data" / "processed" / "final_chunks.jsonl"
KUZU_DB_PATH = str(BASE_DIR / "indices" / "kuzu_db")

# Extraction
LOG_EVERY = 5000

# Traversée
GRAPH_MAX_HOPS = 2
N_SEEDS = 3                  # nb de candidats servant de point de départ
USE_REFERENCES = True
USE_SAME_DOC = True          # équivalent des "siblings" de BSARD
MAX_SAME_DOC = 15            # limite le voisinage pour éviter le bruit

# Retrieval
TOP_K = 10