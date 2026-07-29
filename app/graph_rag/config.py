from pathlib import Path

BASE_DIR = Path(__file__).resolve().parent            # app/graph_rag
PROJECT_ROOT = BASE_DIR.parent.parent                  # rag-maroc2
DATA_DIR = PROJECT_ROOT / "data" / "graph_rag"         # CSV sources du graphe (à déposer ici)
INDICES_DIR = BASE_DIR / "indices"

KUZU_DB_PATH = str(INDICES_DIR / "kuzu_db")

ARTICLES_CSV = str(DATA_DIR / "articles_clean.csv")
TRAIN_CSV = str(DATA_DIR / "train_clean.csv")
TEST_CSV = str(DATA_DIR / "test_clean.csv")

HIERARCHY_LEVELS = ["book", "part", "act", "chapter", "section", "subsection"]

CROSS_REF_BATCH_LOG_EVERY = 5000

GRAPH_WEIGHT = 0.3
BASELINE_WEIGHT = 0.7

TOP_K = 10  # cohérent avec Hit Rate@10 / MRR@10 utilisés dans le protocole de benchmark
GRAPH_MAX_HOPS = 2  # profondeur de traversée du graphe pour le graph_retriever (mode legacy)

# --- GraphRAG fusion (post-ablation) ---
# Modes: "noop" | "bidir_soft" | "controlled_expand" | "controlled_expand_bidir" | "expand_with_siblings"
# Défaut = C4 (meilleure ablation : ΔHR=+0.018, ΔMRR=+0.007, ΔF1=+0.017)
GRAPH_MODE = "expand_with_siblings"
BM25_POOL = 50          # pool lexical dans lequel le graphe peut promouvoir des candidats
EXPAND_N_SEEDS = 3      # seeds BM25 pour l'expansion REFERENCE contrôlée
BIDIR_ALPHA = 0.02      # boost soft borné (fraction du score RRF) pour le degré bidir induit
GRAPH_LIST_WEIGHT = 1.0 # poids RRF de la liste graphe dans controlled_expand
USE_SIBLINGS_DEFAULT = True  # siblings gated au pool BM25 (C4) — utiles, contrairement au re-rank libre
