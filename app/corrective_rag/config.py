"""
Configuration du module Corrective RAG (CRAG) — évaluation sur BSARD.

CRAG n'est pas un retriever : il évalue les documents fournis par un retriever
en amont, puis agit selon trois niveaux de confiance.
  - correct   : documents jugés fiables      -> raffinement
  - ambiguous : signal incertain             -> raffinement permissif
  - incorrect : aucun document convaincant   -> voir INCORRECT_STRATEGY

Retriever de base retenu : Hybrid RAG (HR@10 = 0.586 sur test_clean.csv),
nettement supérieur à BM25 seul (0.414). CRAG sans branche web ne pouvant que
filtrer, il lui faut un retriever qui trouve déjà de bons articles.
"""
from pathlib import Path

# =====================================================================
# Chemins
# =====================================================================
MODULE_DIR = Path(__file__).parent
PROJECT_ROOT = MODULE_DIR.parent.parent
DATA_DIR = PROJECT_ROOT / "data" / "graph_rag"

ARTICLES_CSV = str(DATA_DIR / "articles_clean.csv")
TRAIN_CSV = str(DATA_DIR / "train_clean.csv")
TEST_CSV = str(DATA_DIR / "test_clean.csv")

RESULTS_DIR = MODULE_DIR / "resultats"


# =====================================================================
# Évaluateur de pertinence
# =====================================================================
# Le papier original entraîne un T5 dédié. On réutilise le cross-encoder déjà
# présent dans le pipeline : déterministe, donc reproductible et testable par
# bootstrap comme GraphRAG.
RERANKER_MODEL = "BAAI/bge-reranker-v2-m3"

# CrossEncoder.predict() retourne des logits bruts. On applique une Sigmoid
# pour ramener les scores dans [0, 1], échelle sur laquelle portent les seuils.
NORMALIZE_SCORES = True

# Résumé des scores en un seul indice de confiance.
#   "max"  : le meilleur document suffit (esprit CRAG)
#   "mean" : moyenne (sévère, pénalisé par le bruit)
#   "top3" : moyenne des 3 meilleurs (adapté au multi-articles de BSARD)
SCORE_AGGREGATION = "max"

# Seuils — VALEURS PROVISOIRES, à calibrer sur train_clean.csv.
# Un seuil mal placé rend une branche inatteignable : CRAG tournerait en mode
# dégradé sans que cela se voie (cf. le bug du GraphRAG v1, sans effet mesurable).
TAU_CORRECT = 0.70
TAU_INCORRECT = 0.20


# =====================================================================
# Raffinement
# =====================================================================
# "conservative" : ne retire qu'un document au score très bas.
#                  Garde-fou : CRAG ne peut quasiment pas dégrader la baseline.
# "standard"     : garde les N meilleurs, fidèle au papier.
# "none"         : aucun filtrage, seule la décision à 3 branches est exploitée.
REFINEMENT_MODE = "conservative"

MIN_KEEP_SCORE = 0.10           # mode conservative
KEEP_TOP_N_CORRECT = 5          # mode standard
KEEP_TOP_N_AMBIGUOUS = 8        # mode standard

# Le papier découpe les documents en segments. Sur BSARD, l'évaluation attend
# des identifiants d'articles entiers : le filtrage opère au niveau de l'article.
SEGMENT_LEVEL_REFINEMENT = False


# =====================================================================
# Branche "incorrect"
# =====================================================================
# "keep_base" : retourne le classement de base inchangé (garde-fou).
# "drop"      : retourne une liste vide — fera chuter HR/MRR/F1 sur ces questions.
# La branche web du papier est écartée : BSARD est un corpus fermé, une page web
# ne peut pas produire d'article_id évaluable.
INCORRECT_STRATEGY = "keep_base"


# =====================================================================
# Protocole d'évaluation
# =====================================================================
TOP_K = 10              # identique aux benchmarks BM25, GraphRAG et Hybrid
CANDIDATE_POOL = 30     # candidats fournis par Hybrid avant évaluation

# Le raffinement réduit le nombre de documents retournés, ce qui fausserait la
# comparaison à K=10 (HR et F1 dépendent de la taille de la liste). Avec
# PAD_TO_TOP_K, la liste filtrée est complétée par les candidats suivants.
PAD_TO_TOP_K = True

LOG_BRANCH_DISTRIBUTION = True