"""
Configuration du module Corrective RAG (CRAG) — évaluation sur BSARD.

Principe : évaluer la pertinence des documents récupérés, puis agir selon
trois niveaux de confiance.
  - correct   : documents jugés fiables      -> raffinement
  - ambiguous : signal incertain             -> raffinement permissif
  - incorrect : aucun document convaincant   -> voir INCORRECT_STRATEGY

Protocole : CRAG est branché sur le même retriever BM25 que celui utilisé pour
évaluer GraphRAG, afin que la comparaison entre les deux couches soit valide
(même retriever de base, seule la couche change).
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
# Le papier original entraîne un T5 dédié. On réutilise ici le cross-encoder
# de hybrid_rag : déterministe, donc reproductible et testable par bootstrap,
# comme pour GraphRAG.
RERANKER_MODEL = "BAAI/bge-reranker-v2-m3"

# compute_score(..., normalize=True) borne les scores dans [0, 1].
# Les seuils ci-dessous supposent cette normalisation.
NORMALIZE_SCORES = True

# Comment résumer les scores des candidats en un seul indice de confiance.
#   "max"  : le meilleur document suffit (le plus proche de l'esprit CRAG)
#   "mean" : moyenne sur tous les candidats (sévère, pénalisé par le bruit)
#   "top3" : moyenne des 3 meilleurs (adapté au multi-articles de BSARD)
SCORE_AGGREGATION = "max"

# Seuils de décision — VALEURS PROVISOIRES.
# À calibrer sur train_clean.csv avant toute évaluation : si un seuil est mal
# placé, une des trois branches ne se déclenche jamais et CRAG tourne en mode
# dégradé sans que cela se voie (cf. le bug du GraphRAG v1, sans effet mesurable).
TAU_CORRECT = 0.70      # >= ce score -> "correct"
TAU_INCORRECT = 0.20    # <= ce score -> "incorrect"


# =====================================================================
# Raffinement
# =====================================================================
# "conservative" : ne retire qu'un document au score très bas.
#                  Garde-fou : CRAG ne peut quasiment pas dégrader la baseline.
# "standard"     : garde les N meilleurs, fidèle au papier.
# "none"         : aucun filtrage, on n'exploite que la décision à 3 branches.
# Les trois modes seront comparés en ablation, comme les 5 modes du graphe.
REFINEMENT_MODE = "conservative"

# Mode "conservative" : un document n'est retiré que si son score est
# strictement inférieur à ce seuil absolu.
MIN_KEEP_SCORE = 0.10

# Mode "standard" : nombre de documents conservés selon la branche.
KEEP_TOP_N_CORRECT = 5
KEEP_TOP_N_AMBIGUOUS = 8

# Le papier découpe les documents en segments avant de filtrer. Sur BSARD,
# l'évaluation attend des identifiants d'articles entiers : le filtrage opère
# donc au niveau de l'article, pas du segment.
SEGMENT_LEVEL_REFINEMENT = False


# =====================================================================
# Branche "incorrect"
# =====================================================================
# "keep_base" : retourne le classement de base inchangé.
#               Garde-fou anti-dégradation : sur les questions jugées
#               "incorrect", CRAG ne peut pas faire pire que la baseline.
# "drop"      : retourne une liste vide (mesure du comportement brut ;
#               fera mécaniquement chuter HR/MRR/F1 sur ces questions).
#
# La branche web du papier est écartée : BSARD étant un corpus fermé, une page
# web ne peut pas produire d'article_id évaluable par HR/MRR/F1.
INCORRECT_STRATEGY = "keep_base"


# =====================================================================
# Protocole d'évaluation
# =====================================================================
TOP_K = 10              # identique aux benchmarks BM25 et GraphRAG
CANDIDATE_POOL = 50     # candidats récupérés avant évaluation

# Le raffinement réduit le nombre de documents retournés, ce qui fausserait la
# comparaison avec BM25 et GraphRAG mesurés à K=10 : Hit Rate et F1 dépendent
# directement de la taille de la liste. Avec PAD_TO_TOP_K, la liste filtrée est
# complétée par les candidats suivants pour toujours retourner TOP_K documents.
# On mesure alors le réordonnancement, pas un artefact de troncature.
PAD_TO_TOP_K = True

# Journalise la répartition correct / ambiguous / incorrect par question.
# Permet de vérifier que les trois branches se déclenchent réellement.
LOG_BRANCH_DISTRIBUTION = True