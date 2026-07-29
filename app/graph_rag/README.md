# GraphRAG — module d'évaluation sur le benchmark BSARD

Projet RAG-MAROC2. Ce module évalue l'apport d'un graphe de connaissances
juridiques sur la recherche d'articles de loi, mesuré sur le benchmark belge
BSARD (Louis & Spanakis, ACL 2022).

## Fichiers du module

| Fichier | Rôle |
|---|---|
| `config.py` | Chemins et paramètres centralisés |
| `schema.py` | Schéma du graphe KuzuDB : nœuds Code / Division / Article, relations CONTIENT_* et REFERENCE |
| `entity_extraction.py` | Extraction par regex des renvois croisés entre articles |
| `graph_builder.py` | Construction du graphe : hiérarchie depuis le CSV + résolution des renvois |
| `graph_retriever.py` | Traversée du graphe : expansion REFERENCE, voisinage hiérarchique, degré bidirectionnel |
| `fusion.py` | Fusion BM25 + graphe (5 modes de fusion disponibles) |
| `benchmark.py` | Évaluation : Hit Rate@10, MRR@10, F1@10, Recall@k, MAP@k |
| `bootstrap_c4.py` | Test de significativité statistique par bootstrap |
| `RESULTATS_ABLATION.md` | Résultats détaillés de l'ablation des 5 modes |

Le dossier `resultats/` contient les sorties brutes des évaluations (une ligne par question).

## Installation

```bash
pip install -r requirements.txt
```

## Données requises

Le module attend les 3 fichiers du benchmark BSARD dans `data/graph_rag/`,
deux niveaux au-dessus de ce dossier :
ag-maroc2/
├── data/graph_rag/
│ ├── articles_clean.csv (22 382 articles)
│ ├── train_clean.csv (886 questions)
│ └── test_clean.csv (222 questions)
└── app/graph_rag/ <- ce dossier
Si l'arborescence diffère, ajuster les chemins en haut de `config.py`.

## Utilisation

1. Construire le graphe (une seule fois, ~3 min) :
```bash
python3 graph_builder.py
```

2. Évaluer BM25 seul vs BM25 + graphe :
```bash
python3 -c "from benchmark import run_benchmark; run_benchmark()"
```

3. Comparer les 5 modes de fusion :
```bash
python3 -c "from benchmark import run_ablation; run_ablation()"
```

4. Test de significativité statistique :
```bash
python3 bootstrap_c4.py
```

## Résultat principal

Gain de F1@10 de +0.016 (jeu d'entraînement) à +0.017 (jeu de test),
statistiquement significatif (100 % de tirages positifs au bootstrap),
concentré sur les questions attendant plusieurs articles en réponse.

Détails complets dans le rapport `Resultats_Complets_GraphRAG_BSARD.pdf`.