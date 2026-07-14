# Pipeline des données (corpus RAG)

Ce document décrit comment le contenu aboutit à ce que l’application **recherche réellement** : `data/processed/final_chunks.jsonl` + l’index `vector_store/faiss.index`.

## Fichier « source de vérité » pour le RAG

| Élément | Rôle |
|--------|------|
| `data/processed/final_chunks.jsonl` | Segments (texte + métadonnées) : **seul contenu** utilisé par le retriever à l’exécution. |
| `vector_store/faiss.index` + `faiss_manifest.json` | Vecteurs alignés sur **le même ordre** que les chunks chargés à la construction. |
| `data/sources.csv` | **Inventaire** pratique pour la chaîne PDF (ex. noms de fichiers BO), **pas** le fichier interrogé par le RAG. |

Tout ce qui doit être « dans le RAG » doit apparaître **dans** `final_chunks.jsonl`, puis regénérer **embeddings + FAISS**.

## Chaîne de construction courante

1. **Textes / chunks bruts**  
   - Bulletins : souvent `data/processed/chunks_by_article.jsonl` (dérivé des PDF référencés par le travail d’extraction, dont `data/sources.csv` pour les noms de fichiers).  
   - Pages admin : `data/processed/admin_chunks.jsonl` (ex. sites type Watiqa).

2. **Fusion** — `python scripts/merge_chunks.py`  
   - Assemble `chunks_by_article.jsonl` + `admin_chunks.jsonl` en **`final_chunks.jsonl`** avec un schéma unifié (`source_type`, `category`, etc.).

3. **Embeddings** — `python scripts/build_embeddings.py`  
   - Lit `final_chunks.jsonl` → `vector_store/embeddings.npy` + `vector_store/metadata.jsonl`.

4. **Index** — `python scripts/build_faiss.py`  
   - Produit `vector_store/faiss.index` + `faiss_manifest.json`.

5. **Redémarrer** le serveur web pour recharger l’index.

## File web fallback (enrichissement semi-automatique)

Quand le RAG utilise le **fallback web**, les passages sont enregistrés dans :

`data/processed/web_additions_queue.jsonl`

Pour les intégrer au corpus (idempotent, avec filtres qualité) :

```bash
# Voir le plan sans écrire
python scripts/ingest_web_queue.py

# Ajouter au corpus + nettoyer la file + reconstruire l’index
python scripts/ingest_web_queue.py --apply --prune-queue --rebuild
```

Puis redémarrer uvicorn. Les questions déjà couvertes par une URL/texte présent dans `final_chunks.jsonl` sont ignorées (pas de doublon).

Journal optionnel : `data/processed/web_ingest_log.jsonl`

D’autres scripts peuvent **enrichir** le corpus (chunks curatés CNIE, consulat, etc.) : même règle — tout doit finir dans `final_chunks.jsonl` puis **rebuild** étapes 3–4.

## Inventaire du corpus

Pour un résumé automatique (volume par `source_org`, `doc_id`, comparaison avec FAISS) :

```bash
python scripts/inventory_final_chunks.py
```

Génère notamment `data/processed/corpus_inventory.json` et `data/processed/corpus_by_doc.csv`.

## Cohérence (à surveiller)

Si les logs indiquent un **décalage** entre le nombre de chunks JSONL et le nombre de vecteurs FAISS, regénérer : `build_embeddings.py` puis `build_faiss.py` **sans** modifier l’ordre des chunks entre les deux étapes.
