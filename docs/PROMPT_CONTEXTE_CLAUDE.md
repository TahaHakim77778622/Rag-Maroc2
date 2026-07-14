# Prompt de contexte — RAG-MAROC2 (à coller dans Claude)

Copiez tout le bloc ci-dessous dans une nouvelle conversation Claude.

---

## Rôle

Tu m’aides sur **RAG-MAROC2**, un assistant RAG (Retrieval-Augmented Generation) orienté **administration et droit marocain** : Bulletins officiels (SGG), procédures citoyennes (CNIE, passeport, Watiqa), Code du travail, etc. Réponds en français sauf demande contraire.

## Objectif produit

- Répondre aux questions des utilisateurs sur les **démarches et textes juridiques au Maroc**.
- **Priorité absolue** : répondre depuis le **dataset local** (`final_chunks.jsonl` + FAISS), pas depuis le web.
- **Fallback web** (.ma : consulat.ma, cnie.ma, maroc.ma, etc.) **uniquement** si le corpus ne couvre pas la question après analyse contextuelle.
- Comprendre le **sens de la phrase entière** (négations, objectif, document visé), pas seulement des mots-clés isolés (ex. « sans CNIE » + « passeport » → passeport, pas CNIE).

## Stack technique

| Couche | Technologie |
|--------|-------------|
| Langage | Python 3.11 |
| Embeddings | sentence-transformers (modèle configuré dans `app/config.py`) |
| Vector store | FAISS (IndexFlatIP), `vector_store/faiss.index` |
| LLM | Cohere (`command-r-08-2024`) ou OpenAI-compatible via `app/llm_factory.py` |
| API / UI | FastAPI + Jinja (`webapp/`), option Streamlit (`app/streamlit_app.py`) |
| Auth | JWT + utilisateurs PostgreSQL (`WEBAPP_DATABASE_URL`) ou SQLite (`data/webapp_users.db`) |
| Tests | pytest (~43 tests), CI GitHub Actions |
| macOS | conda pour `faiss` + `pyarrow` ; pip via `requirements-pip.txt` ou `./scripts/install_deps_mac.sh` |

## Arborescence essentielle

```
rag-maroc2/
├── app/                    # Cœur RAG
│   ├── rag_pipeline.py     # Orchestration ask()
│   ├── retriever.py        # FAISS search
│   ├── retrieval_rerank.py # Hybride FAISS + BM25 + keywords (+ CE optionnel)
│   ├── query_understanding.py
│   ├── phrase_context.py   # Contexte phrase (négations, sujet principal)
│   ├── corpus_first.py     # Dataset d'abord → web ensuite
│   ├── corpus_coverage.py  # corpus_covers_question()
│   ├── dataset_registry.py # Tous domaines du dataset (BO, CNIE, passeport, …)
│   ├── portal_cases.py     # Cas CNIE (première demande, perte/vol, renouvellement)
│   ├── passeport_cases.py  # Cas passeport (perte/vol, renouvellement)
│   ├── passeport_corpus.py # Injection chunks consulat.ma
│   ├── labor_corpus.py     # Code du travail LABOR65, heures sup.
│   ├── web_fallback.py     # DuckDuckGo + scrape .ma
│   ├── prompt_builder.py
│   └── llm_cohere.py / llm_openai.py
├── webapp/                 # Plateforme web
│   ├── main.py             # FastAPI, routes, chat
│   ├── rag_service.py        # Pont vers RAGPipeline
│   └── db.py                 # Users PostgreSQL/SQLite
├── data/
│   ├── processed/final_chunks.jsonl  # SOURCE DE VÉRITÉ RAG (~25 567 chunks)
│   ├── sources.csv           # Inventaire PDF BO (SGG0001–SGG0130…)
│   └── processed/web_additions_queue.jsonl
├── vector_store/           # embeddings.npy, faiss.index, metadata.jsonl
├── scripts/                # merge, build_embeddings, build_faiss, rebuild, …
├── tests/
├── requirements.txt
├── requirements-pip.txt    # pip sans faiss-cpu (Mac)
└── .env                    # Clés API (ne pas versionner)
```

## Corpus (données interrogées par le RAG)

- **~25 128** chunks `bulletin_officiel` / catégorie `juridique` (SGG, PDF BO).
- **~404** chunks Watiqa / état civil.
- **~15** CNIE, **~7** passeport (consulat.ma), **8** Code du travail (`LABOR65`).
- Fichier unique : `data/processed/final_chunks.jsonl`.
- Index aligné : `python scripts/build_embeddings.py` puis `python scripts/build_faiss.py`.

Chunks curatés ajoutés par scripts (SPA vides au scrape) :
- `scripts/append_cnie_procedure_chunks.py`
- `scripts/append_passport_consulat_chunks.py`
- `scripts/append_watiqa_procedure_chunks.py`
- `data/processed/labor_code_chunks.jsonl` (art. 201 heures sup., etc.)

## Pipeline d’une question (`POST /api/ask`)

1. **Query understanding** (`analyze_query`) : sujets, hints retrieval, flags (cnie, passeport, labor, BO, master/doctorat).
2. **Phrase context** (`analyze_phrase`) : sujet principal, négations (« sans CNIE »), action (« demander mon passeport »), âge (mineur).
3. **Retrieval FAISS** (pool élargi pour portails / BO / travail).
4. **Rerank hybride** (sémantique + BM25 + mots-clés ; cross-encoder optionnel `RERANK_ENABLE_CROSS_ENCODER`).
5. **prepare_local_hits** (`corpus_first.py`) : filtre domaines, merge labor/passeport/cnie/watiqa, chunks curatés.
6. **corpus_covers_question** : si oui → **pas de web** ; sinon `fetch_web_hits`.
7. **Génération LLM** (Cohere) avec `build_rag_prompt` + citations sources.

Modules clés récents :
- `dataset_registry.py` : mappe question → domaines (bulletin, juridique, cnie, passeport, watiqa, labor, construction, education, fiscalité…).
- `corpus_first.should_use_web_after_corpus()` : décision web centralisée.
- Ne plus valider le corpus sur la seule présence du mot « CNIE » si la phrase vise le passeport.

## Variables `.env` importantes (voir `env.example`)

```env
LLM_PROVIDER=cohere
COHERE_API_KEY=<secret>
COHERE_MODEL=command-r-08-2024
QUERY_REWRITE_LLM=0
RERANK_ENABLE_CROSS_ENCODER=0
WEB_FALLBACK_MAX_SEARCH_QUERIES=2
RAG_PRELOAD=1
WEBAPP_SECRET_KEY=<secret>
WEBAPP_DATABASE_URL=postgresql://user@localhost:5432/rag_maroc2
CORPUS_MIN_TOP_SCORE=0.26
CORPUS_MIN_TERM_RATIO=0.5
```

Profil **rapide** (dev) : pas de réécriture LLM avant search, pas de cross-encoder, modèle Cohere `command-r-08-2024`.

## Commandes opérationnelles

```bash
# Installation (Mac + conda)
./scripts/install_deps_mac.sh

# Rebuild corpus + index après modification chunks
python scripts/rebuild_corpus_and_index.py
# ou embeddings seulement :
python scripts/rebuild_corpus_and_index.py --embeddings-only

# Lancer l’app
kill $(lsof -t -i :8000) 2>/dev/null || true
./scripts/run_webapp.sh
# → http://127.0.0.1:8000

# Tests
pytest tests/ -q

# Éval hit@3
python scripts/evaluate_hit_at_k.py
```

## Problèmes connus / historique debug

| Symptôme | Cause | Direction fix |
|----------|--------|----------------|
| Réponse CNIE pour question passeport | Mot « CNIE » sans contexte | `phrase_context`, `passeport_corpus` |
| Web fallback alors que consulat dans dataset | `portal_local_hits_sufficient` trop strict ou mauvais chunk en tête | `corpus_first`, `passeport_cases` perte/vol |
| « Perte CNIE » → première demande | Pas de chunk perte/vol / mauvais rerank | `portal_cases`, chunks `cnie_procedure_perte_vol_1` |
| Heures sup. → SMIG fiscal BO | Faux positifs BO | `labor_corpus`, art. 201 curated |
| Phrase longue vs courte comportement différent | Mots parasites au score lexical | `_discriminative_terms`, ignore maintenant/comment |
| `pip install -r requirements.txt` échoue (CMake) | faiss/pyarrow compilent | conda + `requirements-pip.txt` |
| `psycopg` manquant | PostgreSQL dans .env | `pip install "psycopg[binary]"` |
| Lenteur / timeout Cohere | 2 appels LLM + CE | `.env` perf ci-dessus |

## Exemples de questions test

- « Perte ou vol de la CNIE : procédure » → chunks CNIE perte/vol, corpus, pas web.
- « je suis un mineur sans CNIE et je veux demander mon passeport » → consulat.ma mineur sans CNIE.
- « maintenant j’ai perdu mon passeport comment déclarer et comment le refaire » → même logique que « Perte passeport : déclaration et refaire ».
- « rémunération heures supplémentaires » → art. 201 loi 65-99 (LABOR65), pas SMIG fiscal BO.
- Questions BO / décrets / normes master → chunks SGG juridique.

## Ce que je attend de toi (Claude)

- Proposer des changements **minimaux** et cohérents avec l’architecture existante.
- Respecter : **dataset d’abord**, **contexte de phrase**, **tests pytest**.
- Ne pas exposer ni inventer de clés API ; renvoyer vers `.env` / `env.example`.
- Citer les fichiers (`app/…`, `webapp/…`, `scripts/…`) quand tu suggères du code.
- Répondre aux questions en t’appuyant sur la logique RAG décrite ci-dessus, pas sur des suppositions hors corpus marocain.

## Fichiers à lire en priorité pour une tâche code

1. `app/rag_pipeline.py` — flux principal
2. `app/corpus_first.py` + `app/corpus_coverage.py` — quand activer le web
3. `app/phrase_context.py` + `app/dataset_registry.py` — compréhension question
4. `app/web_fallback.py` — fallback web
5. `webapp/rag_service.py` + `webapp/main.py` — API
6. `data/PIPELINE.md` — chaîne données

---

*Fin du prompt — projet RAG-MAROC2, workspace local `/Users/tahahakim/Desktop/rag-maroc2`.*
