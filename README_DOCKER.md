# Lancer RAG-MAROC2 avec Docker

## Prérequis

- Docker Desktop installé
- Fichier `.env` configuré (clés API, `WEBAPP_SECRET_KEY`, etc.)
- Index FAISS et corpus présents dans `vector_store/` et `data/`

## Lancer le projet

```bash
docker compose up --build
```

## Accéder à l'interface

http://localhost:8000

Compte démo par défaut : `demo` / `demo123` (voir `WEBAPP_DEMO_USER` / `WEBAPP_DEMO_PASSWORD` dans `.env`).

## Arrêter

```bash
docker compose down
```

## Rebuild après modification

```bash
docker compose up --build --force-recreate
```

## Healthcheck

Le service expose `GET /health` — utilisé par Docker Compose pour vérifier que l'API répond.
