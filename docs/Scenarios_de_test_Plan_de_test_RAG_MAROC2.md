# SCÉNARIOS DE TEST — PLAN DE TEST

**Projet :** RAG-MAROC2 (assistant RAG sur corpus juridique et administratif marocain)  
**Stack de référence :** FastAPI, Jinja2, sessions cookie, index FAISS + `final_chunks.jsonl`, LLM (Cohere / OpenAI), base utilisateurs (PostgreSQL ou SQLite)  
**Date du document :** 24 avril 2026  
**Version application :** 0.1.0  

*Document structuré sur le modèle d’un plan de type livrable de test (intro, périmètre, environnement, scénarios numérotés, non-régression, critères d’acceptation).*

---

## Table des matières

1. [Visa et signatures](#1-visa-et-signatures)  
2. [Introduction et objectifs du plan de test](#2-introduction-et-objectifs-du-plan-de-test)  
3. [Périmètre des tests](#3-périmètre-des-tests)  
4. [Environnement de test](#4-environnement-de-test)  
5. [Types de tests](#5-types-de-tests)  
6. [Scénarios de test — Authentification (navigateur)](#6-scénarios-de-test--authentification-navigateur)  
7. [Scénarios de test — API JWT](#7-scénarios-de-test--api-jwt)  
8. [Scénarios de test — Assistant RAG](#8-scénarios-de-test--assistant-rag)  
9. [Scénarios de test — Site public (pages)](#9-scénarios-de-test--site-public-pages)  
10. [Scénarios de test non fonctionnels](#10-scénarios-de-test-non-fonctionnels)  
11. [Matrice de couverture (résumé)](#11-matrice-de-couverture-résumé)  
12. [Critères d’acceptation](#12-critères-dacceptation)  
13. [Conclusion](#13-conclusion)  

---

## 1. Visa et signatures

| Rôle | Nom | Date | Signature |
|------|-----|------|-----------|
| **Équipe projet** |  |  |  |
| **Encadrant pédagogique** |  |  |  |
| **Validation qualité (optionnel)** |  |  |  |

---

## 2. Introduction et objectifs du plan de test

Ce document définit la stratégie, les scénarios et les critères d’acceptation des tests du projet **RAG-MAROC2**. Il sert de référentiel de validation de la plateforme avant toute **mise en production** ou **soutenance / livrable**.

### 2.1 Objectifs

- Vérifier que les **fonctionnalités principales** (authentification, accès à l’assistant, appels API) sont **implémentées et stables**.  
- Garantir la **cohérence** du pipeline RAG (retrieval FAISS + génération LLM) et l’**affichage des sources** lorsque des passages sont retournés.  
- S’assurer de la **sécurité minimale** : session utilisateur, mots de passe hachés (bcrypt), protection des routes (`/api/ask` et `/chat` réservés aux utilisateurs authentifiés), JWT pour clients API.  
- Valider l’**intégration** LLM (timeouts, message d’erreur utilisateur en cas d’indisponibilité).  
- Couvrir les exigences **non fonctionnelles** : temps de réponse raisonnable, ergonomie de base, comportement en absence de corpus ou d’index désynchronisé.  

### 2.2 Portée

**Inclus :** application web FastAPI, pages Jinja, assistant `/chat`, API `/api/ask`, `/api/auth/login`, `/api/me`, healthcheck, scénario d’inscription si `WEBAPP_ALLOW_REGISTER=1`.  

**Hors périmètre (sauf mention contraire) :** tests de charge massifs, audit de sécurité approfondi, disponibilité des services tiers (Cohere, etc.), application mobile native, CI/CD.  

---

## 3. Périmètre des tests

**Tableau 1 — Modules et fonctionnalités testés**

| Module | Fonctionnalités testées |
|--------|-------------------------|
| **Authentification web** | Connexion / déconnexion par session, redirection `/login` si non connecté, inscription (si autorisée), validation des champs (identifiant, mot de passe, confirmation) |
| **API JWT** | `POST /api/auth/login` (token), `GET /api/me` avec en-tête `Authorization: Bearer` |
| **Assistant RAG** | `POST /api/ask` avec historique, réponse JSON `answer` + `sources`, erreurs 401 / 503 index |
| **Site public** | Accueil, pages informatives, liens, chargement des assets statiques |
| **Back-office données** | Présence de `vector_store/faiss.index` et `data/processed/final_chunks.jsonl` cohérents (avertissements de désynchronisation) |
| **Base utilisateurs** | Création de compte en base (PostgreSQL ou SQLite selon `WEBAPP_DATABASE_URL` / défaut) |

---

## 4. Environnement de test

**Tableau 2 — Configuration type**

| Paramètre | Valeur / remarque |
|-----------|------------------|
| OS | macOS / Linux (dev) |
| Python | 3.11+ (venv recommandé) |
| Démarrage | `uvicorn webapp.main:app --host 127.0.0.1 --port 8001` (ou port indiqué) |
| Fichier d’environnement | `.env` à la racine (`COHERE_API_KEY`, `WEBAPP_SECRET_KEY`, `WEBAPP_DATABASE_URL` optionnel, `WEBAPP_ALLOW_REGISTER`, `LLM_TIMEOUT`, etc.) |
| Index RAG | `vector_store/faiss.index` + `faiss_manifest.json` aligné avec `final_chunks.jsonl` |

**Tableau 3 — Comptes de test (à adapter)**

| Compte | Mot de passe | Usage |
|--------|--------------|--------|
| `demo` | `demo123` | Compte créé auto si la base est vide (variables `WEBAPP_DEMO_USER` / `WEBAPP_DEMO_PASSWORD` possibles) |
| Utilisateur créé en test | (défini à l’inscription) | Tests inscription + connexion si `WEBAPP_ALLOW_REGISTER=1` |
| API JWT | — | Obtenir un `access_token` via `POST /api/auth/login` pour les requêtes `Bearer` |

**Outils :** navigateur (Chrome / Firefox), client HTTP (curl, Thunder Client, Postman), éventuellement `curl -H "Cookie: ..."` pour la session.  

---

## 5. Types de tests

**Tableau 4 — Types de tests et moyens**

| Type | Description | Moyens |
|------|-------------|--------|
| Fonctionnels | Parcours utilisateur et API | Manuel, scénarios ci-dessous |
| Non fonctionnels | Performance perçue (LLM), messages d’erreur | Manuel + observation logs serveur |
| Régression | Vérification après changement de code / données | Sous-ensemble de scénarios marqués « critique » |
| Compatibilité | Différents navigateurs / largeurs d’écran | Manuel responsive |

---

## 6. Scénarios de test — Authentification (navigateur)

| ID | Scénario | Préconditions | Étapes | Résultat attendu | Statut |
|----|----------|---------------|--------|------------------|--------|
| **A-01** | Accès assistant sans session | Aucun cookie de session | Ouvrir `/chat` | Redirection vers `/login` (302) |  |
| **A-02** | Connexion valide | Compte `demo` / compte de test existant | `GET /login` → saisir identifiant + mot de passe → envoyer | Redirection vers `/chat`, affichage du prénom / identifiant |  |
| **A-03** | Connexion refusée | — | Saisir un mauvais mot de passe | Message d’erreur explicite, reste sur `/login` |  |
| **A-04** | Déconnexion | Utilisateur connecté | Cliquer « Déconnexion » (ou `GET /logout`) | Redirection (ex. accueil), `/chat` redirige à nouveau vers `/login` |  |
| **A-05** | Inscription | `WEBAPP_ALLOW_REGISTER=1` | Remplir formulaire (identifiant valide, mot de passe 8+ car., confirmation identique) | Redirection / message de succès, compte en base, connexion possible |  |
| **A-06** | Inscription refusée (validation) | Inscription autorisée | Mots de passe différents ou identifiant invalide | Message d’erreur sans création de compte |  |
| **A-07** | Inscription désactivée | `WEBAPP_ALLOW_REGISTER=0` | Ouvrir `/register` | Redirection vers `/login` |  |  

---

## 7. Scénarios de test — API JWT

| ID | Scénario | Préconditions | Étapes | Résultat attendu | Statut |
|----|----------|---------------|--------|------------------|--------|
| **J-01** | Login JWT | Utilisateur valide | `POST /api/auth/login` JSON `{"username","password"}` | `200`, corps avec `access_token`, `token_type: bearer` |  |
| **J-02** | Login échoué | — | Identifiants incorrects | `401`, détail explicite |  |
| **J-03** | Profil avec token | Token J-01 | `GET /api/me` + `Authorization: Bearer <token>` | `200`, `id`, `username` |  |
| **J-04** | Profil sans / mauvais token | — | `GET /api/me` sans en-tête ou token invalide | `401` |  |  

---

## 8. Scénarios de test — Assistant RAG

| ID | Scénario | Préconditions | Étapes | Résultat attendu | Statut |
|----|----------|---------------|--------|------------------|--------|
| **R-01** | Question authentifiée (session) | Connecté sur le site | POST `/api/ask` (via UI chat) avec question pertinente | `200`, champ `answer` non vide, éventuellement `sources` |  |
| **R-02** | Question authentifiée (JWT) | Token valide | `POST /api/ask` JSON `question`, `top_k`, `history` | Idem, JSON sérialisable |  |
| **R-03** | Sans authentification | Aucun cookie ni Bearer | `POST /api/ask` | `401` |  |
| **R-04** | Historique conversation | Deux questions enchaînées | Deuxième question faisant référence au contexte | Réponse cohérente (comportement à qualifier selon modèle) |  |
| **R-05** | Index désynchronisé (si reproductible) | JSONL / FAISS en nombre de vecteurs incohérent | Appel assistant | `503` ou message d’erreur documenté, hint rebuild |  |
| **R-06** | Paramètre `top_k` | `top_k` entre 1 et 20 | Vérifier nombre de sources affichées / renvoyées | Cohérent avec `k` (dans la limite du retrieval) |  |  

---

## 9. Scénarios de test — Site public (pages)

| ID | Scénario | Étapes | Résultat attendu | Statut |
|----|----------|--------|------------------|--------|
| **P-01** | Page d’accueil | `GET /` | `200`, contenu HTML |  |
| **P-02** | Navigation | Liens Accueil, Documentation, FAQ, etc. | Pages chargées sans erreur 500 |  |
| **P-03** | CSS / JS | Chargement `/static/app.css`, scripts chat si concerné | Pas d’erreurs 404 bloquantes sur les assets |  |  

---

## 10. Scénarios de test non fonctionnels

| ID | Scénario | Résultat attendu | Statut |
|----|----------|------------------|--------|
| **N-01** | Santé | `GET /health` retourne `{"status": "ok"}` |  |
| **N-02** | Timeout LLM (si surcharge ou délai court) | Message utilisateur compréhensible, pas d’écran 500 nu sur l’UI chat |  |
| **N-03** | Affichage mobile / largeur réduite | Login et chat utilisables (scroll, boutons) |  |
| **N-04** | Persistance des comptes | Après redémarrage du serveur, comptes toujours présents (même BDD) |  |  

---

## 11. Matrice de couverture (résumé)

| Zone | Cas référencés |
|------|----------------|
| Auth session | A-01 — A-07 |
| Auth JWT + API protégée | J-01 — J-04, R-02, R-03 |
| RAG / assistant | R-01 — R-06 |
| Public | P-01 — P-03 |
| Non-fonctionnel | N-01 — N-04 |  

---

## 12. Critères d’acceptation

### 12.1 Critères de validation (Go / No-Go)

- **Go** : tous les scénarios **critiques** (A-01, A-02, A-04, R-01, R-03, J-01, N-01) passent sur l’environnement de recette.  
- **No-Go** : échec d’authentification systématique, impossibilité d’obtenir une réponse RAG pour une configuration LLM valide, erreurs 500 non gérées sur le parcours principal.  

### 12.2 Livrables de test

- Ce plan (ce document).  
- Tableau de suivi : colonne **Statut** (Non testé / OK / KO / Bloqué) + date + remarque.  
- Captures d’écran optionnelles pour les KO.  

### 12.3 Légende des statuts (tableaux)

| Symbole / texte | Signification |
|-----------------|---------------|
| vide | Non testé |
| OK | Conforme |
| KO | Non conforme (créer un ticket / bug) |
| Bloqué | En attente d’environnement ou de donnée |  

---

## 13. Conclusion

Le projet **RAG-MAROC2** combine une **authentification** classique, un **index sémantique** local et un **LLM** : la qualité perçue dépend à la fois du **corpus** (`final_chunks.jsonl`), de l’**alignement** FAISS, et de la **disponibilité** des APIs distantes. Ce plan de test fournit une **base reproductible** pour valider chaque version livrable ; il doit être **mis à jour** si de nouvelles routes ou modules sont ajoutés.  

---
*Généré pour le dépôt RAG-MAROC2 — structure inspirée d’un modèle de scénarios / plan de test type livrable (intro, tables, scénarios numérotés, critères d’acceptation).*
