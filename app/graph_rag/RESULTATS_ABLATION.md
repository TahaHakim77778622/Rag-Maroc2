# Résultats d'ablation GraphRAG (benchmark belge BSARD-like)

## Protocole

- Corpus : `articles_clean.csv` (22 382 articles), `test_clean.csv` (**222** questions)
- Métriques : Hit Rate@10, MRR@10, F1@10
- Baseline lexicale : BM25Okapi sur le texte des articles (proxy du pipeline FAISS+BM25)
- Graphe : KuzuDB (`REFERENCE` + hiérarchie `CONTIENT_*`), déjà construit
- Critère de rétention : ΔHR ≥ 0 et ΔF1 ≥ 0 ; viser ΔMRR > 0 reproductible
- Date d'exécution : généré automatiquement par `benchmark.run_ablation()`

## Diagnostic préalable (pourquoi le signal peinait)

1. **Bug no-op (C0)** : le re-rank strict prenait *tout* le top_k comme seeds de `expand()`,
   assignant score=1.0 à chaque candidat → tri stable = ordre BM25 (0 promote / 0 demote sur 92 hits).
2. **REFERENCE ↔ pertinence NL** : seulement ~4.1 % des paires gold multi sont liées par REFERENCE ;
   ~52 % des questions multi ont au moins une arête gold–gold (signal rare, non systématique).
3. **Siblings** : plus fréquents (~70 % des questions multi) mais bruyants (LIMIT 15 sans filtre lexical).
4. **Hubness globale** : articles très référencés = nœuds procéduraux (ΔMRR ≈ −0.023 en offline) — **non retenu**.
5. **Expansion libre** (historique) : injection hors pool lexical → ΔHR ≈ −0.086 — abandonnée.

## Tableau récapitulatif (GLOBAL)

| Config | HR | MRR | F1 | ΔHR | ΔMRR | ΔF1 | Retenu ? |
|---|---:|---:|---:|---:|---:|---:|:---:|
| BM25 seul | 0.414 | 0.237 | 0.084 | — | — | — | baseline |
| C0_noop | 0.414 | 0.237 | 0.084 | +0.000 | +0.000 | +0.000 | contrôle |
| C1_bidir_soft | 0.414 | 0.244 | 0.084 | +0.000 | +0.007 | +0.000 | oui |
| C2_controlled_expand | 0.428 | 0.242 | 0.091 | +0.014 | +0.006 | +0.007 | oui |
| C3_expand_bidir | 0.428 | 0.246 | 0.091 | +0.014 | +0.009 | +0.007 | oui |
| C4_expand_siblings | 0.437 | 0.244 | 0.101 | +0.023 | +0.008 | +0.017 | oui |

**Configuration retenue pour la suite : `C4_expand_siblings`** (meilleure sous ΔHR≥0, ΔF1≥0, ΔMRR>0 ; `GRAPH_MODE=expand_with_siblings`).

## Détail par configuration

### C0_noop — C0 — re-rank strict top_k via expand() (bug no-op historique)

Seeds = tout le top_k BM25 → score graphe uniforme 1.0 → tri stable = ordre BM25. HR/F1/MRR structurellement identiques à la baseline (à bruit numérique près).

| Split | n | HR_b | HR_g | ΔHR | MRR_b | MRR_g | ΔMRR | F1_b | F1_g | ΔF1 |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| ALL | 222 | 0.414 | 0.414 | +0.000 | 0.237 | 0.237 | +0.000 | 0.084 | 0.084 | +0.000 |
| single | 83 | 0.337 | 0.337 | +0.000 | 0.215 | 0.215 | +0.000 | 0.061 | 0.061 | +0.000 |
| multi | 139 | 0.460 | 0.460 | +0.000 | 0.250 | 0.250 | +0.000 | 0.097 | 0.097 | +0.000 |

### C1_bidir_soft — C1 — soft re-rank bidir induit (α=0.02), siblings off

Ne change pas l'ensemble top_k (HR/F1 = BM25). Boost soft proportionnel au degré REFERENCE bidirectionnel dans le top_k.

| Split | n | HR_b | HR_g | ΔHR | MRR_b | MRR_g | ΔMRR | F1_b | F1_g | ΔF1 |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| ALL | 222 | 0.414 | 0.414 | +0.000 | 0.237 | 0.244 | +0.007 | 0.084 | 0.084 | +0.000 |
| single | 83 | 0.337 | 0.337 | +0.000 | 0.215 | 0.218 | +0.003 | 0.061 | 0.061 | +0.000 |
| multi | 139 | 0.460 | 0.460 | +0.000 | 0.250 | 0.259 | +0.010 | 0.097 | 0.097 | +0.000 |

### C2_controlled_expand — C2 — controlled_expand (top-3 → REFERENCE ∩ pool@50 → RRF), siblings off

Expansion REFERENCE gated au pool BM25@50. Gain multi-articles (HR/F1) sans régression single. Base solide sans siblings.

| Split | n | HR_b | HR_g | ΔHR | MRR_b | MRR_g | ΔMRR | F1_b | F1_g | ΔF1 |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| ALL | 222 | 0.414 | 0.428 | +0.014 | 0.237 | 0.242 | +0.006 | 0.084 | 0.091 | +0.007 |
| single | 83 | 0.337 | 0.337 | +0.000 | 0.215 | 0.218 | +0.002 | 0.061 | 0.061 | +0.000 |
| multi | 139 | 0.460 | 0.482 | +0.022 | 0.250 | 0.257 | +0.008 | 0.097 | 0.108 | +0.011 |

### C3_expand_bidir — C3 — controlled_expand + soft bidir sur le top_k final

Combine expansion contrôlée et re-rank bidir soft. Meilleur MRR global, mais ΔMRR single négatif (−0.013) — compromis à signaler.

| Split | n | HR_b | HR_g | ΔHR | MRR_b | MRR_g | ΔMRR | F1_b | F1_g | ΔF1 |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| ALL | 222 | 0.414 | 0.428 | +0.014 | 0.237 | 0.246 | +0.009 | 0.084 | 0.091 | +0.007 |
| single | 83 | 0.337 | 0.337 | +0.000 | 0.215 | 0.203 | -0.013 | 0.061 | 0.061 | +0.000 |
| multi | 139 | 0.460 | 0.482 | +0.022 | 0.250 | 0.272 | +0.023 | 0.097 | 0.108 | +0.011 |

### C4_expand_siblings — C4 — controlled_expand + siblings gated au pool BM25

Siblings + REFERENCE, toujours ∩ pool@50. Contrairement au re-rank libre, le gated sibling améliore HR/F1 (surtout single) sans dégrader la baseline.

| Split | n | HR_b | HR_g | ΔHR | MRR_b | MRR_g | ΔMRR | F1_b | F1_g | ΔF1 |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| ALL | 222 | 0.414 | 0.437 | +0.023 | 0.237 | 0.244 | +0.008 | 0.084 | 0.101 | +0.017 |
| single | 83 | 0.337 | 0.386 | +0.048 | 0.215 | 0.227 | +0.011 | 0.061 | 0.070 | +0.009 |
| multi | 139 | 0.460 | 0.468 | +0.007 | 0.250 | 0.255 | +0.005 | 0.097 | 0.119 | +0.022 |

## Conclusions pour le papier

- Le re-rank graphe naïf (C0) est un **artefact méthodologique** (no-op), pas une ablation neutre informative.
- Un signal graphe **filtré par le pool BM25** (C2/C4) améliore HR/MRR/F1 simultanément vs BM25 seul.
- **C2 (REFERENCE seul)** : gains concentrés sur les questions multi-articles (ΔHR +0.022, ΔF1 +0.011).
- **C4 (REFERENCE + siblings gated)** : meilleure config globale (ΔHR +0.023, ΔF1 +0.017, ΔMRR +0.008) ;
  les siblings aident surtout le single-article (ΔHR +0.048) quand ils sont restreints au pool lexical.
- **C3** : meilleur MRR global (+0.009) mais régresse le MRR single (−0.013) — compromis à expliciter.
- **C1 (bidir soft)** : gain MRR pur (+0.007) sans toucher HR/F1 — utile comme re-ranker conservateur.
- Hubness globale et expansion libre hors pool lexical : rejetées (dégradent le ranking).
