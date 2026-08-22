"""
bsard_benchmark.py
---------------------
Benchmark Hybrid RAG (bge-m3 + FAISS + BM25 + RRF + cross-encoder) sur BSARD.

Mêmes métriques que le benchmark GraphRAG (HR@10, MRR@10, F1@10) et même
bootstrap apparié, pour que les architectures soient directement comparables.

Retrieval uniquement : aucune génération, aucun appel LLM.

Usage :
    python3 app/hybrid_rag/bsard_benchmark.py
    python3 app/hybrid_rag/bsard_benchmark.py --split train
"""

import re
import sys
import time
import pickle
import numpy as np
import pandas as pd
import faiss
from pathlib import Path
from rank_bm25 import BM25Okapi
from sentence_transformers import SentenceTransformer

sys.path.insert(0, str(Path(__file__).resolve().parent))
import config as cfg

cfg.INDICES_DIR.mkdir(parents=True, exist_ok=True)


# ============ CHARGEMENT BSARD ============

def parse_article_ids(raw) -> list:
    """Gold labels au format '[947 948]' (sans virgules) -> [947, 948].
    Même parsing que dans graph_rag/benchmark.py."""
    return [int(x) for x in re.findall(r"\d+", str(raw))]


def detect_text_column(df: pd.DataFrame, exclude: list) -> str:
    """Colonne texte = celle avec la plus grande longueur moyenne."""
    candidates = [c for c in df.columns if c not in exclude]
    avg = {c: df[c].astype(str).str.len().mean() for c in candidates}
    return max(avg, key=avg.get)


def load_articles():
    df = pd.read_csv(cfg.ARTICLES_CSV)
    id_col = "id" if "id" in df.columns else df.columns[0]
    hierarchy = [c for c in ["book", "part", "act", "chapter", "section", "subsection"]
                 if c in df.columns]
    text_col = detect_text_column(df, exclude=[id_col] + hierarchy)

    print(f"[bsard] Colonne ID    : '{id_col}'")
    print(f"[bsard] Colonne texte : '{text_col}'")

    return df[id_col].astype(int).tolist(), df[text_col].astype(str).tolist()


def load_questions(split: str):
    path = cfg.TRAIN_CSV if split == "train" else cfg.TEST_CSV
    df = pd.read_csv(path)
    q_col = "question" if "question" in df.columns else df.columns[0]
    gold_col = ("article_ids" if "article_ids" in df.columns
                else [c for c in df.columns if "article" in c.lower()][0])

    print(f"[bsard] Colonne question : '{q_col}'")
    print(f"[bsard] Colonne gold     : '{gold_col}'")

    return (df[q_col].astype(str).tolist(),
            [parse_article_ids(x) for x in df[gold_col].tolist()])


# ============ INDEXATION ============

def tokenize(text: str) -> list:
    """Tokenisation simple, SANS retrait de stopwords — identique à celle de
    graph_rag/fusion.py, pour que la baseline BM25 soit strictement comparable
    entre les deux benchmarks."""
    return re.findall(r"\b\w+\b", text.lower(), flags=re.UNICODE)


def build_or_load_indices(article_ids, texts):
    if cfg.FAISS_INDEX_PATH.exists() and cfg.BM25_INDEX_PATH.exists():
        print("[bsard] Index existants trouvés, chargement...")
        index = faiss.read_index(str(cfg.FAISS_INDEX_PATH))
        with open(cfg.FAISS_IDMAP_PATH, "rb") as f:
            idmap = pickle.load(f)
        with open(cfg.BM25_INDEX_PATH, "rb") as f:
            data = pickle.load(f)
        return index, idmap, data["bm25"], data["article_ids"]

    print(f"[bsard] Construction des index sur {len(texts)} articles "
          f"(plusieurs heures sur CPU, une seule fois)...")

    model = SentenceTransformer(cfg.EMBED_MODEL)
    model.max_seq_length = cfg.MAX_SEQ_LENGTH

    print("[bsard] Encodage dense (bge-m3)...")
    vectors = model.encode(
        texts,
        batch_size=cfg.EMBED_BATCH_SIZE,
        show_progress_bar=True,
        normalize_embeddings=True,
        convert_to_numpy=True,
    ).astype("float32")

    index = faiss.IndexFlatIP(vectors.shape[1])
    index.add(vectors)
    idmap = {i: article_ids[i] for i in range(len(article_ids))}

    faiss.write_index(index, str(cfg.FAISS_INDEX_PATH))
    with open(cfg.FAISS_IDMAP_PATH, "wb") as f:
        pickle.dump(idmap, f)

    print("[bsard] Construction BM25...")
    bm25 = BM25Okapi([tokenize(t) for t in texts])
    with open(cfg.BM25_INDEX_PATH, "wb") as f:
        pickle.dump({"bm25": bm25, "article_ids": article_ids}, f)

    print("[bsard] Index construits.\n")
    return index, idmap, bm25, article_ids


# ============ RETRIEVAL ============

_reranker = None


def get_reranker():
    global _reranker
    if _reranker is None and cfg.RERANKER_ENABLED:
        from FlagEmbedding import FlagReranker
        print(f"[bsard] Chargement du reranker {cfg.RERANKER_MODEL}...")
        _reranker = FlagReranker(cfg.RERANKER_MODEL, use_fp16=True)
    return _reranker


def reciprocal_rank_fusion(faiss_results, bm25_results):
    scores = {}
    for rank, (aid, _) in enumerate(faiss_results, start=1):
        scores[aid] = scores.get(aid, 0.0) + 1.0 / (cfg.RRF_K + rank)
    for rank, (aid, _) in enumerate(bm25_results, start=1):
        scores[aid] = scores.get(aid, 0.0) + 1.0 / (cfg.RRF_K + rank)
    return sorted(scores.items(), key=lambda x: x[1], reverse=True)[:cfg.TOP_K_HYBRID]


def retrieve_bm25_only(query, bm25, article_ids):
    """Baseline BM25 seul, pour la comparaison appariée."""
    scores = bm25.get_scores(tokenize(query))
    ranked = sorted(zip(article_ids, scores), key=lambda x: x[1], reverse=True)
    return [aid for aid, _ in ranked[:cfg.TOP_K_FINAL]]


def retrieve_hybrid(query, model, index, idmap, bm25, bm25_ids, texts_by_id):
    """Dense + sparse -> fusion RRF -> reranking cross-encoder."""
    q_vec = model.encode([query], normalize_embeddings=True,
                         convert_to_numpy=True).astype("float32")
    scores, indices = index.search(q_vec, cfg.TOP_K_FAISS)
    faiss_results = [(idmap[i], float(s)) for i, s in zip(indices[0], scores[0]) if i != -1]

    bm25_scores = bm25.get_scores(tokenize(query))
    bm25_ranked = sorted(zip(bm25_ids, bm25_scores),
                         key=lambda x: x[1], reverse=True)[:cfg.TOP_K_BM25]

    fused = reciprocal_rank_fusion(faiss_results, bm25_ranked)

    if cfg.RERANKER_ENABLED:
        reranker = get_reranker()
        valid = [aid for aid, _ in fused if aid in texts_by_id]
        pairs = [[query, texts_by_id[aid]] for aid in valid]
        if pairs:
            rr = reranker.compute_score(pairs, normalize=True)
            if isinstance(rr, float):
                rr = [rr]
            fused = sorted(zip(valid, rr), key=lambda x: x[1], reverse=True)

    return [aid for aid, _ in fused[:cfg.TOP_K_FINAL]]


# ============ MÉTRIQUES ============
# Identiques à app/graph_rag/benchmark.py pour que la comparaison soit valide.

def hit_rate_at_k(retrieved, gold):
    return 1.0 if any(g in retrieved for g in gold) else 0.0


def mrr_at_k(retrieved, gold):
    for rank, aid in enumerate(retrieved, start=1):
        if aid in gold:
            return 1.0 / rank
    return 0.0


def f1_at_k(retrieved, gold):
    r, g = set(retrieved), set(gold)
    if not r or not g:
        return 0.0
    tp = len(r & g)
    if tp == 0:
        return 0.0
    precision, recall = tp / len(r), tp / len(g)
    return 2 * precision * recall / (precision + recall)


def bootstrap_significance(deltas, n_resamples=None):
    """% de rééchantillonnages où la moyenne des deltas (Hybrid − BM25) > 0.
    >= 95 % = gain statistiquement significatif."""
    n_resamples = n_resamples or cfg.N_BOOTSTRAP
    deltas = np.array(deltas)
    n = len(deltas)
    positives = sum(1 for _ in range(n_resamples)
                    if np.random.choice(deltas, size=n, replace=True).mean() > 0)
    return 100 * positives / n_resamples


# ============ MAIN ============

def main():
    split = "test"
    if "--split" in sys.argv:
        split = sys.argv[sys.argv.index("--split") + 1]

    print(f"=== Benchmark Hybrid RAG sur BSARD ({split}_clean.csv) ===\n")

    article_ids, texts = load_articles()
    texts_by_id = dict(zip(article_ids, texts))
    questions, gold_list = load_questions(split)
    print(f"[bsard] {len(questions)} questions chargées.\n")

    index, idmap, bm25, bm25_ids = build_or_load_indices(article_ids, texts)

    print("[bsard] Chargement de bge-m3 pour les requêtes...")
    model = SentenceTransformer(cfg.EMBED_MODEL)
    model.max_seq_length = cfg.MAX_SEQ_LENGTH

    hr_h, mrr_h, f1_h = [], [], []
    hr_b, mrr_b, f1_b = [], [], []

    start = time.time()
    for i, (query, gold) in enumerate(zip(questions, gold_list), start=1):
        rh = retrieve_hybrid(query, model, index, idmap, bm25, bm25_ids, texts_by_id)
        rb = retrieve_bm25_only(query, bm25, bm25_ids)

        hr_h.append(hit_rate_at_k(rh, gold))
        mrr_h.append(mrr_at_k(rh, gold))
        f1_h.append(f1_at_k(rh, gold))

        hr_b.append(hit_rate_at_k(rb, gold))
        mrr_b.append(mrr_at_k(rb, gold))
        f1_b.append(f1_at_k(rb, gold))

        if i % 20 == 0:
            print(f"  {i}/{len(questions)} questions traitées...")

    print(f"\nÉvaluation terminée en {(time.time() - start) / 60:.1f} minutes.\n")

    print("=== RÉSULTATS ===")
    print(f"{'Métrique':<10} {'Hybrid RAG':<12} {'BM25 seul':<12} {'Delta':<10}")
    for name, h, b in [(f"HR@{cfg.TOP_K_FINAL}", hr_h, hr_b),
                       (f"MRR@{cfg.TOP_K_FINAL}", mrr_h, mrr_b),
                       (f"F1@{cfg.TOP_K_FINAL}", f1_h, f1_b)]:
        mh, mb = np.mean(h), np.mean(b)
        print(f"{name:<10} {mh:<12.3f} {mb:<12.3f} {mh - mb:+.3f}")

    print("\n=== Bootstrap (% de rééchantillonnages avec delta > 0) ===")
    print(f"HR  : {bootstrap_significance(np.array(hr_h) - np.array(hr_b)):.1f}%")
    print(f"MRR : {bootstrap_significance(np.array(mrr_h) - np.array(mrr_b)):.1f}%")
    print(f"F1  : {bootstrap_significance(np.array(f1_h) - np.array(f1_b)):.1f}%")
    print("(>= 95 % = statistiquement significatif)")

    print("\n=== Référence GraphRAG (test_clean.csv, 222 questions) ===")
    print("BM25 seul     : HR@10 = 0.414, MRR@10 = 0.237, F1@10 = 0.084")
    print("GraphRAG (C4) : HR@10 = 0.437, MRR@10 = 0.244, F1@10 = 0.101")

    out = cfg.BASE_DIR / f"hybrid_bsard_results_{split}.csv"
    pd.DataFrame({
        "question": questions,
        "gold_ids": gold_list,
        "hr_hybrid": hr_h, "mrr_hybrid": mrr_h, "f1_hybrid": f1_h,
        "hr_bm25": hr_b, "mrr_bm25": mrr_b, "f1_bm25": f1_b,
    }).to_csv(out, index=False)
    print(f"\nRésultats détaillés sauvegardés dans {out}")


if __name__ == "__main__":
    main()