
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

# ============ CONFIG ============
DATA_DIR = Path("data/graph_rag")  # ajuste si tes fichiers BSARD sont ailleurs
ARTICLES_CSV = DATA_DIR / "articles_clean.csv"

INDICES_DIR = Path("app/hybrid_rag/indices_bsard")
INDICES_DIR.mkdir(parents=True, exist_ok=True)
FAISS_INDEX_PATH = INDICES_DIR / "faiss_bsard.bin"
FAISS_IDMAP_PATH = INDICES_DIR / "faiss_idmap_bsard.pkl"
BM25_INDEX_PATH = INDICES_DIR / "bm25_bsard.pkl"

EMBED_MODEL = "BAAI/bge-m3"
RERANKER_MODEL = "BAAI/bge-reranker-v2-m3"
RERANKER_ENABLED = True

TOP_K_FAISS = 30
TOP_K_BM25 = 30
RRF_K = 60
TOP_K_HYBRID = 15
TOP_K_FINAL = 10  # HR@10 / MRR@10 / F1@10, comme GraphRAG

N_BOOTSTRAP = 1000


# ============ CHARGEMENT BSARD ============

def parse_article_ids(raw) -> list:
    """Parse une chaîne d'IDs d'articles gold (espace-séparés, bug connu du dataset BSARD)."""
    return [int(x) for x in re.findall(r"\d+", str(raw))]


def detect_text_column(df: pd.DataFrame, exclude: list) -> str:
    """Détecte automatiquement la colonne contenant le texte de l'article
    (la colonne restante avec la plus grande longueur moyenne)."""
    candidates = [c for c in df.columns if c not in exclude]
    avg_lengths = {c: df[c].astype(str).str.len().mean() for c in candidates}
    return max(avg_lengths, key=avg_lengths.get)


def load_articles():
    df = pd.read_csv(ARTICLES_CSV)
    id_col = "id" if "id" in df.columns else df.columns[0]
    hierarchy_cols = [c for c in ["book", "part", "act", "chapter", "section", "subsection"] if c in df.columns]
    text_col = detect_text_column(df, exclude=[id_col] + hierarchy_cols)

    print(f"[bsard] Colonne ID détectée   : '{id_col}'")
    print(f"[bsard] Colonne texte détectée : '{text_col}'")

    article_ids = df[id_col].astype(int).tolist()
    texts = df[text_col].astype(str).tolist()
    return article_ids, texts


def load_questions(split: str):
    path = DATA_DIR / f"{split}_clean.csv"
    df = pd.read_csv(path)
    question_col = "question" if "question" in df.columns else df.columns[0]
    gold_candidates = [c for c in df.columns if "article" in c.lower()]
    gold_col = "article_ids" if "article_ids" in df.columns else gold_candidates[0]

    print(f"[bsard] Colonne question détectée : '{question_col}'")
    print(f"[bsard] Colonne gold IDs détectée  : '{gold_col}'")

    questions = df[question_col].astype(str).tolist()
    gold_ids = [parse_article_ids(x) for x in df[gold_col].tolist()]
    return questions, gold_ids


# ============ INDEXATION ============

def tokenize(text: str) -> list:
    text = text.lower()
    return re.findall(r"\b\w+\b", text, flags=re.UNICODE)


def build_or_load_indices(article_ids, texts):
    if FAISS_INDEX_PATH.exists() and BM25_INDEX_PATH.exists():
        print("[bsard] Index existants trouvés, chargement...")
        index = faiss.read_index(str(FAISS_INDEX_PATH))
        with open(FAISS_IDMAP_PATH, "rb") as f:
            idmap = pickle.load(f)
        with open(BM25_INDEX_PATH, "rb") as f:
            bm25_data = pickle.load(f)
        return index, idmap, bm25_data["bm25"], bm25_data["article_ids"]

    print(f"[bsard] Construction des index sur {len(texts)} articles (peut prendre plusieurs heures sur CPU)...")

    print("[bsard] Chargement bge-m3...")
    model = SentenceTransformer(EMBED_MODEL)
    model.max_seq_length = 512

    print("[bsard] Encodage FAISS...")
    vectors = model.encode(
        texts, batch_size=8, show_progress_bar=True,
        normalize_embeddings=True, convert_to_numpy=True
    ).astype("float32")

    dim = vectors.shape[1]
    index = faiss.IndexFlatIP(dim)
    index.add(vectors)
    idmap = {i: article_ids[i] for i in range(len(article_ids))}

    faiss.write_index(index, str(FAISS_INDEX_PATH))
    with open(FAISS_IDMAP_PATH, "wb") as f:
        pickle.dump(idmap, f)

    print("[bsard] Construction BM25...")
    tokenized_corpus = [tokenize(t) for t in texts]
    bm25 = BM25Okapi(tokenized_corpus)
    with open(BM25_INDEX_PATH, "wb") as f:
        pickle.dump({"bm25": bm25, "article_ids": article_ids}, f)

    print("[bsard] Index construits.\n")
    return index, idmap, bm25, article_ids


# ============ RETRIEVAL ============

_reranker = None


def get_reranker():
    global _reranker
    if _reranker is None and RERANKER_ENABLED:
        from FlagEmbedding import FlagReranker
        print(f"[bsard] Chargement reranker {RERANKER_MODEL}...")
        _reranker = FlagReranker(RERANKER_MODEL, use_fp16=True)
    return _reranker


def reciprocal_rank_fusion(faiss_results, bm25_results, k=RRF_K, top_k=TOP_K_HYBRID):
    scores = {}
    for rank, (aid, _) in enumerate(faiss_results, start=1):
        scores[aid] = scores.get(aid, 0.0) + 1.0 / (k + rank)
    for rank, (aid, _) in enumerate(bm25_results, start=1):
        scores[aid] = scores.get(aid, 0.0) + 1.0 / (k + rank)
    fused = sorted(scores.items(), key=lambda x: x[1], reverse=True)
    return fused[:top_k]


def retrieve_bm25_only(query, bm25, bm25_article_ids, top_k=TOP_K_FINAL):
    """Baseline : BM25 seul, pour la comparaison bootstrap."""
    tokenized_q = tokenize(query)
    scores = bm25.get_scores(tokenized_q)
    ranked = sorted(zip(bm25_article_ids, scores), key=lambda x: x[1], reverse=True)
    return [aid for aid, _ in ranked[:top_k]]


def retrieve_hybrid(query, model, index, idmap, bm25, bm25_article_ids, texts_by_id, top_k=TOP_K_FINAL):
    q_vec = model.encode([query], normalize_embeddings=True, convert_to_numpy=True).astype("float32")
    scores, indices = index.search(q_vec, TOP_K_FAISS)
    faiss_results = [(idmap[idx], float(score)) for idx, score in zip(indices[0], scores[0]) if idx != -1]

    tokenized_q = tokenize(query)
    bm25_scores = bm25.get_scores(tokenized_q)
    bm25_ranked = sorted(zip(bm25_article_ids, bm25_scores), key=lambda x: x[1], reverse=True)[:TOP_K_BM25]

    fused = reciprocal_rank_fusion(faiss_results, bm25_ranked)

    if RERANKER_ENABLED:
        reranker = get_reranker()
        valid_ids = [aid for aid, _ in fused if aid in texts_by_id]
        pairs = [[query, texts_by_id[aid]] for aid in valid_ids]
        if pairs:
            rerank_scores = reranker.compute_score(pairs, normalize=True)
            if isinstance(rerank_scores, float):
                rerank_scores = [rerank_scores]
            fused = sorted(zip(valid_ids, rerank_scores), key=lambda x: x[1], reverse=True)

    return [aid for aid, _ in fused[:top_k]]


# ============ METRIQUES ============

def hit_rate_at_k(retrieved, gold):
    return 1.0 if any(g in retrieved for g in gold) else 0.0


def mrr_at_k(retrieved, gold):
    for rank, aid in enumerate(retrieved, start=1):
        if aid in gold:
            return 1.0 / rank
    return 0.0


def f1_at_k(retrieved, gold):
    retrieved_set = set(retrieved)
    gold_set = set(gold)
    if not retrieved_set or not gold_set:
        return 0.0
    tp = len(retrieved_set & gold_set)
    precision = tp / len(retrieved_set)
    recall = tp / len(gold_set)
    if precision + recall == 0:
        return 0.0
    return 2 * precision * recall / (precision + recall)


# ============ BOOTSTRAP ============

def bootstrap_significance(deltas, n_resamples=N_BOOTSTRAP):
    """% de resamples où la moyenne des deltas (Hybrid - BM25) > 0."""
    deltas = np.array(deltas)
    n = len(deltas)
    positive_count = 0
    for _ in range(n_resamples):
        sample = np.random.choice(deltas, size=n, replace=True)
        if sample.mean() > 0:
            positive_count += 1
    return 100 * positive_count / n_resamples


# ============ MAIN ============

def main():
    split = "test"
    if "--split" in sys.argv:
        split = sys.argv[sys.argv.index("--split") + 1]

    print(f"=== Benchmark Hybrid RAG sur BSARD ({split}_clean.csv) ===\n")

    article_ids, texts = load_articles()
    texts_by_id = dict(zip(article_ids, texts))
    questions, gold_ids_list = load_questions(split)
    print(f"[bsard] {len(questions)} questions chargées.\n")

    index, idmap, bm25, bm25_article_ids = build_or_load_indices(article_ids, texts)

    print("[bsard] Chargement bge-m3 pour les requêtes...")
    model = SentenceTransformer(EMBED_MODEL)
    model.max_seq_length = 512

    hr_hybrid, mrr_hybrid, f1_hybrid = [], [], []
    hr_bm25, mrr_bm25, f1_bm25 = [], [], []

    start = time.time()
    for i, (query, gold) in enumerate(zip(questions, gold_ids_list), start=1):
        retrieved_hybrid = retrieve_hybrid(query, model, index, idmap, bm25, bm25_article_ids, texts_by_id)
        retrieved_bm25 = retrieve_bm25_only(query, bm25, bm25_article_ids)

        hr_hybrid.append(hit_rate_at_k(retrieved_hybrid, gold))
        mrr_hybrid.append(mrr_at_k(retrieved_hybrid, gold))
        f1_hybrid.append(f1_at_k(retrieved_hybrid, gold))

        hr_bm25.append(hit_rate_at_k(retrieved_bm25, gold))
        mrr_bm25.append(mrr_at_k(retrieved_bm25, gold))
        f1_bm25.append(f1_at_k(retrieved_bm25, gold))

        if i % 20 == 0:
            print(f"  {i}/{len(questions)} questions traitées...")

    elapsed = time.time() - start
    print(f"\nÉvaluation terminée en {elapsed / 60:.1f} minutes.\n")

    print("=== RÉSULTATS ===")
    print(f"{'Métrique':<10} {'Hybrid RAG':<12} {'BM25 seul':<12} {'Delta':<10}")
    for name, h, b in [
        (f"HR@{TOP_K_FINAL}", hr_hybrid, hr_bm25),
        (f"MRR@{TOP_K_FINAL}", mrr_hybrid, mrr_bm25),
        (f"F1@{TOP_K_FINAL}", f1_hybrid, f1_bm25),
    ]:
        mh, mb = np.mean(h), np.mean(b)
        print(f"{name:<10} {mh:<12.3f} {mb:<12.3f} {mh - mb:+.3f}")

    print("\n=== Bootstrap (significativité, % resamples avec delta > 0) ===")
    hr_deltas = np.array(hr_hybrid) - np.array(hr_bm25)
    mrr_deltas = np.array(mrr_hybrid) - np.array(mrr_bm25)
    f1_deltas = np.array(f1_hybrid) - np.array(f1_bm25)

    print(f"HR@{TOP_K_FINAL}  : {bootstrap_significance(hr_deltas):.1f}% positif")
    print(f"MRR@{TOP_K_FINAL} : {bootstrap_significance(mrr_deltas):.1f}% positif")
    print(f"F1@{TOP_K_FINAL}  : {bootstrap_significance(f1_deltas):.1f}% positif")
    print("(>= 95% = statistiquement significatif, comme dans ton benchmark GraphRAG)")

    print("\n=== Rappel GraphRAG (mémoire du projet, sur test_clean.csv) ===")
    print("BM25 baseline : Recall@10 = 0.199, MRR = 0.228")
    print("GraphRAG (C4) : Recall@10 = 0.233, MRR = 0.232")

    results_df = pd.DataFrame({
        "question": questions,
        "gold_ids": gold_ids_list,
        "hr_hybrid": hr_hybrid, "mrr_hybrid": mrr_hybrid, "f1_hybrid": f1_hybrid,
        "hr_bm25": hr_bm25, "mrr_bm25": mrr_bm25, "f1_bm25": f1_bm25,
    })
    out_path = f"hybrid_bsard_results_{split}.csv"
    results_df.to_csv(out_path, index=False)
    print(f"\nRésultats détaillés sauvegardés dans {out_path}")


if __name__ == "__main__":
    main()