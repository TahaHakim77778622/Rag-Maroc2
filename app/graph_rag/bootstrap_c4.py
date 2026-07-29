import time
import random
import pandas as pd
import re
from config import ARTICLES_CSV, TRAIN_CSV
from fusion import HybridGraphRetriever
from benchmark import hit_rate_at_k, mrr_at_k, f_measure_at_k

articles_df = pd.read_csv(ARTICLES_CSV)
train_df = pd.read_csv(TRAIN_CSV)
train_df["gold"] = train_df["article_ids"].apply(lambda x: re.findall(r"\d+", str(x)))

retriever = HybridGraphRetriever(articles_df)

rows = []
t0 = time.time()
for i, row in train_df.iterrows():
    question, gold = row["question"], row["gold"]
    baseline = retriever.retrieve(question, top_k=10, use_graph=False)
    graph = retriever.retrieve(question, top_k=10, use_graph=True, mode="expand_with_siblings", use_siblings=True)
    rows.append({
        "hr_b": hit_rate_at_k(baseline, gold), "hr_g": hit_rate_at_k(graph, gold),
        "mrr_b": mrr_at_k(baseline, gold), "mrr_g": mrr_at_k(graph, gold),
        "f1_b": f_measure_at_k(baseline, gold), "f1_g": f_measure_at_k(graph, gold),
    })
    if (i + 1) % 200 == 0:
        print(f"  ... {i+1}/{len(train_df)} ({round(time.time()-t0,1)}s)")
retriever.close()

df = pd.DataFrame(rows)
df.to_csv("bootstrap_c4_train.csv", index=False)

random.seed(42)
n = len(df)
n_bootstrap = 1000
deltas = {"HR": [], "MRR": [], "F1": []}

for _ in range(n_bootstrap):
    idx = [random.randint(0, n - 1) for _ in range(n)]
    sample = df.iloc[idx]
    deltas["HR"].append(sample["hr_g"].mean() - sample["hr_b"].mean())
    deltas["MRR"].append(sample["mrr_g"].mean() - sample["mrr_b"].mean())
    deltas["F1"].append(sample["f1_g"].mean() - sample["f1_b"].mean())

def pct_positive(vals):
    return 100 * sum(1 for d in vals if d > 0) / len(vals)

print(f"\n{'Metrique':<10}{'Delta observe':>16}{'% tirages > 0':>16}")
for m in ["HR", "MRR", "F1"]:
    observed = df[f"{m.lower()}_g"].mean() - df[f"{m.lower()}_b"].mean()
    print(f"{m:<10}{observed:>+16.4f}{pct_positive(deltas[m]):>15.1f}%")