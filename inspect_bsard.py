
import pandas as pd
from pathlib import Path

# Ajuste ce chemin si besoin — normalement data/graph_rag/ vu ton implémentation GraphRAG
DATA_DIR = Path("data/graph_rag")

for filename in ["articles_clean.csv", "train_clean.csv", "test_clean.csv"]:
    path = DATA_DIR / filename
    print(f"\n{'=' * 50}")
    print(f"Fichier : {path}")
    if not path.exists():
        print("INTROUVABLE à ce chemin.")
        continue

    df = pd.read_csv(path)
    print(f"Shape : {df.shape}")
    print(f"Colonnes : {list(df.columns)}")
    print("\nPremière ligne :")
    print(df.iloc[0].to_dict())