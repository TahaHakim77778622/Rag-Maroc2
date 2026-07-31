
import pickle
import re
from typing import List, Tuple
from rank_bm25 import BM25Okapi

from . import config

FRENCH_STOPWORDS = {
    "le", "la", "les", "de", "des", "du", "un", "une", "et", "ou", "à", "au", "aux",
    "en", "dans", "sur", "pour", "par", "est", "sont", "que", "qui", "ce", "cette",
    "ces", "son", "sa", "ses", "avec", "il", "elle", "ne", "pas", "se", "d", "l",
}


def tokenize(text: str) -> List[str]:
    """Tokenisation simple : minuscule + suppression ponctuation + retrait stopwords."""
    text = text.lower()
    tokens = re.findall(r"\b\w+\b", text, flags=re.UNICODE)
    return [t for t in tokens if t not in FRENCH_STOPWORDS and len(t) > 1]


def build_bm25_index(chunk_ids: List[str], texts: List[str]) -> BM25Okapi:
    """Construit et sauvegarde l'index BM25."""
    tokenized_corpus = [tokenize(t) for t in texts]
    bm25 = BM25Okapi(tokenized_corpus)

    config.INDICES_DIR.mkdir(parents=True, exist_ok=True)
    with open(config.BM25_INDEX_PATH, "wb") as f:
        pickle.dump({"bm25": bm25, "chunk_ids": chunk_ids}, f)

    print(f"[bm25_retriever] Index BM25 construit sur {len(texts)} chunks.")
    return bm25


def load_bm25_index():
    """Charge l'index BM25 et la liste des chunk_ids alignée."""
    if not config.BM25_INDEX_PATH.exists():
        raise FileNotFoundError(
            f"Index BM25 introuvable : {config.BM25_INDEX_PATH}. Lance build_indices.py d'abord."
        )

    with open(config.BM25_INDEX_PATH, "rb") as f:
        data = pickle.load(f)
    return data["bm25"], data["chunk_ids"]


def search_bm25(query: str, top_k: int = None) -> List[Tuple[str, float]]:
    """Recherche les top_k chunks les plus pertinents lexicalement. Retourne [(chunk_id, score), ...]."""
    top_k = top_k or config.TOP_K_BM25
    bm25, chunk_ids = load_bm25_index()

    tokenized_query = tokenize(query)
    scores = bm25.get_scores(tokenized_query)

    ranked = sorted(zip(chunk_ids, scores), key=lambda x: x[1], reverse=True)
    return ranked[:top_k]