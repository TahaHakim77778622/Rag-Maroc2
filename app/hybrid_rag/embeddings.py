

import pickle
import numpy as np
import faiss
from typing import List, Tuple
from sentence_transformers import SentenceTransformer

from . import config

_model = None


def get_embed_model() -> SentenceTransformer:
    """Charge le modèle bge-m3 une seule fois (singleton)."""
    global _model
    if _model is None:
        print(f"[embeddings] Chargement du modèle {config.EMBED_MODEL} ...")
        _model = SentenceTransformer(config.EMBED_MODEL)
        _model.max_seq_length = 512  # évite les blocages sur chunks anormalement longs (CPU)
    return _model


def embed_texts(texts: List[str], batch_size: int = 8) -> np.ndarray:
    """Encode une liste de textes en vecteurs denses normalisés (pour cosine via IP)."""
    model = get_embed_model()
    vectors = model.encode(
        texts,
        batch_size=batch_size,
        show_progress_bar=True,
        normalize_embeddings=True,  # cosine similarity via produit scalaire
        convert_to_numpy=True,
    )
    return vectors.astype("float32")


def build_faiss_index(chunk_ids: List[str], texts: List[str]) -> Tuple[faiss.Index, dict]:
    """Construit un index FAISS (IndexFlatIP) à partir des textes, et sauvegarde sur disque."""
    vectors = embed_texts(texts)
    dim = vectors.shape[1]

    index = faiss.IndexFlatIP(dim)
    index.add(vectors)

    idmap = {i: chunk_id for i, chunk_id in enumerate(chunk_ids)}

    config.INDICES_DIR.mkdir(parents=True, exist_ok=True)
    faiss.write_index(index, str(config.FAISS_INDEX_PATH))
    with open(config.FAISS_IDMAP_PATH, "wb") as f:
        pickle.dump(idmap, f)

    print(f"[embeddings] Index FAISS construit : {index.ntotal} vecteurs, dim={dim}")
    return index, idmap


def load_faiss_index() -> Tuple[faiss.Index, dict]:
    """Charge l'index FAISS et l'idmap depuis le disque."""
    if not config.FAISS_INDEX_PATH.exists():
        raise FileNotFoundError(
            f"Index FAISS introuvable : {config.FAISS_INDEX_PATH}. Lance build_indices.py d'abord."
        )

    index = faiss.read_index(str(config.FAISS_INDEX_PATH))
    with open(config.FAISS_IDMAP_PATH, "rb") as f:
        idmap = pickle.load(f)
    return index, idmap


def search_faiss(query: str, top_k: int = None) -> List[Tuple[str, float]]:
    """Recherche les top_k chunks les plus proches d'une requête. Retourne [(chunk_id, score), ...]."""
    top_k = top_k or config.TOP_K_FAISS
    index, idmap = load_faiss_index()

    q_vector = embed_texts([query])
    scores, indices = index.search(q_vector, top_k)

    results = []
    for idx, score in zip(indices[0], scores[0]):
        if idx == -1:
            continue
        results.append((idmap[idx], float(score)))
    return results