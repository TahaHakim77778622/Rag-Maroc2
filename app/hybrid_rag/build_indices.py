""

from . import config
from .data_loader import load_corpus, get_texts_and_ids
from .embeddings import build_faiss_index
from .bm25_retriever import build_bm25_index


def main():
    print("=== Construction des index Hybrid RAG ===")
    chunks = load_corpus(config.FINAL_CHUNKS_PATH)
    chunk_ids, texts = get_texts_and_ids(chunks)

    print("\n--- FAISS (dense, bge-m3) ---")
    build_faiss_index(chunk_ids, texts)

    print("\n--- BM25 (sparse) ---")
    build_bm25_index(chunk_ids, texts)

    print("\nIndex construits avec succès dans :", config.INDICES_DIR)


if __name__ == "__main__":
    main()