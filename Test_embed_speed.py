
import time
from app.hybrid_rag import config
from app.hybrid_rag.data_loader import load_corpus, get_texts_and_ids
from sentence_transformers import SentenceTransformer

print("Chargement du corpus...")
chunks = load_corpus(config.FINAL_CHUNKS_PATH)
chunk_ids, texts = get_texts_and_ids(chunks)

print(f"Corpus total : {len(texts)} chunks")

# Stats sur la longueur des textes (en caractères, proxy grossier des tokens)
lengths = [len(t) for t in texts]
print(f"Longueur texte - min: {min(lengths)}, max: {max(lengths)}, moyenne: {sum(lengths)//len(lengths)}")

# Test sur un petit échantillon
N_TEST = 20
sample = texts[:N_TEST]

print(f"\nChargement du modèle {config.EMBED_MODEL}...")
model = SentenceTransformer(config.EMBED_MODEL)

# Limiter la longueur max pour éviter les chunks anormalement longs
model.max_seq_length = 512
print(f"max_seq_length fixé à {model.max_seq_length}")

print(f"\nEncodage de {N_TEST} chunks (test de vitesse)...")
start = time.time()
vectors = model.encode(sample, batch_size=8, show_progress_bar=True, normalize_embeddings=True)
elapsed = time.time() - start

print(f"\n=== RÉSULTAT ===")
print(f"{N_TEST} chunks encodés en {elapsed:.1f} secondes")
print(f"Vitesse : {elapsed / N_TEST:.2f} sec/chunk")

total_estimated_sec = (elapsed / N_TEST) * len(texts)
print(f"\nEstimation pour {len(texts)} chunks : {total_estimated_sec / 60:.1f} minutes (~{total_estimated_sec / 3600:.1f} heures)")