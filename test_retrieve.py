
from app.hybrid_rag.pipeline import retrieve

query = "Quelles sont les conditions pour créer une SARL au Maroc ?"

print(f"Requête : {query}\n")
results = retrieve(query)

print(f"=== {len(results)} chunks récupérés ===\n")
for i, r in enumerate(results, start=1):
    print(f"{i}. [{r['score']:.4f}] {r.get('title', 'N/A')}")
    print(f"   {r['text'][:200]}...")
    print()