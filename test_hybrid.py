
from app.hybrid_rag.pipeline import answer_query

query = "Quelles sont les conditions pour créer une SARL au Maroc ?"

print(f"Requête : {query}\n")

result = answer_query(query)

print("=== RÉPONSE ===")
print(result["answer"])

print("\n=== SOURCES ===")
for i, s in enumerate(result["sources"], start=1):
    print(f"{i}. [{s['score']:.4f}] {s.get('title', 'N/A')}")