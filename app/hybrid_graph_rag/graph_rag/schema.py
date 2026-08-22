"""
Schéma du graphe pour le corpus marocain.

Nœuds :
  - Document : un texte du Bulletin Officiel (doc_id)
  - Chunk    : une unité de retrieval (chunk_id), avec son label normalisé

Relations :
  - CONTIENT   : Document -> Chunk
  - REFERENCE  : Chunk -> Chunk (renvoi interne résolu)
  - MEME_DOC   : Chunk -> Chunk (voisinage dans le même document)
"""
import kuzu
from config import KUZU_DB_PATH


def create_schema(db_path: str = KUZU_DB_PATH):
    db = kuzu.Database(db_path)
    conn = kuzu.Connection(db)

    conn.execute("""
        CREATE NODE TABLE IF NOT EXISTS Document(
            id STRING, title STRING, doc_type STRING,
            bo_number STRING, PRIMARY KEY (id)
        )
    """)
    conn.execute("""
        CREATE NODE TABLE IF NOT EXISTS Chunk(
            id STRING, doc_id STRING, texte_seq INT64,
            label STRING, label_norm STRING, texte STRING,
            PRIMARY KEY (id)
        )
    """)

    conn.execute("CREATE REL TABLE IF NOT EXISTS CONTIENT(FROM Document TO Chunk)")
    conn.execute("CREATE REL TABLE IF NOT EXISTS REFERENCE(FROM Chunk TO Chunk)")
    conn.execute("CREATE REL TABLE IF NOT EXISTS MEME_TEXTE(FROM Chunk TO Chunk)")
    print("Schéma créé dans", db_path)
    conn.close()
    db.close()


if __name__ == "__main__":
    create_schema()