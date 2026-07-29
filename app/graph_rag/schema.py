"""
Schéma du graphe juridique (graph_rag).

Nœuds :
  - Code      : ex. "Code Civil", "Code Pénal"
  - Division  : nœud générique pour book/part/act/chapter/section/subsection
                (niveau exact stocké dans l'attribut `level`)
  - Article   : unité de retrieval finale

Relations :
  - CONTIENT_CD : Code -> Division      (division de premier niveau)
  - CONTIENT_DD : Division -> Division  (sous-division)
  - CONTIENT_DA : Division -> Article
  - CONTIENT_CA : Code -> Article       (article directement sous le code, sans division)
  - REFERENCE   : Article -> Article    (référence croisée détectée par regex — étape entity_extraction)
"""

import kuzu
from config import KUZU_DB_PATH


def create_schema(db_path: str = KUZU_DB_PATH):
    db = kuzu.Database(db_path)
    conn = kuzu.Connection(db)

    conn.execute("""
        CREATE NODE TABLE IF NOT EXISTS Code(
            id STRING, name STRING, law_type STRING, PRIMARY KEY (id)
        )
    """)
    conn.execute("""
        CREATE NODE TABLE IF NOT EXISTS Division(
            id STRING, level STRING, name STRING, code_name STRING, PRIMARY KEY (id)
        )
    """)
    conn.execute("""
        CREATE NODE TABLE IF NOT EXISTS Article(
            id STRING, article_ref_id STRING, reference STRING,
            texte STRING, code_name STRING, PRIMARY KEY (id)
        )
    """)

    conn.execute("CREATE REL TABLE IF NOT EXISTS CONTIENT_CD(FROM Code TO Division)")
    conn.execute("CREATE REL TABLE IF NOT EXISTS CONTIENT_DD(FROM Division TO Division)")
    conn.execute("CREATE REL TABLE IF NOT EXISTS CONTIENT_DA(FROM Division TO Article)")
    conn.execute("CREATE REL TABLE IF NOT EXISTS CONTIENT_CA(FROM Code TO Article)")
    conn.execute("CREATE REL TABLE IF NOT EXISTS REFERENCE(FROM Article TO Article)")

    print("Schéma créé dans", db_path)
    conn.close()
    db.close()


if __name__ == "__main__":
    create_schema()