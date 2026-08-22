"""
Graph retriever pour le corpus marocain.

Étend un ensemble de chunks "seeds" (fournis par Hybrid RAG) en parcourant le
graphe KuzuDB. Deux mécanismes, activables séparément :

  1. REFERENCE : un chunk cité par / citant un seed. Signal fort — c'est un
     renvoi juridique explicite entre articles du même document.
  2. MEME_TEXTE : les chunks voisins dans le même document. Signal plus faible,
     équivalent des "siblings" de la version BSARD, mais fondé sur la proximité
     dans le document plutôt que sur le chapitre (le corpus marocain n'a pas de
     hiérarchie multi-niveaux).

Le score de chaque candidat décroît avec la distance au seed.
"""

import kuzu
from config import KUZU_DB_PATH, GRAPH_MAX_HOPS, MAX_SAME_DOC


class GraphRetriever:
    def __init__(self, db_path: str = KUZU_DB_PATH):
        self.db = kuzu.Database(db_path)
        self.conn = kuzu.Connection(self.db)

    def _neighbors_via_reference(self, chunk_id: str):
        """Chunks liés par REFERENCE, dans les deux sens (1 saut)."""
        q1 = self.conn.execute(
            "MATCH (a:Chunk {id:$id})-[:REFERENCE]->(b:Chunk) RETURN b.id",
            {"id": chunk_id},
        )
        q2 = self.conn.execute(
            "MATCH (a:Chunk {id:$id})<-[:REFERENCE]-(b:Chunk) RETURN b.id",
            {"id": chunk_id},
        )
        out = []
        for q in (q1, q2):
            while q.has_next():
                out.append(q.get_next()[0])
        return out

    def _neighbors_same_doc(self, chunk_id: str):
        """Chunks voisins dans le même document, via MEME_TEXTE (deux sens)."""
        q1 = self.conn.execute(
            f"""MATCH (a:Chunk {{id:$id}})-[:MEME_TEXTE]->(b:Chunk)
                RETURN DISTINCT b.id LIMIT {MAX_SAME_DOC}""",
            {"id": chunk_id},
        )
        q2 = self.conn.execute(
            f"""MATCH (a:Chunk {{id:$id}})<-[:MEME_TEXTE]-(b:Chunk)
                RETURN DISTINCT b.id LIMIT {MAX_SAME_DOC}""",
            {"id": chunk_id},
        )
        out = []
        for q in (q1, q2):
            while q.has_next():
                out.append(q.get_next()[0])
        return out

    def expand(
        self,
        seed_ids: list,
        max_hops: int = GRAPH_MAX_HOPS,
        use_references: bool = True,
        use_same_doc: bool = True,
    ):
        """
        Prend une liste de chunk_id et retourne
        {chunk_id: {"score": float, "hop": int}}, seeds inclus (score 1.0).
        """
        candidates = {sid: {"score": 1.0, "hop": 0} for sid in seed_ids}
        frontier = list(seed_ids)

        if use_references:
            for hop in range(1, max_hops + 1):
                next_frontier = []
                decay = 0.5 ** hop
                for cid in frontier:
                    for nid in self._neighbors_via_reference(cid):
                        if nid not in candidates:
                            candidates[nid] = {"score": decay, "hop": hop}
                            next_frontier.append(nid)
                        else:
                            candidates[nid]["score"] = max(candidates[nid]["score"], decay)
                frontier = next_frontier
                if not frontier:
                    break

        if use_same_doc:
            for cid in seed_ids:
                for nid in self._neighbors_same_doc(cid):
                    if nid not in candidates:
                        candidates[nid] = {"score": 0.3, "hop": 1}
                    else:
                        candidates[nid]["score"] = max(candidates[nid]["score"], 0.3)

        return candidates

    def close(self):
        self.conn.close()
        self.db.close()


if __name__ == "__main__":
    retriever = GraphRetriever()

    q = retriever.conn.execute(
        "MATCH (a:Chunk)-[:REFERENCE]->(b:Chunk) RETURN a.id LIMIT 1"
    )
    if q.has_next():
        seed = q.get_next()[0]
        result = retriever.expand([seed])
        print(f"Seed : {seed}")
        print(f"Candidats après expansion : {len(result)}")
        for cid, info in list(result.items())[:10]:
            print(" ", cid, info)
    else:
        print("Aucune arête REFERENCE dans le graphe — lance graph_builder.py d'abord.")

    retriever.close()