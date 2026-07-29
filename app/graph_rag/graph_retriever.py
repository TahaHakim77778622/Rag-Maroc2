"""
Graph retriever : étend un ensemble de candidats "seeds" (trouvés par un retriever
lexical/vectoriel classique) en parcourant le graphe KuzuDB.

Deux mécanismes d'expansion :
  1. REFERENCE (1 à GRAPH_MAX_HOPS sauts) : un article cité par / citant un seed
     est un candidat fort (cas explicite de renvoi juridique).
  2. Voisinage hiérarchique : les articles de la MÊME division immédiate
     (chapitre/section) qu'un seed sont des candidats plus faibles
     (souvent pertinents pour les questions qui couvrent tout un régime juridique).

Le score de chaque candidat ajouté décroît avec la distance au seed.
"""

import kuzu
from config import KUZU_DB_PATH, GRAPH_MAX_HOPS


class GraphRetriever:
    def __init__(self, db_path: str = KUZU_DB_PATH):
        self.db = kuzu.Database(db_path)
        self.conn = kuzu.Connection(self.db)

    def _neighbors_via_reference(self, article_id: str):
        """Articles liés par REFERENCE dans les deux sens (1 saut)."""
        q1 = self.conn.execute(
            "MATCH (a:Article {id:$id})-[:REFERENCE]->(b:Article) RETURN b.id, b.article_ref_id",
            {"id": article_id},
        )
        q2 = self.conn.execute(
            "MATCH (a:Article {id:$id})<-[:REFERENCE]-(b:Article) RETURN b.id, b.article_ref_id",
            {"id": article_id},
        )
        neighbors = []
        for q in (q1, q2):
            while q.has_next():
                row = q.get_next()
                neighbors.append({"id": row[0], "article_ref_id": row[1]})
        return neighbors

    def _siblings_same_division(self, article_id: str):
        """Autres articles rattachés à la même Division immédiate (même chapitre/section)."""
        q = self.conn.execute(
            """MATCH (d:Division)-[:CONTIENT_DA]->(a:Article {id:$id}),
                     (d)-[:CONTIENT_DA]->(sibling:Article)
               WHERE sibling.id <> $id
               RETURN DISTINCT sibling.id, sibling.article_ref_id
               LIMIT 15""",
            {"id": article_id},
        )
        siblings = []
        while q.has_next():
            row = q.get_next()
            siblings.append({"id": row[0], "article_ref_id": row[1]})
        return siblings

    def expand(self, seed_article_ids: list[str], max_hops: int = GRAPH_MAX_HOPS,
               use_references: bool = True, use_siblings: bool = True):
        """
        Prend une liste d'IDs internes (`Article.id`, pas `article_ref_id`) et retourne
        un dict {article_id: {"score": float, "article_ref_id": str, "hop": int}}
        incluant les seeds (score=1.0) et les candidats étendus.
        """
        candidates = {sid: {"score": 1.0, "hop": 0} for sid in seed_article_ids}
        frontier = list(seed_article_ids)

        if use_references:
            for hop in range(1, max_hops + 1):
                next_frontier = []
                decay = 0.5 ** hop  # score décroissant avec la distance
                for aid in frontier:
                    for n in self._neighbors_via_reference(aid):
                        nid = n["id"]
                        if nid not in candidates:
                            candidates[nid] = {"score": decay, "hop": hop}
                            next_frontier.append(nid)
                        else:
                            candidates[nid]["score"] = max(candidates[nid]["score"], decay)
                frontier = next_frontier
                if not frontier:
                    break

        if use_siblings:
            for aid in seed_article_ids:
                for s in self._siblings_same_division(aid):
                    sid = s["id"]
                    sibling_score = 0.3
                    if sid not in candidates:
                        candidates[sid] = {"score": sibling_score, "hop": 1}
                    else:
                        candidates[sid]["score"] = max(candidates[sid]["score"], sibling_score)

        return candidates

    def close(self):
        self.conn.close()
        self.db.close()


if __name__ == "__main__":
    # Test rapide : prendre un article au hasard ayant des références, et vérifier l'expansion
    retriever = GraphRetriever()
    q = retriever.conn.execute(
        "MATCH (a:Article)-[:REFERENCE]->(b:Article) RETURN a.id LIMIT 1"
    )
    seed_id = q.get_next()[0]
    result = retriever.expand([seed_id])
    print(f"Seed: {seed_id}")
    print(f"Candidats trouvés après expansion: {len(result)}")
    for aid, info in list(result.items())[:10]:
        print(" ", aid, info)
    retriever.close()