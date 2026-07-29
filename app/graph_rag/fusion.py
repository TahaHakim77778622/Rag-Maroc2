"""
Fusion GraphRAG = baseline lexicale (BM25) + signal graphe (REFERENCE / siblings).

Modes de fusion (voir config.GRAPH_MODE) :
  - noop : ancien re-rank strict top_k via expand() — bug : scores seeds uniformes → no-op
  - bidir_soft : re-rank soft dans le top_k via degré REFERENCE bidirectionnel induit
  - controlled_expand : seeds = top EXPAND_N_SEEDS BM25 → voisins REFERENCE ∩ pool BM25
                         → fusion RRF (peut faire entrer des docs du pool dans le top_k)
  - controlled_expand_bidir : controlled_expand + boost soft bidir sur le ranking final
  - expand_with_siblings : controlled_expand + voisins siblings (contrôle négatif)
"""

from collections import defaultdict
import re
import pandas as pd
from rank_bm25 import BM25Okapi

from config import (
    TOP_K, GRAPH_MODE, BM25_POOL, EXPAND_N_SEEDS, BIDIR_ALPHA,
    GRAPH_LIST_WEIGHT, USE_SIBLINGS_DEFAULT,
)
from graph_retriever import GraphRetriever

RRF_K = 60  # constante standard de la littérature RRF (Cormack et al.)

_TOKEN_RE = re.compile(r"\w+", re.UNICODE)


def tokenize(text: str):
    return _TOKEN_RE.findall(text.lower())


class HybridGraphRetriever:
    """Baseline BM25 + expansion / re-rank graphe, avec fusion des scores."""

    def __init__(self, articles_df: pd.DataFrame, db_path: str = None,
                 bm25_k1: float = 1.5, bm25_b: float = 0.75):
        self.df = articles_df.reset_index(drop=True)
        self.article_ref_to_row = {str(r["id"]): i for i, r in self.df.iterrows()}
        corpus_tokens = [tokenize(str(t)) for t in self.df["article"].fillna("")]
        self.bm25 = BM25Okapi(corpus_tokens, k1=bm25_k1, b=bm25_b)

        self.graph_retriever = GraphRetriever(db_path) if db_path else GraphRetriever()
        self._build_id_maps()
        self._load_graph_adjacency()

    def _build_id_maps(self):
        """Correspondance Article.id (kuzu) <-> article_ref_id (CSV)."""
        q = self.graph_retriever.conn.execute("MATCH (a:Article) RETURN a.id, a.article_ref_id")
        self.kuzu_id_to_ref = {}
        self.ref_to_kuzu_id = {}
        while q.has_next():
            row = q.get_next()
            self.kuzu_id_to_ref[row[0]] = row[1]
            self.ref_to_kuzu_id[row[1]] = row[0]

    def _load_graph_adjacency(self):
        """Charge les arêtes REFERENCE (dirigées) et les siblings hiérarchiques en mémoire."""
        self.ref_out = defaultdict(set)
        q = self.graph_retriever.conn.execute(
            "MATCH (a:Article)-[:REFERENCE]->(b:Article) "
            "RETURN a.article_ref_id, b.article_ref_id"
        )
        while q.has_next():
            a, b = q.get_next()
            self.ref_out[a].add(b)

        self.ref_und = defaultdict(set)
        for a, targets in self.ref_out.items():
            for b in targets:
                self.ref_und[a].add(b)
                self.ref_und[b].add(a)

        div_members = defaultdict(set)
        q = self.graph_retriever.conn.execute(
            "MATCH (d:Division)-[:CONTIENT_DA]->(a:Article) "
            "RETURN d.id, a.article_ref_id"
        )
        while q.has_next():
            did, aid = q.get_next()
            div_members[did].add(aid)

        self.siblings = defaultdict(set)
        for members in div_members.values():
            for m in members:
                self.siblings[m] = members - {m}

    def _bm25_pool(self, question: str, pool_size: int):
        scores = self.bm25.get_scores(tokenize(question))
        ranked_idx = sorted(range(len(scores)), key=lambda i: scores[i], reverse=True)[:pool_size]
        refs = [str(self.df.iloc[i]["id"]) for i in ranked_idx]
        return refs

    def _bidir_degree(self, candidates: list[str]) -> dict[str, float]:
        """Degré REFERENCE bidirectionnel induit dans l'ensemble candidats."""
        cand_set = set(candidates)
        deg = {}
        for r in candidates:
            deg[r] = sum(
                1 for x in cand_set
                if x != r and x in self.ref_out[r] and r in self.ref_out[x]
            )
        return deg

    def _soft_bidir_rerank(self, ranked: list[str], alpha: float = BIDIR_ALPHA) -> list[str]:
        """Score = RRF(BM25 rank) + alpha * bidir_norm — conserve l'ordre BM25 si signal nul."""
        bidir = self._bidir_degree(ranked)
        mx = max(bidir.values()) if bidir else 0
        scores = {}
        for rank, ref_id in enumerate(ranked, start=1):
            boost = (bidir[ref_id] / mx) if mx > 0 else 0.0
            scores[ref_id] = 1.0 / (RRF_K + rank) + alpha * boost
        return sorted(scores, key=scores.get, reverse=True)

    def _controlled_expand(
        self,
        pool: list[str],
        n_seeds: int = EXPAND_N_SEEDS,
        use_siblings: bool = False,
        graph_list_weight: float = GRAPH_LIST_WEIGHT,
    ) -> list[str]:
        """
        Seeds = top n_seeds du pool BM25.
        Candidats graphe = voisins REFERENCE (et optionnellement siblings) déjà présents
        dans le pool BM25 — pas d'injection hors signal lexical.
        Fusion RRF entre la liste BM25 (pool) et la liste graphe (seeds + hits).
        """
        seeds = pool[:n_seeds]
        pool_set = set(pool)
        seen = set(seeds)
        graph_hits = []

        for s in seeds:
            neighbors = set(self.ref_und.get(s, ()))
            if use_siblings:
                neighbors |= self.siblings.get(s, set())
            # Tri stable : l'ordre d'itération des set Python n'est pas déterministe
            # (hash randomization) et changeait les rangs RRF d'un run à l'autre.
            for n in sorted(neighbors):
                if n in pool_set and n not in seen:
                    graph_hits.append(n)
                    seen.add(n)

        graph_list = seeds + graph_hits
        scores = {}
        for rank, ref_id in enumerate(pool, start=1):
            scores[ref_id] = scores.get(ref_id, 0.0) + 1.0 / (RRF_K + rank)
        for rank, ref_id in enumerate(graph_list, start=1):
            scores[ref_id] = scores.get(ref_id, 0.0) + graph_list_weight / (RRF_K + rank)

        return sorted(scores, key=scores.get, reverse=True)

    def _noop_rerank(self, top_k_refs: list[str], use_references: bool, use_siblings: bool) -> list[str]:
        """Ancien comportement (bugué) : seeds = tout le top_k → scores uniformes → no-op."""
        seed_kuzu_ids = [
            self.ref_to_kuzu_id[ref_id]
            for ref_id in top_k_refs
            if ref_id in self.ref_to_kuzu_id
        ]
        graph_candidates = self.graph_retriever.expand(
            seed_kuzu_ids, use_references=use_references, use_siblings=use_siblings
        )
        graph_boost = {}
        for kuzu_id, info in graph_candidates.items():
            ref_id = self.kuzu_id_to_ref.get(kuzu_id)
            if ref_id in top_k_refs:
                graph_boost[ref_id] = info["score"]
        return sorted(top_k_refs, key=lambda r: graph_boost.get(r, 0.0), reverse=True)

    def retrieve(
        self,
        question: str,
        top_k: int = TOP_K,
        use_graph: bool = True,
        bm25_pool: int = BM25_POOL,
        mode: str = None,
        use_references: bool = True,
        use_siblings: bool = None,
        n_seeds: int = EXPAND_N_SEEDS,
        bidir_alpha: float = BIDIR_ALPHA,
    ):
        """
        Retourne une liste ordonnée de article_ref_id (IDs CSV = gold labels).
        """
        if mode is None:
            mode = GRAPH_MODE
        if use_siblings is None:
            use_siblings = USE_SIBLINGS_DEFAULT

        pool = self._bm25_pool(question, bm25_pool)
        baseline = pool[:top_k]

        if not use_graph:
            return baseline

        if mode == "noop":
            return self._noop_rerank(baseline, use_references, use_siblings)

        if mode == "bidir_soft":
            return self._soft_bidir_rerank(baseline, alpha=bidir_alpha)

        if mode == "controlled_expand":
            ranked = self._controlled_expand(
                pool, n_seeds=n_seeds, use_siblings=False
            )
            return ranked[:top_k]

        if mode == "controlled_expand_bidir":
            ranked = self._controlled_expand(
                pool, n_seeds=n_seeds, use_siblings=False
            )
            return self._soft_bidir_rerank(ranked[:top_k], alpha=bidir_alpha)

        if mode == "expand_with_siblings":
            ranked = self._controlled_expand(
                pool, n_seeds=n_seeds, use_siblings=True
            )
            return ranked[:top_k]

        raise ValueError(f"Unknown graph fusion mode: {mode!r}")

    def close(self):
        self.graph_retriever.close()


if __name__ == "__main__":
    from config import ARTICLES_CSV

    df = pd.read_csv(ARTICLES_CSV)
    retriever = HybridGraphRetriever(df)

    question = "A-t-on droit à l'allocation de naissance en cas de fausse couche ?"
    print("Sans graphe :", retriever.retrieve(question, use_graph=False))
    print("controlled_expand :", retriever.retrieve(question, use_graph=True, mode="controlled_expand"))
    print("bidir_soft :", retriever.retrieve(question, use_graph=True, mode="bidir_soft"))
    retriever.close()
