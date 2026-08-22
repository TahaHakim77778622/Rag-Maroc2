"""
Construction du graphe KuzuDB pour le corpus marocain (SGG / Bulletin Officiel).

Le problème résolu ici : un doc_id ne désigne pas un texte juridique mais un
NUMÉRO de Bulletin Officiel, qui publie des dizaines de décrets, arrêtés et lois
indépendants. Chacun a son propre « article premier », son propre « article 2 ».
Dans SGG0071, 117 chunks portent le label « article premier » — ce sont 117 textes
différents, pas 117 morceaux du même article.

Résoudre un renvoi sur la seule paire (doc_id, label) créait donc des liens vers
les articles homonymes de tous les autres textes du BO, majoritairement faux.

La solution : découper séquentiellement. En parcourant les chunks d'un BO dans
l'ordre, chaque label normalisé « 1 » marque le début d'un nouveau texte. On
obtient un texte_seq, et la résolution se fait sur (doc_id, texte_seq, label).

Limite connue : quand l'article premier d'un texte a été raté à l'extraction, ses
articles sont rattachés au texte précédent. Le découpage reste approximatif, mais
infiniment plus précis que sans lui.
"""

import json
import collections
import os
import shutil
import kuzu

from config import CHUNKS_PATH, KUZU_DB_PATH, LOG_EVERY, MAX_SAME_DOC
from schema import create_schema
from entity_extraction import extract_references, normalize_label


def load_chunks():
    """Charge le JSONL, normalise les labels et attribue un texte_seq."""
    chunks = []
    with open(CHUNKS_PATH, encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            o = json.loads(line)
            o["label_norm"] = normalize_label(o.get("label"))
            chunks.append(o)

    # --- Découpage séquentiel en textes juridiques ---
    # Les chunks sont supposés dans l'ordre du document. Chaque label « 1 »
    # ouvre un nouveau texte au sein du même BO.
    seq_by_doc = collections.defaultdict(int)
    for c in chunks:
        doc_id = c["doc_id"]
        if c["label_norm"] == "1":
            seq_by_doc[doc_id] += 1
        # Si un texte commence sans article premier détecté, seq vaut 0 :
        # ses chunks forment alors un groupe « en-tête » du document.
        c["texte_seq"] = seq_by_doc[doc_id]

    n_textes = sum(seq_by_doc.values())
    print(f"{len(chunks)} chunks chargés.")
    print(f"{n_textes} textes juridiques identifiés dans {len(seq_by_doc)} documents "
          f"(moyenne {n_textes/max(len(seq_by_doc),1):.1f} textes/BO).")
    return chunks


def build_hierarchy(chunks, conn):
    """
    Crée les nœuds Document et Chunk, les arêtes CONTIENT et MEME_TEXTE.
    Retourne le lookup (doc_id, texte_seq, label_norm) -> [chunk_id, ...]
    """
    docs_seen = set()
    lookup = collections.defaultdict(list)
    by_texte = collections.defaultdict(list)

    for i, c in enumerate(chunks, 1):
        doc_id = c["doc_id"]

        if doc_id not in docs_seen:
            conn.execute(
                "MERGE (d:Document {id: $id}) SET d.title = $title, "
                "d.doc_type = $dt, d.bo_number = $bo",
                {
                    "id": doc_id,
                    "title": str(c.get("title") or ""),
                    "dt": str(c.get("document_type") or ""),
                    "bo": str(c.get("bo_number") or ""),
                },
            )
            docs_seen.add(doc_id)

        conn.execute(
            "MERGE (k:Chunk {id: $id}) SET k.doc_id = $doc_id, k.texte_seq = $seq, "
            "k.label = $label, k.label_norm = $ln, k.texte = $texte",
            {
                "id": c["chunk_id"],
                "doc_id": doc_id,
                "seq": int(c["texte_seq"]),
                "label": str(c.get("label") or ""),
                "ln": str(c.get("label_norm") or ""),
                "texte": str(c.get("text") or "")[:8000],
            },
        )
        conn.execute(
            "MATCH (d:Document {id:$did}), (k:Chunk {id:$kid}) "
            "CREATE (d)-[:CONTIENT]->(k)",
            {"did": doc_id, "kid": c["chunk_id"]},
        )

        key = (doc_id, c["texte_seq"])
        if c["label_norm"]:
            lookup[(doc_id, c["texte_seq"], c["label_norm"])].append(c["chunk_id"])
        by_texte[key].append(c["chunk_id"])

        if i % LOG_EVERY == 0:
            print(f"  ... {i} chunks insérés")

    print(f"Hiérarchie : {len(docs_seen)} documents, {len(chunks)} chunks.")

    # --- Arêtes MEME_TEXTE ---
    # Remplace les « siblings » de BSARD. Le voisinage est désormais le texte
    # juridique, pas le BO entier : bien plus pertinent sémantiquement.
    n_same = 0
    for (doc_id, seq), chunk_ids in by_texte.items():
        for idx, kid in enumerate(chunk_ids):
            for other in chunk_ids[idx + 1: idx + 1 + MAX_SAME_DOC]:
                conn.execute(
                    "MATCH (a:Chunk {id:$a}), (b:Chunk {id:$b}) "
                    "CREATE (a)-[:MEME_TEXTE]->(b)",
                    {"a": kid, "b": other},
                )
                n_same += 1
    print(f"Arêtes MEME_TEXTE : {n_same} (sur {len(by_texte)} textes)")

    return lookup


def build_references(chunks, conn, lookup):
    """Résout les renvois internes en arêtes REFERENCE, dans le périmètre du texte."""
    stats = collections.Counter()
    seen_pairs = set()
    resolved = 0
    unresolved = 0

    for i, c in enumerate(chunks, 1):
        src = c["chunk_id"]
        key_prefix = (c["doc_id"], c["texte_seq"])

        for num, kind in extract_references(c.get("text", "")):
            stats[kind] += 1

            # Les visas externes pointent hors corpus : on les écarte.
            if kind == "externe":
                continue

            target_label = normalize_label(num)
            if not target_label:
                continue

            # Résolution dans le périmètre du TEXTE, plus du BO entier.
            targets = lookup.get(key_prefix + (target_label,), [])
            if not targets:
                unresolved += 1
                continue

            for tgt in targets:
                if tgt == src:
                    continue
                pair = (src, tgt)
                if pair in seen_pairs:
                    continue
                seen_pairs.add(pair)

                conn.execute(
                    "MATCH (a:Chunk {id:$a}), (b:Chunk {id:$b}) "
                    "CREATE (a)-[:REFERENCE]->(b)",
                    {"a": src, "b": tgt},
                )
                resolved += 1

        if i % LOG_EVERY == 0:
            print(f"  ... {i} chunks analysés, {resolved} arêtes REFERENCE")

    total = sum(stats.values())
    print(f"\nRenvois détectés : {total}")
    for k, v in stats.most_common():
        print(f"  {k:<10} : {v:>6} ({100*v/total:.0f}%)")
    interne = stats["interne"] + stats["ambigu"]
    print(f"\nRenvois exploitables (interne + ambigu) : {interne}")
    print(f"  non résolus (cible absente du texte)   : {unresolved}")
    print(f"Arêtes REFERENCE créées : {resolved}")


def build_graph():
    # kuzu crée parfois un fichier, parfois un dossier — gérer les deux cas
    if os.path.isdir(KUZU_DB_PATH):
        shutil.rmtree(KUZU_DB_PATH)
    elif os.path.exists(KUZU_DB_PATH):
        os.remove(KUZU_DB_PATH)
    for ext in (".wal", ".tmp"):
        p = KUZU_DB_PATH + ext
        if os.path.exists(p):
            os.remove(p)
    os.makedirs(os.path.dirname(KUZU_DB_PATH), exist_ok=True)

    create_schema()

    chunks = load_chunks()
    db = kuzu.Database(KUZU_DB_PATH)
    conn = kuzu.Connection(db)

    conn.execute("BEGIN TRANSACTION")
    lookup = build_hierarchy(chunks, conn)
    build_references(chunks, conn, lookup)
    conn.execute("COMMIT")

    conn.close()
    db.close()
    print("\nGraphe construit dans", KUZU_DB_PATH)


if __name__ == "__main__":
    build_graph()