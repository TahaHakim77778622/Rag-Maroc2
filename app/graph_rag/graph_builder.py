"""
Construction du graphe KuzuDB complet :
  1. Hiérarchie Code -> Division -> Article (depuis les colonnes structurées)
  2. Relations REFERENCE (Article -> Article), résolues à partir des références
     croisées extraites par entity_extraction.py
"""

import hashlib
import re
import pandas as pd
import kuzu

from config import KUZU_DB_PATH, ARTICLES_CSV, HIERARCHY_LEVELS
from schema import create_schema
from entity_extraction import extract_all, normalize_number

_REF_NUM_RE = re.compile(r"^Art\.?\s*([^,]+),")


def make_id(*parts: str) -> str:
    raw = "||".join(str(p) for p in parts)
    return hashlib.md5(raw.encode("utf-8")).hexdigest()[:16]


def build_hierarchy(df: pd.DataFrame, conn: kuzu.Connection):
    """Construit Code -> Division -> Article. Retourne le lookup (code, numéro) -> article_id."""
    codes_seen, divisions_seen = {}, {}
    rel_cd_seen, rel_dd_seen = set(), set()
    lookup = {}  # (code_name, normalized_number) -> article_id
    n_articles = n_divisions = n_codes = 0

    for _, row in df.iterrows():
        code_name = row["code"]
        law_type = row["law_type"] if pd.notna(row["law_type"]) else "inconnu"
        code_id = make_id("CODE", code_name)

        if code_id not in codes_seen:
            conn.execute(
                "MERGE (c:Code {id: $id}) SET c.name = $name, c.law_type = $law_type",
                {"id": code_id, "name": code_name, "law_type": law_type},
            )
            codes_seen[code_id] = code_name
            n_codes += 1

        path_parts = [("CODE", code_name)]
        parent_id, parent_type = code_id, "Code"

        for level in HIERARCHY_LEVELS:
            val = row[level]
            if pd.isna(val):
                continue
            path_parts.append((level, val))
            div_id = make_id(*[f"{lvl}:{name}" for lvl, name in path_parts])

            if div_id not in divisions_seen:
                conn.execute(
                    "MERGE (d:Division {id: $id}) SET d.level=$level, d.name=$name, d.code_name=$code_name",
                    {"id": div_id, "level": level, "name": str(val), "code_name": code_name},
                )
                divisions_seen[div_id] = True
                n_divisions += 1

            key = (parent_id, div_id)
            if parent_type == "Code" and key not in rel_cd_seen:
                conn.execute(
                    "MATCH (c:Code {id:$cid}),(d:Division {id:$did}) CREATE (c)-[:CONTIENT_CD]->(d)",
                    {"cid": parent_id, "did": div_id},
                )
                rel_cd_seen.add(key)
            elif parent_type == "Division" and key not in rel_dd_seen:
                conn.execute(
                    "MATCH (p:Division {id:$pid}),(d:Division {id:$did}) CREATE (p)-[:CONTIENT_DD]->(d)",
                    {"pid": parent_id, "did": div_id},
                )
                rel_dd_seen.add(key)

            parent_id, parent_type = div_id, "Division"

        article_id = make_id("ARTICLE", row["id"])
        conn.execute(
            "MERGE (a:Article {id:$id}) SET a.article_ref_id=$ref_id, a.reference=$reference, "
            "a.texte=$texte, a.code_name=$code_name",
            {
                "id": article_id, "ref_id": str(row["id"]),
                "reference": str(row["reference"]) if pd.notna(row["reference"]) else "",
                "texte": str(row["article"]) if pd.notna(row["article"]) else "",
                "code_name": code_name,
            },
        )
        n_articles += 1

        if parent_type == "Division":
            conn.execute(
                "MATCH (d:Division {id:$did}),(a:Article {id:$aid}) CREATE (d)-[:CONTIENT_DA]->(a)",
                {"did": parent_id, "aid": article_id},
            )
        else:
            conn.execute(
                "MATCH (c:Code {id:$cid}),(a:Article {id:$aid}) CREATE (c)-[:CONTIENT_CA]->(a)",
                {"cid": parent_id, "aid": article_id},
            )

        # Alimenter le lookup pour la résolution des références croisées
        ref_match = _REF_NUM_RE.match(str(row["reference"])) if pd.notna(row["reference"]) else None
        if ref_match:
            num_norm = normalize_number(ref_match.group(1))
            lookup[(code_name, num_norm)] = article_id

        if n_articles % 5000 == 0:
            print(f"  ... hiérarchie: {n_articles} articles traités")

    print(f"Hiérarchie construite : {n_codes} codes, {n_divisions} divisions, {n_articles} articles.")
    return lookup


def build_references(df: pd.DataFrame, conn: kuzu.Connection, lookup: dict):
    """Résout les références croisées extraites en relations REFERENCE (Article -> Article)."""
    raw_refs = extract_all(df)

    resolved, seen_pairs = 0, set()
    for r in raw_refs:
        source_id = make_id("ARTICLE", r["source_article_ref_id"])
        target_key = (r["target_code_name"], r["target_number"])
        target_id = lookup.get(target_key)

        if target_id is None or target_id == source_id:
            continue  # référence non résolue ou auto-référence

        pair = (source_id, target_id)
        if pair in seen_pairs:
            continue
        seen_pairs.add(pair)

        conn.execute(
            "MATCH (a:Article {id:$sid}),(b:Article {id:$tid}) CREATE (a)-[:REFERENCE]->(b)",
            {"sid": source_id, "tid": target_id},
        )
        resolved += 1

    print(f"Relations REFERENCE créées : {resolved} / {len(raw_refs)} références brutes "
          f"({round(100 * resolved / max(len(raw_refs), 1), 1)}% résolues).")


def build_graph(csv_path: str = ARTICLES_CSV, db_path: str = KUZU_DB_PATH, limit: int = None):
    df = pd.read_csv(csv_path)
    if limit:
        df = df.head(limit)

    db = kuzu.Database(db_path)
    conn = kuzu.Connection(db)

    lookup = build_hierarchy(df, conn)
    build_references(df, conn, lookup)

    conn.close()
    db.close()


if __name__ == "__main__":
    import shutil, os

    # kuzu stocke la base tantôt en dossier, tantôt en fichier unique selon la version.
    if os.path.isdir(KUZU_DB_PATH):
        shutil.rmtree(KUZU_DB_PATH)
    elif os.path.exists(KUZU_DB_PATH):
        os.remove(KUZU_DB_PATH)

    create_schema()
    build_graph()