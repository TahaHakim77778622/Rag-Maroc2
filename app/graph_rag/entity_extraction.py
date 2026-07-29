"""
Extraction des relations (rôle "RE" du NER/RE) pour le benchmark belge.

Contrairement au corpus SGG marocain, les "entités" (Code, Division) sont déjà
données par les colonnes structurées du CSV -> pas besoin de NER ici.
Le seul travail d'extraction nécessaire est de détecter les RÉFÉRENCES CROISÉES
entre articles à l'intérieur du texte brut (ex: "conformément à l'article 1382
du Code civil", "visé aux articles 137 et 140").

Sortie : liste de tuples (source_article_id, target_code_name, target_number_raw)
que graph_builder.py résout ensuite en relations REFERENCE (Article -> Article).
"""

import re
import pandas as pd

# Numéro d'article : chiffres/points/slashes/tirets + suffixe latin optionnel (bis, ter, quater...)
# Note : chaque segment doit contenir au moins un chiffre (lookahead), sinon un mot ordinaire
# collé sans espace se fait lire comme un numéro : "1.4.6.Il soumet" -> "1.4.6.IL"
# ou "2.2.22, chargée" -> numéros ["2.2.22", "C"] (I, L, C etc. sont aussi des chiffres romains).
_NUM_SEG = r"(?=[\divxlcmIVXLCM]*\d)[\divxlcmIVXLCM]+"
_NUM_TOKEN = rf"{_NUM_SEG}(?:[./:\-]{_NUM_SEG})*(?:bis|ter|quater|quinquies|sexies|septies|octies)?"

# "article(s) 1382" / "articles 137 et 140 à 140septies" / "art. 6"
ARTICLE_MENTION_RE = re.compile(
    rf"articles?\s+({_NUM_TOKEN}(?:\s*(?:,|et|à)\s*{_NUM_TOKEN})*)",
    re.IGNORECASE,
)

# Nom de code mentionné juste après ("du Code civil", "du Code pénal")
CODE_MENTION_RE = re.compile(r"du\s+(Code\s+[A-ZÉÀ][^,.;]{2,40})", re.IGNORECASE)

_SPLIT_RE = re.compile(r"\s*(?:,|et|à)\s*")


def normalize_number(raw: str) -> str:
    """Normalise un numéro d'article pour matcher le format de la colonne `reference`."""
    return raw.strip().upper().replace(" ", "")


def extract_cross_references(article_text: str, source_code_name: str):
    """
    Retourne une liste de dicts {target_code_name, target_number} pour un texte d'article.
    Si aucun nom de code n'est explicitement mentionné, on suppose une référence
    au même code que l'article source (cas le plus fréquent dans ce corpus).
    """
    if not isinstance(article_text, str) or not article_text:
        return []

    results = []
    for match in ARTICLE_MENTION_RE.finditer(article_text):
        numbers_blob = match.group(1)
        numbers = [normalize_number(n) for n in _SPLIT_RE.split(numbers_blob) if n.strip()]

        # Chercher un nom de code mentionné juste après la référence (fenêtre de 60 caractères)
        window = article_text[match.end(): match.end() + 60]
        code_match = CODE_MENTION_RE.search(window)
        target_code = code_match.group(1).strip() if code_match else source_code_name

        for num in numbers:
            results.append({"target_code_name": target_code, "target_number": num})

    return results


def extract_all(articles_df: pd.DataFrame, log_every: int = 5000):
    """
    Parcourt tout le corpus d'articles et retourne la liste complète des références
    croisées détectées : [{source_article_ref_id, source_code_name, target_code_name, target_number}, ...]
    """
    all_refs = []
    for i, row in articles_df.iterrows():
        refs = extract_cross_references(row["article"], row["code"])
        for r in refs:
            all_refs.append({
                "source_article_ref_id": str(row["id"]),
                "source_code_name": row["code"],
                "target_code_name": r["target_code_name"],
                "target_number": r["target_number"],
            })
        if (i + 1) % log_every == 0:
            print(f"  ... {i + 1} articles analysés, {len(all_refs)} références brutes trouvées")

    print(f"Extraction terminée : {len(all_refs)} références croisées brutes détectées "
          f"sur {len(articles_df)} articles.")
    return all_refs


if __name__ == "__main__":
    from config import ARTICLES_CSV

    df = pd.read_csv(ARTICLES_CSV)
    refs = extract_all(df)

    # Aperçu rapide
    print("\nExemples de références extraites :")
    for r in refs[:10]:
        print(" ", r)