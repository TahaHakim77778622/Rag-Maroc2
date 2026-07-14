"""
Détecte le type de question pour décider corpus vs web fallback.
"""

from __future__ import annotations

import re
import unicodedata
from typing import Any


def _fold(s: str) -> str:
    return "".join(
        c for c in unicodedata.normalize("NFD", s or "")
        if unicodedata.category(c) != "Mn"
    ).lower()


# Patterns indiquant une donnée actuelle chiffrée
_CURRENT_DATA_PATTERNS = (
    # Montants et chiffres
    "montant exact",
    "montant actuel",
    "montant en vigueur",
    "combien",
    "quel montant",
    "quel prix",
    "quel tarif",
    "quel taux",
    "taux actuel",
    "taux en vigueur",
    "quel salaire",
    "salaire exact",
    "salaire fixe",
    "salaire fixé",
    "valeur actuelle",
    "valeur en vigueur",
    # SMIG / salaires
    "montant du smig",
    "combien est le smig",
    "combien vaut le smig",
    "quel est le smig",
    "smig 2024",
    "smig 2025",
    "smig 2026",
    "smig horaire",
    "smig mensuel",
    # Données fiscales chiffrées
    "taux tva",
    "taux ir",
    "taux is",
    "bareme fiscal",
    "bareme impot",
    "tranche impot",
    # Données sociales chiffrées
    "taux cnss",
    "cotisation cnss",
    "taux amo",
    "indemnite journaliere montant",
    "montant indemnite",
    # Statistiques et données actuelles
    "statistique",
    "chiffre actuel",
    "donnee actuelle",
    "en vigueur actuellement",
    "actuellement fixe",
    # Années récentes (données susceptibles d'avoir changé)
    "en 2024",
    "en 2025",
    "en 2026",
    "cette annee",
    "dernier arrete",
    "dernier decret",
    "derniere decision",
    "nouveau montant",
    "nouveau taux",
    "nouvelle valeur",
    # Prix et coûts
    "cout",
    "frais exact",
    "frais actuel",
    "timbre fiscal montant",
    "droit de timbre",
)

# Patterns indiquant une question de cadre légal → corpus
_LEGAL_FRAMEWORK_PATTERNS = (
    "comment",
    "procedure",
    "procédure",
    "demarche",
    "démarche",
    "quelles pieces",
    "quels documents",
    "piece a fournir",
    "conditions",
    "quelles conditions",
    "quels droits",
    "obligation",
    "sanction",
    "article",
    "loi",
    "decret",
    "dahir",
    "code du travail",
    "code travail",
    "comment obtenir",
    "comment demander",
    "comment faire",
    "etapes",
    "étapes",
    "delai",
    "délai reglementaire",
)

# Questions hors périmètre (cuisine, sport, tech générale…) — pas de RAG BO
_GENERAL_KNOWLEDGE_MARKERS = (
    "messi",
    "ronaldo",
    "mbappe",
    "neymar",
    "coupe du monde",
    "championnat du monde",
    "ligue 1",
    "real madrid",
    "barcelona",
    "psg",
    "match de foot",
    "football",
    "machine learning",
    "deep learning",
    "intelligence artificielle",
    "chatgpt",
    "bitcoin",
    "crypto monnaie",
    "recette de",
    "recette du",
    "recette ",
    "ingredient du",
    "ingredient de",
    "ingredients du",
    "ingredients de",
    "ingrédient du",
    "ingrédient de",
    "ingrédients du",
    "ingrédients de",
    "repas italien",
    "repas francais",
    "repas français",
    "cuisine italienne",
    "cuisine francaise",
    "cuisine française",
    "comment cuisiner",
    "comment preparer un",
    "comment préparer un",
    "faire un gateau",
    "faire un gâteau",
    "netflix",
    "serie tv",
    "série tv",
)

_CULINARY_RECIPE_MARKERS = (
    "recette",
    "ingredient",
    "ingredients",
    "ingrédient",
    "ingrédients",
    "comment preparer",
    "comment préparer",
    "comment cuisiner",
    "repas italien",
    "repas francais",
    "repas français",
    "construire un repas",
    "faire un repas",
)

_FOOD_LAW_MARKERS = (
    "norme",
    "nomenclature",
    "bulletin officiel",
    "journal officiel",
    "code aliment",
    "produit alimentaire",
    "denree",
    "denrée",
    "etiquetage",
    "étiquetage",
    "hygiene alimentaire",
    "hygiène alimentaire",
    "arrete du",
    "arrêté du",
)

_MAROC_ADMIN_MARKERS = (
    "maroc",
    "marocain",
    "marocaine",
    "morocco",
    "cnie",
    "passeport",
    "watiqa",
    "etat civil",
    "état civil",
    "permis de construire",
    "autorisation de construire",
    "urbanisme",
    "rokhas",
    "code du travail",
    "licenciement",
    "preavis",
    "préavis",
    "smig",
    "bulletin officiel",
    "journal officiel",
    "dahir",
    "decret",
    "décret",
    "arrete",
    "arrêté",
    "demarche",
    "démarche",
    "service-public",
    "service public",
    "carte nationale",
    "acte de naissance",
    "consulat",
    "cnss",
    "timbre fiscal",
)


def has_maroc_admin_scope(question: str) -> bool:
    """True si la question relève du droit / admin marocain (dataset ou web .ma)."""
    q = _fold(question)
    if any(m in q for m in _MAROC_ADMIN_MARKERS):
        return True
    try:
        from app.phrase_context import analyze_phrase

        if analyze_phrase(question).primary_subject:
            return True
    except ImportError:
        pass
    try:
        from app.dataset_registry import active_domains

        if active_domains(question):
            return True
    except ImportError:
        pass
    return False


def is_general_knowledge_question(question: str) -> bool:
    """
    True pour cuisine grand public, sport, tech générale, etc.
    Ces questions ne doivent pas être couvertes par des extraits BO.
    """
    q = _fold(question)
    if len(q.strip()) < 4:
        return False

    if any(m in q for m in _FOOD_LAW_MARKERS):
        return False

    if any(m in q for m in _GENERAL_KNOWLEDGE_MARKERS):
        return True

    if any(m in q for m in _CULINARY_RECIPE_MARKERS):
        return True

    if re.search(r"\b(?:c\s+est\s+quoi|qu\s+est\s+ce\s+que|dis\s+moi\s+c\s+est\s+quoi)\b", q):
        if any(
            t in q
            for t in (
                "machine learning",
                "deep learning",
                "intelligence artificielle",
                "football",
                "messi",
                "bitcoin",
            )
        ):
            return True

    if any(
        dish in q
        for dish in ("couscous", "tajine", "pastilla", "harira")
    ) and any(w in q for w in ("ingredient", "ingredients", "ingrédient", "recette", "preparer", "préparer")):
        if not has_maroc_admin_scope(question):
            return True

    return False


def out_of_scope_reply(question: str) -> str:
    """Réponse fixe sans retrieval ni citations BO."""
    _ = question
    return (
        "Cette question sort du périmètre de l’assistant : droit, administration et "
        "démarches au Maroc (documents officiels, urbanisme, travail, fiscalité, textes "
        "publiés au Bulletin officiel, etc.).\n\n"
        "Je ne m’appuie pas sur le Bulletin officiel pour répondre aux questions de "
        "cuisine, de sport, d’actualité ou de culture générale (recettes, football, "
        "informatique générale, etc.).\n\n"
        "Reformulez une question liée au Maroc — par exemple : pièces pour un passeport, "
        "permis de construire, SMIG, délais ou procédure CNIE."
    )


def is_current_data_question(question: str) -> bool:
    """
    True si la question porte sur une donnée actuelle chiffrée
    qui nécessite le web si absente du corpus.
    """
    q = _fold(question)

    # Si contient des patterns de cadre légal fort → pas current data
    legal_score = sum(1 for p in _LEGAL_FRAMEWORK_PATTERNS if p in q)
    current_score = sum(1 for p in _CURRENT_DATA_PATTERNS if p in q)

    # Question mixte : cadre légal domine → corpus
    if legal_score >= 2 and current_score <= 1:
        return False

    # Question clairement chiffrée/actuelle
    if current_score >= 1:
        return True

    # Détecter questions courtes avec "combien" ou interrogation chiffrée
    if re.search(r"\bcombien\b", q):
        return True
    if re.search(r"\bquel\s+(est\s+le?\s+)?(montant|taux|prix|tarif|salaire|cout)\b", q):
        return True

    return False


def corpus_should_suffice(question: str, hits: list, top_score: float) -> bool:
    """
    True si le corpus suffit sans web :
    - Question de cadre légal ET score FAISS correct
    - Ou corpus contient la donnée chiffrée demandée
    """
    if not hits:
        return False

    if is_general_knowledge_question(question):
        return False

    q = _fold(question)

    # Cadre légal : au moins 2 indices ou sujet admin explicite (évite « comment » seul)
    legal_score = sum(1 for p in _LEGAL_FRAMEWORK_PATTERNS if p in q)
    if legal_score >= 2 and top_score >= 0.45:
        return True
    if legal_score >= 1 and top_score >= 0.45 and has_maroc_admin_scope(question):
        return True

    # Si données chiffrées dans les hits → corpus suffit
    if is_current_data_question(question):
        # Chercher si les hits contiennent des chiffres pertinents
        all_text = " ".join(
            (h.get("text") or "") for h in hits[:3]
        ).lower()
        # Présence de montants chiffrés (DH, %, chiffres)
        has_numbers = bool(re.search(
            r'\d+[,.]?\d*\s*(dh|dirham|%|pour\s*cent)', all_text
        ))
        if has_numbers and top_score >= 0.45:
            return True
        return False

    return top_score >= 0.35
