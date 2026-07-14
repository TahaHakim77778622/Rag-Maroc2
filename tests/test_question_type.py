"""Détection questions hors périmètre Maroc / admin."""

from app.corpus_coverage import corpus_covers_question
from app.question_type import (
    corpus_should_suffice,
    has_maroc_admin_scope,
    is_general_knowledge_question,
    out_of_scope_reply,
)


def test_general_knowledge_sports():
    assert is_general_knowledge_question("est ce que Messi va jouer en la coupe du monde ?")


def test_general_knowledge_ml():
    assert is_general_knowledge_question("dis moi c quoi machine learning ?")


def test_general_knowledge_couscous_recipe():
    assert is_general_knowledge_question(
        "est ce que tu peux me donner les ingrédients du couscous ?"
    )


def test_general_knowledge_italian_meal():
    assert is_general_knowledge_question(
        "comment construire un repas italien ?"
    )


def test_maroc_admin_not_general():
    assert not is_general_knowledge_question("comment obtenir la CNIE au Maroc ?")
    assert has_maroc_admin_scope("comment obtenir la CNIE au Maroc ?")


def test_food_law_stays_in_scope():
    assert not is_general_knowledge_question(
        "normes du couscous au Bulletin officiel marocain nomenclature"
    )


def test_corpus_should_suffice_rejects_comment_only():
    hits = [{"score": 0.8, "text": "Article 2 couscous produit minoterie"}]
    assert not corpus_should_suffice("comment construire un repas italien ?", hits, 0.8)


def test_corpus_covers_rejects_recipe_with_bo_food_hit():
    hits = [
        {
            "score": 0.75,
            "rerank_score": 0.75,
            "text": "Le couscous est le produit de la minoterie des cereales sans fermentation",
            "metadata": {"source_type": "bulletin_officiel", "label": "ART. 2"},
        }
    ]
    q = "ingredients du couscous"
    assert is_general_knowledge_question(q)
    assert not corpus_covers_question(q, hits)


def test_out_of_scope_reply_mentions_perimeter():
    text = out_of_scope_reply("Messi")
    assert "périmètre" in text.lower() or "perimetre" in text.lower()
    assert "Bulletin officiel" in text
