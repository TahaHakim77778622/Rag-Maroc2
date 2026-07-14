"""Tests ingestion file web fallback."""

from __future__ import annotations

from app.web_queue_ingest import plan_ingest, queue_row_to_chunk


def test_queue_row_rejects_short_text():
    assert queue_row_to_chunk({"source_url": "https://www.cnie.ma/x", "text": "court"}) is None


def test_queue_row_rejects_non_ma_domain():
    rec = {
        "source_url": "https://example.com/page",
        "text": "x" * 200,
        "title": "Test",
    }
    assert queue_row_to_chunk(rec) is None


def test_queue_row_builds_cnie_category():
    rec = {
        "source_url": "https://www.cnie.ma/procedure",
        "text": "Première demande CNIE : formulaire, acte de naissance, domicile, photos, timbre fiscal. " * 2,
        "title": "CNIE procédure",
    }
    ch = queue_row_to_chunk(rec)
    assert ch is not None
    assert ch["category"] == "cnie"
    assert ch["chunk_id"].startswith("web_ingest_")


def test_portal_intent_watiqa_before_acte_naissance():
    from app.web_fallback import _portal_intent

    assert _portal_intent("Comment commander un acte de naissance sur Watiqa ?") == "watiqa"


def test_portal_local_insufficient_on_cnie_top_for_watiqa():
    from app.web_fallback import portal_local_hits_sufficient

    hits = [
        {
            "metadata": {
                "chunk_id": "cnie_procedure_premiere_demande_maroc_1",
                "category": "cnie",
                "source_url": "https://www.cnie.ma/static/procedure",
            },
            "text": "Première demande CNIE pièces à fournir formulaire acte de naissance",
        }
    ]
    q = "Comment commander un acte de naissance sur Watiqa ?"
    assert portal_local_hits_sufficient(q, "watiqa", hits) is False


def test_portal_local_sufficient_watiqa_procedure():
    from app.web_fallback import portal_local_hits_sufficient

    hits = [
        {
            "metadata": {
                "chunk_id": "watiqa_procedure_acte_naissance_commande_1",
                "category": "etat_civil",
                "source_url": "https://www.watiqa.ma/?page=citoyen.GuichetActe",
                "source_org": "Watiqa",
            },
            "text": (
                "Commander un acte de naissance sur Watiqa : Nouvelle demande, "
                "Commencer la démarche, payer les frais, suivre une commande."
            ),
        }
    ]
    q = "Comment commander un acte de naissance sur Watiqa ?"
    assert portal_local_hits_sufficient(q, "watiqa", hits) is True


def test_should_use_web_fallback_for_smig_without_smig_in_hits():
    from app.web_fallback import should_use_web_fallback

    hits = [
        {
            "metadata": {
                "source_type": "bulletin_officiel",
                "label": "article 15",
            },
            "text": "article 15 du décret relatif aux marchés publics salaire de référence 50 pour cent",
            "rerank_score": 0.8,
        }
    ]
    q = "Quel est le SMIG salaire minimum au Maroc selon l'arrêté ?"
    assert should_use_web_fallback(q, hits) is True


def test_smig_false_positive_salaire_minimum_legal_only():
    from app.web_fallback import local_hits_substantively_answer

    hits = [
        {
            "text": (
                "article 14 revenu forfaitaire fixé à 0.75 fois la valeur du "
                "salaire minimum légal dans les activités non agricoles"
            ),
            "metadata": {"source_type": "bulletin_officiel"},
        }
    ]
    q = "Quel est le SMIG salaire minimum au Maroc ?"
    assert local_hits_substantively_answer(q, hits) is False


def test_should_use_web_fallback_for_watiqa_with_weak_hits():
    from app.web_fallback import should_use_web_fallback

    hits = [
        {
            "metadata": {"category": "etat_civil", "source_url": "https://www.watiqa.ma/"},
            "text": "Guichet électronique Accueil FAQ liens utiles",
        }
    ]
    q = "Comment commander un acte de naissance sur Watiqa ?"
    assert should_use_web_fallback(q, hits) is True


def test_cnie_hit_filtered_for_watiqa():
    from app.web_fallback import _web_hit_is_cnie_not_watiqa

    hit = {
        "metadata": {
            "chunk_id": "cnie_procedure_premiere_demande_maroc_1",
            "category": "cnie",
            "source_url": "https://www.cnie.ma/static/procedure",
        },
        "text": "Première demande CNIE pièces à fournir",
    }
    assert _web_hit_is_cnie_not_watiqa(hit) is True


def test_plan_ingest_skips_duplicate_url(tmp_path):
    corpus = tmp_path / "final.jsonl"
    url = "https://www.maroc.ma/demarche-test"
    existing = {
        "chunk_id": "existing_1",
        "doc_id": "d1",
        "title": "T",
        "source_url": url,
        "text": "Texte déjà présent dans le corpus pour la démarche administrative marocaine. " * 3,
        "source_type": "admin",
        "category": "services_publics",
    }
    corpus.write_text(__import__("json").dumps(existing, ensure_ascii=False) + "\n", encoding="utf-8")

    queue = [
        {
            "source_url": url,
            "text": "Nouveau texte différent mais même URL officielle marocaine pour test dédup. " * 4,
            "title": "Maroc.ma",
        }
    ]
    to_add, rejected, stats = plan_ingest(queue, corpus_path=corpus)
    assert stats["skipped_duplicate"] == 1
    assert len(to_add) == 0


def test_passeport_fee_not_covered_by_bo_budget_hits():
    from app.corpus_coverage import corpus_covers_question
    from app.corpus_first import should_use_web_after_corpus
    from app.web_fallback import passeport_fee_hits_substantive

    q = "combien coûte le timbre fiscal passeport Maroc 2025 ?"
    bo_hits = [
        {
            "score": 0.8,
            "rerank_score": 0.8,
            "text": (
                "Article 41 loi de finances 2025 recettes ordinaires du budget general "
                "421325037000 dirhams droits d'enregistrement et de timbre 21979673000"
            ),
            "metadata": {
                "chunk_id": "SGG0007_unit_210",
                "source_type": "bulletin_officiel",
                "label": "Article 41",
            },
        }
    ]
    assert not passeport_fee_hits_substantive(q, bo_hits)
    assert not corpus_covers_question(q, bo_hits)
    need_web, _ = should_use_web_after_corpus(q, bo_hits, top_k=5)
    assert need_web is True


def test_fetch_web_hits_passeport_timbre_curated_without_ddg(monkeypatch):
    from app.web_fallback import fetch_web_hits

    monkeypatch.setattr("app.web_fallback._search_duckduckgo", lambda _q: [])
    monkeypatch.setattr("app.web_fallback._scrape_hits", lambda *_a, **_k: [])
    hits = fetch_web_hits("combien coûte le timbre fiscal passeport Maroc 2025 ?", top_k=3)
    assert hits
    assert hits[0]["metadata"]["chunk_id"] == "curated::passeport_etimbre"
    assert "e-timbre" in hits[0]["text"].lower()


def test_fetch_web_hits_smig_curated_when_only_irrelevant_pages(monkeypatch):
    from app.web_fallback import fetch_web_hits

    monkeypatch.setattr("app.web_fallback._search_duckduckgo", lambda _q: [])
    monkeypatch.setattr(
        "app.web_fallback._scrape_hits",
        lambda _q, _urls, **kwargs: [
            {
                "index": 0,
                "score": 0.5,
                "rerank_score": 0.5,
                "metadata": {
                    "chunk_id": "web::maroc.ma",
                    "source_url": "https://www.maroc.ma/fr",
                    "source_type": "admin",
                },
                "text": "Discours royal sur la transformation du pays.",
            }
        ],
    )
    hits = fetch_web_hits("Quel est le SMIG au Maroc ?", top_k=3)
    assert hits[0]["metadata"]["chunk_id"] == "curated::smig_info"
    assert "SMIG" in hits[0]["text"]
