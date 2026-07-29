"""Compréhension contextuelle des phrases (sujet principal, négations)."""

from app.Rag_classique.corpus_coverage import corpus_covers_question
from app.Rag_classique.phrase_context import analyze_phrase, primary_portal_intent, subject_mentioned_but_not_target
from app.Rag_classique.portal_cases import cnie_case_from_question
from app.Rag_classique.web_fallback import _curated_passeport_hits, _portal_intent, portal_local_hits_sufficient


def test_mineur_sans_cnie_veut_passeport_primary_is_passeport():
    q = "je suis un mineur sans CNIE et je veux demander a mon passeport"
    ctx = analyze_phrase(q)
    assert ctx.primary_subject == "passeport"
    assert "cnie" in ctx.negated_subjects or subject_mentioned_but_not_target(q, "cnie")
    assert _portal_intent(q) == "passeport"
    assert cnie_case_from_question(q) is None


def test_perte_vol_cnie_still_cnie():
    q = "Perte ou vol de la CNIE : procédure"
    assert primary_portal_intent(q) == "cnie"
    assert cnie_case_from_question(q) == "perte_vol"


def test_passeport_dataset_blocks_web_fallback():
    from app.Rag_classique.passeport_corpus import best_passeport_hits_for, merge_passeport_hits
    from app.Rag_classique.web_fallback import should_use_web_fallback

    q = "je suis un mineur sans CNIE et je veux demander mon passeport"
    hits = merge_passeport_hits(q, [], top_k=3)
    assert hits
    text = hits[0]["text"].lower()
    assert "passeport" in text
    assert "mineur" in text or "moins de 12" in text
    assert not should_use_web_fallback(q + " Maroc", hits)


def test_curated_passeport_covers_mineur_sans_cnie():
    q = "je suis un mineur sans CNIE et je veux demander mon passeport"
    hits = _curated_passeport_hits(q)
    assert corpus_covers_question(q, hits)
    assert portal_local_hits_sufficient(q, "passeport", hits)
    assert "passeport" in hits[0]["text"].lower()
    assert "mineur" in hits[0]["text"].lower()


def test_long_passeport_loss_same_as_short():
    from app.Rag_classique.corpus_first import should_use_web_after_corpus

    q_long = "maintenant j ai perdu mon passeport comment déclarer et comment le refaire Maroc"
    q_short = "Perte passeport : déclaration et refaire Maroc"
    need_long, hits_long = should_use_web_after_corpus(q_long, [], top_k=3)
    need_short, hits_short = should_use_web_after_corpus(q_short, [], top_k=3)
    assert not need_long and not need_short
    assert "perte" in hits_long[0]["text"].lower() or "vol" in hits_long[0]["text"].lower()
    assert "passeport" in hits_long[0]["text"].lower()
    assert corpus_covers_question(q_long, hits_long)
    assert corpus_covers_question(q_short, hits_short)
