"""Fallback web pour questions marocaines non couvertes par le corpus local."""

from __future__ import annotations

import json
import logging
import os
import re
import warnings
from datetime import datetime, timezone
from pathlib import Path
from typing import Any
from urllib.parse import parse_qs, quote_plus, unquote, urlparse

import requests
from bs4 import BeautifulSoup

from app.config import PROJECT_ROOT

logger = logging.getLogger(__name__)

# Après un HTTP 202 sur html.duckduckgo.com, on n’y retourne plus dans ce processus.
_ddg_html_blocked = False

# Sites souvent en 403/WAF ou SPA vide au scrape HTTP — on utilise curated / dataset / extraits DDG.
_SCRAPE_SKIP_HOSTS = frozenset(
    {
        "consulat.ma",
        "legalisation.consulat.ma",
        "cnie.ma",
        "watiqa.ma",
    }
)

OFFICIAL_MA_DOMAINS = (
    "cnie.ma",
    "passeport.ma",
    "watiqa.ma",
    "tax.gov.ma",
    "service-public.ma",
    "maroc.ma",
    "mre.gov.ma",
    "interieur.gov.ma",
    "justice.gov.ma",
    "finances.gov.ma",
    "emploi.gov.ma",
    "miepeec.gov.ma",
    "tax.gov.ma",
    "diplomatie.ma",
    "consulat.ma",
    "sgg.gov.ma",
    "rokhas.ma",
)

# Domaines optionnels (désactivés par défaut pour un fallback strictement officiel .ma).
EXTRA_ALLOWED_DOMAINS = (
    "wikipedia.org",
    "mapnews.ma",
    "hcp.ma",
    "oncf.ma",
    "adm.co.ma",
    "onda.ma",
    "bankalmaghrib.ma",
)

HEADERS = {
    "User-Agent": "Mozilla/5.0 (compatible; RAG-Maroc2/1.0; +https://127.0.0.1)"
}

SEARCH_TIMEOUT = 6
FETCH_TIMEOUT = 8
MAX_RESULTS = 2
WEB_QUEUE_PATH = PROJECT_ROOT / "data" / "processed" / "web_additions_queue.jsonl"


def _bool_env(name: str, default: bool) -> bool:
    raw = os.environ.get(name)
    if raw is None:
        return default
    return raw.strip().lower() in ("1", "true", "yes", "on")


def _int_env(name: str, default: int) -> int:
    raw = os.environ.get(name)
    if raw is None:
        return default
    try:
        return int(raw)
    except ValueError:
        return default

_MOROCCO_HINTS = (
    "maroc",
    "marocaine",
    "marocain",
    "cnie",
    "carte nationale",
    "passeport",
    "etat civil",
    "acte de naissance",
    "casier judiciaire",
    "visa",
    "consulat",
    "demarche",
    "document",
    "piece",
    "administratif",
    "loi",
    "juridique",
    "urbanisme",
    "construction",
    "construire",
    "batir",
    "bâtir",
    "bâtiment",
    "batiment",
    "immeuble",
    "terrain",
    "maison",
    "habiter",
    "logement",
    "promotion immobiliere",
    "rokhas",
    "smig",
    "salaire",
    "travail",
    "code de la route",
    "route",
    "conduite",
    "conduire",
    "permis",
    "casablanca",
    "rabat",
    "fes",
    "fès",
    "marrakech",
    "tanger",
    "agadir",
    "oujda",
    "kenitra",
    "meknes",
    "morocco",
    "droit marocain",
    "code penal",
    "code pénal",
    "code civil",
    "obligations et contrats",
    "famille",
    "fiscal",
    "impot",
    "impôt",
    "entreprise",
    "travail",
    "nationalite marocaine",
    "cuisine",
    "recette",
    "couscous",
    "tajine",
    "tagine",
    "the marocain",
    "thé marocain",
    "atay",
)

_PROCEDURE_HINTS = (
    "piece",
    "pièce",
    "pieces",
    "pièces",
    "document",
    "demande",
    "demander",
    "renouvel",
    "delai",
    "délai",
    "premiere",
    "première",
    "obtenir",
    "fournir",
    "procedure",
    "procédure",
    "deposer",
    "déposer",
    "formulaire",
    "photo",
    "norme",
    "comment",
    "commander",
    "suivre",
)

# Pages .ma à tenter en direct (sites Angular souvent vides au scrape classique).
# Texte de secours si le web ne renvoie que du passeport / pages vides (cnie.ma = SPA).
_CURATED_WATIQA_ACTE_NAISSANCE = (
    "Commander un acte de naissance sur Watiqa (www.watiqa.ma) — procédure en ligne : "
    "1) Accéder au guichet « Acte de naissance » (page citoyen.GuichetActe) ; "
    "2) Choisir « Extrait » ou « Copie intégrale » selon le document voulu ; "
    "3) Cliquer sur « Nouvelle demande » puis « Commencer la démarche » ; "
    "4) Renseigner les 6 étapes du formulaire : Administration (province, commune, bureau d'état civil d'origine), "
    "Document (type d'acte), Livraison (adresse postale), Validation des informations, "
    "Règlement sécurisé par carte bancaire des frais d'envoi, Confirmation ; "
    "5) Conserver le numéro de commande et l'e-mail pour « Suivre une commande » ; "
    "6) Réception des documents par courrier recommandé à l'adresse indiquée. "
    "Watiqa s'adresse aux personnes inscrites aux registres d'état civil marocains ; "
    "les MRE peuvent utiliser le guichet selon les conditions affichées sur le portail. "
    "Ne pas confondre avec la demande de CNIE (cnie.ma) ni le passeport biométrique."
)

_CURATED_CNIE_PREMIERE_DEMANDE = (
    "Première demande de Carte Nationale d'Identité Électronique (CNIE) au Maroc — pièces à fournir : "
    "1) Formulaire de demande (cnie.ma ou sur place), signé ; "
    "2) Extrait ou copie intégrale d'acte de naissance (souvent de moins de six mois) ; "
    "3) Justificatif de domicile de moins de trois mois (facture, attestation de résidence, etc.) ; "
    "4) Photographies d'identité aux normes CNIE (format 35×45 mm, fond clair, visage dégagé) ; "
    "5) Timbre fiscal / frais de délivrance ; "
    "6) Présence du demandeur pour biométrie (empreintes, photo). "
    "Dépôt au commissariat de police ou service d'arrondissement du lieu de résidence, ou en ligne sur www.cnie.ma. "
    "Récépissé délivré, exigé au retrait. Ne pas confondre avec les pièces du passeport biométrique (consulat.ma)."
)

_CURATED_CNIE_PERTE_VOL = (
    "CNIE perdue ou volée au Maroc — procédure (renouvellement en cas de perte ou de vol, pas une première demande) : "
    "1) Déclarer la perte ou le vol de la carte auprès des autorités compétentes "
    "(commissariat de police / service d'arrondissement — déclaration de perte ou de vol, selon les consignes du guichet) ; "
    "2) Constituer un dossier de renouvellement pour perte/vol : formulaire de demande CNIE, "
    "extrait ou copie intégrale d'acte de naissance (souvent récent), justificatif de domicile de moins de trois mois, "
    "photographies aux normes CNIE, timbre fiscal / frais ; "
    "3) Joindre la déclaration de perte ou de vol (ou attestation équivalente selon le service) ; "
    "4) Présence du demandeur pour biométrie ; "
    "5) Ne pas confondre avec la première demande (aucune ancienne carte à présenter) ni avec le passeport biométrique. "
    "Dépôt au commissariat ou via www.cnie.ma selon les cas ouverts en ligne. "
    "Consulter cnie.ma / le guichet pour le modèle de déclaration et les délais de délivrance."
)

_CURATED_CNIE_RENOUVELLEMENT = (
    "Renouvellement de la CNIE au Maroc (carte valide ou expirée, sans perte ni vol déclaré) — pièces usuelles : "
    "ancienne carte nationale d'identité (originale) ; formulaire de renouvellement ; "
    "justificatif de domicile récent si changement d'adresse ; photographies aux normes CNIE ; "
    "timbre fiscal / frais ; présence du titulaire pour mise à jour biométrique si exigée. "
    "Dépôt au commissariat, service d'arrondissement ou www.cnie.ma. "
    "En cas de perte ou vol, voir la procédure dédiée (déclaration de perte/vol + dossier de renouvellement pour perte/vol)."
)

_PORTAL_SEED_URLS: dict[str, list[str]] = {
    "cnie": [
        "https://www.maroc.ma/fr/services-numeriques",
    ],
    "passeport": [
        "https://www.passeport.ma/Home/PiecesAuMaroc",
        "https://www.maroc.ma/fr/services-numeriques",
        "https://www.mre.gov.ma/fr/e-services/guide-des-prestations-consulaires-en-ligne",
    ],
    "watiqa": [
        "https://www.watiqa.ma/",
    ],
}

_TOPIC_SEED_URLS_EXTRA: dict[str, list[str]] = {
    "passeport_fees": [
        "https://www.passeport.ma/",
        "https://www.tax.gov.ma/",
    ],
}

_TOPIC_SEED_URLS: dict[str, list[str]] = {
    "smig": [
        "https://www.miepeec.gov.ma/",
        "https://www.maroc.ma/fr",
    ],
    "travail": [
        "https://www.miepeec.gov.ma/",
    ],
}

_CURATED_SMIG_INFO = (
    "SMIG (salaire minimum interprofessionnel garanti) au Maroc : le montant applicable est fixé "
    "par la loi et révisé par décret ou arrêté publié au Bulletin officiel (sgg.gov.ma). "
    "Pour le montant en vigueur à une date précise, consulter le dernier texte de revalorisation "
    "au Bulletin officiel ou les communications du ministère de l'Emploi (miepeec.gov.ma / emploi.gov.ma). "
    "Le SMIG concerne les salariés du secteur privé selon le Code du travail marocain ; "
    "certaines branches peuvent disposer de minima conventionnels différents. "
    "Ne pas confondre avec des mentions de « salaire minimum légal » dans d'autres contextes "
    "(fiscalité, pêche, marchés publics) qui ne fixent pas le SMIG national."
)

_FALLBACK_PRIORITY_HINTS = (
    "code de la route",
    "route",
    "conduite",
    "conduire",
    "permis",
    "infractions",
    "amende",
    "pv",
    "construction",
    "construire",
    "maison",
    "batir",
    "bâtir",
    "bâtiment",
    "batiment",
    "urbanisme",
    "rokhas",
    "autorisation de construire",
    "permis de construire",
    "fonds de commerce",
    "immeuble",
    "logement",
    "smig",
    "salaire minimum",
    "code du travail",
    "licenciement",
    "préavis",
    "preavis",
    "congé annuel",
    "conge annuel",
)

_WEAK_QUERY_TERMS = frozenset(
    {
        "maroc",
        "marocaine",
        "marocain",
        "arrete",
        "arrêté",
        "arreté",
        "article",
        "loi",
        "lois",
        "decret",
        "décret",
        "dahir",
        "officiel",
        "bulletin",
        "quelle",
        "quelles",
        "comment",
        "procedure",
        "procédure",
        "texte",
        "dispositions",
    }
)

_STOP_TERMS = {
    "pour",
    "avec",
    "sans",
    "dans",
    "sur",
    "une",
    "des",
    "les",
    "que",
    "qui",
    "quoi",
    "comment",
    "maroc",
    "marocaine",
    "marocain",
    "loi",
    "lois",
}


def is_morocco_admin_query(question: str) -> bool:
    """Question éligible au corpus Maroc (dataset + web .ma en secours)."""
    q = question.strip().lower()
    if not q:
        return False
    if "maroc" in q or "morocco" in q:
        return True
    if any(k in q for k in _MOROCCO_HINTS):
        return True
    try:
        from app.dataset_registry import active_domains, infer_domains_from_question

        if active_domains(question) or infer_domains_from_question(question):
            return True
    except ImportError:
        pass
    # Application RAG Maroc : toute question substantielle tente le dataset d'abord
    return len(q) >= 12


def _portal_intent(question: str) -> str | None:
    """CNIE / passeport / Watiqa — selon le sens de la phrase, pas le premier mot trouvé."""
    q = question.strip().lower()
    try:
        from app.phrase_context import analyze_phrase, primary_portal_intent

        ctx = analyze_phrase(question)
        portal = primary_portal_intent(question)
        if portal:
            return portal
        if ctx.primary_subject in ("cnie", "passeport", "watiqa"):
            return ctx.primary_subject
    except ImportError:
        pass
    if not any(h in q for h in _PROCEDURE_HINTS):
        return None
    if "watiqa" in q:
        return "watiqa"
    if "passeport" in q:
        return "passeport"
    if any(k in q for k in ("cnie", "carte nationale", "identite electronique", "identité électronique", "cine")):
        return "cnie"
    if any(k in q for k in ("etat civil", "état civil", "acte de naissance", "acte de mariage", "acte de deces", "acte de décès")):
        return "watiqa"
    return None


def _is_stub_chunk(text: str) -> bool:
    t = (text or "").strip()
    tl = t.lower()
    if _chunk_is_watiqa_procedure(t) or _bo_chunk_is_cnie_procedure(t) or _chunk_is_passeport_procedure(t):
        return len(t) < 80
    if len(t) < 150:
        return True
    if "portail cnie.ma" in tl and len(t) < 280:
        return True
    if "portail watiqa" in tl and len(t) < 280:
        return True
    return False


def _chunk_is_watiqa_procedure(text: str) -> bool:
    tl = (text or "").lower()
    markers = (
        "commencer la démarche",
        "commencer la demarche",
        "nouvelle demande",
        "suivre une commande",
        "créer une demande",
        "creer une demande",
        "commander électroniquement",
        "commander electroniquement",
        "guichet électronique",
        "guichet electronique",
        "étape 1",
        "etape 1",
        "formulairecommande",
        "extrait et copie intégrale",
        "extrait et copie integrale",
        "payer les frais",
        "numéro de la commande",
        "numero de la commande",
    )
    return any(m in tl for m in markers)


def _bo_chunk_is_cnie_procedure(text: str) -> bool:
    """Un extrait BO qui cite la CNIE dans un autre contexte ne couvre pas une procédure CNIE."""
    tl = (text or "").lower()
    markers = (
        "première demande",
        "premiere demande",
        "pièces à fournir",
        "pieces a fournir",
        "pièces a fournir",
        "demande de la cnie",
        "demande de cnie",
        "délivrance de la cnie",
        "delivrance de la cnie",
        "renouvellement de la cnie",
        "renouvellement cnie",
        "perte ou vol",
        "perte de la cnie",
        "vol de la cnie",
        "declaration de perte",
        "déclaration de perte",
        "carte perdue",
        "carte volée",
        "photographie d'identité",
        "photographie d identité",
        "normes des photographies",
        "déposer une demande de cnie",
    )
    return any(m in tl for m in markers)


def _question_lexical_coverage(question: str, text: str, *, min_ratio: float = 0.28) -> bool:
    """Le passage principal recoupe-t-il les mots-clés utiles de la question (pas les mots de liaison)."""
    try:
        from app.corpus_coverage import _discriminative_terms

        terms = _discriminative_terms(question)
    except ImportError:
        q = question.strip().lower()
        terms = [
            t
            for t in re.findall(r"[a-zA-Z0-9àâäéèêëïîôùûœç]{4,}", q)
            if t not in _STOP_TERMS and t not in ("watiqa", "maroc")
        ]
    if not terms:
        return True
    tl = (text or "").lower()
    found = sum(1 for t in terms if t in tl)
    if len(terms) <= 2:
        return found >= 1
    return found / max(1, len(terms)) >= min_ratio


def _chunk_is_passeport_procedure(text: str) -> bool:
    tl = (text or "").lower()
    markers = (
        "passeport biométrique",
        "passeport biometrique",
        "pièces à fournir",
        "pieces a fournir",
        "immatriculation consulaire",
        "délivrance du passeport",
        "delivrance du passeport",
        "suivi de la demande",
        "numéro de dossier",
        "numero de dossier",
        "mineur de moins de 12 ans sans cnie",
        "mineurs de 12 à 18 ans sans cnie",
        "sans cnie",
        "curated::passeport",
    )
    return any(m in tl for m in markers)


def portal_local_hits_sufficient(
    question: str, portal: str, hits: list[dict[str, Any]]
) -> bool:
    """
    Le meilleur hit local permet-il de répondre utilement (procédure sur le bon portail) ?
    Sinon → le pipeline doit basculer sur le fallback web, pas répondre avec du bruit (menu, BO, CNIE).
    """
    if not hits:
        return False
    top = hits[0]
    meta = top.get("metadata") or {}
    text = str(top.get("text") or "")
    url = str(meta.get("source_url") or "").lower()
    cat = str(meta.get("category") or "").lower()
    cid = str(meta.get("chunk_id") or "")
    org = str(meta.get("source_org") or "").lower()

    if _is_stub_chunk(text):
        return False

    if portal == "watiqa":
        if _web_hit_is_cnie_not_watiqa(top):
            return False
        if not ("watiqa.ma" in url or org == "watiqa" or cid.startswith("watiqa_procedure_")):
            return False
        if not (_chunk_is_watiqa_procedure(text) or cid.startswith("watiqa_procedure_") or cid.startswith("curated::watiqa")):
            return False
        return _question_lexical_coverage(question, text)

    if portal == "cnie":
        from app.portal_cases import cnie_case_aligned, cnie_case_from_question

        if _web_hit_is_passeport_not_cnie(top):
            return False
        if not (
            cat == "cnie"
            or "cnie.ma" in url
            or cid.startswith("cnie_procedure_")
            or cid.startswith("curated::cnie")
        ):
            return False
        if not (_bo_chunk_is_cnie_procedure(text) or cid.startswith("cnie_procedure_") or cid.startswith("curated::cnie")):
            return False
        if cnie_case_from_question(question) and not cnie_case_aligned(question, top):
            return False
        return _question_lexical_coverage(question, text)

    if portal == "passeport":
        try:
            from app.phrase_context import hit_matches_primary_subject
            from app.passeport_cases import passeport_case_aligned, passeport_case_from_question

            if not hit_matches_primary_subject(question, top):
                return False
            if passeport_case_from_question(question) and not passeport_case_aligned(
                question, top
            ):
                return False
        except ImportError:
            pass
        if not (
            cat == "passeport"
            or "passeport" in url
            or "consulat.ma" in url
            or cid.startswith("consulat_passeport_")
            or cid.startswith("curated::passeport")
        ):
            return False
        if not (_chunk_is_passeport_procedure(text) or cid.startswith("curated::passeport")):
            return False
        return _question_lexical_coverage(question, text, min_ratio=0.22)

    return False


def _hits_cover_portal(intent: str, hits: list[dict[str, Any]]) -> bool:
    """Le corpus local couvre-t-il vraiment la procédure demandée (pas une simple mention CNIE dans le BO) ?"""
    for h in hits[:3]:
        meta = h.get("metadata") or {}
        text = str(h.get("text") or "")
        if _is_stub_chunk(text):
            continue
        st = str(meta.get("source_type") or "")
        if st == "web_fallback":
            return True
        cat = str(meta.get("category") or "").lower()
        url = str(meta.get("source_url") or "").lower()
        org = str(meta.get("source_org") or "").lower()

        if intent == "cnie":
            if cat == "cnie" or "cnie.ma" in url or "cnie maroc" in org:
                return True
            if st == "bulletin_officiel" and not _bo_chunk_is_cnie_procedure(text):
                continue
            if st != "bulletin_officiel" and len(text) >= 200 and "cnie" in text.lower():
                return True
        elif intent == "passeport":
            if cat == "passeport" or "passeport.ma" in url or "consulat" in url:
                return True
            if st == "bulletin_officiel":
                continue
        elif intent == "watiqa":
            if cat == "cnie" or "cnie.ma" in url:
                continue
            if ("watiqa.ma" in url or org == "watiqa") and _chunk_is_watiqa_procedure(text):
                return True
            if cat in ("watiqa", "etat_civil", "état_civil") and _chunk_is_watiqa_procedure(text):
                return True
            if st == "bulletin_officiel":
                continue
    return False


def _discriminative_query_terms(question: str) -> list[str]:
    """Mots qui doivent apparaître dans les extraits pour considérer que le corpus répond."""
    q = question.strip().lower()
    terms = [
        t
        for t in re.findall(r"[a-zA-Z0-9àâäéèêëïîôùûœç]{3,}", q)
        if t not in _STOP_TERMS and t not in _WEAK_QUERY_TERMS
    ]
    # Phrases métier courtes
    if "salaire minimum" in q and "salaire" not in terms:
        terms.append("salaire")
    if "code du travail" in q:
        terms.extend(["travail", "code"])
    seen: set[str] = set()
    out: list[str] = []
    for t in terms:
        if t not in seen:
            seen.add(t)
            out.append(t)
    return out


def local_hits_substantively_answer(question: str, hits: list[dict[str, Any]]) -> bool:
    """Alias : le corpus couvre-t-il la question (voir app/corpus_coverage.py)."""
    from app.corpus_coverage import corpus_covers_question

    return corpus_covers_question(question, hits)


def should_use_web_fallback(question: str, hits: list[dict[str, Any]]) -> bool:
    """
    Dataset d'abord (avec compréhension du contexte de la phrase), web seulement si insuffisant.
    """
    try:
        from app.corpus_first import should_use_web_after_corpus

        need_web, _prepared = should_use_web_after_corpus(question, hits, top_k=8)
        return need_web
    except ImportError:
        pass

    if not is_morocco_admin_query(question):
        return False

    from app.corpus_coverage import corpus_covers_question, explain_coverage

    if not hits:
        logger.info("Web fallback (aucun extrait corpus)")
        return True

    if corpus_covers_question(question, hits):
        return False

    logger.info(
        "Web fallback (corpus insuffisant) : %s — %s",
        question[:80],
        explain_coverage(question, hits),
    )
    return True


def _domain_priority(host: str) -> float:
    h = (host or "").lower()
    priorities = (
        ("sgg.gov.ma", 0.30),
        ("service-public.ma", 0.28),
        ("interieur.gov.ma", 0.27),
        ("justice.gov.ma", 0.27),
        ("finances.gov.ma", 0.27),
        ("tax.gov.ma", 0.25),
        ("mre.gov.ma", 0.25),
        ("rokhas.ma", 0.25),
        ("cnie.ma", 0.26),
        ("passeport.ma", 0.26),
        ("watiqa.ma", 0.26),
        ("maroc.ma", 0.22),
    )
    for d, s in priorities:
        if h == d or h.endswith(f".{d}"):
            return s
    return 0.08


def _is_allowed(url: str) -> bool:
    host = (urlparse(url).netloc or "").lower()
    if not host:
        return False
    strict_official = _bool_env("WEB_FALLBACK_OFFICIAL_ONLY", True)
    if strict_official:
        if any(host == d or host.endswith(f".{d}") for d in OFFICIAL_MA_DOMAINS):
            return True
        # Sites .ma crédibles si la recherche HTML DDG est bloquée (voir WEB_FALLBACK_ALLOW_MA_BROAD).
        if _bool_env("WEB_FALLBACK_ALLOW_MA_BROAD", True):
            return host.endswith(".ma") or ".gov.ma" in host
        return False
    all_domains = OFFICIAL_MA_DOMAINS + EXTRA_ALLOWED_DOMAINS
    return any(host == d or host.endswith(f".{d}") for d in all_domains)


def _clean_text(text: str) -> str:
    text = re.sub(r"\s+", " ", text or "").strip()
    return text


def _extract_text_from_html(html: str) -> str:
    soup = BeautifulSoup(html, "html.parser")
    for tag in soup(["script", "style", "noscript", "svg"]):
        tag.decompose()
    main = soup.find("main") or soup.find("article") or soup.body or soup
    text = _clean_text(main.get_text(" ", strip=True))
    return text


def _lexical_overlap(question: str, text: str) -> float:
    q_terms = set(re.findall(r"[a-zA-Z0-9]{3,}", question.lower()))
    if not q_terms:
        return 0.0
    t = text.lower()
    found = sum(1 for term in q_terms if term in t)
    return found / max(1, len(q_terms))


def _url_blocked_for_portal(url: str, portal: str, question: str) -> bool:
    u = (url or "").lower()
    q = question.lower()
    if portal == "cnie" and "passeport" not in q:
        if "passeport" in u:
            return True
        if "consulat.ma" in u and "cnie" not in u:
            return True
    return False


def _web_hit_is_passeport_not_cnie(hit: dict[str, Any]) -> bool:
    text = (hit.get("text") or "").lower()
    meta = hit.get("metadata") or {}
    label = str(meta.get("label") or "").lower()
    url = str(meta.get("source_url") or "").lower()
    if "passeport biométrique" in text or "passeport biometrique" in text:
        if "première demande" not in text and "premiere demande" not in text:
            if "cnie" in label and "passeport" not in label:
                return False
            return True
    if "consulat.ma" in url and "passeport" in url:
        return True
    if "pieces a fournir - majeur" in label and "passeport" in label:
        return True
    return False


def _web_hit_is_cnie_not_watiqa(hit: dict[str, Any]) -> bool:
    """Exclure les chunks CNIE quand la question porte sur Watiqa / état civil en ligne."""
    meta = hit.get("metadata") or {}
    cat = str(meta.get("category") or "").lower()
    url = str(meta.get("source_url") or "").lower()
    org = str(meta.get("source_org") or "").lower()
    cid = str(meta.get("chunk_id") or "")
    if cat == "cnie" or "cnie.ma" in url or "cnie maroc" in org:
        return True
    if cid.startswith("cnie_procedure_") or cid.startswith("curated::cnie"):
        return True
    text = (hit.get("text") or "").lower()
    if "première demande" in text and "cnie" in text and "watiqa" not in text:
        if "watiqa.ma" not in url:
            return True
    return False


def _curated_watiqa_hits() -> list[dict[str, Any]]:
    return [
        {
            "index": -1,
            "score": 0.99,
            "rerank_score": 0.99,
            "metadata": {
                "chunk_id": "curated::watiqa_acte_naissance",
                "title": "Watiqa — commander un acte de naissance",
                "label": "Acte de naissance - commande en ligne",
                "source_type": "admin",
                "source_url": "https://www.watiqa.ma/?page=citoyen.GuichetActe",
                "category": "etat_civil",
                "source_org": "Watiqa",
            },
            "text": _CURATED_WATIQA_ACTE_NAISSANCE,
        }
    ]


def _prioritize_watiqa_hits(hits: list[dict[str, Any]]) -> list[dict[str, Any]]:
    def _rank(h: dict[str, Any]) -> tuple[int, float]:
        meta = h.get("metadata") or {}
        url = str(meta.get("source_url") or "").lower()
        text = str(h.get("text") or "")
        if "watiqa.ma" in url and _chunk_is_watiqa_procedure(text):
            return (0, -float(h.get("rerank_score", h.get("score", 0.0)) or 0.0))
        if str(meta.get("source_org") or "").lower() == "watiqa":
            return (1, -float(h.get("rerank_score", h.get("score", 0.0)) or 0.0))
        return (2, -float(h.get("rerank_score", h.get("score", 0.0)) or 0.0))

    return sorted(hits, key=_rank)


def _curated_cnie_hit_for_case(case: str) -> dict[str, Any]:
    if case == "perte_vol":
        return {
            "index": -1,
            "score": 0.99,
            "rerank_score": 0.99,
            "metadata": {
                "chunk_id": "curated::cnie_perte_vol",
                "title": "CNIE — perte ou vol",
                "label": "Perte ou vol - procedure",
                "source_type": "admin",
                "source_url": "https://www.cnie.ma/static/procedure",
                "category": "cnie",
                "source_org": "CNIE Maroc",
            },
            "text": _CURATED_CNIE_PERTE_VOL,
        }
    if case == "renouvellement":
        return {
            "index": -1,
            "score": 0.99,
            "rerank_score": 0.99,
            "metadata": {
                "chunk_id": "curated::cnie_renouvellement",
                "title": "CNIE — renouvellement",
                "label": "Renouvellement - pieces",
                "source_type": "admin",
                "source_url": "https://www.cnie.ma/static/procedure",
                "category": "cnie",
                "source_org": "CNIE Maroc",
            },
            "text": _CURATED_CNIE_RENOUVELLEMENT,
        }
    return {
        "index": -1,
        "score": 0.99,
        "rerank_score": 0.99,
        "metadata": {
            "chunk_id": "curated::cnie_premiere_demande",
            "title": "CNIE — première demande au Maroc",
            "label": "Premiere demande - pieces a fournir",
            "source_type": "admin",
            "source_url": "https://www.cnie.ma/static/procedure",
            "category": "cnie",
            "source_org": "CNIE Maroc",
        },
        "text": _CURATED_CNIE_PREMIERE_DEMANDE,
    }


def _curated_cnie_hits(question: str = "") -> list[dict[str, Any]]:
    from app.portal_cases import cnie_case_from_question

    case = cnie_case_from_question(question) or "premiere"
    return [_curated_cnie_hit_for_case(case)]


_CURATED_PASSEPORT_MINEUR_SANS_CNIE = (
    "Passeport biométrique pour mineur sans CNIE (consulat.ma / prestations consulaires) : "
    "Mineur de moins de 12 ans sans CNIE : dossier établi sur la base de l'extrait d'acte de naissance "
    "(ou copie intégrale de moins de six mois, ou copie du livret de famille avec noms en arabe et en latin). "
    "Mineur de 12 à 18 ans sans CNIE : présence obligatoire au service consulaire pour prise d'empreintes digitales ; "
    "immatriculation consulaire à jour ; photos d'identité 35×45 mm fond clair ; "
    "si seul reçu de dépôt CNIE : joindre extrait ou copie intégrale d'acte de naissance récent. "
    "Dépôt par le mineur accompagné du père, de la mère ou du représentant légal ; "
    "pièce d'identité du représentant (original + photocopie) ; justificatif de qualité de représentant si besoin. "
    "Ne pas confondre avec la procédure CNIE (cnie.ma) ni avec les pièces d'une première demande de carte seule."
)

_CURATED_PASSEPORT_GENERAL = (
    "Demande de passeport biométrique marocain : dépôt auprès du consulat de résidence ou selon passeport.ma au Maroc. "
    "Majeur : CNIE valide ou reçu de demande CNIE + pièces (immatriculation consulaire, photos 35×45 mm, ancien passeport si renouvellement). "
    "Mineur : présence du mineur et du représentant légal, acte de naissance ou livret de famille, photos, immatriculation consulaire. "
    "Sans CNIE : règles spécifiques selon l'âge (extrait d'acte de naissance pour moins de 12 ans ; empreintes pour 12-18 ans)."
)

_CURATED_PASSEPORT_PERTE_VOL = (
    "Passeport biométrique perdu ou volé (consulat.ma) : "
    "1) Déclaration de perte ou de vol de l'ancien passeport — déclaration sur l'honneur de l'intéressé(e), "
    "du tuteur ou du représentant légal, légalisée auprès des services consulaires ou des autorités locales du pays d'accueil ; "
    "modèle PDF sur la page délivrance du passeport biométrique (consulat.ma). "
    "2) Déposer un nouveau dossier de passeport biométrique (renouvellement après perte/vol) : "
    "immatriculation consulaire à jour, photos 35×45 mm, CNIE ou reçu CNIE + acte de naissance récent si besoin. "
    "3) Au retrait : récépissé de dépôt ; si perte du récépissé, nouvelle déclaration sur l'honneur légalisée. "
    "MRE au Maroc : vol/perte/expiration — dépôt selon passeport.ma (renvoi consulat.ma pour pièces détaillées)."
)

_CURATED_PASSEPORT_ETIMBRE = (
    "Timbre fiscal et frais — passeport biométrique marocain (Maroc) : "
    "les droits de timbre s’acquittent via un code e-timbre à 16 chiffres, acheté en ligne sur le portail officiel "
    "(rubrique « Acheter votre e-Timbre » sur www.passeport.ma, ou plateforme e-timbre de l’administration fiscale "
    "tax.gov.ma). Le montant exact dépend du type de demande (première demande, renouvellement, mineur, etc.) "
    "et est affiché au moment de l’achat du e-timbre — il n’est en général pas fixé dans un seul article du "
    "Bulletin officiel indexé ici. Procédure : 1) acheter le e-timbre correspondant à votre cas sur passeport.ma ; "
    "2) conserver le code à 16 chiffres ; 3) joindre ce code au dossier (formulaire + pièces sur passeport.ma / "
    "consulat). Ne pas confondre avec le timbre ou les frais de la CNIE (montants et nature différents). "
    "Pour le barème en vigueur (2025 ou année courante), vérifier directement sur passeport.ma au moment de "
    "l’achat du e-timbre."
)


_BO_BUDGET_NOISE_MARKERS = (
    "loi de finances",
    "annee budgetaire",
    "année budgétaire",
    "recettes ordinaires",
    "budget general",
    "budget général",
    "evaluation globale des recettes",
    "droits d'enregistrement et de timbre",
    "impots directs",
    "impôts directs",
)


def passeport_fee_hits_substantive(
    question: str, hits: list[dict[str, Any]]
) -> bool:
    """
    True si les extraits parlent vraiment du timbre / e-timbre passeport
    (pas d'un article BO budget général qui cite « timbre »).
    """
    if not hits or not _is_passeport_fee_question(question):
        return False
    for h in hits[:6]:
        cid = str((h.get("metadata") or {}).get("chunk_id") or "")
        if cid.startswith("curated::passeport"):
            return True
        if cid.startswith("consulat_passeport"):
            txt = (h.get("text") or "").lower()
            if any(
                m in txt
                for m in (
                    "e-timbre",
                    "etimbre",
                    "droits de timbre",
                    "timbre fiscal",
                    "code e-timbre",
                )
            ):
                return True
    combined = " ".join(
        (h.get("text") or "") for h in hits[:5]
    ).lower()
    if any(n in combined for n in _BO_BUDGET_NOISE_MARKERS):
        if not any(
            m in combined
            for m in (
                "passeport biometrique",
                "passeport biométrique",
                "e-timbre",
                "etimbre",
                "passeport.ma",
                "curated::passeport",
            )
        ):
            return False
    return any(
        m in combined
        for m in (
            "e-timbre",
            "etimbre",
            "passeport.ma",
            "code e-timbre",
            "acheter votre e-timbre",
            "droits de timbre sont acquittes",
            "droits de timbre sont acquittés",
        )
    )


def _is_passeport_fee_question(question: str) -> bool:
    q = (question or "").lower()
    if "passeport" not in q and "e-timbre" not in q and "etimbre" not in q:
        return False
    return any(
        x in q
        for x in (
            "timbre",
            "e-timbre",
            "etimbre",
            "frais",
            "cout",
            "coût",
            "combien",
            "montant",
            "prix",
            "tarif",
            "redevance",
            "droits de timbre",
        )
    )


def _curated_passeport_fees_hits() -> list[dict[str, Any]]:
    return [
        {
            "index": -1,
            "score": 0.99,
            "rerank_score": 0.99,
            "metadata": {
                "chunk_id": "curated::passeport_etimbre",
                "title": "Passeport — timbre fiscal / e-timbre",
                "label": "Frais et e-timbre passeport",
                "source_type": "admin",
                "source_url": "https://www.passeport.ma/",
                "category": "passeport",
                "source_org": "passeport.ma",
            },
            "text": _CURATED_PASSEPORT_ETIMBRE,
        }
    ]


def _dedupe_web_hits(hits: list[dict[str, Any]]) -> list[dict[str, Any]]:
    seen: set[str] = set()
    out: list[dict[str, Any]] = []
    for h in hits:
        meta = h.get("metadata") or {}
        key = str(meta.get("chunk_id") or meta.get("source_url") or id(h))
        if key in seen:
            continue
        seen.add(key)
        out.append(h)
    return out


def _dataset_passeport_hits(question: str, *, limit: int = 4) -> list[dict[str, Any]]:
    try:
        from app.passeport_corpus import best_passeport_hits_for

        return best_passeport_hits_for(question)[:limit]
    except ImportError:
        return []


def _official_bootstrap_hits(
    question: str,
    portal: str | None,
    *,
    max_results: int,
) -> list[dict[str, Any]]:
    """Sources officielles locales (curated + JSONL) sans DuckDuckGo ni scrape."""
    hits: list[dict[str, Any]] = []
    if _is_passeport_fee_question(question):
        hits.extend(_curated_passeport_fees_hits())
    if portal == "cnie":
        hits.extend(_curated_cnie_hits(question))
    elif portal == "passeport":
        hits.extend(_curated_passeport_hits(question))
        hits.extend(_dataset_passeport_hits(question))
    elif portal == "watiqa":
        hits.extend(_curated_watiqa_hits())
    if _labor_topic_question(question):
        hits.extend(_curated_smig_hits())
    return _dedupe_web_hits(hits)[:max_results]


def _finalize_portal_hits(
    question: str,
    portal: str | None,
    hits: list[dict[str, Any]],
    *,
    max_results: int,
) -> list[dict[str, Any]]:
    """Injecte curated / dataset quand le scrape ou DDG ont échoué."""
    out = list(hits)
    if portal == "cnie":
        out = [h for h in out if not _web_hit_is_passeport_not_cnie(h)]
        curated = _curated_cnie_hits(question)
        cid = curated[0]["metadata"]["chunk_id"]
        out = [h for h in out if (h.get("metadata") or {}).get("chunk_id") != cid]
        out = curated + out
    elif portal == "passeport":
        bootstrap = _official_bootstrap_hits(question, "passeport", max_results=max_results)
        out = _dedupe_web_hits(bootstrap + out)
    elif portal == "watiqa":
        out = [h for h in out if not _web_hit_is_cnie_not_watiqa(h)]
        out = _prioritize_watiqa_hits(out)
        if not _hits_cover_portal("watiqa", out):
            out = _curated_watiqa_hits() + out
    elif _is_passeport_fee_question(question):
        out = _dedupe_web_hits(_curated_passeport_fees_hits() + out)
    elif _labor_topic_question(question) and not local_hits_substantively_answer(
        question, out
    ):
        curated = _curated_smig_hits()
        cid = curated[0]["metadata"]["chunk_id"]
        out = curated + [h for h in out if (h.get("metadata") or {}).get("chunk_id") != cid]
    out.sort(key=lambda h: float(h.get("rerank_score", 0.0)), reverse=True)
    return out[:max_results]


def curated_fallback_hits(question: str, *, top_k: int = 5) -> list[dict[str, Any]]:
    """Dernier recours quand fetch_web_hits() est vide (réseau / DDG / 403)."""
    portal = _portal_intent(question)
    hits = _official_bootstrap_hits(question, portal, max_results=top_k)
    if hits:
        return _finalize_portal_hits(question, portal, hits, max_results=top_k)
    if _labor_topic_question(question):
        return _curated_smig_hits()[:top_k]
    return []


def _curated_passeport_hits(question: str = "") -> list[dict[str, Any]]:
    try:
        from app.passeport_cases import passeport_case_from_question
        from app.phrase_context import analyze_phrase

        if passeport_case_from_question(question) == "perte_vol":
            text = _CURATED_PASSEPORT_PERTE_VOL
            label = "Perte ou vol - declaration et refaire"
        else:
            ctx = analyze_phrase(question)
            mineur = ctx.age_hint == "mineur" or "mineur" in ctx.question_folded
            sans_cnie = "cnie" in ctx.negated_subjects or "sans cnie" in ctx.question_folded
            text = _CURATED_PASSEPORT_MINEUR_SANS_CNIE if (mineur and sans_cnie) else _CURATED_PASSEPORT_GENERAL
            label = "Mineur sans CNIE - passeport" if (mineur and sans_cnie) else "Passeport - procedure et pieces"
    except ImportError:
        text = _CURATED_PASSEPORT_GENERAL
        label = "Passeport - procedure et pieces"
    return [
        {
            "index": -1,
            "score": 0.99,
            "rerank_score": 0.99,
            "metadata": {
                "chunk_id": "curated::passeport_procedure",
                "title": "Passeport biométrique — procédure",
                "label": label,
                "source_type": "admin",
                "source_url": "https://consulat.ma/index.php/fr/delivrance-du-passeport-biometrique",
                "category": "passeport",
                "source_org": "consulat.ma",
            },
            "text": text,
        }
    ]


def _construction_intent(question: str) -> bool:
    q = question.strip().lower()
    keys = (
        "construire",
        "construction",
        "maison",
        "batir",
        "bâtir",
        "immeuble",
        "logement",
        "terrain",
        "urbanisme",
        "permis",
        "autorisation de construire",
        "permis de construire",
        "rokhas",
    )
    return any(k in q for k in keys)


def _unwrap_ddg_href(href: str) -> str:
    """DuckDuckGo HTML renvoie des redirections //duckduckgo.com/l/?uddg=… — extraire l’URL cible."""
    h = (href or "").strip()
    if not h:
        return ""
    if h.startswith("//"):
        h = "https:" + h
    if "duckduckgo.com/l/" in h:
        parsed = urlparse(h)
        target = parse_qs(parsed.query).get("uddg", [""])[0]
        return unquote(target).strip()
    return h


def _ddg_html_enabled() -> bool:
    """HTML DDG souvent en 202 ; par défaut on utilise uniquement l’API DDGS."""
    global _ddg_html_blocked
    if _ddg_html_blocked:
        return False
    return _bool_env("WEB_FALLBACK_USE_DDG_HTML", False)


def _import_ddgs_class():
    """ddgs (nouveau paquet) ou duckduckgo_search, sans avertissement de renommage."""
    with warnings.catch_warnings():
        warnings.filterwarnings(
            "ignore",
            message=".*renamed to `ddgs`.*",
            category=RuntimeWarning,
        )
        try:
            from ddgs import DDGS  # type: ignore[import-untyped]  # noqa: PLC0415

            return DDGS
        except ImportError:
            from duckduckgo_search import DDGS  # noqa: PLC0415

            return DDGS


def _search_duckduckgo_html(query: str) -> list[tuple[str, str]]:
    """Retourne [(url, extrait), …] via la page HTML DDG (souvent bloquée en 202)."""
    global _ddg_html_blocked
    url = f"https://html.duckduckgo.com/html/?q={quote_plus(query)}"
    r = requests.get(
        url,
        headers={
            **HEADERS,
            "User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64) Chrome/120.0.0.0 Safari/537.36",
        },
        timeout=SEARCH_TIMEOUT,
    )
    if r.status_code in (202, 403, 429):
        _ddg_html_blocked = True
        logger.debug("DuckDuckGo HTML indisponible (HTTP %s)", r.status_code)
        return []
    r.raise_for_status()
    soup = BeautifulSoup(r.text, "html.parser")
    out: list[tuple[str, str]] = []
    seen: set[str] = set()
    for block in soup.select(".result"):
        a = block.select_one("a.result__a")
        if not a:
            continue
        href = _unwrap_ddg_href(a.get("href") or "")
        if not href.startswith("http") or not _is_allowed(href):
            continue
        if href in seen:
            continue
        sn_el = block.select_one(".result__snippet")
        snippet = _clean_text(sn_el.get_text(" ", strip=True) if sn_el else "")
        seen.add(href)
        out.append((href, snippet))
        if len(out) >= 10:
            break
    return out


def _search_duckduckgo_api(query: str) -> list[tuple[str, str]]:
    """API DDGS (pip install ddgs ou duckduckgo-search)."""
    if not _bool_env("WEB_FALLBACK_USE_DDGS", True):
        return []
    try:
        DDGS = _import_ddgs_class()
    except ImportError:
        logger.warning(
            "Recherche web DDGS indisponible — pip install ddgs (ou duckduckgo-search)"
        )
        return []
    out: list[tuple[str, str]] = []
    seen: set[str] = set()
    try:
        with warnings.catch_warnings():
            warnings.filterwarnings(
                "ignore",
                message=".*renamed to `ddgs`.*",
                category=RuntimeWarning,
            )
            with DDGS() as ddgs:
                rows = list(ddgs.text(query, max_results=12))
    except Exception:
        logger.debug("Recherche DDGS sans résultat pour: %s", query[:80], exc_info=True)
        return []
    for row in rows:
        href = str(row.get("href") or "").strip()
        body = _clean_text(str(row.get("body") or ""))
        if not href.startswith("http") or not _is_allowed(href) or href in seen:
            continue
        seen.add(href)
        out.append((href, body))
    return out


def _search_duckduckgo(query: str) -> list[tuple[str, str]]:
    pairs: list[tuple[str, str]] = []
    if _ddg_html_enabled():
        pairs = _search_duckduckgo_html(query)
    if not pairs:
        pairs = _search_duckduckgo_api(query)
    return pairs


def _topic_seed_urls(question: str) -> list[str]:
    q = question.lower()
    urls: list[str] = []
    if _is_passeport_fee_question(question):
        urls.extend(_TOPIC_SEED_URLS_EXTRA.get("passeport_fees", []))
    if "smig" in q or "salaire minimum" in q:
        urls.extend(_TOPIC_SEED_URLS.get("smig", []))
    if any(k in q for k in ("code du travail", "licenciement", "préavis", "preavis", "travail")):
        urls.extend(_TOPIC_SEED_URLS.get("travail", []))
    seen: set[str] = set()
    out: list[str] = []
    for u in urls:
        if u not in seen:
            seen.add(u)
            out.append(u)
    return out


def _hits_from_snippets_only(
    question: str,
    pairs: list[tuple[str, str]],
    *,
    top_k: int,
) -> list[dict[str, Any]]:
    """Utilise les extraits de recherche quand le scrape des pages .ma échoue (403, WAF)."""
    qlow = question.lower()
    min_overlap = float(os.environ.get("WEB_FALLBACK_MIN_OVERLAP", "0.05") or 0.05)
    hits: list[dict[str, Any]] = []
    for url, snippet in pairs:
        if len(hits) >= top_k:
            break
        if len(snippet) < 80:
            continue
        if "smig" in qlow and "smig" not in snippet.lower():
            if "salaire minimum" not in snippet.lower():
                continue
        ov = _lexical_overlap(question, snippet)
        if ov < min_overlap:
            continue
        host = (urlparse(url).netloc or "").lower()
        score = float(ov + _domain_priority(host))
        hits.append(
            {
                "index": -1,
                "score": score,
                "rerank_score": score,
                "metadata": {
                    "chunk_id": f"web_snip::{host}",
                    "title": f"Extrait web ({host})",
                    "label": "Web fallback (extrait)",
                    "source_type": "web_fallback",
                    "source_url": url,
                    "domain_trust": round(_domain_priority(host), 3),
                },
                "text": snippet[:2000],
            }
        )
    hits.sort(key=lambda h: float(h.get("rerank_score", 0.0)), reverse=True)
    return hits[:top_k]


def _curated_smig_hits() -> list[dict[str, Any]]:
    return [
        {
            "index": -1,
            "score": 0.99,
            "rerank_score": 0.99,
            "metadata": {
                "chunk_id": "curated::smig_info",
                "title": "SMIG — salaire minimum au Maroc",
                "label": "SMIG - texte de reference",
                "source_type": "admin",
                "source_url": "https://www.miepeec.gov.ma/",
                "category": "travail",
                "source_org": "Ministère de l'Emploi",
            },
            "text": _CURATED_SMIG_INFO,
        }
    ]


def _labor_topic_question(question: str) -> bool:
    try:
        from app.labor_corpus import is_labor_code_question

        return is_labor_code_question(question)
    except ImportError:
        pass
    q = question.lower()
    return any(
        k in q
        for k in (
            "smig",
            "salaire minimum",
            "code du travail",
            "licenciement",
            "préavis",
            "preavis",
            "heures supplémentaires",
            "heures supplementaires",
            "congé annuel",
            "conge annuel",
            "démission",
            "demission",
        )
    )


def _scrape_hits(
    question: str,
    urls: list[str],
    *,
    top_k: int,
    url_snippets: dict[str, str] | None = None,
) -> list[dict[str, Any]]:
    match_str = question.strip()
    if _construction_intent(question):
        match_str = f"{match_str} Maroc autorisations territoriales Rokhas permis urbanisme"
    hits: list[dict[str, Any]] = []
    min_overlap = float(os.environ.get("WEB_FALLBACK_MIN_OVERLAP", "0.05") or 0.05)
    for url in urls:
        if len(hits) >= top_k:
            break
        host = (urlparse(url).netloc or "").lower()
        ddg_snippet = (url_snippets or {}).get(url, "").strip()
        if any(host == h or host.endswith(f".{h}") for h in _SCRAPE_SKIP_HOSTS):
            if len(ddg_snippet) >= 80:
                ov = _lexical_overlap(match_str, ddg_snippet)
                if ov >= min_overlap:
                    hits.append(
                        {
                            "index": -1,
                            "score": float(ov + _domain_priority(host)),
                            "rerank_score": float(ov + _domain_priority(host)),
                            "metadata": {
                                "chunk_id": f"web_snip::{host}",
                                "title": f"Extrait web ({host})",
                                "label": "Web fallback (extrait)",
                                "source_type": "web_fallback",
                                "source_url": url,
                                "domain_trust": round(_domain_priority(host), 3),
                            },
                            "text": ddg_snippet[:2000],
                        }
                    )
            continue
        try:
            r = requests.get(url, headers=HEADERS, timeout=FETCH_TIMEOUT)
            r.raise_for_status()
        except requests.RequestException:
            if len(ddg_snippet) >= 80:
                ov = _lexical_overlap(match_str, ddg_snippet)
                if ov >= min_overlap:
                    hits.append(
                        {
                            "index": -1,
                            "score": float(ov + _domain_priority(host)),
                            "rerank_score": float(ov + _domain_priority(host)),
                            "metadata": {
                                "chunk_id": f"web_snip::{host}",
                                "title": f"Extrait web ({host})",
                                "label": "Web fallback (extrait)",
                                "source_type": "web_fallback",
                                "source_url": url,
                            },
                            "text": ddg_snippet[:2000],
                        }
                    )
            continue

        text = _extract_text_from_html(r.text)
        ddg_snippet = (url_snippets or {}).get(url, "").strip()
        if len(text) < 120 and len(ddg_snippet) >= 80:
            text = ddg_snippet
        if len(text) < 80:
            continue

        ov = _lexical_overlap(match_str, text)
        if ov < min_overlap and len(ddg_snippet) >= 80:
            ov = _lexical_overlap(match_str, ddg_snippet)
        if ov < min_overlap:
            continue

        snippet = text[:1800]
        host = (urlparse(url).netloc or "").lower()
        score = float(ov + _domain_priority(host))
        if portal := _portal_intent(question):
            boost_text = f"{snippet} {ddg_snippet}".lower()
            if portal == "cnie" and any(
                x in boost_text for x in ("cnie", "carte nationale", "identité", "identite", "pièces", "pieces")
            ):
                score += 0.15
            if portal == "passeport" and "passeport" in boost_text:
                score += 0.15
            if portal == "watiqa" and "watiqa" in boost_text:
                score += 0.15
        hits.append(
            {
                "index": -1,
                "score": score,
                "rerank_score": score,
                "metadata": {
                    "chunk_id": f"web::{host}",
                    "title": f"Source web ({host})",
                    "label": "Web fallback",
                    "source_type": "web_fallback",
                    "source_url": url,
                    "domain_trust": round(_domain_priority(host), 3),
                },
                "text": snippet,
            }
        )
    return hits


def _intent_queries(question: str) -> list[str]:
    q0 = question.strip()
    qn = q0.lower()
    queries = [f"{q0} maroc officiel"]
    if _construction_intent(q0):
        queries.insert(0, f"{q0} Maroc rokhas.ma autorisation de construire")
        queries.append("Maroc rokhas autorisations territoriales guichet unique")
    if any(k in qn for k in ("cnie", "carte nationale", "identite")):
        queries.insert(0, f"site:maroc.ma {q0} CNIE pièces")
        queries.insert(0, f"site:mre.gov.ma {q0} carte nationale identité")
        queries.insert(0, f"{q0} cnie.ma documents requis")
    if "passeport" in qn:
        queries.insert(0, f"site:passeport.ma {q0} e-timbre timbre fiscal")
        queries.insert(0, f"{q0} passeport.ma pieces a fournir e-timbre")
        queries.insert(0, f"site:tax.gov.ma passeport e-timbre Maroc")
    if _is_passeport_fee_question(q0):
        queries.insert(0, f"site:passeport.ma timbre fiscal passeport Maroc e-timbre")
        queries.insert(0, f"{q0} e-timbre passeport biométrique montant")
    if any(k in qn for k in ("etat civil", "acte de naissance", "watiqa")):
        queries.insert(0, f"site:watiqa.ma {q0}")
        queries.insert(0, f"{q0} watiqa.ma etat civil")
    if any(k in qn for k in ("smig", "salaire minimum", "code du travail", "licenciement", "préavis", "preavis")):
        queries.insert(0, f"{q0} SMIG Maroc montant bulletin officiel")
        queries.insert(0, f"site:miepeec.gov.ma SMIG salaire minimum")
    if any(k in qn for k in ("bulletin officiel", "dahir", "decret", "arrêté", "arrete", "loi")):
        queries.insert(0, f"{q0} sgg.gov.ma bulletin officiel")

    # dédup
    seen: set[str] = set()
    out: list[str] = []
    for q in queries:
        qq = " ".join(q.split())
        if not qq or qq in seen:
            continue
        seen.add(qq)
        out.append(qq)
    return out


def fetch_web_hits(question: str, top_k: int = MAX_RESULTS) -> list[dict[str, Any]]:
    q0 = question.strip()
    max_results = _int_env("WEB_FALLBACK_MAX_RESULTS", top_k)
    portal = _portal_intent(q0)

    bootstrap = _official_bootstrap_hits(q0, portal, max_results=max_results)
    if _bool_env("WEB_FALLBACK_CURATED_FIRST", True):
        if _is_passeport_fee_question(q0) and bootstrap:
            logger.info("Web fallback : e-timbre passeport (curated, sans recherche DDG)")
            return _finalize_portal_hits(q0, portal, bootstrap, max_results=max_results)
        if portal in ("cnie", "passeport", "watiqa") and bootstrap:
            if portal != "passeport" or _dataset_passeport_hits(q0) or _is_passeport_fee_question(
                q0
            ):
                if local_hits_substantively_answer(q0, bootstrap) or portal == "passeport":
                    logger.info(
                        "Web fallback : sources officielles locales (%s), recherche DDG évitée",
                        portal,
                    )
                    return _finalize_portal_hits(q0, portal, bootstrap, max_results=max_results)

    all_urls: list[str] = []
    url_snippets: dict[str, str] = {}
    search_pairs: list[tuple[str, str]] = []

    if portal:
        for seed in _PORTAL_SEED_URLS.get(portal, []):
            if seed not in all_urls and not _url_blocked_for_portal(seed, portal, q0):
                all_urls.append(seed)

    for seed in _topic_seed_urls(q0):
        if seed not in all_urls:
            all_urls.append(seed)

    max_search_q = _int_env("WEB_FALLBACK_MAX_SEARCH_QUERIES", 3)
    for q in _intent_queries(q0)[:max_search_q]:
        try:
            pairs = _search_duckduckgo(q)
        except requests.RequestException:
            logger.debug("Recherche web interrompue: %s", q[:80])
            continue
        for u, snippet in pairs:
            if _url_blocked_for_portal(u, portal or "", q0):
                continue
            search_pairs.append((u, snippet))
            if u not in all_urls:
                all_urls.append(u)
            if snippet and u not in url_snippets:
                url_snippets[u] = snippet
            if len(all_urls) >= 18:
                break
        if len(all_urls) >= 18:
            break

    hits = _scrape_hits(q0, all_urls, top_k=max_results, url_snippets=url_snippets)
    if not hits and search_pairs:
        hits = _hits_from_snippets_only(q0, search_pairs, top_k=max_results)

    if not hits and bootstrap:
        hits = bootstrap
    elif bootstrap:
        hits = _dedupe_web_hits(bootstrap + hits)

    return _finalize_portal_hits(q0, portal, hits, max_results=max_results)


def save_web_hits_to_queue(question: str, hits: list[dict[str, Any]]) -> None:
    if not hits:
        return
    WEB_QUEUE_PATH.parent.mkdir(parents=True, exist_ok=True)
    now = datetime.now(timezone.utc).isoformat()
    with WEB_QUEUE_PATH.open("a", encoding="utf-8") as f:
        for h in hits:
            meta = h.get("metadata") or {}
            row = {
                "collected_at": now,
                "question": question.strip(),
                "title": meta.get("title"),
                "source_url": meta.get("source_url"),
                "source_type": "web_fallback",
                "text": h.get("text", ""),
                "score": h.get("rerank_score", h.get("score", 0.0)),
            }
            f.write(json.dumps(row, ensure_ascii=False) + "\n")
