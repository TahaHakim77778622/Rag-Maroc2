#!/usr/bin/env python3
"""
Ajoute au dataset des chunks passeport biométrique (pièces / procédure)
sourcés consulat.ma — idempotent (skip si chunk_id déjà présent).

Après exécution : relancer build_embeddings.py + build_faiss.py.
"""

from __future__ import annotations

import json
from pathlib import Path

PROJECT = Path(__file__).resolve().parent.parent
DATASET = PROJECT / "data" / "processed" / "final_chunks.jsonl"

NEW_ROWS = [
    {
        "chunk_id": "consulat_passeport_biometrique_presentation_1",
        "doc_id": "consulat_passeport_biometrique",
        "title": "Passeport biométrique — présentation et procédure",
        "source_org": "consulat.ma",
        "source_url": "https://consulat.ma/index.php/fr/delivrance-du-passeport-biometrique",
        "filename": "consulat_ma_passeport_biometrique",
        "page_start": 1,
        "source_type": "admin",
        "category": "passeport",
        "label": "Presentation et procedure",
        "text": (
            "Passeport biométrique (prestations consulaires) : titre de voyage individuel délivré sans condition d’âge "
            "à tout citoyen marocain qui en fait la demande, sauf décision judiciaire contraire. "
            "Durée de validité : 5 ans si plus de 3 ans ; 3 ans pour les enfants de moins de 3 ans. "
            "Le passeport n’est pas prorogeable. Dépôt de la demande auprès du service consulaire compétent contre récépissé. "
            "Pour les majeurs : passeport établi sur la base de la CNIE en cours de validité ou du reçu de demande de CNIE "
            "délivré par le service consulaire. Pour les mineurs de moins de 12 ans sans CNIE : sur la base de l’extrait d’acte de naissance. "
            "Les mineurs de 12 à 18 ans sans CNIE doivent se présenter au service consulaire pour la prise d’empreintes digitales. "
            "Dépôt du dossier par l’intéressé(e), son représentant ou tuteur légal ; présence obligatoire du demandeur au dépôt ; "
            "mineurs accompagnés du représentant ou tuteur. Le représentant ou tuteur présente sa pièce d’identité (original + photocopie) "
            "et un justificatif de qualité de représentant légal s’il n’est ni père ni mère. Le récépissé est exigé au retrait. "
            "En cas d’altération, vol ou perte de l’ancien passeport : déclaration sur l’honneur de l’intéressé(e), du tuteur ou du représentant légal, "
            "légalisée auprès des services consulaires ou des autorités locales."
        ),
    },
    {
        "chunk_id": "consulat_passeport_biometrique_pieces_majeur_1",
        "doc_id": "consulat_passeport_biometrique",
        "title": "Passeport biométrique — pièces (majeur)",
        "source_org": "consulat.ma",
        "source_url": "https://consulat.ma/index.php/fr/delivrance-du-passeport-biometrique",
        "filename": "consulat_ma_passeport_biometrique",
        "page_start": 1,
        "source_type": "admin",
        "category": "passeport",
        "label": "Pieces a fournir - majeur",
        "text": (
            "Cas personne majeure — pièces à fournir (délivrance passeport biométrique) : "
            "immatriculation consulaire (à jour) ; "
            "CNIE en cours de validité (original + photocopie) ou photocopie du reçu de dépôt de demande de délivrance ou de renouvellement de la CNIE ; "
            "si présentation du seul reçu de dépôt de demande de CNIE : fournir en plus un extrait d’acte de naissance ou copie intégrale de l’acte de naissance "
            "datant de moins de 6 mois ; "
            "en cas de renouvellement : présenter l’ancien passeport ; "
            "deux photographies d’identité identiques et récentes, fond blanc ou bleu clair, format 35 mm × 45 mm."
        ),
    },
    {
        "chunk_id": "consulat_passeport_biometrique_pieces_tutelle_mineurs_1",
        "doc_id": "consulat_passeport_biometrique",
        "title": "Passeport biométrique — majeur sous tutelle et mineurs",
        "source_org": "consulat.ma",
        "source_url": "https://consulat.ma/index.php/fr/delivrance-du-passeport-biometrique",
        "filename": "consulat_ma_passeport_biometrique",
        "page_start": 1,
        "source_type": "admin",
        "category": "passeport",
        "label": "Pieces a fournir - tutelle et mineurs",
        "text": (
            "Cas personne majeure sous tutelle — en plus des documents du majeur : "
            "pièce d’identité du représentant légal (original + copie) ; "
            "justificatif de la qualité de représentant légal et du lien avec la personne sous tutelle si le représentant n’est ni père ni mère ; "
            "présence obligatoire de la personne sous tutelle et de son représentant légal. "
            "Cas mineur 12 à 18 ans : immatriculation consulaire (à jour) ; "
            "CNIE en cours de validité (original + copie) ou photocopie du reçu de dépôt de demande de CNIE ; "
            "si reçu CNIE seul : extrait ou copie intégrale d’acte de naissance de moins de 6 mois ; "
            "deux photos identiques 35×45 mm fond blanc ou bleu clair ; "
            "en cas de renouvellement : ancien passeport. "
            "Cas enfant mineur de moins de 12 ans : immatriculation consulaire ; "
            "extrait d’acte de naissance ou copie intégrale de moins de 6 mois, ou copie du livret de famille "
            "(l’ancien livret d’identité et d’état civil n’est pas accepté) ; "
            "le document doit mentionner nom et prénom de l’enfant et noms des parents en arabe et en caractères latins ; "
            "deux photos identiques 35×45 mm ; en cas de renouvellement : ancien passeport."
        ),
    },
    {
        "chunk_id": "consulat_passeport_biometrique_formulaires_1",
        "doc_id": "consulat_passeport_biometrique",
        "title": "Passeport biométrique — formulaires",
        "source_org": "consulat.ma",
        "source_url": "https://consulat.ma/index.php/fr/delivrance-du-passeport-biometrique",
        "filename": "consulat_ma_passeport_biometrique",
        "page_start": 1,
        "source_type": "admin",
        "category": "passeport",
        "label": "Formulaires PDF",
        "text": (
            "Formulaires et modèles indiqués sur la page « Délivrance du passeport biométrique » : "
            "fiche d’immatriculation (PDF sur consulat.ma) ; "
            "formulaire de demande de passeport biométrique (PDF) ; "
            "modèle de déclaration en cas d’altération, de vol ou de perte de l’ancien passeport, à légaliser au consulat ou auprès des autorités du pays d’accueil (PDF). "
            "Les liens exacts sont ceux publiés sur la même page officielle consulat.ma."
        ),
    },
    {
        "chunk_id": "consulat_passeport_mre_maroc_procedure_1",
        "doc_id": "consulat_passeport_mre_maroc",
        "title": "Passeport biométrique au Maroc (MRE en séjour)",
        "source_org": "consulat.ma",
        "source_url": "https://consulat.ma/index.php/fr/delivrance-du-passeport-biometrique-au-maroc-au-profit-des-mre",
        "filename": "consulat_ma_passeport_mre_maroc",
        "page_start": 1,
        "source_type": "admin",
        "category": "passeport",
        "label": "Procedure et pieces",
        "text": (
            "MRE en séjour au Maroc : en cas de vol, perte, altération ou expiration du passeport, le renouvellement se dépose "
            "auprès de l’annexe administrative ou du caïdat du lieu de résidence habituelle au Maroc. "
            "Sans justification de résidence au Maroc : dépôt à la préfecture ou la province du lieu de séjour ; "
            "le demandeur doit joindre une attestation d’immatriculation consulaire délivrée par le consulat de son lieu de résidence à l’étranger ; "
            "l’adresse sur le passeport sera celle figurant sur cette attestation. "
            "Pour la liste détaillée des pièces à fournir dans ce cadre, la page renvoie explicitement au site www.passeport.ma."
        ),
    },
    {
        "chunk_id": "consulat_passeport_retrait_pieces_1",
        "doc_id": "consulat_passeport_retrait",
        "title": "Retrait du passeport biométrique",
        "source_org": "consulat.ma",
        "source_url": "https://consulat.ma/index.php/fr/retrait-du-passeport-biometrique",
        "filename": "consulat_ma_passeport_retrait",
        "page_start": 1,
        "source_type": "admin",
        "category": "passeport",
        "label": "Procedure et pieces a fournir",
        "text": (
            "Retrait du passeport biométrique : retrait personnel par le titulaire au lieu de dépôt du dossier ou par son représentant légal, "
            "sur présentation du récépissé et d’un document d’identité. "
            "Le titulaire restitue le récépissé daté et signé après vérification des données sur la page 2 du passeport et signature sur le registre de remise. "
            "En cas de renouvellement : le nouveau passeport n’est retiré qu’après présentation de l’ancien passeport pour annulation et oblitération (perforation) ; "
            "restitution de l’ancien passeport au demandeur. "
            "Vol/perte/altération de l’ancien passeport : déclaration sur l’honneur légalisée au consulat ou aux autorités du pays d’accueil. "
            "Si l’ancien passeport contient des visas valides, le signaler pour éviter le cachet d’annulation sur la page du visa. "
            "Pièces à fournir au retrait : récépissé de dépôt (en cas de perte du récépissé : déclaration sur l’honneur légalisée) ; "
            "carte d’identité de la personne qui retire le passeport ; "
            "retrait pour mineur : père, mère avec autorité parentale ou procuration du père, ou mandataire ; "
            "personne sous tutelle : seul le tuteur légal peut retirer ou donner procuration écrite à un tiers. "
            "Passeports non retirés dans les six mois suivant le dépôt : annulés et détruits ; nouvelle demande avec paiement des droits."
        ),
    },
    {
        "chunk_id": "consulat_passeport_suivi_procedure_1",
        "doc_id": "consulat_passeport_suivi",
        "title": "Suivi de la demande du passeport biométrique",
        "source_org": "consulat.ma",
        "source_url": "https://consulat.ma/index.php/fr/suivi-de-la-demande-du-passeport-biometrique",
        "filename": "consulat_ma_passeport_suivi",
        "page_start": 1,
        "source_type": "admin",
        "category": "passeport",
        "label": "Procedure",
        "text": (
            "Suivi de la demande de passeport biométrique : le demandeur peut suivre l’évolution du traitement sur le portail www.passeport.ma, "
            "rubrique « suivi de la demande », en saisissant soit le numéro de dossier de 16 chiffres figurant sur le récépissé "
            "(pays 3 chiffres + consulat 3 chiffres + année 4 chiffres + numéro d’ordre 6 chiffres), "
            "soit les informations personnelles (nom, prénom, date de naissance)."
        ),
    },
]


def _existing_chunk_ids(path: Path) -> set[str]:
    text = path.read_text(encoding="utf-8")
    dec = json.JSONDecoder()
    i = 0
    ids: set[str] = set()
    while i < len(text):
        while i < len(text) and text[i].isspace():
            i += 1
        if i >= len(text):
            break
        try:
            obj, j = dec.raw_decode(text, i)
        except json.JSONDecodeError:
            i += 1
            continue
        if isinstance(obj, dict) and obj.get("chunk_id"):
            ids.add(str(obj["chunk_id"]))
        i = j
    return ids


def main() -> None:
    if not DATASET.is_file():
        raise SystemExit(f"Fichier introuvable : {DATASET}")

    existing = _existing_chunk_ids(DATASET)
    to_add = [r for r in NEW_ROWS if r["chunk_id"] not in existing]
    if not to_add:
        print("Déjà présent : aucun chunk ajouté.")
        return

    tail = DATASET.read_text(encoding="utf-8")
    with DATASET.open("a", encoding="utf-8") as f:
        if tail and not tail.endswith("\n"):
            f.write("\n")
        for r in to_add:
            f.write(json.dumps(r, ensure_ascii=False) + "\n")

    print(f"Ajouté {len(to_add)} chunk(s). Relance : python scripts/build_embeddings.py && python scripts/build_faiss.py")


if __name__ == "__main__":
    main()
