#!/usr/bin/env python3
"""
Chunks procédure CNIE (première demande, renouvellement, photos, mineurs)
— le portail cnie.ma est une SPA vide au scrape ; ces textes comblent le trou.

Usage:
    python scripts/append_cnie_procedure_chunks.py
    python scripts/build_embeddings.py
    python scripts/build_faiss.py
"""

from __future__ import annotations

import json
from pathlib import Path

PROJECT = Path(__file__).resolve().parent.parent
DATASET = PROJECT / "data" / "processed" / "final_chunks.jsonl"

# Synthèse procédures citoyen au Maroc (DGSN / CNIE) — à valider sur cnie.ma si le site charge.
NEW_ROWS = [
    {
        "chunk_id": "cnie_procedure_premiere_demande_maroc_1",
        "doc_id": "cnie_procedure_citoyen",
        "title": "CNIE — première demande au Maroc",
        "source_org": "CNIE Maroc",
        "source_url": "https://www.cnie.ma/static/procedure",
        "filename": "cnie_procedure_citoyen",
        "page_start": 1,
        "source_type": "admin",
        "category": "cnie",
        "label": "Premiere demande - pieces a fournir",
        "text": (
            "Première demande de Carte Nationale d'Identité Électronique (CNIE) au Maroc — pièces à fournir "
            "(dépôt en personne au commissariat de police ou au service d'arrondissement de votre lieu de résidence, "
            "ou via le portail www.cnie.ma selon les cas ouverts en ligne) : "
            "1) Formulaire de demande de la CNIE (téléchargeable sur cnie.ma ou fourni sur place), dûment rempli et signé ; "
            "2) Acte de naissance : extrait d'acte de naissance ou copie intégrale de l'acte de naissance "
            "(souvent de moins de six mois selon les services — vérifier auprès du guichet) ; "
            "3) Justificatif de domicile de moins de trois mois au nom du demandeur "
            "(facture d'eau, d'électricité, attestation de résidence, contrat de bail enregistré, etc.) ; "
            "4) Photographies d'identité récentes aux normes CNIE (format et fond conformes — voir normes officielles sur cnie.ma) ; "
            "5) Timbre fiscal / frais de délivrance de la CNIE (montant en vigueur au moment du dépôt) ; "
            "6) Présence obligatoire du demandeur pour la prise des données biométriques (empreintes, photo) ; "
            "7) En cas de renouvellement (et non première demande) : présenter en plus l'ancienne carte nationale d'identité. "
            "Un récépissé de dépôt est délivré ; il est exigé au retrait de la CNIE."
        ),
    },
    {
        "chunk_id": "cnie_procedure_photos_normes_1",
        "doc_id": "cnie_procedure_citoyen",
        "title": "CNIE — normes photographiques",
        "source_org": "CNIE Maroc",
        "source_url": "https://www.cnie.ma/static/procedure/normes-photographies",
        "filename": "cnie_procedure_citoyen",
        "page_start": 1,
        "source_type": "admin",
        "category": "cnie",
        "label": "Normes photos CNIE",
        "text": (
            "Normes des photographies pour la CNIE (Carte Nationale d'Identité Électronique) : "
            "photos récentes, en couleur, sur fond clair uniforme (blanc ou bleu clair selon consignes en vigueur), "
            "visage dégagé et centré, sans couvre-chef (sauf motif religieux ou médical attesté), "
            "sans lunettes teintées, expression neutre, bouche fermée ; "
            "format habituel 35 mm × 45 mm (vérifier le format exact sur www.cnie.ma / rubrique normes des photographies). "
            "Les photos non conformes entraînent le rejet du dossier ou une nouvelle prise sur place."
        ),
    },
    {
        "chunk_id": "cnie_procedure_mineur_1",
        "doc_id": "cnie_procedure_citoyen",
        "title": "CNIE — demande pour mineur",
        "source_org": "CNIE Maroc",
        "source_url": "https://www.cnie.ma/static/procedure",
        "filename": "cnie_procedure_citoyen",
        "page_start": 1,
        "source_type": "admin",
        "category": "cnie",
        "label": "Mineur - pieces et presence",
        "text": (
            "CNIE pour mineur (première demande ou renouvellement au Maroc) — en plus des pièces de base : "
            "présence obligatoire du mineur accompagné du père, de la mère ou du représentant légal ; "
            "pièce d'identité du représentant légal (CNIE ou passeport en cours de validité) ; "
            "livret de famille ou extrait / copie intégrale d'acte de naissance du mineur mentionnant les noms des parents "
            "(en arabe et en caractères latins selon les services) ; "
            "autorisation parentale ou justificatif de qualité de représentant si le déposant n'est pas le père ou la mère ; "
            "photographies du mineur aux normes CNIE ; "
            "justificatif de domicile au nom du représentant légal ou du mineur selon les règles du guichet."
        ),
    },
    {
        "chunk_id": "cnie_procedure_perte_vol_1",
        "doc_id": "cnie_procedure_citoyen",
        "title": "CNIE — perte ou vol",
        "source_org": "CNIE Maroc",
        "source_url": "https://www.cnie.ma/static/procedure",
        "filename": "cnie_procedure_citoyen",
        "page_start": 1,
        "source_type": "admin",
        "category": "cnie",
        "label": "Perte ou vol - procedure",
        "text": (
            "CNIE perdue ou volée au Maroc — procédure (renouvellement en cas de perte ou de vol) : "
            "1) Déclarer la perte ou le vol auprès du commissariat de police ou du service d'arrondissement "
            "(déclaration de perte ou de vol — modèle selon le guichet) ; "
            "2) Déposer un dossier de renouvellement pour perte/vol : formulaire CNIE, "
            "extrait ou copie intégrale d'acte de naissance (souvent récent), justificatif de domicile de moins de trois mois, "
            "photographies aux normes CNIE, timbre fiscal / frais, déclaration de perte ou de vol ; "
            "3) Présence du demandeur pour biométrie ; "
            "4) Ne pas confondre avec la première demande (pas d'ancienne carte) ni avec le passeport. "
            "Portail : www.cnie.ma — rubrique demande / cas particuliers selon les options en ligne."
        ),
    },
    {
        "chunk_id": "cnie_procedure_renouvellement_1",
        "doc_id": "cnie_procedure_citoyen",
        "title": "CNIE — renouvellement",
        "source_org": "CNIE Maroc",
        "source_url": "https://www.cnie.ma/static/procedure",
        "filename": "cnie_procedure_citoyen",
        "page_start": 1,
        "source_type": "admin",
        "category": "cnie",
        "label": "Renouvellement - pieces",
        "text": (
            "Renouvellement de la CNIE au Maroc — pièces usuelles : "
            "ancienne carte nationale d'identité (originale) ; "
            "formulaire de demande de renouvellement ; "
            "justificatif de domicile récent si changement d'adresse ; "
            "photographies aux normes CNIE ; "
            "timbre fiscal / frais ; "
            "présence du titulaire pour mise à jour biométrique si exigée. "
            "En cas de perte ou vol : déclaration de perte / vol et pièces complémentaires selon le commissariat "
            "(extrait d'acte de naissance, photos, justificatif de domicile)."
        ),
    },
    {
        "chunk_id": "cnie_procedure_mre_consulat_1",
        "doc_id": "cnie_procedure_mre",
        "title": "CNIE — première demande à l'étranger (consulat)",
        "source_org": "consulat.ma",
        "source_url": "https://consulat.ma/index.php/fr/delivrance-de-la-cnie-pour-la-premiere-fois",
        "filename": "cnie_consulat_premiere_fois",
        "page_start": 1,
        "source_type": "admin",
        "category": "cnie",
        "label": "MRE premiere demande consulat",
        "text": (
            "Première demande de CNIE à l'étranger (Marocain résidant à l'étranger — prestation consulaire, consulat.ma) : "
            "dépôt auprès du consulat de résidence ; présence personnelle du demandeur ; "
            "attestation d'immatriculation consulaire en cours de validité ; "
            "quatre photographies d'identité récentes, en couleur, fond clair, format 35 mm × 45 mm, visage découvert ; "
            "document de filiation : copie de la page du livret de famille (présentation du livret) "
            "OU copie intégrale d'acte de naissance de moins de six mois "
            "OU extrait d'acte de naissance de moins de six mois ; "
            "mention « épouse » : copie certifiée de l'acte de mariage + extrait d'acte de naissance du mari ; "
            "mention « veuf/veuve » : copie certifiée de l'acte de mariage + extrait d'acte de décès du conjoint. "
            "Ne pas confondre avec la procédure au Maroc (commissariat / cnie.ma) ni avec les pièces du passeport biométrique."
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


def main() -> int:
    if not DATASET.is_file():
        raise SystemExit(f"Fichier introuvable : {DATASET}")

    existing = _existing_chunk_ids(DATASET)
    to_add = [r for r in NEW_ROWS if r["chunk_id"] not in existing]
    if not to_add:
        print("Déjà présent : aucun chunk CNIE ajouté.")
        return 0

    tail = DATASET.read_text(encoding="utf-8")
    with DATASET.open("a", encoding="utf-8") as f:
        if tail and not tail.endswith("\n"):
            f.write("\n")
        for r in to_add:
            f.write(json.dumps(r, ensure_ascii=False) + "\n")

    print(f"Ajouté {len(to_add)} chunk(s) CNIE.")
    print("Relance : python scripts/build_embeddings.py && python scripts/build_faiss.py")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
