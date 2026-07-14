#!/usr/bin/env python3
"""
Chunks procédure Watiqa (commande acte de naissance, suivi) — idempotent.

Usage:
    python scripts/append_watiqa_procedure_chunks.py
    python scripts/build_embeddings.py && python scripts/build_faiss.py
"""

from __future__ import annotations

import json
from pathlib import Path

PROJECT = Path(__file__).resolve().parent.parent
DATASET = PROJECT / "data" / "processed" / "final_chunks.jsonl"

NEW_ROWS = [
    {
        "chunk_id": "watiqa_procedure_acte_naissance_commande_1",
        "doc_id": "watiqa_procedure_citoyen",
        "title": "Watiqa — commander un acte de naissance",
        "source_org": "Watiqa",
        "source_url": "https://www.watiqa.ma/?page=citoyen.GuichetActe",
        "filename": "watiqa_procedure_citoyen",
        "page_start": 1,
        "source_type": "admin",
        "category": "etat_civil",
        "label": "Acte de naissance - commande en ligne",
        "text": (
            "Commander un acte de naissance sur Watiqa (www.watiqa.ma) : accéder au guichet « Acte de naissance » "
            "(page citoyen.GuichetActe). Choisir le type de document : extrait d'acte de naissance ou copie intégrale. "
            "Cliquer sur « Nouvelle demande » puis « Commencer la démarche ». "
            "Le formulaire comporte six étapes : (1) Administration — province/préfecture, commune, bureau d'état civil "
            "(ou inscription à l'étranger si applicable) ; (2) Document — type d'acte demandé ; "
            "(3) Livraison — adresse postale de réception ; (4) Validation des informations ; "
            "(5) Règlement par carte bancaire des frais d'envoi ; (6) Confirmation avec numéro de commande. "
            "Les documents sont envoyés par courrier recommandé. Service réservé aux personnes inscrites aux registres "
            "d'état civil marocains (conditions MRE selon le portail)."
        ),
    },
    {
        "chunk_id": "watiqa_procedure_suivi_commande_1",
        "doc_id": "watiqa_procedure_citoyen",
        "title": "Watiqa — suivi de commande",
        "source_org": "Watiqa",
        "source_url": "https://www.watiqa.ma/?page=citoyen.GuichetActe",
        "filename": "watiqa_procedure_citoyen",
        "page_start": 1,
        "source_type": "admin",
        "category": "etat_civil",
        "label": "Suivre une commande",
        "text": (
            "Suivre une commande Watiqa : sur le guichet acte de naissance ou la page d'accueil, "
            "rubrique « Suivre une commande ». Saisir le numéro de commande et l'adresse e-mail utilisée lors du dépôt. "
            "Le portail affiche l'état d'avancement (création, paiement, traitement, envoi postal). "
            "Pour une réclamation sur le traitement, utiliser l'espace support / réclamation indiqué sur watiqa.ma."
        ),
    },
    {
        "chunk_id": "watiqa_procedure_comment_ca_marche_1",
        "doc_id": "watiqa_procedure_citoyen",
        "title": "Watiqa — comment ça marche",
        "source_org": "Watiqa",
        "source_url": "https://www.watiqa.ma/",
        "filename": "watiqa_procedure_citoyen",
        "page_start": 1,
        "source_type": "admin",
        "category": "etat_civil",
        "label": "Comment ca marche - 4 etapes",
        "text": (
            "Watiqa — fonctionnement du guichet électronique : "
            "(1) Créer une demande en ligne (acte de naissance, copie intégrale, ou attestation d'immatriculation consulaire) ; "
            "(2) Payer les frais de la commande en ligne par carte bancaire ; "
            "(3) Suivre la commande pas à pas avec le numéro de dossier ; "
            "(4) Recevoir les documents par courrier postal à l'adresse renseignée. "
            "La plateforme permet de commander électroniquement des documents administratifs d'état civil."
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
        print("Déjà présent : aucun chunk Watiqa ajouté.")
        return 0

    tail = DATASET.read_text(encoding="utf-8")
    with DATASET.open("a", encoding="utf-8") as f:
        if tail and not tail.endswith("\n"):
            f.write("\n")
        for r in to_add:
            f.write(json.dumps(r, ensure_ascii=False) + "\n")

    print(f"Ajouté {len(to_add)} chunk(s) Watiqa.")
    print("Relance : python scripts/build_embeddings.py && python scripts/build_faiss.py")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
