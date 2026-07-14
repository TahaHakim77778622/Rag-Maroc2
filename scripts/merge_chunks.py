import json
import re
import unicodedata
from pathlib import Path

BO_FILE = Path("data/processed/chunks_by_article.jsonl")
ADMIN_FILE = Path("data/processed/admin_chunks.jsonl")
LABOR_FILE = Path("data/processed/labor_code_chunks.jsonl")
OUTPUT_FILE = Path("data/processed/final_chunks.jsonl")
def _iter_json_objects(raw: str):
    decoder = json.JSONDecoder()
    idx = 0
    n = len(raw)
    while idx < n:
        while idx < n and raw[idx].isspace():
            idx += 1
        if idx >= n:
            break
        record, end = decoder.raw_decode(raw, idx)
        yield record
        idx = end




def _strip_accents(text: str) -> str:
    return "".join(ch for ch in unicodedata.normalize("NFD", text or "") if unicodedata.category(ch) != "Mn")


def _norm(text: str) -> str:
    return _strip_accents((text or "").lower())


def _extract_bo_number(text: str, filename: str = "") -> str | None:
    t = text or ""
    # 1) formes explicites autour de "Bulletin Officiel"
    m = re.search(
        r"(?:BULLETIN\s+OFFICIEL[^\\n]{0,40}?(?:N[º°o]\s*|n[º°o]\s*)(\d{3,5}(?:\s*bis)?)(?!\s*-\s*\d{2,4})|"
        r"(?:N[º°o]\s*|n[º°o]\s*)(\d{3,5}(?:\s*bis)?)(?!\s*-\s*\d{2,4})\s*[-–][^\\n]{0,24}BULLETIN\s+OFFICIEL)",
        t,
        flags=re.IGNORECASE,
    )
    if m:
        val = m.group(1) or m.group(2)
        if val:
            return " ".join(val.split())
    if filename:
        mf = re.search(r"bo_(\d{4})", filename.lower())
        if mf:
            return str(int(mf.group(1)))
    return None


def _extract_law_date(text: str) -> str | None:
    t = text or ""
    m = re.search(r"\((\d{1,2}[/-]\d{1,2}[/-]\d{4})\)", t)
    if m:
        return m.group(1)
    m2 = re.search(
        r"\b(\d{1,2}\s+(?:janvier|fevrier|février|mars|avril|mai|juin|juillet|aout|août|septembre|octobre|novembre|decembre|décembre)\s+\d{4})\b",
        _norm(t),
    )
    if m2:
        return m2.group(1)
    return None


def _extract_document_type(text: str, title: str = "", source_type: str = "") -> str:
    blob = _norm(" ".join([text or "", title or ""]))
    if source_type == "admin":
        return "page_administrative"
    if "dahir" in blob:
        return "dahir"
    if "decret" in blob:
        return "decret"
    if "arrete" in blob:
        return "arrete"
    if "circulaire" in blob:
        return "circulaire"
    if "loi" in blob:
        return "loi"
    if "rapport" in blob:
        return "rapport"
    if "bulletin officiel" in blob:
        return "publication_bo"
    return "document_officiel"


def _extract_ministry(text: str, source_org: str = "", title: str = "") -> str | None:
    blob = _norm(" ".join([text or "", title or ""]))
    # motifs fréquents dans le corpus
    patterns = (
        ("ministere de l'enseignement superieur", "Ministère de l'Enseignement supérieur"),
        ("ministere de l'interieur", "Ministère de l'Intérieur"),
        ("ministere de la justice", "Ministère de la Justice"),
        ("ministere des finances", "Ministère des Finances"),
        ("ministere de l'emploi", "Ministère de l'Emploi"),
        ("ministere de l'equipement", "Ministère de l'Équipement"),
        ("ministere de la sante", "Ministère de la Santé"),
        ("ministere de l'education", "Ministère de l'Éducation"),
    )
    for needle, label in patterns:
        if needle in blob:
            return label
    if source_org:
        so = _norm(source_org)
        if so == "sgg":
            return "SGG"
        if "watiqa" in so:
            return "Watiqa / État civil"
        return source_org
    return None


def _infer_category(text: str, title: str, source_type: str, seed_category: str | None = None) -> str:
    if seed_category:
        return seed_category
    blob = _norm(" ".join([text or "", title or ""]))
    rules = (
        ("etat civil", "etat_civil"),
        ("cnie", "identite"),
        ("carte nationale", "identite"),
        ("passeport", "voyage"),
        ("fiscal", "fiscalite"),
        ("impot", "fiscalite"),
        ("taxe", "fiscalite"),
        ("marche public", "marches_publics"),
        ("urbanisme", "urbanisme"),
        ("construction", "urbanisme"),
        ("permis de construire", "urbanisme"),
        ("travail", "travail"),
        ("emploi", "travail"),
        ("sante", "sante"),
        ("enseignement", "education"),
        ("universite", "education"),
        ("master", "education"),
    )
    for needle, cat in rules:
        if needle in blob:
            return cat
    return "administratif" if source_type == "admin" else "juridique"


def _infer_language(text: str) -> str:
    t = text or ""
    has_ar = bool(re.search(r"[\u0600-\u06FF]", t))
    has_lat = bool(re.search(r"[A-Za-zÀ-ÿ]", t))
    if has_ar and has_lat:
        return "mixte"
    if has_ar:
        return "ar"
    return "fr"


def enrich_metadata(base: dict, *, source_type: str) -> dict:
    text = str(base.get("text") or "")
    title = str(base.get("title") or "")
    filename = str(base.get("filename") or "")
    source_org = str(base.get("source_org") or "")
    bo_num = _extract_bo_number(text, filename)
    enriched = dict(base)
    enriched["bo_number"] = bo_num
    enriched["law_date"] = _extract_law_date(text)
    enriched["document_type"] = _extract_document_type(text, title=title, source_type=source_type)
    enriched["ministry"] = _extract_ministry(text, source_org=source_org, title=title)
    enriched["category"] = _infer_category(
        text,
        title,
        source_type,
        seed_category=base.get("category"),
    )
    enriched["language"] = _infer_language(text)
    return enriched


def normalize_bo(record):
    base = {
        "chunk_id": record.get("chunk_id"),
        "doc_id": record.get("doc_id"),
        "title": record.get("title"),
        "source_org": record.get("source_org"),
        "source_url": record.get("source_url"),
        "filename": record.get("filename"),
        "page_start": record.get("page_start"),
        "source_type": "bulletin_officiel",
        "category": "juridique",
        "label": record.get("article_label"),
        "text": record.get("text"),
    }
    return enrich_metadata(base, source_type="bulletin_officiel")


def normalize_admin(record):
    base = {
        "chunk_id": record.get("chunk_id"),
        "doc_id": record.get("doc_id"),
        "title": record.get("title"),
        "source_org": record.get("source_org"),
        "source_url": record.get("source_url"),
        "filename": record.get("filename"),
        "page_start": record.get("page_start"),
        "source_type": "admin",
        "category": record.get("category", "administratif"),
        "label": record.get("section_label", record.get("label")),
        "text": record.get("text"),
    }
    return enrich_metadata(base, source_type="admin")


def normalize_labor(record):
    """Code du travail (loi 65-99) — complément au corpus BO."""
    base = {
        "chunk_id": record.get("chunk_id"),
        "doc_id": record.get("doc_id"),
        "title": record.get("title"),
        "source_org": record.get("source_org"),
        "source_url": record.get("source_url"),
        "filename": record.get("filename"),
        "page_start": record.get("page_start"),
        "source_type": record.get("source_type", "loi"),
        "category": record.get("category", "code_travail"),
        "label": record.get("label"),
        "text": record.get("text"),
    }
    return enrich_metadata(base, source_type=str(base["source_type"]))


def merge_file(input_file, normalizer, out_f):
    count = 0
    with open(input_file, "r", encoding="utf-8") as f:
        raw = f.read()
    for record in _iter_json_objects(raw):
        if not isinstance(record, dict):
            continue
        normalized = normalizer(record)
        out_f.write(json.dumps(normalized, ensure_ascii=False) + "\n")
        count += 1
    return count


def main():
    total = 0
    with open(OUTPUT_FILE, "w", encoding="utf-8") as out_f:
        total += merge_file(BO_FILE, normalize_bo, out_f)
        total += merge_file(ADMIN_FILE, normalize_admin, out_f)
        if LABOR_FILE.is_file():
            total += merge_file(LABOR_FILE, normalize_labor, out_f)
        else:
            print(f"AVIS: {LABOR_FILE} absent — Code du travail non fusionné.")

    print(f"Fusion terminée. Total chunks: {total}")
    print(f"Fichier créé: {OUTPUT_FILE}")


if __name__ == "__main__":
    main()