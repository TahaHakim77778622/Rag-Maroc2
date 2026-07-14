import json
import csv
from pathlib import Path
from pypdf import PdfReader

RAW_DIR = Path("data/raw_sgg_bo")
SOURCES_CSV = Path("data/sources.csv")
OUT_DIR = Path("data/processed/pages_jsonl")
OUT_DIR.mkdir(parents=True, exist_ok=True)

def load_sources():
    sources = {}
    with open(SOURCES_CSV, "r", encoding="utf-8-sig") as f:
        reader = csv.DictReader(f)
        for row in reader:
            sources[row["filename"]] = row
    return sources

def extract_pdf(pdf_path: Path, source_meta: dict):
    reader = PdfReader(str(pdf_path))
    out_path = OUT_DIR / f"{pdf_path.stem}.jsonl"

    with out_path.open("w", encoding="utf-8") as f:
        for i, page in enumerate(reader.pages):
            text = page.extract_text() or ""
            record = {
                "doc_id": source_meta["doc_id"],
                "title": source_meta["title"],
                "source_org": source_meta["source_org"],
                "source_url": source_meta["source_url"],
                "filename": pdf_path.name,
                "page": i + 1,
                "text": text.strip()
            }
            f.write(json.dumps(record, ensure_ascii=False) + "\n")

def main():
    sources = load_sources()
    pdfs = sorted(RAW_DIR.glob("*.pdf"))

    print(f"Nombre de PDF trouvés : {len(pdfs)}")

    if not pdfs:
        print("Aucun PDF trouvé.")
        return

    for pdf in pdfs:
        if pdf.name not in sources:
            print(f"⚠ {pdf.name} absent de sources.csv")
            continue

        print(f"Extraction de {pdf.name}")
        try:
            extract_pdf(pdf, sources[pdf.name])
        except Exception as e:
            print(f"Erreur avec {pdf.name} -> {e}")

    print("sExtraction terminée.")

if __name__ == "__main__":
    main()