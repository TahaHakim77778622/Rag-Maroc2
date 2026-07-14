import json
import re
import time
from pathlib import Path
from urllib.parse import urlparse

import fitz  # PyMuPDF
import requests
from bs4 import BeautifulSoup

RAW_DIR = Path("data/raw_admin_docs")
PROCESSED_DIR = Path("data/processed/admin_pages_jsonl")

RAW_DIR.mkdir(parents=True, exist_ok=True)
PROCESSED_DIR.mkdir(parents=True, exist_ok=True)

# Sources officielles vérifiées
SOURCES = [
    # CNIE
    {"url": "https://www.cnie.ma/", "category": "cnie"},
    {"url": "https://www.cnie.ma/static/procedure", "category": "cnie"},
    {"url": "https://www.cnie.ma/static/procedure/normes-photographies", "category": "cnie"},
    {"url": "https://www.cnie.ma/static/about", "category": "cnie"},
    {"url": "https://cnie.ma/request-select-type", "category": "cnie"},

    # Passeport
    {"url": "https://www.passeport.ma/", "category": "passeport"},
    {"url": "https://www.passeport.ma/Home/PiecesAuMaroc", "category": "passeport"},
    {"url": "https://www.passeport.ma/Procedure/GuideAuMaroc", "category": "passeport"},
    {"url": "https://www.passeport.ma/ConditionsDelivrance/DelAuMaroc", "category": "passeport"},
    {"url": "https://www.passeport.ma/FormDemande/FormDemande", "category": "passeport"},

    # Watiqa
    {"url": "https://www.watiqa.ma/", "category": "etat_civil"},
    {"url": "https://www.watiqa.ma/?page=citoyen.GuichetActe", "category": "etat_civil"},
    {"url": "https://www.watiqa.ma/index.php5?page=citoyen.Faq", "category": "etat_civil"},
    {"url": "https://www.watiqa.ma/index.php5?page=common.aideEnLigne", "category": "etat_civil"},

    # Portail national services publics
    {"url": "https://www.maroc.ma/fr/services-numeriques", "category": "services_publics"},
]

HEADERS = {
    "User-Agent": "Mozilla/5.0 (compatible; RAG-Maroc-Project/1.0)"
}

REQUEST_TIMEOUT = 30
SLEEP_SECONDS = 1.0


def slugify(text: str) -> str:
    text = text.lower().strip()
    text = re.sub(r"https?://", "", text)
    text = re.sub(r"[^a-zA-Z0-9]+", "_", text)
    text = re.sub(r"_+", "_", text).strip("_")
    return text[:140]


def clean_text(text: str) -> str:
    if not text:
        return ""
    text = text.replace("\u00a0", " ")
    text = text.replace("\r", "\n")
    text = text.replace("￾", " ")
    text = re.sub(r"[ \t]+", " ", text)
    text = re.sub(r"\n{3,}", "\n\n", text)
    return text.strip()


def source_org_from_url(url: str) -> str:
    domain = urlparse(url).netloc.lower()
    if "passeport.ma" in domain:
        return "Passeport Maroc"
    if "cnie.ma" in domain:
        return "CNIE Maroc"
    if "watiqa.ma" in domain:
        return "Watiqa"
    if "maroc.ma" in domain:
        return "Maroc.ma"
    return domain


def build_doc_id(url: str) -> str:
    return slugify(url)


def build_filename(url: str, content_type: str) -> str:
    slug = slugify(url)
    path = urlparse(url).path.lower()
    if "pdf" in content_type or path.endswith(".pdf"):
        return f"{slug}.pdf"
    return f"{slug}.html"


def save_binary_file(path: Path, content: bytes):
    with open(path, "wb") as f:
        f.write(content)


def save_text_file(path: Path, content: str):
    with open(path, "w", encoding="utf-8") as f:
        f.write(content)


def extract_pdf_text(pdf_path: Path):
    doc = fitz.open(pdf_path)
    pages = []

    for i, page in enumerate(doc, start=1):
        text = clean_text(page.get_text("text"))
        pages.append({
            "page": i,
            "text": text
        })

    title = pdf_path.name
    if pages and pages[0]["text"]:
        first_lines = pages[0]["text"].splitlines()
        if first_lines:
            title = first_lines[0][:200]

    doc.close()
    return title, pages


def extract_html_text(html: str):
    soup = BeautifulSoup(html, "html.parser")

    for tag in soup(["script", "style", "noscript"]):
        tag.decompose()

    title = ""
    if soup.title and soup.title.string:
        title = clean_text(soup.title.string)

    text = clean_text(soup.get_text("\n"))

    # On stocke chaque page web comme une "page 1"
    pages = [{
        "page": 1,
        "text": text
    }]

    return title, pages


def process_source(source: dict):
    url = source["url"]
    category = source["category"]

    print(f"Traitement: {url}")

    response = requests.get(url, headers=HEADERS, timeout=REQUEST_TIMEOUT)
    response.raise_for_status()

    content_type = response.headers.get("Content-Type", "").lower()
    filename = build_filename(url, content_type)

    raw_path = RAW_DIR / filename
    out_path = PROCESSED_DIR / f"{build_doc_id(url)}.jsonl"

    source_org = source_org_from_url(url)
    doc_id = build_doc_id(url)

    if "pdf" in content_type or filename.endswith(".pdf"):
        save_binary_file(raw_path, response.content)
        title, pages = extract_pdf_text(raw_path)
        doc_type = "admin_pdf"
    else:
        html = response.text
        save_text_file(raw_path, html)
        title, pages = extract_html_text(html)
        doc_type = "admin_web"

    with open(out_path, "w", encoding="utf-8") as f:
        for page in pages:
            record = {
                "doc_id": doc_id,
                "title": title or filename,
                "source_org": source_org,
                "source_url": url,
                "filename": filename,
                "page": page["page"],
                "doc_type": doc_type,
                "category": category,
                "text": page["text"]
            }
            f.write(json.dumps(record, ensure_ascii=False) + "\n")

    print(f"  -> sauvegardé: {out_path}")


def main():
    success = 0
    failed = 0

    for source in SOURCES:
        try:
            process_source(source)
            success += 1
            time.sleep(SLEEP_SECONDS)
        except Exception as e:
            failed += 1
            print(f"  -> erreur: {e}")

    print(f"\n✅ Terminé. Succès: {success} | Échecs: {failed}")
    print(f"✅ Dossier brut: {RAW_DIR}")
    print(f"✅ Dossier JSONL: {PROCESSED_DIR}")


if __name__ == "__main__":
    main()