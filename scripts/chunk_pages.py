import json
import re
from pathlib import Path

INPUT_DIR = Path("data/processed/cleaned_pages_jsonl")
OUTPUT_FILE = Path("data/processed/chunks_by_article.jsonl")

# Détection souple des articles
# Exemples reconnus :
# - ARTICLE PREMIER
# - Article 1
# - Art. 2
# - art 26
ARTICLE_PATTERN = re.compile(
    r'(?i)\b(?:article|art\.?)\s+(?:premier|1er|\d+)\b'
)


def load_pages(file_path: Path) -> list[dict]:
    pages = []
    with open(file_path, "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if line:
                pages.append(json.loads(line))
    return pages


def normalize_text(text: str) -> str:
    if not text:
        return ""

    text = text.replace("\u00a0", " ")   # espace insécable
    text = text.replace("￾", " ")        # artefact PDF fréquent
    text = text.replace("\r", "\n")

    # Nettoyage léger sans casser la structure
    text = re.sub(r"[ \t]+", " ", text)
    text = re.sub(r"\n{3,}", "\n\n", text)

    return text.strip()


def build_full_text(pages: list[dict]) -> tuple[str, list[dict]]:
    full_text_parts = []
    page_offsets = []
    cursor = 0

    for p in pages:
        page_text = normalize_text(p.get("text", ""))
        if not page_text:
            continue

        start = cursor
        full_text_parts.append(page_text)
        full_text_parts.append("\n\n")
        cursor += len(page_text) + 2

        page_offsets.append({
            "page": p.get("page"),
            "start": start,
            "end": cursor
        })

    return "".join(full_text_parts).strip(), page_offsets


def find_page(char_pos: int, page_offsets: list[dict]):
    for p in page_offsets:
        if p["start"] <= char_pos < p["end"]:
            return p["page"]
    return page_offsets[0]["page"] if page_offsets else None


def clean_chunk_text(text: str) -> str:
    text = text.strip()
    text = re.sub(r"\n{3,}", "\n\n", text)
    return text


def split_articles(full_text: str) -> list[dict]:
    matches = list(ARTICLE_PATTERN.finditer(full_text))
    chunks = []

    if not matches:
        return chunks

    for i, m in enumerate(matches):
        start = m.start()
        end = matches[i + 1].start() if i + 1 < len(matches) else len(full_text)

        article_text = clean_chunk_text(full_text[start:end])
        label = m.group(0).strip()

        if article_text:
            chunks.append({
                "label": label,
                "start": start,
                "text": article_text
            })

    return chunks


def process_file(file_path: Path, out_f) -> int:
    pages = load_pages(file_path)
    if not pages:
        return 0

    first = pages[0]

    doc_id = first.get("doc_id", file_path.stem)
    title = first.get("title")
    source_org = first.get("source_org")
    source_url = first.get("source_url")
    filename = first.get("filename", f"{file_path.stem}.pdf")

    full_text, page_offsets = build_full_text(pages)
    if not full_text:
        return 0

    chunks = split_articles(full_text)

    # Fallback essentiel :
    # si aucun article n'est détecté, on garde tout le document comme un seul chunk
    if len(chunks) == 0:
        chunks = [{
            "label": "FULL_TEXT",
            "start": 0,
            "text": clean_chunk_text(full_text)
        }]

    count = 0

    for i, chunk in enumerate(chunks, start=1):
        page_start = find_page(chunk["start"], page_offsets)

        record = {
            "chunk_id": f"{doc_id}_unit_{i}",
            "doc_id": doc_id,
            "title": title,
            "source_org": source_org,
            "source_url": source_url,
            "filename": filename,
            "page_start": page_start,
            "article_label": chunk["label"],
            "text": chunk["text"]
        }

        out_f.write(json.dumps(record, ensure_ascii=False) + "\n")
        count += 1

    return count


def main():
    files = sorted(INPUT_DIR.glob("*.jsonl"))

    if not files:
        print("Aucun fichier trouvé dans cleaned_pages_jsonl.")
        return

    total = 0

    with open(OUTPUT_FILE, "w", encoding="utf-8") as out_f:
        for file_path in files:
            try:
                n = process_file(file_path, out_f)
                print(f"{file_path.name} → {n} unités")
                total += n
            except Exception as e:
                print(f"Erreur avec {file_path.name}: {e}")

    print(f"\n✅ TOTAL UNITÉS: {total}")
    print(f"✅ Fichier généré: {OUTPUT_FILE}")


if __name__ == "__main__":
    main()