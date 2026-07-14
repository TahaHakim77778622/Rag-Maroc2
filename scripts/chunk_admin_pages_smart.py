import json
import re
from pathlib import Path

INPUT_DIR = Path("data/processed/admin_pages_jsonl")
OUTPUT_FILE = Path("data/processed/admin_chunks.jsonl")

MAX_CHARS = 1000
# Article IEEE : fenêtre glissante ~50 tokens (~4 car./token en français).
TOKEN_OVERLAP = 50
CHARS_PER_TOKEN = 4
OVERLAP = TOKEN_OVERLAP * CHARS_PER_TOKEN  # 200 caractères


def load_pages(file_path):
    pages = []
    with open(file_path, "r", encoding="utf-8") as f:
        for line in f:
            if line.strip():
                pages.append(json.loads(line))
    return pages


def clean_text(text):
    text = text.replace("\u00a0", " ")
    text = text.replace("￾", " ")
    text = text.replace("\r", "\n")

    text = re.sub(r"[ \t]+", " ", text)
    text = re.sub(r"\n{3,}", "\n\n", text)

    return text.strip()


# 🔥 Détection de titres (sections)
def split_sections(text):
    lines = text.split("\n")

    sections = []
    current = []

    for line in lines:
        line = line.strip()

        # détecte un titre (majuscule ou court)
        if (
            len(line) < 80
            and line.isupper()
            and len(current) > 0
        ):
            sections.append("\n".join(current))
            current = [line]
        else:
            current.append(line)

    if current:
        sections.append("\n".join(current))

    return sections


def split_paragraphs(text):
    return [p.strip() for p in re.split(r"\n\s*\n", text) if p.strip()]


def build_chunks(paragraphs):
    chunks = []
    current = ""

    for para in paragraphs:
        if len(para) > MAX_CHARS:
            # découpe gros bloc
            start = 0
            while start < len(para):
                end = min(len(para), start + MAX_CHARS)
                chunks.append(para[start:end])
                start = end - OVERLAP
            continue

        candidate = f"{current}\n\n{para}".strip() if current else para

        if len(candidate) <= MAX_CHARS:
            current = candidate
        else:
            chunks.append(current)

            # overlap
            tail = current[-OVERLAP:]
            current = f"{tail}\n\n{para}"

    if current:
        chunks.append(current)

    return chunks


def process_file(file_path, out_f):
    pages = load_pages(file_path)
    if not pages:
        return 0

    meta = pages[0]

    doc_id = meta.get("doc_id")
    title = meta.get("title")
    source_org = meta.get("source_org")
    source_url = meta.get("source_url")
    filename = meta.get("filename")
    category = meta.get("category")

    count = 0

    for page in pages:
        text = clean_text(page.get("text", ""))
        page_num = page.get("page", 1)

        if not text:
            continue

        # 🔥 étape 1 : sections
        sections = split_sections(text)

        for sec_id, section in enumerate(sections):
            paragraphs = split_paragraphs(section)

            # 🔥 étape 2 : chunks
            chunks = build_chunks(paragraphs)

            for i, chunk in enumerate(chunks):
                record = {
                    "chunk_id": f"{doc_id}_p{page_num}_s{sec_id}_c{i}",
                    "doc_id": doc_id,
                    "title": title,
                    "source_org": source_org,
                    "source_url": source_url,
                    "filename": filename,
                    "page_start": page_num,
                    "category": category,
                    "text": chunk
                }

                out_f.write(json.dumps(record, ensure_ascii=False) + "\n")
                count += 1

    return count


def main():
    files = list(INPUT_DIR.glob("*.jsonl"))

    total = 0

    with open(OUTPUT_FILE, "w", encoding="utf-8") as out_f:
        for f in files:
            try:
                n = process_file(f, out_f)
                print(f"{f.name} → {n} chunks")
                total += n
            except Exception as e:
                print(f"Erreur {f.name}: {e}")

    print(f"\n✅ TOTAL: {total}")


if __name__ == "__main__":
    main()