import json
from pathlib import Path

INPUT = Path("data/processed/final_chunks.jsonl")
OUTPUT = Path("data/processed/final_chunks_cleaned.jsonl")

MAX_CHARS = 1200
MIN_CHARS = 80
OVERLAP = 150


def split_large(text):
    chunks = []
    start = 0

    while start < len(text):
        end = min(len(text), start + MAX_CHARS)
        chunk = text[start:end].strip()

        if chunk:
            chunks.append(chunk)

        if end == len(text):
            break

        start = end - OVERLAP

    return chunks


total = 0
kept = 0

with open(INPUT, "r", encoding="utf-8") as f, open(OUTPUT, "w", encoding="utf-8") as out:
    for line in f:
        if not line.strip():
            continue

        total += 1
        data = json.loads(line)
        text = data.get("text", "").strip()

        # ❌ supprimer petits chunks
        if len(text) < MIN_CHARS:
            continue

        # ✅ recouper gros chunks
        if len(text) > MAX_CHARS:
            parts = split_large(text)

            for i, p in enumerate(parts):
                new = data.copy()
                new["chunk_id"] = f"{data['chunk_id']}_fix{i}"
                new["text"] = p

                out.write(json.dumps(new, ensure_ascii=False) + "\n")
                kept += 1
        else:
            out.write(json.dumps(data, ensure_ascii=False) + "\n")
            kept += 1


print("===== CLEAN DONE =====")
print(f"Original: {total}")
print(f"Final: {kept}")
print(f"Saved: {OUTPUT}")