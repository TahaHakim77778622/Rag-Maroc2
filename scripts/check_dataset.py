import json
from pathlib import Path

FILE = Path("data/processed/final_chunks_cleaned.jsonl")
total = 0
empty = 0
too_small = 0
too_big = 0

lengths = []

with open(FILE, "r", encoding="utf-8") as f:
    for line in f:
        if not line.strip():
            continue

        total += 1
        data = json.loads(line)
        text = data.get("text", "")

        l = len(text)
        lengths.append(l)

        if l == 0:
            empty += 1
        elif l < 100:
            too_small += 1
        elif l > 1500:
            too_big += 1

print("====== DATASET CHECK ======")
print(f"Total chunks: {total}")
print(f"Empty: {empty}")
print(f"Too small (<100): {too_small}")
print(f"Too big (>1500): {too_big}")

if lengths:
    print(f"Min length: {min(lengths)}")
    print(f"Max length: {max(lengths)}")
    print(f"Avg length: {sum(lengths)//len(lengths)}")