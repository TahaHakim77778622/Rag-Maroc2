import json
import re
from pathlib import Path

IN_DIR = Path("data/processed/pages_jsonl")
OUT_DIR = Path("data/processed/cleaned_pages_jsonl")
OUT_DIR.mkdir(parents=True, exist_ok=True)

def clean_text(text: str) -> str:
    # remplace retours ligne multiples
    text = re.sub(r"\s+", " ", text)

    # supprime espaces inutiles
    text = text.strip()

    return text

def main():
    files = sorted(IN_DIR.glob("*.jsonl"))

    if not files:
        print("Aucun fichier pages_jsonl trouvé.")
        return

    for fp in files:
        out_path = OUT_DIR / fp.name

        with open(fp, "r", encoding="utf-8") as fin, open(out_path, "w", encoding="utf-8") as fout:
            for line in fin:
                rec = json.loads(line)
                rec["text"] = clean_text(rec.get("text", ""))

                if rec["text"]:
                    fout.write(json.dumps(rec, ensure_ascii=False) + "\n")

        print(f"Nettoyé : {fp.name}")

    print("Nettoyage terminé.")

if __name__ == "__main__":
    main()