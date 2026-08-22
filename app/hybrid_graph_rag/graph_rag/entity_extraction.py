"""
Extraction des renvois entre chunks.

Contrairement à BSARD, il faut d'abord CLASSIFIER le renvoi :
seuls les renvois internes (au même document) sont résolubles de façon fiable.
Les visas externes ("Vu la loi n° 43-05 ... ses articles 13-1 et 13-2") pointent
majoritairement vers des textes absents du corpus — on les écarte.
"""
import re

# "articles 13, 15 et 16" / "article 28-1" / "articles 140 à 145"
ARTICLE_MENTION = re.compile(
    r"articles?\s+(\d+(?:-\d+)?(?:\s*(?:,|et|à)\s*\d+(?:-\d+)?)*)",
    re.IGNORECASE,
)
SPLIT_NUMS = re.compile(r"\s*(?:,|et|à)\s*")

# Marqueurs de renvoi interne au même document
INTERNE = re.compile(r"ci-dessus|ci-dessous|pr[ée]c[ée]dent|du pr[ée]sent", re.I)
# Marqueurs de visa / renvoi externe
EXTERNE = re.compile(r"\bvu\b|susvis|pr[ée]cit|loi\s+n|d[ée]cret\s+n|arr[êe]t[ée]\s+n|dahir", re.I)


def normalize_label(raw: str) -> str:
    """Normalise un numéro d'article pour matcher le champ `label` des chunks."""
    if not raw:
        return None
    l = raw.lower().strip()
    l = re.sub(r"^art(icle)?\.?\s*", "", l)
    if "premier" in l or l == "1er":
        return "1"
    m = re.match(r"(\d+(?:-\d+)?)", l)
    return m.group(1) if m else None


def classify(context: str) -> str:
    """interne | externe | ambigu"""
    if INTERNE.search(context):
        return "interne"
    if EXTERNE.search(context):
        return "externe"
    return "ambigu"


def extract_references(text: str):
    """
    Retourne [(numero_article, type), ...] pour un texte de chunk.
    type ∈ {interne, externe, ambigu}
    """
    if not isinstance(text, str) or not text:
        return []

    out = []
    for m in ARTICLE_MENTION.finditer(text):
        ctx = text[max(0, m.start() - 80): m.end() + 60]
        kind = classify(ctx)
        for num in SPLIT_NUMS.split(m.group(1)):
            num = num.strip()
            if num:
                out.append((num, kind))
    return out