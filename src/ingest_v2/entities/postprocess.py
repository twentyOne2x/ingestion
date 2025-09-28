# src/ingest_v2/entities/postprocess.py
import re

STOP_TYPES = {"money_amount","date","time","number","quantity","duration","percent"}
STOP_WORDS = {"dollar","dollars","percent","today","yesterday","tomorrow"}

ORG_TOKENS = {"inc","inc.","llc","ltd","co","co.","company","corp","corp.","industries",
              "foundation","protocol","dao","labs","ventures","capital","research","network",
              "university","bank","studio","studios","media","markets","exchange","protocols"}

def _titlecase(s: str) -> str:
    return " ".join(w.capitalize() if w.isalpha() else w for w in s.split())

def _canon_org(s: str) -> str:
    # collapse variants like "Ford", "Ford Industries" → "Ford"
    parts = [p for p in re.split(r"\s+", s.strip()) if p]
    if len(parts) >= 2 and parts[-1].lower().rstrip(".,") in ORG_TOKENS:
        parts = parts[:-1]
    return _titlecase(" ".join(parts)) or s.strip()

def postprocess_aai_entities(aai_entities, max_n=64):
    out = set()
    for e in aai_entities or []:
        txt = (e.get("text") or "").strip()
        typ = (e.get("entity_type") or "").lower()
        if not txt:
            continue
        if typ in STOP_TYPES:
            continue
        low = txt.lower().strip(".,")
        if low in STOP_WORDS:
            continue

        if txt.startswith("$"):         # $SOL
            out.add(txt.upper())
            continue
        if txt.startswith("@"):         # @anatoly
            out.add(txt.lower())
            continue

        # keep likely org/protocol names (2–4 TitleCase words or ends with org token)
        words = txt.split()
        if (1 < len(words) <= 6 and sum(w[:1].isupper() for w in words) >= 2) or \
           (words and words[-1].lower().rstrip(".,") in ORG_TOKENS):
            out.add(_canon_org(txt))
    return sorted(out)[:max_n]
