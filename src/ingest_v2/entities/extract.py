import re
from typing import List

TICKER_RE = re.compile(r"\B\$[A-Z]{2,6}\b")
HANDLE_RE = re.compile(r"@[\w\d_]{2,30}")

def extract_entities(text: str) -> List[str]:
    ents = set()
    for m in TICKER_RE.findall(text):
        ents.add(m.upper())
    for m in HANDLE_RE.findall(text):
        ents.add(m)
    return sorted(ents)[:32]
