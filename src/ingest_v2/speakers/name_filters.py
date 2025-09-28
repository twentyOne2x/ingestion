# src/ingest_v2/speakers/name_filters.py
from __future__ import annotations

import re
from typing import Iterable, List

# ──────────────────────────────────────────────────────────────────────────────
# Config toggles
# ──────────────────────────────────────────────────────────────────────────────

# Mononyms cause many false positives in crypto/media corpora ("Zora", "Doodles",
# "Ethereum", "Cloud", "Noise", "Delta", etc.). Keep OFF unless you curate.
ALLOW_MONONYMS = False

# If you later enable mononyms, keep this short and curated to *real* first names
# you see in your channels.
CURATED_MONONYMS = {
    "jose", "jordan", "tom", "john", "harry", "jeremy", "lars", "craig", "kevin",
    "dylan", "michael", "alex", "mustafa", "akshat", "chang", "toli", "nor",
    "hermes", "karan", "ejaz", "cheyenne", "rosin",
}

# ──────────────────────────────────────────────────────────────────────────────
# Normalization helpers
# ──────────────────────────────────────────────────────────────────────────────

_WS = re.compile(r"\s+")
# Obvious garbage / non-person signals
_BAD_TOKENS = re.compile(
    r"(?:\d|\.com|\.xyz|http|www|/|\\|\bhost\b|\bspeaker\s*[a-d]\b)", re.I
)

# Strip a trailing possessive (e.g., "nothreadguy's")
_POSSESSIVE = re.compile(r"^[\s\S]*?\b('s|’s)$", re.I)


def _norm(s: str | None) -> str:
    s = (s or "").strip()
    if not s:
        return ""
    # remove trailing possessive "'s"
    if s.endswith("'s") or s.endswith("’s"):
        s = s[:-2]
    return _WS.sub(" ", s).strip()


def _is_title_word(w: str) -> bool:
    # Simple "Name-like" token: Michael, Alex, Bennington
    return w and w[0].isupper() and w[1:].islower() and w.isalpha()


def _all_title_words(parts: List[str]) -> bool:
    return all(_is_title_word(p) for p in parts if p.isalpha())

# ──────────────────────────────────────────────────────────────────────────────
# Org / brand hints
# (lowercased; we match by substring for phrases and by token for tokens)
# ──────────────────────────────────────────────────────────────────────────────

ORG_PHRASE_HINTS = {
    # sectors / descriptors
    "capital", "ventures", "labs", "foundation", "protocol", "protocols", "dao",
    "markets", "network", "networks", "pay", "exchange", "exchanges", "research",
    "digital", "finance", "fund", "funds", "studio", "studios", "collective",
    "guild", "gov", "capital markets", "university", "college", "institute",
    "bank", "media",
    # brands / teams common in your logs
    "solana", "hyper liquid", "hyperliquid", "gnosis", "google", "coinbase",
    "kraken", "delphi digital", "bankless", "attention capital markets", "legion",
    "pump", "pumpfun", "pump dot fun", "on chain", "global crossing",
    "ethereum", "zora", "doodles",
}

ORG_TOKEN_HINTS = {
    "dao", "labs", "protocol", "research", "network", "networks", "capital",
    "ventures", "foundation", "media", "bank", "college", "university",
    "institute", "studio", "studios", "exchange", "exchanges", "markets",
    "finance", "digital", "chain", "global", "crossing",
    # common mononym-y brands/terms we don't want
    "ethereum", "zora", "doodles", "cookie", "cloud", "noise", "delta", "maybe",
}

EXACT_BANNED = {
    "everyone", "speaker a", "speaker b", "speaker c", "speaker d", "host"
}

# If a handle contains any of these substrings (after removing '@'), treat it as org-ish.
HANDLE_ORG_SUBSTRS = {
    "labs", "dao", "protocol", "foundation", "fund", "ventures", "capital",
    "digital", "media", "research", "exchange", "markets", "network", "networks",
    "college", "university", "institute", "studio", "studios", "bank",
    # well-known brands seen in the data
    "solana", "gnosis", "coinbase", "kraken", "delphi", "bankless",
}

# ──────────────────────────────────────────────────────────────────────────────
# Alias normalization
# ──────────────────────────────────────────────────────────────────────────────

_ALIAS_PATTERNS: list[tuple[re.Pattern[str], str]] = [
    # notthreadguy variants
    (
        re.compile(
            r"^(?:@)?(?:(?:no?t\s*thread|nothread)\s*guy|thread\s*guy|threadguy)$",
            re.I,
        ),
        "@notthreadguy",
    ),
    # AccelerateWithMert variants
    (
        re.compile(
            r"^(?:@)?(?:accelerate\s*with\s*mert|acceleratewithmert)$",
            re.I,
        ),
        "@AccelerateWithMert",
    ),
]


def normalize_alias(label: str) -> str:
    """
    Collapse common alias/format variants into a canonical form.
    Currently normalizes Threadguy/Thread Guy/nothreadguy → @notthreadguy
    and AccelerateWithMert → @AccelerateWithMert.
    """
    n = _norm(label)
    low = n.lower()
    for pat, repl in _ALIAS_PATTERNS:
        if pat.match(low):
            return repl
    return n

# ──────────────────────────────────────────────────────────────────────────────
# Core checks
# ──────────────────────────────────────────────────────────────────────────────

def _looks_like_handle(n: str) -> bool:
    """Accept @handles that don't smell like orgs."""
    if not n.startswith("@"):
        return False
    base = n[1:].lower()
    if not base or len(base) < 3:
        return False
    # reject obvious org handles
    if any(sub in base for sub in HANDLE_ORG_SUBSTRS):
        return False
    return True


def _contains_org_hints(n: str) -> bool:
    low = n.lower()
    if any(p in low for p in ORG_PHRASE_HINTS):
        return True
    # token check
    parts = low.split()
    if any(tok in ORG_TOKEN_HINTS for tok in parts):
        return True
    return False


def looks_like_person(
    name: str,
    host_names: Iterable[str] = (),
    allow_mononyms: bool | None = None,
) -> bool:
    """
    Lightweight person-likeness heuristic:

      - reject if equals/contains org hints or garbage tokens
      - accept @handles unless they look like orgs
      - accept 2+ TitleCase tokens (e.g., "Tom Shaughnessy")
      - (optional) accept curated mononyms only
      - reject if equal to any host/channel name (after alias normalization)
    """
    if allow_mononyms is None:
        allow_mononyms = ALLOW_MONONYMS

    n = normalize_alias(name)
    n = _norm(n)
    if not n or len(n) < 2:
        return False

    # host equality check (normalize hosts + aliases)
    host_lc = {normalize_alias(h).lower() for h in (host_names or [])}
    if n.lower() in host_lc:
        return False

    # junk / banned labels
    if _BAD_TOKENS.search(n):
        return False
    if n.lower() in EXACT_BANNED:
        return False
    if _contains_org_hints(n):
        return False

    # handles first
    if _looks_like_handle(n):
        return True

    # multi-token Title-Case names
    parts = n.split()
    if len(parts) >= 2 and _all_title_words(parts):
        return True

    # mononyms (extremely conservative; OFF by default)
    if len(parts) == 1:
        word = parts[0]
        if allow_mononyms:
            return word.lower() in CURATED_MONONYMS and _is_title_word(word)
        return False

    return False


def filter_to_people(candidates: Iterable[str], host_names: Iterable[str] = ()) -> List[str]:
    """
    Normalize, alias-map, dedupe (case-insensitively), and keep only person-like labels.
    """
    seen = set()
    out: List[str] = []
    for raw in candidates or []:
        n = normalize_alias(raw)
        n = _norm(n)
        if not n:
            continue
        low = n.lower()
        if low in seen:
            continue
        if looks_like_person(n, host_names=host_names):
            out.append(n)
            seen.add(low)
    return out
