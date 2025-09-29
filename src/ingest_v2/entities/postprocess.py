# src/ingest_v2/entities/postprocess.py
import os, re, json, unicodedata
from collections import Counter
from typing import Any, Dict, List, Optional, Tuple

# NOTE: we will special-case 'money_amount' and 'quantity' below.
STOP_TYPES = {"date","time","number","duration","percent"}  # removed 'quantity'
STOP_WORDS = {"dollar","dollars","percent","today","yesterday","tomorrow",
              "coin","coins","usd"}  # drop generic money words

ORG_TOKENS = {
    "inc","inc.","llc","ltd","co","co.","company","corp","corp.","industries",
    "foundation","protocol","dao","labs","ventures","capital","research","network",
    "university","bank","studio","studios","media","markets","exchange","protocols",
    "finance"
}

# ── helpers ─────────────────────────────────────────────────────────────────────
def _titlecase(s: str) -> str:
    return " ".join(w.capitalize() if w.isalpha() else w for w in (s or "").split())

def _canon_org(s: str) -> str:
    parts = [p for p in re.split(r"\s+", (s or "").strip()) if p]
    if len(parts) >= 2 and parts[-1].lower().rstrip(".,") in ORG_TOKENS:
        parts = parts[:-1]
    return _titlecase(" ".join(parts)) or (s or "").strip()

def _norm_key(s: str) -> str:
    s = unicodedata.normalize("NFKD", (s or "")).encode("ascii", "ignore").decode("ascii")
    s = s.lower()
    s = re.sub(r"[^\w\s.-]", "", s)
    s = re.sub(r"\s+", " ", s).strip()
    return s

def _is_titlecase_single(s: str) -> bool:
    return bool(s) and s[:1].isupper() and s[1:].islower() and s.isalpha()

def _lev(a: str, b: str) -> int:
    if a == b: return 0
    la, lb = len(a), len(b)
    if la == 0 or lb == 0: return max(la, lb)
    dp = list(range(lb + 1))
    for i in range(1, la + 1):
        prev = dp[0]; dp[0] = i; ca = a[i-1]
        for j in range(1, lb + 1):
            cur = dp[j]; cb = b[j-1]
            dp[j] = min(dp[j] + 1, dp[j-1] + 1, prev + (0 if ca == cb else 1))
            prev = cur
    return dp[-1]

# ── aliases (extend via ENTITIES_ALIAS_PATH or aliases.json next to this file) ──
DEFAULT_ALIASES = {
    # orgs / products
    "masari": "Messari",
    "messari": "Messari",

    "hyperliquid": "Hyper Liquid",
    "hyper liquid": "Hyper Liquid",

    "kamino": "Kamino",
    "kamino finance": "Kamino Finance",
    "camino": "Kamino",  # common ASR slip

    # Hypernative(.io) vs Hibernative slips
    "hibernative": "Hypernative",
    "hibernative io": "Hypernative.io",
    "hibernative.io": "Hypernative.io",
    "hypernative io": "Hypernative.io",
    "hypernative.io": "Hypernative.io",
    "hybernative": "Hypernative",
    "hibernative-io": "Hypernative.io",
    "hibernativeio": "Hypernative.io",

    # people
    "kyle samani": "Kyle Samani",
    "kyle somani": "Kyle Samani",
    "kyle simone": "Kyle Samani",

    # tokens / assets (names map to canonical names; tickers handled below)
    "btc": "BTC", "eth": "ETH", "sol": "SOL", "bnb": "BNB",
    "bitcoin": "Bitcoin", "ethereum": "Ethereum", "solana": "Solana",
    "salana": "Solana", "soul": "Solana",

    # user-provided
    "takabu": "Tokabu",
}

def _load_aliases() -> Dict[str, str]:
    aliases = {k: v for k, v in DEFAULT_ALIASES.items()}
    here = os.path.dirname(__file__)
    default_path = os.path.join(here, "aliases.json")
    path = os.getenv("ENTITIES_ALIAS_PATH", default_path)
    try:
        if os.path.exists(path):
            with open(path, "r", encoding="utf-8") as f:
                user_map = json.load(f)
            for k, v in (user_map or {}).items():
                if isinstance(k, str) and isinstance(v, str):
                    aliases[_norm_key(k)] = v.strip()
    except Exception:
        pass
    return aliases

_ALIASES = _load_aliases()
_ALIAS_KEYS = list(_ALIASES.keys())

def _alias_lookup(txt: str) -> Optional[str]:
    k = _norm_key(txt)
    if k in _ALIASES: return _ALIASES[k]
    k_nospace = k.replace(" ", "")
    if k_nospace in _ALIASES: return _ALIASES[k_nospace]
    # light fuzzy
    best_key, best_d = None, 10**9
    for key in _ALIAS_KEYS:
        d = _lev(k, key)
        if d < best_d:
            best_d, best_key = d, key
        if best_d == 0: break
    if best_key is not None:
        L = max(len(k), len(best_key))
        if best_d <= (1 if L <= 6 else 2):
            return _ALIASES.get(best_key)
    return None

# ── coin names ↔ ticker map ────────────────────────────────────────────────────
COIN_NAME_TO_TICKER = {
    "bitcoin": "BTC",
    "ethereum": "ETH",
    "solana": "SOL",
    "dogecoin": "DOGE",
    "binance coin": "BNB",
    "pepe": "PEPE",
}
# allow single-word fallbacks
for k in ["btc","eth","sol","bnb","doge","usdc","usdt","dai","pepe"]:
    COIN_NAME_TO_TICKER[k] = k.upper()
# frequent ASR slips
COIN_NAME_TO_TICKER["soul"] = "SOL"
COIN_NAME_TO_TICKER["salana"] = "SOL"
COIN_NAME_TO_TICKER["binancecoin"] = "BNB"

NUMBER_WORDS = {
    "one","two","three","four","five","six","seven","eight","nine","ten",
    "eleven","twelve","thirteen","fourteen","fifteen","sixteen","seventeen",
    "eighteen","nineteen","twenty","thirty","forty","fifty","sixty","seventy",
    "eighty","ninety","hundred","thousand","million","billion","trillion"
}

def _maybe_token_from_quantity(txt: str) -> Optional[str]:
    """
    Extract ticker when text looks like a quantity + unit (e.g., "8 SOL", "eight soul", "280 solana").
    We ignore the numeric value and just return the ticker.
    """
    if not txt: return None
    s = txt.lower()
    # normalize separators to spaces
    s = re.sub(r"[^\w.+-]+", " ", s).strip()

    # patterns like "<num> <unit>" or "<num><unit>"
    m = re.search(
        r"\b(?P<num>(?:\d[\d,]*(?:\.\d+)?|"
        + "|".join(re.escape(w) for w in sorted(NUMBER_WORDS, key=len, reverse=True))
        + r"))\s*(?:x|×|-)?\s*(?P<unit>[a-z][a-z0-9.-]{2,20})s?\b",
        s, flags=re.IGNORECASE,
    )
    if not m:
        return None

    unit = m.group("unit")
    unit_norm = _norm_key(unit).replace("-", "").replace(".", "")
    # try exact
    if unit_norm in COIN_NAME_TO_TICKER:
        return COIN_NAME_TO_TICKER[unit_norm]

    # try alias, then coin-name-to-ticker
    ali = _alias_lookup(unit_norm)
    if ali:
        key = _norm_key(ali)
        key_nopunct = key.replace("-", "").replace(".", "")
        if key_nopunct in COIN_NAME_TO_TICKER:
            return COIN_NAME_TO_TICKER[key_nopunct]

    return None

# ── money_amount crypto keeper ──────────────────────────────────────────────────
def _maybe_keep_money_token(txt: str) -> Optional[str]:
    """
    Keep crypto-style money_amounts when not caught by quantity extractor:
      - $TICKER  → upper
      - ALLCAPS 2–6 letters (BTC, SOL, BNB) → upper
      - Coin names → canonical TitleCase (or mapped), but if it’s a known coin name we prefer ticker.
    """
    t = (txt or "").strip()
    if not t: return None
    if t.startswith("$"):
        return t.upper()

    core = t.strip(".,;:!?")

    # ALLCAPS ticker
    if re.fullmatch(r"[A-Z]{2,6}", core):
        return core

    # lowercase ticker
    if re.fullmatch(r"[a-z]{2,6}", core) and core in COIN_NAME_TO_TICKER:
        return COIN_NAME_TO_TICKER[core]

    # coin names (TitleCase or lowercase)
    k = _norm_key(core).replace("-", "").replace(".", " ")
    k_compact = k.replace(" ", "")
    if k in COIN_NAME_TO_TICKER:
        return COIN_NAME_TO_TICKER[k]
    if k_compact in COIN_NAME_TO_TICKER:
        return COIN_NAME_TO_TICKER[k_compact]

    # fallback: alias → then titlecase
    mapped = _alias_lookup(core)
    if mapped:
        km = _norm_key(mapped)
        if km in COIN_NAME_TO_TICKER:
            return COIN_NAME_TO_TICKER[km]
        kmc = km.replace(" ", "")
        if kmc in COIN_NAME_TO_TICKER:
            return COIN_NAME_TO_TICKER[kmc]
        return _titlecase(mapped)

    if _is_titlecase_single(core) and len(core) >= 3:
        return _titlecase(core)

    return None

# ── main ───────────────────────────────────────────────────────────────────────
def postprocess_aai_entities(aai_entities: List[Dict[str, Any]], max_n: int = 64) -> List[str]:
    aai_entities = aai_entities or []

    raw_counts = Counter()
    for e in aai_entities:
        tx = (e.get("text") or "").strip()
        if tx:
            raw_counts[tx.strip(".,").lower()] += 1

    selected: List[Tuple[str, int]] = []

    for idx, e in enumerate(aai_entities):
        txt = (e.get("text") or "").strip()
        typ = (e.get("entity_type") or "").lower()
        if not txt:
            continue

        low = txt.lower().strip(".,")
        if low in STOP_WORDS:
            continue

        # ── amounts & quantities: try to extract token ticker first ──
        if typ in {"money_amount", "quantity"}:
            tkr = _maybe_token_from_quantity(txt)
            if tkr:
                selected.append((tkr, idx))
                continue
            # fallback: keep standalone token mentions for money_amount
            if typ == "money_amount":
                kept = _maybe_keep_money_token(txt)
                if kept:
                    selected.append((kept, idx))
            continue  # done handling for amounts/quantities

        if typ in STOP_TYPES:
            continue

        if txt.startswith("$"):
            selected.append((txt.upper(), idx)); continue
        if txt.startswith("@"):
            selected.append((txt.lower(), idx)); continue

        words = txt.split()

        # single-word proper nouns (Aster)
        if len(words) == 1 and _is_titlecase_single(txt) and typ in {
            "organization","product","event","protocol","project","person_name"
        }:
            if len(txt) >= 5 or raw_counts[low] >= 2:
                selected.append((_titlecase(txt), idx))
                continue

        # multi-word org/protocol names or ends with org token
        if (1 < len(words) <= 6 and sum(w[:1].isupper() for w in words) >= 2) or \
           (words and words[-1].lower().rstrip(".,") in ORG_TOKENS):
            selected.append((_canon_org(txt), idx))
            continue

    # alias + rank
    first_seen: Dict[str, int] = {}
    counts: Counter = Counter()
    out_set = set()

    def _to_canon(s: str) -> str:
        # Try alias; if alias is a coin name, convert to ticker when applicable
        ali = _alias_lookup(s)
        if ali:
            key = _norm_key(ali)
            if key in COIN_NAME_TO_TICKER:
                return COIN_NAME_TO_TICKER[key]
            keyc = key.replace(" ", "")
            if keyc in COIN_NAME_TO_TICKER:
                return COIN_NAME_TO_TICKER[keyc]
            return _canon_org(ali)
        # if looks like a coin name directly
        key = _norm_key(s)
        if key in COIN_NAME_TO_TICKER:
            return COIN_NAME_TO_TICKER[key]
        keyc = key.replace(" ", "")
        if keyc in COIN_NAME_TO_TICKER:
            return COIN_NAME_TO_TICKER[keyc]
        return _canon_org(s)

    for s, i in selected:
        c = _to_canon(s)
        if c not in out_set:
            out_set.add(c)
            first_seen[c] = i
        base = s.strip(".,").lower()
        counts[c] += max(1, raw_counts.get(base, 1))

    ranked = sorted(out_set, key=lambda k: (-counts[k], first_seen.get(k, 10**9), k))
    return ranked[:max_n]
