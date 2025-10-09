from __future__ import annotations
import concurrent.futures
import json
import logging
import os
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple
from src.ingest_v2.speakers.enroll_guards import safe_auto_enroll
from src.ingest_v2.speakers.name_filters import filter_to_people, looks_like_person, normalize_alias
from src.ingest_v2.configs.settings import settings_v2
from src.utils.global_thread_guard import get_global_thread_limiter


# ----------------------------
# Public API
# ----------------------------
def resolve_speakers(
    meta: Dict[str, Any],
    raw: Dict[str, Any],
    audio_hint_path: Optional[Path] = None,
) -> Dict[str, Any]:
    """
    Returns a speaker map + primary guess, merging Tier-1 + Tier-2 run in parallel.
    Also:
      - optional auto-enroll of primary speaker into library.json (if enabled)
      - save per-video speaker_map to SPEAKER_MAP_DIR
    """
    limiter = get_global_thread_limiter()
    with limiter.claim(2, label="speakers-resolve"):
        with concurrent.futures.ThreadPoolExecutor(max_workers=2) as pool:
            f1 = pool.submit(_tier1_names_from_text, meta, raw)
            f2 = pool.submit(_tier2_voice_match, meta, raw, audio_hint_path)
            t1 = _safe_result(f1, fallback={})
            t2 = _safe_result(f2, fallback={})

    merged = _merge_tier1_tier2(t1, t2)
    primary = _guess_primary(raw)
    merged["speaker_primary"] = primary

    # Drop org/brand "guests" from the final map; keep host no matter what
    merged["speaker_map"] = _filter_map_people(
        merged.get("speaker_map") or {}, keep_keys={primary}, channel_name=meta.get("channel_name") or ""
    )

    # ── Auto-enroll the primary speaker if no library match and config allows ──
    try:
        _maybe_auto_enroll_primary(meta, raw, audio_hint_path, merged, t2)
    except Exception as e:
        logging.warning("[speakers] auto-enroll failed (non-fatal): %s", e)

    # ── Persist per-video speaker_map for auditability ──
    try:
        _write_speaker_map(meta, merged)
    except Exception as e:
        logging.warning("[speakers] failed to write speaker_map export: %s", e)

    return merged


# ----------------------------
# Tier 1: transcript/heuristics/metadata
# ----------------------------
def _tier1_names_from_text(meta: Dict[str, Any], raw: Dict[str, Any]) -> Dict[str, Any]:
    """
    Tier-1 speaker hints (no LLM). Uses:
      - talk-time heuristic for host,
      - regex self-introductions in early window,
      - candidate names/handles pulled ONLY from AssemblyAI entities.
    """
    channel_raw = (meta.get("channel_name") or "")
    channel = channel_raw.lstrip("@")
    segs: List[Dict[str, Any]] = raw.get("segments") or []

    # accumulate talking time
    talk_time: Dict[str, float] = {}
    for s in segs:
        spk = s.get("speaker") or "S1"
        start = float(s.get("start") or 0.0)
        end = float(s.get("end") or start)
        talk_time[spk] = talk_time.get(spk, 0.0) + max(0.0, end - start)

    import re
    NAME_RE = re.compile(
        r"\b(?:I am|I'm|This is|with|joined by|today we have)\s+([A-Z][a-z]+(?:\s+[A-Z][a-z]+){0,3})\b"
    )
    counts: Dict[Tuple[str, str], int] = {}
    first_window_s = 5 * 60.0

    for s in segs:
        if not s.get("text"):
            continue
        if float(s.get("start") or 0.0) > first_window_s:
            continue
        for m in NAME_RE.finditer(s["text"]):
            name = m.group(1).strip()
            if not looks_like_person(name, host_names=[channel_raw]):
                continue
            spk = s.get("speaker") or "S1"
            counts[(spk, name)] = counts.get((spk, name), 0) + 1

    # default labels
    speaker_map: Dict[str, Dict[str, Any]] = {}
    for spk in talk_time.keys():
        speaker_map[spk] = {
            "name": f"Speaker {spk.replace('S','')}",
            "role": "Guest",
            "confidence": 0.55,
            "source": "tier1",
        }

    # host guess (most talk time)
    host_spk = None
    if talk_time:
        host_spk = max(talk_time, key=talk_time.get)
        speaker_map[host_spk]["role"] = "Host"

        # Prefer a person/handle as host name; avoid org labels
        if channel_raw.startswith("@") or looks_like_person(channel, host_names=[]):
            speaker_map[host_spk]["name"] = channel_raw or channel
            speaker_map[host_spk]["confidence"] = max(0.70, speaker_map[host_spk]["confidence"])

    # apply extracted self-introduced names
    for (spk, name), c in counts.items():
        prev = speaker_map.get(spk)
        if not prev:
            prev = {"name": f"Speaker {spk.replace('S','')}", "role": "Guest", "confidence": 0.55, "source": "tier1"}
            speaker_map[spk] = prev
        conf = min(0.9, 0.55 + 0.15 * min(3, c))
        # don't overwrite an identified host name
        if prev.get("role") != "Host" or "Speaker " in prev.get("name", ""):
            prev["name"] = normalize_alias(name)
        prev["confidence"] = max(prev.get("confidence", 0.0), conf)

    # AAI entities → candidate people/handles only (no LLM, no title parsing)
    try:
        cands = _people_from_aai_entities(raw, channel_raw)
        if cands and host_spk:
            speaker_map = _assign_candidates_to_speakers(
                speaker_map=speaker_map,
                segs=segs,
                talk_time=talk_time,
                host_spk=host_spk,
                channel_raw=channel_raw,
                candidates=cands,
            )
            logging.info("[speakers/tier1] aai-entities candidates -> %s", cands)
    except Exception as e:
        logging.warning("[speakers/tier1] aai-entities pass failed (non-fatal): %s", e)

    return {"speaker_map": speaker_map}


def _people_from_aai_entities(raw: Dict[str, Any], channel_raw: str) -> list[str]:
    """
    Extract person-like labels from AssemblyAI's entities payload.
    Accepts @handles and entity_type including 'person'.
    """
    ents = (raw or {}).get("entities") or []
    cands: list[str] = []
    for e in ents:
        try:
            txt = (e.get("text") or "").strip()
            typ = (e.get("entity_type") or "").lower()
        except Exception:
            continue
        if not txt:
            continue
        if txt.startswith("@"):
            cands.append(txt)
            continue
        if "person" in typ:
            cands.append(txt)
    # Normalize + drop org-like names; dedupe case-insensitively
    return filter_to_people(cands, host_names=[channel_raw or ""])


# ----------------------------
# Tier 2: voice embeddings (local)
# ----------------------------
def _tier2_voice_match(
    meta: Dict[str, Any],
    raw: Dict[str, Any],
    audio_hint_path: Optional[Path],
) -> Dict[str, Any]:
    """
    Local embedding backend using Resemblyzer (no external API).
    VOICE_EMBED_BACKEND (default 'auto'):
      - 'auto' / '' / 'resemblyzer' / 'local' -> compute local embeddings & match library
      - 'none' / 'off' / 'disable'            -> skip
      - 'assemblyai'                          -> not implemented here
    """
    backend = os.getenv("VOICE_EMBED_BACKEND", "off").lower()
    if backend in ("none", "off", "disable"):
        return {}
    if backend in ("assemblyai", "aai"):
        logging.warning("[speakers/tier2] 'assemblyai' backend not implemented. Use 'resemblyzer' or 'auto'.")
        return {}

    audio_path = _find_audio_for_meta(meta, audio_hint_path)
    if not audio_path or not audio_path.exists():
        logging.info("[speakers/tier2] no audio found for vid=%s; skip tier2", meta.get("video_id"))
        return {}

    voices_dir = Path(settings_v2.VOICE_LIBRARY_DIR)
    lib_path = voices_dir / "library.json"
    library = _load_voice_library(lib_path)

    # Filter library to person-like labels only (prevents org matches like 'Solana', 'Delphi Digital', etc.)
    channel_raw = (meta.get("channel_name") or "")
    people_lib = {k: v for k, v in (library or {}).items() if looks_like_person(k, host_names=[channel_raw])}
    if library and not people_lib:
        logging.info("[speakers/tier2] library had only non-person labels; skipping matches.")
        return {}
    library = people_lib

    # Try local Resemblyzer backend
    try:
        from .resemblyzer_backend import embed_speakers_from_audio
    except Exception as e:
        logging.warning(
            "[speakers/tier2] local backend unavailable (install: resemblyzer, librosa, soundfile, numpy): %s", e
        )
        return {}

    try:
        emb_by_spk = embed_speakers_from_audio(str(audio_path), raw, meta=meta)
    except Exception as e:
        logging.warning("[speakers/tier2] embedding failed: %s", e)
        return {}

    if not library:
        logging.info("[speakers/tier2] no voice library at %s; skip matching (auto-enroll may populate it).", lib_path)
        return {}

    threshold = float(os.getenv("VOICE_MATCH_THRESHOLD", "0.80"))
    speaker_map: Dict[str, Dict[str, Any]] = {}
    for spk, vec in emb_by_spk.items():
        best_name, best_sim = _best_match(vec, library)
        if best_name and best_sim >= threshold:
            speaker_map[spk] = {
                "name": best_name,
                "role": "Guest",
                "confidence": float(min(0.99, max(0.0, best_sim))),
                "source": "tier2",
            }

    return {"speaker_map": speaker_map}


# ----------------------------
# Auto-enroll + exports
# ----------------------------
def _maybe_auto_enroll_primary(
    meta: Dict[str, Any],
    raw: Dict[str, Any],
    audio_hint_path: Optional[Path],
    merged_out: Dict[str, Any],
    t2_out: Dict[str, Any],
) -> None:
    # gate by env
    if os.getenv("VOICE_AUTO_ENROLL", "yes").lower() not in ("1", "true", "yes", "y"):
        return
    backend = os.getenv("VOICE_EMBED_BACKEND", "off").lower()
    if backend in ("none", "off", "disable"):
        return

    primary = merged_out.get("speaker_primary") or "S1"
    t2map = (t2_out or {}).get("speaker_map") or {}
    if primary in t2map and t2map[primary].get("name"):
        # already matched in library; no enrollment needed
        return

    min_talk = float(os.getenv("VOICE_ENROLL_MIN_TALK_S", "60"))
    if _talk_time_seconds(raw, primary) < min_talk:
        logging.info("[speakers/enroll] primary %s spoke < %.1fs; skip auto-enroll", primary, min_talk)
        return

    audio_path = _find_audio_for_meta(meta, audio_hint_path)
    if not audio_path or not audio_path.exists():
        return

    try:
        from .resemblyzer_backend import embed_speakers_from_audio
    except Exception as e:
        logging.warning("[speakers/enroll] resemblyzer unavailable: %s", e)
        return

    emb_by_spk = embed_speakers_from_audio(str(audio_path), raw, meta=meta)
    vec = emb_by_spk.get(primary)
    if not vec:
        logging.info("[speakers/enroll] no embedding vector for primary %s; skip", primary)
        return

    # Decide on a label to enroll that looks like a person or a handle (@user)
    forced = os.getenv("VOICE_AUTO_ENROLL_NAME", "").strip()
    spmap = (merged_out or {}).get("speaker_map") or {}
    primary_name = (spmap.get(primary, {}) or {}).get("name") or ""
    channel_raw = (meta.get("channel_name") or "")

    candidates = []
    if forced:
        candidates.append(forced)
    if primary_name:
        candidates.append(primary_name)
    if channel_raw.startswith("@"):
        candidates.append(channel_raw)

    label = None
    for cand in candidates:
        if looks_like_person(cand, host_names=[]):
            label = cand
            break
    if not label:
        # avoid enrolling org-y labels like 'Solana', 'Delphi Digital'
        logging.info("[speakers/enroll] no person-like label for primary; skipping auto-enroll.")
        return

    voices_dir = Path(settings_v2.VOICE_LIBRARY_DIR)
    lib_path = voices_dir / "library.json"

    # Atomic, locked, person-only write
    enrolled = safe_auto_enroll(str(lib_path), label=label, embedding=vec, host_names=())
    if enrolled:
        logging.info("[speakers/enroll] auto-enrolled '%s' into %s", label, lib_path)
        sp = merged_out.setdefault("speaker_map", {}).setdefault(primary, {"role": "Host"})
        sp["name"] = label
        sp["source"] = (sp.get("source") or "") + "|auto-enroll"
        sp["confidence"] = max(0.8, float(sp.get("confidence") or 0.0))

def _write_speaker_map(meta: Dict[str, Any], merged: Dict[str, Any]) -> None:
    outdir = Path(settings_v2.SPEAKER_MAP_DIR)
    outdir.mkdir(parents=True, exist_ok=True)
    vid = meta.get("video_id") or "unknown"
    out = {
        "video_id": vid,
        "title": meta.get("title"),
        "channel_name": meta.get("channel_name"),
        "speaker_primary": merged.get("speaker_primary"),
        "speaker_map": merged.get("speaker_map") or {},
    }
    (outdir / f"{vid}.json").write_text(json.dumps(out, ensure_ascii=False, indent=2), encoding="utf-8")

def _talk_time_seconds(raw: Dict[str, Any], speaker: str) -> float:
    total = 0.0
    for s in (raw.get("segments") or []):
        spk = s.get("speaker") or "S1"
        if spk != speaker:
            continue
        st = float(s.get("start") or 0.0)
        et = float(s.get("end") or st)
        total += max(0.0, et - st)
    return total


# ----------------------------
# Helpers
# ----------------------------
def _safe_result(fut, fallback):
    timeout_s = float(os.getenv("SPEAKERS_STAGE_TIMEOUT_S", "120"))
    try:
        return fut.result(timeout=timeout_s)
    except Exception as e:
        logging.warning("[speakers] parallel stage failed: %s", e)
        return fallback

def _merge_tier1_tier2(t1: Dict[str, Any], t2: Dict[str, Any]) -> Dict[str, Any]:
    out = {"speaker_map": {}}
    m1 = (t1 or {}).get("speaker_map", {})
    m2 = (t2 or {}).get("speaker_map", {})
    speakers = set(m1.keys()) | set(m2.keys())
    for spk in speakers:
        a, b = m1.get(spk), m2.get(spk)
        if a and b:
            final = dict(a)
            if b.get("confidence", 0) > a.get("confidence", 0):
                final["name"] = b.get("name", final.get("name"))
                final["confidence"] = b.get("confidence")
                final["source"] = b.get("source", final.get("source"))
            if b.get("role") == "Host":
                final["role"] = "Host"
            out["speaker_map"][spk] = final
        elif b:
            out["speaker_map"][spk] = dict(b)
        elif a:
            out["speaker_map"][spk] = dict(a)
    return out

def _guess_primary(raw: Dict[str, Any]) -> str:
    segs: List[Dict[str, Any]] = raw.get("segments") or []
    talk: Dict[str, float] = {}
    for s in segs:
        spk = s.get("speaker") or "S1"
        dur = max(0.0, float(s.get("end") or 0.0) - float(s.get("start") or 0.0))
        talk[spk] = talk.get(spk, 0.0) + dur
    return max(talk, key=talk.get) if talk else "S1"

def _find_audio_for_meta(meta: Dict[str, Any], audio_hint_path: Optional[Path]) -> Optional[Path]:
    return audio_hint_path if (audio_hint_path and audio_hint_path.exists()) else None

def _load_voice_library(p: Path) -> Dict[str, List[float]]:
    try:
        if not p.exists():
            return {}
        raw = p.read_text(encoding="utf-8")
        if not raw.strip():
            return {}
        try:
            obj = json.loads(raw)
        except json.JSONDecodeError:
            # In case of a concurrent writer finishing a rename at the same time:
            # try one more time after a tiny delay
            try:
                import time as _t; _t.sleep(0.05)
                obj = json.loads(p.read_text(encoding="utf-8"))
            except Exception:
                logging.warning("[speakers/tier2] voice lib invalid JSON at %s; treating as empty", p)
                return {}
        if isinstance(obj, dict):
            # keep only {str: List[number]}
            out = {}
            for k, v in obj.items():
                if isinstance(k, str) and isinstance(v, list) and all(isinstance(x, (int, float)) for x in v):
                    out[k] = v
            return out
    except Exception as e:
        logging.warning("[speakers/tier2] failed loading voice lib %s: %s", p, e)
    return {}

def _best_match(vec: List[float], library: Dict[str, List[float]]) -> Tuple[Optional[str], float]:
    import math
    def cosine(a, b):
        dot = sum(x*y for x,y in zip(a,b))
        na = math.sqrt(sum(x*x for x in a))
        nb = math.sqrt(sum(x*x for x in b))
        if na == 0 or nb == 0:
            return 0.0
        return dot / (na*nb)
    best_name, best_sim = None, 0.0
    for name, ref in library.items():
        sim = cosine(vec, ref)
        if sim > best_sim:
            best_name, best_sim = name, sim
    return best_name, float(best_sim)


def _atomic_write_json(path: Path, obj: Any) -> None:
    tmp = path.with_suffix(path.suffix + ".tmp")
    tmp.write_text(json.dumps(obj, ensure_ascii=False, indent=2), encoding="utf-8")
    os.replace(tmp, path)


def _filter_map_people(spmap: Dict[str, Any], keep_keys: set[str], channel_name: str = "") -> Dict[str, Any]:
    """Keep speakers in keep_keys (e.g., primary) + any entries whose 'name' looks like a person."""
    out = {}
    hosts = [channel_name] if channel_name else []
    for k, info in (spmap or {}).items():
        if k in keep_keys:
            out[k] = info
            continue
        nm = (info.get("name") or "").strip()
        if nm and looks_like_person(nm, host_names=hosts):
            out[k] = info
    return out


import re
from typing import Iterable

def _people_from_title(title: str, channel_raw: str) -> list[str]:
    """
    Extract @handles and 2-4 token TitleCase names from the title.
    Filter to person-like (orgs dropped), normalize + dedupe.
    """
    t = (title or "").strip()
    if not t:
        return []

    # 1) @handles
    handles = re.findall(r"@[A-Za-z0-9_]{3,}", t)

    # 2) Split by common guest separators, then scan for TitleCase person names
    # “with”, “w/”, “feat/ft”, “x”, “&”, “,”, "-", "—", "|"
    sep_norm = re.sub(r"(?:\bw\/\b|\bwith\b|\bfeat\.?\b|\bfeaturing\b|\bft\.?\b|\bx\b)", ",", t, flags=re.I)
    chunks = re.split(r"[,&|\-\u2014]+", sep_norm)  # -, —, &, |, comma

    name_like = []
    NAME_SEQ = re.compile(r"\b([A-Z][a-z]+(?:\s+[A-Z][a-z]+){1,3})\b")
    for ch in chunks:
        for m in NAME_SEQ.finditer(ch):
            name_like.append(m.group(1).strip())

    cands = handles + name_like
    return filter_to_people(cands, host_names=[channel_raw or ""])

def _early_text_preview(segs: list[dict], cap_s: float = 5 * 60.0, char_cap: int = 1600) -> str:
    acc = []
    total = 0
    for s in segs:
        st = float(s.get("start") or 0.0)
        if st > cap_s:
            break
        txt = (s.get("text") or "").strip()
        if not txt:
            continue
        if total + len(txt) > char_cap:
            acc.append(txt[: max(0, char_cap - total)])
            break
        acc.append(txt)
        total += len(txt)
    return " ".join(acc)

def _assign_candidates_to_speakers(
    speaker_map: dict[str, dict],
    segs: list[dict],
    talk_time: dict[str, float],
    host_spk: str,
    channel_raw: str,
    candidates: Iterable[str],
) -> dict[str, dict]:
    """
    Conservatively assign title/LLM candidate names to unlabeled non-host speakers.
    - Prefer not to overwrite Tier-2 matches or non-default names.
    - If a candidate appears in early text spoken by some S*, favor that S*.
    - Else map in talk-time order (2nd-most talker gets first remaining candidate), low-ish confidence.
    """
    if not candidates:
        return speaker_map

    # Build quick index of which S* mentioned which candidate in the early window
    early = [s for s in segs if float(s.get("start") or 0.0) <= 6 * 60.0]
    mention_by_spk: dict[str, set[str]] = {}
    for s in early:
        txt = (s.get("text") or "").lower()
        spk = s.get("speaker") or "S1"
        for c in candidates:
            c_low = normalize_alias(c).lower()
            # match either full handle or a simple last-name fallback for TitleCase names
            ok = False
            if c_low.startswith("@") and c_low in txt:
                ok = True
            else:
                parts = [p for p in c.split() if p and p[0].isupper()]
                if parts:
                    last = parts[-1].lower()
                    if last and re.search(rf"\b{re.escape(last)}\b", txt):
                        ok = True
            if ok:
                mention_by_spk.setdefault(spk, set()).add(normalize_alias(c))

    # Non-host speakers ordered by talk-time (desc)
    non_host = [k for k in sorted(talk_time, key=talk_time.get, reverse=True) if k != host_spk]

    # Unlabeled slots we feel okay to overwrite: default "Speaker N" with source tier1 only
    def _is_default_slot(info: dict) -> bool:
        nm = (info.get("name") or "")
        src = (info.get("source") or "")
        return nm.startswith("Speaker ") and ("tier2" not in src)

    # Try precise assignment by mention; then fallback by talk-time
    assigned: set[str] = set()
    for spk in non_host:
        info = speaker_map.get(spk) or {}
        if not _is_default_slot(info):
            continue
        # pick a candidate that this spk mentioned
        options = list((mention_by_spk.get(spk) or set()) - assigned)
        chosen = options[0] if options else None
        if chosen:
            info["name"] = chosen
            info["source"] = (info.get("source") or "") + "|title"
            info["confidence"] = max(float(info.get("confidence") or 0.55), 0.70)
            speaker_map[spk] = info
            assigned.add(chosen)

    # Fallback pass: map remaining candidates by talk-time
    remaining = [normalize_alias(c) for c in candidates if normalize_alias(c) not in assigned]
    slots = [spk for spk in non_host if _is_default_slot(speaker_map.get(spk) or {})]
    for spk, cand in zip(slots, remaining):
        info = speaker_map.get(spk) or {"role": "Guest", "source": "tier1", "confidence": 0.55}
        info["name"] = cand
        info["source"] = (info.get("source") or "") + "|title"
        info["confidence"] = max(float(info.get("confidence") or 0.55), 0.62)
        speaker_map[spk] = info
        assigned.add(cand)

    return speaker_map

def _llm_people_hints(meta: dict, segs: list[dict]) -> list[str]:
    """
    Optional: ask a tiny model (gpt-4o-mini) for likely human names/handles.
    Returns a list of canonicalized strings (e.g., '@handle' or 'First Last').
    Controlled via SPEAKERS_LLM_HINTS=true.
    """
    if os.getenv("SPEAKERS_LLM_HINTS", "false").lower() not in ("1", "true", "yes", "y"):
        return []

    try:
        from openai import OpenAI
    except Exception:
        return []

    title = meta.get("title") or ""
    channel = meta.get("channel_name") or ""
    preview = _early_text_preview(segs)

    prompt_user = {
        "title": title,
        "channel_name": channel,
        "early_transcript": preview[:1600],
        "want": "Return JSON with a short list of likely human speakers (host + guests). Use @handles when obvious; otherwise 'First Last'. No org/brand names. 6 items max.",
        "format": {"type": "object", "properties": {"people": {"type": "array", "items": {"type": "string"}}}, "required": ["people"], "additionalProperties": False},
    }
    sys = (
        "You normalize names for a podcast/video. Return STRICT JSON only. "
        "Keep only humans (@handles or First Last). Drop orgs/brands."
    )
    model = os.getenv("SPEAKERS_LLM_MODEL", "gpt-4o-mini")

    try:
        client = OpenAI()
        resp = client.chat.completions.create(
            model=model,
            temperature=0.0,
            response_format={"type": "json_object"},
            messages=[
                {"role": "system", "content": sys},
                {"role": "user", "content": json.dumps(prompt_user, ensure_ascii=False)},
            ],
            timeout=30,
        )
        raw = resp.choices[0].message.content or "{}"
        data = json.loads(raw)
        cands = data.get("people") or []
    except Exception as e:
        logging.info("[speakers/llm] hint call failed: %s", e)
        return []

    return filter_to_people([normalize_alias(x) for x in cands], host_names=[channel or ""])
