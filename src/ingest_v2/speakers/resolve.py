from __future__ import annotations
import concurrent.futures
import json
import logging
import os
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

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
    with concurrent.futures.ThreadPoolExecutor(max_workers=2) as pool:
        f1 = pool.submit(_tier1_names_from_text, meta, raw)
        f2 = pool.submit(_tier2_voice_match, meta, raw, audio_hint_path)
        t1 = _safe_result(f1, fallback={})
        t2 = _safe_result(f2, fallback={})

    merged = _merge_tier1_tier2(t1, t2)
    primary = _guess_primary(raw)
    merged["speaker_primary"] = primary

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
    channel = (meta.get("channel_name") or "").lstrip("@")
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

    # host guess
    if channel and talk_time:
        host_spk = max(talk_time, key=talk_time.get)
        speaker_map[host_spk]["role"] = "Host"
        human_channel = channel.replace("Fndn", "Foundation").replace("_", " ").strip()
        if len(human_channel) > 2:
            speaker_map[host_spk]["name"] = human_channel
            speaker_map[host_spk]["confidence"] = max(0.70, speaker_map[host_spk]["confidence"])

    # apply extracted names
    for (spk, name), c in counts.items():
        prev = speaker_map.get(spk)
        if not prev:
            prev = {"name": f"Speaker {spk.replace('S','')}", "role": "Guest", "confidence": 0.55, "source": "tier1"}
            speaker_map[spk] = prev
        conf = min(0.9, 0.55 + 0.15 * min(3, c))
        if prev.get("role") != "Host" or "Speaker " in prev.get("name", ""):
            prev["name"] = name
        prev["confidence"] = max(prev.get("confidence", 0.0), conf)

    return {"speaker_map": speaker_map}


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
    VOICE_EMBED_BACKEND:
      - 'resemblyzer' / 'local' -> compute local embeddings & match library
      - 'none' (default)        -> skip
      - 'assemblyai'            -> not implemented here
    """
    backend = os.getenv("VOICE_EMBED_BACKEND", "none").lower()
    if backend in ("none", ""):
        return {}
    if backend in ("assemblyai", "aai"):
        logging.warning("[speakers/tier2] 'assemblyai' backend not implemented. Use 'resemblyzer'.")
        return {}
    if backend not in ("resemblyzer", "local"):
        logging.warning("[speakers/tier2] unknown backend=%s; expected 'resemblyzer' or 'none'", backend)
        return {}

    audio_path = _find_audio_for_meta(meta, audio_hint_path)
    if not audio_path or not audio_path.exists():
        logging.info("[speakers/tier2] no audio found for vid=%s; skip tier2", meta.get("video_id"))
        return {}

    voices_dir = Path(os.getenv("VOICE_LIBRARY_DIR", "pipeline_storage_v2/voices"))
    lib_path = voices_dir / "library.json"
    library = _load_voice_library(lib_path)
    if not library:
        logging.info("[speakers/tier2] no voice library at %s; skip tier2", lib_path)
        # We still return empty; auto-enroll may fill it later
        return {}

    # Local embedding (Resemblyzer)
    try:
        from .resemblyzer import embed_speakers_from_audio
    except Exception as e:
        logging.warning("[speakers/tier2] resemblyzer backend not available: %s", e)
        return {}

    emb_by_spk = embed_speakers_from_audio(str(audio_path), raw, meta=meta)

    # Compare each speaker vector with library; pick best if above threshold
    threshold = float(os.getenv("VOICE_MATCH_THRESHOLD", "0.80"))
    speaker_map: Dict[str, Dict[str, Any]] = {}
    for spk, vec in emb_by_spk.items():
        best_name, best_sim = _best_match(vec, library)
        if best_name and best_sim >= threshold:
            speaker_map[spk] = {
                "name": best_name,
                "role": "Guest",  # merged with Tier-1 later
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
    if os.getenv("VOICE_AUTO_ENROLL", "yes").lower() not in ("1", "true", "yes", "y"):
        return
    backend = os.getenv("VOICE_EMBED_BACKEND", "none").lower()
    if backend not in ("resemblyzer", "local"):
        return

    # If Tier-2 already identified the primary with a library match, nothing to do
    primary = merged_out.get("speaker_primary") or "S1"
    t2map = (t2_out or {}).get("speaker_map") or {}
    if primary in t2map and t2map[primary].get("name"):
        return

    # Require minimum talk time
    min_talk = float(os.getenv("VOICE_ENROLL_MIN_TALK_S", "60"))
    if _talk_time_seconds(raw, primary) < min_talk:
        logging.info("[speakers/enroll] primary %s spoke < %.1fs; skip auto-enroll", primary, min_talk)
        return

    # Compute embedding for the primary and write to library
    audio_path = _find_audio_for_meta(meta, audio_hint_path)
    if not audio_path or not audio_path.exists():
        return

    try:
        from .resemblyzer import embed_speakers_from_audio
    except Exception as e:
        logging.warning("[speakers/enroll] resemblyzer unavailable: %s", e)
        return

    emb_by_spk = embed_speakers_from_audio(str(audio_path), raw, meta=meta)
    vec = emb_by_spk.get(primary)
    if not vec:
        logging.info("[speakers/enroll] no embedding vector for primary %s; skip", primary)
        return

    # Decide the name
    forced = os.getenv("VOICE_AUTO_ENROLL_NAME", "").strip()
    if forced:
        name = forced
    else:
        ch = (meta.get("channel_name") or "").lstrip("@").strip()
        name = ch.replace("_", " ").replace("Fndn", "Foundation") if ch else "Host"

    voices_dir = Path(os.getenv("VOICE_LIBRARY_DIR", "pipeline_storage_v2/voices"))
    lib_path = voices_dir / "library.json"
    library = _load_voice_library(lib_path)
    library[name] = vec  # overwrite/refresh if exists
    lib_path.parent.mkdir(parents=True, exist_ok=True)
    lib_path.write_text(json.dumps(library, ensure_ascii=False, indent=2), encoding="utf-8")
    logging.info("[speakers/enroll] auto-enrolled '%s' into %s", name, lib_path)

    # Also update merged map so this run shows the name immediately
    sp = merged_out.setdefault("speaker_map", {}).setdefault(primary, {"role": "Host"})
    sp["name"] = name
    sp["source"] = (sp.get("source") or "") + "|auto-enroll"
    sp["confidence"] = max(0.8, float(sp.get("confidence") or 0.0))

def _write_speaker_map(meta: Dict[str, Any], merged: Dict[str, Any]) -> None:
    outdir = Path(os.getenv("SPEAKER_MAP_DIR", "pipeline_storage_v2/speaker_maps"))
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
    try:
        return fut.result(timeout=30)
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
        if p.exists():
            return json.loads(p.read_text(encoding="utf-8"))
    except Exception as e:
        logging.warning("[speakers/tier2] failed loading voice lib: %s", e)
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
