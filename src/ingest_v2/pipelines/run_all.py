import argparse
import asyncio
import concurrent.futures
import json
import logging
import math
import os
import re
import time
from pathlib import Path
from typing import Any, Dict, Iterable, List, Optional, Tuple
import hashlib  # add to imports


from src.ingest_v2.configs.settings import settings_v2
from src.ingest_v2.pipelines.build_children import build_children_from_raw
from src.ingest_v2.pipelines.build_parents import run_build_parents
from src.ingest_v2.pipelines.upsert_parents import upsert_parents
from src.ingest_v2.pipelines.upsert_pinecone import upsert_children
from src.ingest_v2.router.cache import load as router_cache_load
from src.ingest_v2.router.cache import save as router_cache_save
from src.ingest_v2.router.enrich_parent import (
    enrich_parent_router_fields_async,
)
from src.ingest_v2.speakers.resolve import resolve_speakers
from src.ingest_v2.transcripts.normalize import normalize_to_sentences
from src.ingest_v2.utils.logging import setup_logger
from src.ingest_v2.utils.progress import progress

from src.Llama_index_sandbox import YOUTUBE_VIDEO_DIRECTORY

from src.ingest_v2.speakers.name_filters import looks_like_person, filter_to_people, normalize_alias

# ────────────────────────────────────────────────────────────────────────────────
# Small helpers
# ────────────────────────────────────────────────────────────────────────────────

# ────────────────────────────────────────────────────────────────────────────────
# Fast JSON + entity cache helpers
# ────────────────────────────────────────────────────────────────────────────────

def _fast_json_load(path: Path) -> Any:
    """Prefer orjson for speed; fall back to stdlib json."""
    data = path.read_bytes()
    try:
        import orjson  # type: ignore
        return orjson.loads(data)
    except Exception:
        return json.loads(data.decode("utf-8"))

def _entities_cache_dir() -> Path:
    # dedicated on-disk cache for cleaned entity lists
    p = Path(os.getenv("ENTITIES_CACHE_DIR", "pipeline_storage_v2/entities_cache"))
    p.mkdir(parents=True, exist_ok=True)
    return p

def _entities_sidecar_path(json_path: Path) -> Path:
    stem = json_path.stem.replace("_diarized_content", "")
    return json_path.with_name(f"{stem}_entities.json")

def _entities_cache_key(json_path: Path, obj: Any) -> str:
    """
    Stable key based on sidecar (if present) or main JSON mtime/size.
    Include FAST flag so caches don’t cross-pollute modes.
    """
    fast_flag = "fast" if os.getenv("ENTITIES_FAST", "0").lower() in ("1","true","yes","y") else "full"
    sidecar = _entities_sidecar_path(json_path)
    if sidecar.exists():
        st = sidecar.stat()
        base = f"sc::{sidecar.as_posix()}::{st.st_mtime_ns}::{st.st_size}::{fast_flag}"
    else:
        st = json_path.stat()
        n = 0
        if isinstance(obj, dict) and isinstance(obj.get("entities"), list):
            n = len(obj["entities"])
        base = f"tl::{json_path.as_posix()}::{st.st_mtime_ns}::{st.st_size}::n{n}::{fast_flag}"
    return hashlib.sha1(base.encode("utf-8")).hexdigest()

def _postprocess_entities_with_cache(raw_entities: List[Dict[str, Any]], cache_key: str) -> List[str]:
    """
    Run postprocess_aai_entities with an on-disk read-through cache.
    """
    cache_file = _entities_cache_dir() / f"{cache_key}.json"
    if cache_file.exists():
        try:
            return _fast_json_load(cache_file)
        except Exception:
            pass

    from src.ingest_v2.entities.postprocess import postprocess_aai_entities
    cleaned = postprocess_aai_entities(raw_entities)
    try:
        cache_file.write_text(json.dumps(cleaned, ensure_ascii=False), encoding="utf-8")
    except Exception:
        pass
    return cleaned


def _snip(s: str, n: int) -> str:
    s = (s or "").strip()
    return s if len(s) <= n else (s[: max(0, n - 1)] + "…")

def _ascii(s: str, max_len: int = 400) -> str:
    s = _snip(s or "", max_len)
    try:
        return s.encode("ascii", "ignore").decode("ascii")
    except Exception:
        return s

def _safe_float(x: Any) -> Optional[float]:
    try:
        if x is None:
            return None
        return float(x)
    except Exception:
        return None

def _ms_to_s(x: Any) -> Optional[float]:
    val = _safe_float(x)
    return None if val is None else (val / 1000.0)

# ────────────────────────────────────────────────────────────────────────────────
# Detect trivial/empty rows
# ────────────────────────────────────────────────────────────────────────────────

def _is_trivial_raw_segment(seg: Dict[str, Any]) -> bool:
    no_text = not (seg.get("text") or "").strip()
    no_words = not seg.get("words")
    start_none = seg.get("start") is None
    end_none = seg.get("end") is None
    return no_text and no_words and start_none and end_none

def _looks_like_single_empty_file(obj: Any) -> bool:
    try:
        if not isinstance(obj, list) or len(obj) != 1:
            return False
        seg = obj[0]
        return (
            isinstance(seg, dict)
            and (seg.get("text") in ("", None))
            and seg.get("start") is None
            and seg.get("end") is None
            and isinstance(seg.get("words"), list)
            and len(seg["words"]) == 0
        )
    except Exception:
        return False

# ────────────────────────────────────────────────────────────────────────────────
# AssemblyAI → v2 raw normalizer
# ────────────────────────────────────────────────────────────────────────────────

def _convert_assemblyai_json_to_raw(obj: Any) -> Dict[str, Any]:
    """
    Normalize AssemblyAI-like payloads into:
        {"segments": [{"start": float|None, "end": float|None, "speaker": str, "text": str, "words": [...]?}, ...]}

    Supports:
      - {"segments": [...]}  (preferred)
      - [{"..."}]            (legacy)
      - {"utterances": [...]} (fallback)
    """
    segs_in = None

    if isinstance(obj, dict):
        if "segments" in obj and isinstance(obj["segments"], list):
            segs_in = obj["segments"]
        elif "utterances" in obj and isinstance(obj["utterances"], list):
            # Map utterance shape → segment shape best-effort
            segs_in = []
            for utt in obj["utterances"]:
                # Common utterance fields: start, end, speaker, text, words
                segs_in.append({
                    "start": utt.get("start"),
                    "end": utt.get("end"),
                    "speaker": utt.get("speaker") or utt.get("speaker_label"),
                    "text": utt.get("text"),
                    "words": utt.get("words") or [],  # often present already
                })
    elif isinstance(obj, list):
        segs_in = obj

    if not isinstance(segs_in, list):
        raise ValueError("Expected top-level list or dict with 'segments'/'utterances'")

    out_segments: List[Dict[str, Any]] = []
    for seg in segs_in:
        if _is_trivial_raw_segment(seg):
            continue

        words_in = seg.get("words")
        out_words = None
        if isinstance(words_in, list) and words_in:
            out_words = []
            for w in words_in:
                wtext = (w.get("text") or "").strip()
                if not wtext:
                    continue
                out_words.append({
                    "text": wtext,
                    "start": _ms_to_s(w.get("start")),
                    "end": _ms_to_s(w.get("end")),
                    "speaker": w.get("speaker") or seg.get("speaker") or "S1",
                })

        start_s = _ms_to_s(seg.get("start"))
        end_s = _ms_to_s(seg.get("end"))
        text = (seg.get("text") or "").strip()

        if start_s is None and end_s is None and not text and not out_words:
            continue

        out_segments.append({
            "start": start_s,
            "end": end_s,
            "speaker": seg.get("speaker") or seg.get("speaker_label") or "S1",
            "text": text,
            **({"words": out_words} if out_words is not None else {}),
        })

    return {"segments": out_segments}

# ────────────────────────────────────────────────────────────────────────────────
# Robust YouTube ID extraction
# ────────────────────────────────────────────────────────────────────────────────
# We expect <YYYY-MM-DD>_<VIDEOID>_<TITLE>, and IDs can contain underscores/hyphens.
# We handle: single/double/triple underscores, anywhere in the component; if all else
# fails, we pick the first 11-char token.
_DATE_ID_RE = re.compile(r"(?P<date>\d{4}-\d{2}-\d{2})_(?P<id>[A-Za-z0-9_-]{11})_")
_MULTI_US_RE = re.compile(r"_{1,3}([A-Za-z0-9_-]{11})_")
_TOKEN_11_RE = re.compile(r"[A-Za-z0-9_-]{11}")

def _extract_video_id_from_path(path: Path) -> Optional[str]:
    pieces = [
        path.name, path.stem, path.as_posix(),
        path.parent.name, getattr(path.parent, "stem", path.parent.name),
        path.parent.parent.name if path.parent and path.parent.parent else "",
    ]
    # 1) <date>_<id>_
    for s in pieces:
        m = _DATE_ID_RE.search(s)
        if m:
            return m.group("id")
    # 2) _{1..3}<id>_
    for s in pieces:
        m = _MULTI_US_RE.search(s)
        if m:
            return m.group(1)
    # 3) first 11-char token
    for s in pieces:
        m = _TOKEN_11_RE.search(s)
        if m:
            return m.group(0)
    return None

def _extract_title_from_dir(path: Path) -> str:
    leaf = path.parent.name.strip()
    if leaf and "_diarized_content" not in leaf:
        return leaf
    return path.stem.replace("_diarized_content", "")

def _extract_channel_from_path(path: Path) -> Optional[str]:
    return path.parent.parent.name  # remove the startswith("@") gate

def _extract_date_prefix(title_or_dir: str) -> Optional[str]:
    m = re.match(r"^(\d{4}-\d{2}-\d{2})\b", title_or_dir)
    return m.group(1) if m else None

# ────────────────────────────────────────────────────────────────────────────────
# Iterator
# ────────────────────────────────────────────────────────────────────────────────

def _iter_youtube_assets_from_fs(
    root_dir: Path, prune_empty: bool = False
) -> Iterable[Tuple[Dict[str, Any], Dict[str, Any], Path]]:
    """
    Yield (meta, raw_for_segmenter, json_path) for each AssemblyAI diarized JSON.

    Uses fast JSON loads, entity cache, optional FAST mode caps,
    and concurrent per-video processing.
    """
    paths = list(root_dir.rglob("*_diarized_content.json"))
    if not paths:
        return

    # workers: default ~70% of CPUs, but at least 1
    cpu = os.cpu_count() or 4
    workers_env = os.getenv("ENTITIES_WORKERS", "").strip()
    workers = int(workers_env) if workers_env.isdigit() and int(workers_env) > 0 else max(1, math.floor(0.7 * cpu))

    def _process_one(p: Path) -> Optional[Tuple[Dict[str, Any], Dict[str, Any], Path]]:
        try:
            obj = _fast_json_load(p)
        except Exception:
            logging.warning(f"[v2] skip unreadable JSON: {p}")
            return None

        if prune_empty and _looks_like_single_empty_file(obj):
            try:
                p.unlink(missing_ok=True)
                logging.info(f"[v2] pruned empty AssemblyAI file: {p}")
            except Exception as e:
                logging.warning(f"[v2] failed to prune {p}: {_ascii(str(e))})")
            return None

        try:
            raw_norm = _convert_assemblyai_json_to_raw(obj)
        except Exception as e:
            logging.warning(f"[v2] skip malformed AssemblyAI JSON: {p} ({_ascii(str(e))})")
            return None

        segs = raw_norm.get("segments", [])
        if not segs:
            logging.info(f"[v2] no non-trivial segments after normalization: {p}")
            return None

        # duration
        ends = [float(s.get("end")) for s in segs if isinstance(s.get("end"), (int, float))]
        duration_s = max(ends) if ends else 0.0

        video_id = _extract_video_id_from_path(p)
        if not video_id:
            logging.warning(f"[v2] SKIP: could not find YouTube video_id in path={p}")
            return None

        title = _extract_title_from_dir(p)
        channel_name = _extract_channel_from_path(p)
        published_at = _extract_date_prefix(title)

        # Sidecar/top-level entities (raw) -> cleaned canonical set (cached)
        aai_entities_raw = _load_entities_for_json_path(p, obj)

        # Optional raw cap (in addition to postprocess-side cap)
        try:
            if os.getenv("ENTITIES_FAST", "0").lower() in ("1","true","yes","y"):
                cap = int(os.getenv("ENTITIES_MAX_RAW", "400"))
                if cap > 0 and len(aai_entities_raw) > cap:
                    aai_entities_raw = aai_entities_raw[:cap]
        except Exception:
            pass

        cache_key = _entities_cache_key(p, obj)
        cleaned_entities = _postprocess_entities_with_cache(aai_entities_raw, cache_key)
        logging.info("[v2/entities] vid=%s cleaned=%d sample=%s", video_id, len(cleaned_entities), cleaned_entities[:8])

        meta = {
            "video_id": video_id,
            "title": title,
            "description": "",
            "channel_name": channel_name,
            "speaker_primary": None,
            "published_at": published_at,
            "duration_s": duration_s,
            "url": f"https://www.youtube.com/watch?v={video_id}",
            "thumbnail_url": None,
            "language": "en",
            "entities": cleaned_entities,
            "chapters": None,
            "document_type": "youtube_video",
            "source": "youtube",
        }
        raw_for_segmenter = {
            "segments": segs,
            "caption_lines": [],
            "diarization": [],
            "entities": aai_entities_raw,  # pass raw for child-level tagging
        }
        return meta, raw_for_segmenter, p

    with concurrent.futures.ThreadPoolExecutor(max_workers=workers) as pool:
        futs = [pool.submit(_process_one, p) for p in paths]
        for fut in concurrent.futures.as_completed(futs):
            item = fut.result()
            if item is not None:
                yield item

# ────────────────────────────────────────────────────────────────────────────────
# Main
# ────────────────────────────────────────────────────────────────────────────────

async def _enrich_async(meta: Dict[str, Any], raw: Dict[str, Any]) -> Dict[str, Any]:
    try:
        sentences = normalize_to_sentences(raw)
    except Exception:
        sentences = []
    return await enrich_parent_router_fields_async(meta, sentences)

def _prioritize_assets(
    metas_raw_paths: List[Tuple[Dict[str, Any], Dict[str, Any], Path]],
    deprioritize_channels: Optional[List[str]] = None,
    push_to_end_regexes: Optional[List[str]] = None,
) -> List[Tuple[Dict[str, Any], Dict[str, Any], Path]]:
    """
    Return a new list where items from the given channels (exact match) and/or
    whose channel name matches any regex are pushed to the end. Within priority
    groups, keep a stable ordering using (published_at, video_id).

    Example:
        metas_raw_paths = _prioritize_assets(
            metas_raw_paths,
            deprioritize_channels=["@SolanaFndn"],
            push_to_end_regexes=[r"^@Solana.*"]  # optional
        )
    """
    deprio_set = set(deprioritize_channels or [])
    regex_objs = [re.compile(p) for p in (push_to_end_regexes or [])]

    def _is_deprio(channel: Optional[str]) -> bool:
        ch = (channel or "").strip()
        if ch in deprio_set:
            return True
        return any(rx.search(ch) for rx in regex_objs)

    def _key(item: Tuple[Dict[str, Any], Dict[str, Any], Path]):
        meta, _raw, _p = item
        ch = meta.get("channel_name")
        date = meta.get("published_at") or "0000-00-00"
        vid = meta.get("video_id") or ""
        # True sorts after False → deprioritized to the end
        return (_is_deprio(ch), date, vid)

    return sorted(metas_raw_paths, key=_key)


def _filter_speaker_map_people(spmap: Dict[str, Any], keep_keys: set[str], host_names=()) -> Dict[str, Any]:
    """
    Keep (a) any speaker keys in keep_keys (e.g., primary host), and
    (b) any entries whose 'name' looks like a person.
    """
    out = {}
    for k, info in (spmap or {}).items():
        if k in keep_keys:
            out[k] = info
            continue
        nm = (info.get("name") or "").strip()
        if nm and looks_like_person(nm, host_names=host_names):
            out[k] = info
    return out


def _get_ingested_parent_ids(index_name: str, namespace: str) -> set[str]:
    """
    Query Pinecone for existing child vectors and extract unique parent_ids.
    Works efficiently when you have <10k parents (even if >10k children total).
    """
    api_key = os.getenv("PINECONE_API_KEY")
    if not api_key:
        raise RuntimeError("PINECONE_API_KEY not set")

    from pinecone import Pinecone
    pc = Pinecone(api_key=api_key)
    idx = pc.Index(index_name)

    # Detect dimension (same logic as node_summary_cleanup.py)
    stats = idx.describe_index_stats()
    dimension = None
    for ns_stats in stats.get('namespaces', {}).values():
        if 'dimension' in ns_stats:
            dimension = ns_stats['dimension']
            break

    if dimension is None:
        for try_dim in [3072, 1536, 768]:
            try:
                idx.query(
                    namespace=namespace,
                    filter={"node_type": "child"},
                    top_k=1,
                    include_metadata=True,
                    vector=[0.0] * try_dim
                )
                dimension = try_dim
                logging.info(f"[dedupe] detected dimension={dimension}")
                break
            except:
                continue

    if dimension is None:
        raise RuntimeError("Could not determine index dimension")

    logging.info("[dedupe] querying Pinecone for existing parent_ids...")
    result = idx.query(
        namespace=namespace,
        filter={"node_type": "child"},
        top_k=10000,  # Covers >1739 parents via their children
        include_metadata=True,
        vector=[0.0] * dimension
    )

    parent_ids = {
        match['metadata']['parent_id']
        for match in result.get('matches', [])
        if match.get('metadata', {}).get('parent_id')
    }

    logging.info(f"[dedupe] found {len(parent_ids)} unique parent_ids already in Pinecone")
    return parent_ids

def _guess_audio_path(json_path: Path) -> Optional[Path]:
    # Look for a sibling .mp3/.wav using the same stem
    stem = json_path.stem.replace("_diarized_content", "")
    for ext in (".mp3", ".wav", ".m4a"):
        cand = json_path.with_name(f"{stem}{ext}")
        if cand.exists():
            return cand
    # or parent dir with same basename
    for ext in (".mp3", ".wav", ".m4a"):
        cand = json_path.parent / f"{stem}{ext}"
        if cand.exists():
            return cand
    return None

def _load_entities_for_json_path(content_path: Path, obj: Any) -> List[Dict[str, Any]]:
    """
    Prefer a sibling '*_entities.json'. Fallback to top-level obj['entities'].
    Accepts:
      - list[dict{text, entity_type, ...}] (AAI style)
      - list[str]                           (we wrap into dicts)
      - dict with key 'entities'            (we take that list)
    """
    def _norm(payload) -> List[Dict[str, Any]]:
        if isinstance(payload, dict) and isinstance(payload.get("entities"), list):
            payload = payload["entities"]
        if isinstance(payload, list):
            if payload and isinstance(payload[0], dict):
                return payload
            if payload and isinstance(payload[0], str):
                return [{"text": s, "entity_type": "custom"} for s in payload if isinstance(s, str)]
            return []
        return []

    sidecar = _entities_sidecar_path(content_path)
    try:
        if sidecar.exists():
            raw = _fast_json_load(sidecar)
            ents = _norm(raw)
            logging.info("[v2/entities] using sidecar for %s (%d items)", content_path.name, len(ents))
            return ents
    except Exception as e:
        logging.warning("[v2/entities] sidecar read failed %s: %s", sidecar, e)

    # SAFE: only read top-level entities if the top-level object is a dict
    top_payload = obj.get("entities") if isinstance(obj, dict) else None
    ents = _norm(top_payload)
    if ents:
        logging.info("[v2/entities] using top-level 'entities' for %s (%d items)", content_path.name, len(ents))
    else:
        logging.info("[v2/entities] no entities for %s", content_path.name)
    return ents


# --- namespace-aware channels config (ingestion) ------------------------------
def _parse_list_env(var: str) -> list[str]:
    raw = os.getenv(var, "")
    return [x.strip() for x in re.split(r"[,\s]+", raw) if x.strip()]


def _load_namespace_channels(namespace: str) -> list[str]:
    """
    Resolve channels for a given namespace.
    Priority:
      1) YT_NAMESPACE_CONFIG_JSON  (inline JSON string)
      2) YT_NAMESPACE_CONFIG       (path to JSON/YAML file)
      3) Hardcoded fallbacks (optional, keep empty by default)
    Returns [] if not found.
    Shape:
    {
      "namespaces": {
        "bnb":    { "channels": ["@BinanceYoutube", "..."] },
        "videos": { "channels": ["@Delphi_Digital", "..."] }
      }
    }
    """
    import json as _json

    ns = (namespace or "").strip() or "default"

    # (1) Inline JSON
    raw_inline = os.getenv("YT_NAMESPACE_CONFIG_JSON")
    if raw_inline:
        try:
            cfg = _json.loads(raw_inline)
            chans = (cfg.get("namespaces", {}).get(ns, {}) or {}).get("channels", [])
            if chans: return [c.strip() for c in chans if c.strip()]
        except Exception:
            pass

    # (2) File (JSON or YAML)
    cfg_path = os.getenv("YT_NAMESPACE_CONFIG")
    if cfg_path:
        from pathlib import Path
        p = Path(cfg_path).expanduser().resolve()
        if p.exists():
            text = p.read_text(encoding="utf-8")
            try:
                if p.suffix.lower() in (".yaml", ".yml"):
                    import yaml  # type: ignore
                    cfg = yaml.safe_load(text) or {}
                else:
                    cfg = _json.loads(text)
                chans = (cfg.get("namespaces", {}).get(ns, {}) or {}).get("channels", [])
                if chans: return [c.strip() for c in chans if c.strip()]
            except Exception:
                pass

    # (3) Optional hardcoded fallbacks (leave empty in ingestion repo)
    # if ns == "bnb": return ["@BinanceYoutube", "@binanceacademy"]
    # if ns == "videos": return ["@Delphi_Digital"]
    return []


def main():
    setup_logger("ingest_v2")

    ap = argparse.ArgumentParser()
    ap.add_argument("--doc-type", default="youtube_video", choices=["youtube_video", "stream"])
    ap.add_argument("--namespace", default=os.getenv("PINECONE_NAMESPACE", "videos"),
                    help="Pinecone namespace to write to (defaults to env or 'videos').")
    ap.add_argument(
        "--include-channels",
        nargs="*",
        default=None,  # explicit override only
        help="Only ingest these channel folders (e.g. @BinanceYoutube @notthreadguy).",
    )
    ap.add_argument("--backfill-days", type=int, default=settings_v2.BACKFILL_DAYS)
    ap.add_argument("--root", default=YOUTUBE_VIDEO_DIRECTORY, help="Root dir with diarized JSONs")
    ap.add_argument("--prune-empty", action="store_true", help="Delete obviously empty AssemblyAI files")
    ap.add_argument("--concurrency", type=int, default=int(os.getenv("ROUTER_GEN_CONCURRENCY", "6")))
    ap.add_argument("--speakers-workers", type=int, default=int(os.getenv("SPEAKERS_WORKERS", "0")),
                    help="Workers for speaker resolution across videos (default: ~70% of CPUs when 0)")
    ap.add_argument("--skip-speakers", action="store_true",
                    help="Skip cross-video speaker resolution/enroll stages")
    ap.add_argument("--skip-dedupe", action="store_true",
                    help="Skip Pinecone deduplication check (process all filesystem assets)")
    args = ap.parse_args()

    index_name = os.getenv("PINECONE_INDEX_NAME", "icmfyi-v2")
    ns = args.namespace
    logging.info("[cfg] pinecone index=%s  namespace=%s", index_name, ns)

    # 1) Explicit CLI wins
    channels = [c for c in (args.include_channels or []) if c and c.strip()]

    # 2) If not provided, load from namespace config
    if not channels:
        channels = _load_namespace_channels(ns)
        if channels:
            logging.info("[cfg] loaded %d channel(s) from namespace '%s'", len(channels), ns)

    # 3) Backward-compat env (deprecated)
    if not channels:
        legacy = _parse_list_env("INGEST_INCLUDE_CHANNELS")
        if legacy:
            logging.warning(
                "INGEST_INCLUDE_CHANNELS is DEPRECATED. Move to YT_NAMESPACE_CONFIG or "
                "YT_NAMESPACE_CONFIG_JSON keyed by namespace='%s'.", ns
            )
            channels = legacy

    # Final check
    if not channels:
        ap.error(
            "No channels resolved. Provide --include-channels, or configure YT_NAMESPACE_CONFIG "
            "(JSON/YAML) or YT_NAMESPACE_CONFIG_JSON for namespace='%s'." % ns
        )

    logging.info("[cfg] namespace=%s channels=%s", ns, channels)

    # Normalize back onto args for downstream use
    args.include_channels = channels

    # Ensure Tier-2 voice matching is OFF by default; set VOICE_EMBED_BACKEND=auto to enable.
    os.environ.setdefault("VOICE_EMBED_BACKEND", "off")

    # Make namespace choice global for all downstream helpers (dedupe, upsert, parent_resolver, etc.)
    os.environ["PINECONE_NAMESPACE"] = args.namespace
    # If you also keep a streams space, you can auto-derive it:
    os.environ.setdefault("PINECONE_STREAMS_NS", f"{args.namespace}_streams")

    if args.doc_type != "youtube_video":
        logging.info("[v2] 'stream' not wired yet in run_all; nothing to do.")
        return

    root = Path(args.root).expanduser().resolve()
    logging.info(f"[v2] scanning for AssemblyAI JSON under: {root} (prune_empty={args.prune_empty})")

    metas_raw_paths = list(_iter_youtube_assets_from_fs(root, prune_empty=args.prune_empty))
    if not metas_raw_paths:
        logging.info("[v2] no youtube assets found; exiting.")
        return

    # ── INCLUDE FILTER: restrict to specific channels (exact folder names like '@notthreadguy')
    if args.include_channels:
        allow = {c.strip() for c in args.include_channels if c.strip()}
        before = len(metas_raw_paths)
        metas_raw_paths = [
            (m, r, p) for (m, r, p) in metas_raw_paths
            if (m.get("channel_name") or "").strip() in allow
        ]
        logging.info("[v2/filter] channels=%s kept=%d/%d",
                     sorted(allow), len(metas_raw_paths), before)
        if not metas_raw_paths:
            logging.info("[v2] no assets match include filter; exiting.")
            return

    # ── FILTER: Remove already-ingested parents ────────────────────────────────
    if not args.skip_dedupe:
        logging.info("[v2/dedupe] checking which parents are already in Pinecone...")
        already_ingested = _get_ingested_parent_ids(
            index_name=os.getenv("PINECONE_INDEX_NAME", "icmfyi-v2"),
            namespace=args.namespace,
        )

        before_count = len(metas_raw_paths)
        metas_raw_paths = [
            (meta, raw, path)
            for (meta, raw, path) in metas_raw_paths
            if meta["video_id"] not in already_ingested
        ]
        after_count = len(metas_raw_paths)

        logging.info(
            f"[v2/dedupe] parent filter: {before_count} discovered, "
            f"{len(already_ingested)} already ingested, "
            f"{after_count} remaining to process"
        )

        if not metas_raw_paths:
            logging.info("[v2] all parents already in Pinecone. Nothing to do.")
            return
    else:
        logging.info("[v2/dedupe] skipping deduplication check (--skip-dedupe)")

    # ── Prioritize assets (stable within groups) ────────────────────────────────
    metas_raw_paths = _prioritize_assets(
        metas_raw_paths,
        deprioritize_channels=["@SolanaFndn", "@timroughgardenlectures1861"]
    )

    # ── Speakers stage (skippable) ─────────────────────────────────────────────
    metas_raw_paths2: List[Tuple[Dict[str, Any], Dict[str, Any], Path]] = []
    t_speakers_total = 0.0

    if args.skip_speakers:
        logging.info("[v2/speakers] skipping speaker resolution (--skip-speakers)")
        metas_raw_paths2 = [(m, r, p) for (m, r, p) in metas_raw_paths]
    else:
        cpu = os.cpu_count() or 4
        workers = args.speakers_workers or max(1, math.floor(0.7 * cpu))
        logging.info(f"[v2/speakers] resolving across {len(metas_raw_paths)} assets with {workers} workers")

        def _resolve_one(item: Tuple[Dict[str, Any], Dict[str, Any], Path]):
            meta, raw, json_path = item
            t0 = time.perf_counter()
            audio_path = _guess_audio_path(json_path)
            try:
                spk = resolve_speakers(meta, raw, audio_hint_path=audio_path)
            except Exception as e:
                logging.warning("[v2/speakers] resolve failed vid=%s: %s", meta.get("video_id"), _ascii(str(e)))
                spk = {}
            dt = time.perf_counter() - t0

            # Merge results into meta
            if spk.get("speaker_map"):
                meta["speaker_map"] = spk["speaker_map"]
            if spk.get("speaker_primary"):
                meta["speaker_primary"] = spk["speaker_primary"]

            # normalize + filter people
            if meta.get("speaker_primary"):
                meta["speaker_primary"] = normalize_alias(meta["speaker_primary"])
            primary = meta.get("speaker_primary")

            spmap = meta.get("speaker_map") or {}
            keep = {primary} if primary else set()
            spmap = _filter_speaker_map_people(spmap, keep_keys=keep, host_names=[primary or ""])
            meta["speaker_map"] = spmap

            # friendly log
            vid = meta.get("video_id")
            host_name = (spmap.get(primary, {}) or {}).get("name") or "Host"
            guest_names = []
            for spk_label, info in spmap.items():
                if spk_label == primary:
                    continue
                nm = (info.get("name") or "").strip()
                if nm:
                    guest_names.append(normalize_alias(nm))
            guest_names = filter_to_people(guest_names, host_names=[primary or ""])
            guests = []
            for spk_label, info in spmap.items():
                nm = (info.get("name") or "").strip()
                if not nm:
                    continue
                if normalize_alias(nm) in guest_names:
                    guests.append(f"{nm}({info.get('confidence', 0.0):.2f},{info.get('source', '')})")

            logging.info("[v2/speakers] vid=%s resolved in %.2fs host=%r guests=%s",
                         vid, dt, host_name, guests or "[]")

            return meta, raw, json_path, dt

        with concurrent.futures.ThreadPoolExecutor(max_workers=workers) as pool:
            futures = [pool.submit(_resolve_one, item) for item in metas_raw_paths]
            for fut in concurrent.futures.as_completed(futures):
                meta, raw, json_path, dt = fut.result()
                metas_raw_paths2.append((meta, raw, json_path))
                t_speakers_total += dt

    # ── Enrich via LLM (with cache) ────────────────────────────────────────────
    enriched_metas: List[Dict[str, Any]] = []
    enriched = 0
    cached_hits = 0
    failed_enrich = 0
    t_enrich_total = 0.0

    to_enrich: List[Tuple[Dict[str, Any], Dict[str, Any]]] = []
    for (meta, raw, _p) in metas_raw_paths2:
        pid = meta["video_id"]
        enrich = None
        try:
            enrich = router_cache_load(pid)
        except Exception as e:
            logging.warning(f"[v2/router/cache] read failed for {pid}: {_ascii(str(e))}")

        if enrich is not None:
            cached_hits += 1
            n = int(os.getenv("ROUTER_LOG_DESC_CHARS", "240"))
            logging.info(
                "[router/cache] vid=%s boost=%s tags=%s desc_preview=%r topic_preview=%r",
                pid,
                enrich.get("router_boost"),
                (enrich.get("router_tags") or [])[:6],
                _ascii(_snip(enrich.get("description") or "", n)),
                _ascii(_snip(enrich.get("topic_summary") or "", max(80, n // 2))),
            )
            merged = dict(meta)
            merged.update({
                "description": enrich.get("description", merged.get("description", "")),
                "topic_summary": enrich.get("topic_summary"),
                "router_tags": enrich.get("router_tags"),
                "aliases": enrich.get("aliases"),
                "canonical_entities": enrich.get("canonical_entities"),
                "is_explainer": enrich.get("is_explainer"),
                "router_boost": enrich.get("router_boost"),
            })
            enriched_metas.append(merged)
        else:
            to_enrich.append((meta, raw))

    if to_enrich:
        async def _batch_enrich(pairs: List[Tuple[Dict[str, Any], Dict[str, Any]]], concurrency: int):
            sem = asyncio.Semaphore(concurrency)

            async def _one(meta: Dict[str, Any], raw: Dict[str, Any]):
                pid = meta["video_id"]
                try:
                    t0 = time.perf_counter()
                    async with sem:
                        enrich = await _enrich_async(meta, raw)
                    dt = time.perf_counter() - t0
                    logging.info("[v2/router] vid=%s enriched in %.2fs", pid, dt)
                    try:
                        router_cache_save(pid, enrich)
                    except Exception as e:
                        logging.warning(f"[v2/router/cache] save failed for {pid}: {_ascii(str(e))}")
                    return meta, enrich, None, dt
                except Exception as e:
                    return meta, None, e, 0.0

            tasks = [asyncio.create_task(_one(m, r)) for (m, r) in pairs]
            return await asyncio.gather(*tasks)

        results = asyncio.run(_batch_enrich(to_enrich, args.concurrency))
        for meta, enrich, err, dt in results:
            pid = meta["video_id"]
            if err or enrich is None:
                logging.warning(f"[v2/router] enrichment failed for {pid}: {_ascii(str(err))}")
                failed_enrich += 1
                enrich = {
                    "description": meta.get("description") or "",
                    "topic_summary": "",
                    "router_tags": [],
                    "aliases": [],
                    "canonical_entities": [],
                    "is_explainer": False,
                    "router_boost": 1.0,
                }
            else:
                enriched += 1
                t_enrich_total += dt

            merged = dict(meta)
            merged.update({
                "description": enrich.get("description", merged.get("description", "")),
                "topic_summary": enrich.get("topic_summary"),
                "router_tags": enrich.get("router_tags"),
                "aliases": enrich.get("aliases"),
                "canonical_entities": enrich.get("canonical_entities"),
                "is_explainer": enrich.get("is_explainer"),
                "router_boost": enrich.get("router_boost"),
            })
            enriched_metas.append(merged)

    logging.info(
        f"[v2/router] total={len(enriched_metas)} cached_hits={cached_hits} "
        f"fresh_enriched={enriched} failed={failed_enrich}"
    )

    # ── Build parents ───────────────────────────────────────────────────────────
    parents = run_build_parents(metas=[{
        "video_id": m["video_id"],
        "title": m.get("title", ""),
        "description": m.get("description", ""),
        "channel_name": m.get("channel_name"),
        "speaker_primary": m.get("speaker_primary"),
        "published_at": m.get("published_at"),
        "duration_s": m.get("duration_s", 0.0),
        "url": m.get("url"),
        "thumbnail_url": m.get("thumbnail_url"),
        "language": m.get("language", "en"),
        "entities": m.get("entities", []),
        "chapters": m.get("chapters"),
        "document_type": "youtube_video",
        "source": "youtube",
        "router_tags": m.get("router_tags"),
        "aliases": m.get("aliases"),
        "canonical_entities": m.get("canonical_entities"),
        "is_explainer": m.get("is_explainer"),
        "router_boost": m.get("router_boost"),
        "topic_summary": m.get("topic_summary"),
        # speakers
        "speaker_map": m.get("speaker_map"),
    } for m in enriched_metas])

    progress.set_parents_total(len(parents))

    # parents_map
    parents_map = {p.parent_id: p.model_dump(mode="json") for p in parents}

    # upsert
    upsert_parents([{**p.model_dump(mode="json"), "parent_id": p.parent_id} for p in parents])

    # ── Build + upsert children ────────────────────────────────────────────────
    total_children = 0
    missing_parents = 0
    skipped_empty = 0
    t_children_total = 0.0

    for (meta, raw, _path) in metas_raw_paths2:
        pid = meta["video_id"]
        parent = parents_map.get(pid)
        if not parent:
            logging.warning(f"[v2] missing parent for video_id={pid}; skipping")
            missing_parents += 1
            continue

        t0 = time.perf_counter()
        children = build_children_from_raw(parent, raw)
        if not children:
            logging.info(f"[v2] no children emitted for {pid} ({_ascii(meta.get('title') or '')})")
            skipped_empty += 1
            continue

        t0 = time.perf_counter()
        stats = upsert_children(children)
        dt = time.perf_counter() - t0
        t_children_total += dt
        total_children += len(children)
        logging.info(
            "[v2/children] vid=%s upserted=%d in %.2fs (embed=%.2fs upsert=%.2fs embed_reqs=%d pinecone_batches=%d)",
            pid, len(children), dt, stats["t_embed"], stats["t_upsert"], stats["embed_reqs"], stats["pinecone_batches"]
        )

    logging.info(
        f"[ingest_v2] finished upserting {total_children} child segments "
        f"from {len(parents)} parents "
        f"(missing_parents={missing_parents}, skipped_empty={skipped_empty}, "
        f"router_cached={cached_hits}, router_enriched={enriched}, router_failed={failed_enrich})."
    )

    # Timing summary
    avg_speakers = t_speakers_total / max(1, len(metas_raw_paths2))
    logging.info(
        "[timing] speakers_total=%.2fs (~%.2fs/video), enrich_total=%.2fs, children_total=%.2fs",
        t_speakers_total, avg_speakers, t_enrich_total, t_children_total,
    )




if __name__ == "__main__":
    main()
