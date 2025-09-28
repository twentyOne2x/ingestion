# src/ingest_v2/pipelines/build_children.py
import json
import logging
import os
import time
import uuid
from typing import Any, Dict, List, Optional, Tuple

from ..configs.settings import settings_v2
from ..entities.extract import extract_entities
from ..segmenter.segmenter import build_segments
from ..transcripts.normalize import normalize_to_sentences
from ..utils.ids import segment_uuid, sha1_hex
from ..utils.timefmt import floor_s, s_to_hms_ms
from ..validators.runtime import validate_child_runtime
import re
from openai import OpenAI  # <- for the entity LLM pass
from ..speakers.name_filters import filter_to_people, looks_like_person, normalize_alias


# ────────────────────────────────────────────────────────────────────────────────
# Helpers
# ────────────────────────────────────────────────────────────────────────────────

def _chapters_lookup(chapters: Optional[List[Dict[str, Any]]]):
    """Return a function(start_s) -> chapter_title or None."""
    if not chapters:
        return None
    rows = [c if isinstance(c, dict) else c.dict() for c in chapters]
    rows = sorted(rows, key=lambda x: x.get("start_s", 0.0))

    def lookup(s: float) -> Optional[str]:
        last = None
        for ch in rows:
            if float(ch.get("start_s", 0.0)) <= s:
                last = ch.get("title")
            else:
                break
        return last

    return lookup


def _sentences_in_range(
    sentences: List[Dict[str, Any]],
    start_s: float,
    end_s: float,
) -> List[Dict[str, Any]]:
    """Pick sentences whose midpoint falls inside [start_s, end_s]."""
    out: List[Dict[str, Any]] = []
    for s in sentences:
        ss = float(s.get("start_s", 0.0))
        ee = float(s.get("end_s", ss))
        mid = (ss + ee) / 2.0
        if start_s - 1e-6 <= mid <= end_s + 1e-6:
            out.append(s)
    return out


def _split_by_sentences_with_overlap(
    sentences: List[Dict[str, Any]],
    max_window_s: float,
    overlap_s: float,
    parent: Dict[str, Any],
    clip_base_url: Optional[str],
) -> List[Dict[str, Any]]:
    """
    Greedy sentence-aligned packing with time-based overlap.
    Emits complete child payloads (deterministic IDs, HMS, clip_url, source_hash).
    """
    children: List[Dict[str, Any]] = []
    i, n = 0, len(sentences)
    parent_id = parent["parent_id"]

    while i < n:
        # Grow window from i
        j = i
        while j < n:
            seg_start = float(sentences[i]["start_s"])
            seg_end = float(sentences[j]["end_s"])
            dur = seg_end - seg_start

            # If we exceeded the cap and we have >1 sentence, back off 1 sentence
            if dur > max_window_s and j > i:
                j -= 1
                break

            # If single giant sentence exceeds cap: hard cut
            if dur > max_window_s and j == i:
                cut_end = seg_start + max_window_s
                text = (sentences[i].get("text") or "").strip()
                start_hms = s_to_hms_ms(seg_start)
                end_hms = s_to_hms_ms(cut_end)
                seg_id = segment_uuid(parent_id, seg_start, cut_end)
                clip_url = f"{clip_base_url}&t={floor_s(seg_start)}s" if clip_base_url else None

                payload = {
                    "node_type": "child",
                    "segment_id": seg_id,
                    "parent_id": parent_id,
                    "document_type": parent["document_type"],
                    "text": text,
                    "start_s": seg_start,
                    "end_s": cut_end,
                    "start_hms": start_hms,
                    "end_hms": end_hms,
                    "clip_url": clip_url,
                    "speaker": sentences[i].get("speaker", "S1"),
                    "entities": [],
                    "chapter": sentences[i].get("chapter"),
                    "language": parent.get("language", "en"),
                    "confidence_asr": None,
                    "has_music": False,
                    "flags": [],
                    "rights": parent.get("rights", "public_reference_only"),
                    "ingest_version": 2,
                }
                raw_bytes = json.dumps(
                    {"text": text, "start": seg_start, "end": cut_end},
                    sort_keys=True,
                ).encode("utf-8")
                payload["source_hash"] = sha1_hex(raw_bytes)
                children.append(payload)

                # Advance by overlap logic below
                next_start_time = cut_end - overlap_s
                k = i + 1
                while k < n and float(sentences[k]["end_s"]) <= next_start_time:
                    k += 1
                i = k
                break

            j += 1

        if j >= n:
            j = n - 1

        # If the loop broke due to single-sentence hard cut, continue
        if i >= n:
            break
        # If previous branch emitted an item and advanced i, continue outer loop
        if float(sentences[i]["end_s"]) > float(sentences[j]["end_s"]):
            # Defensive guard, but should not happen
            j = i

        seg_start = float(sentences[i]["start_s"])
        seg_end = float(sentences[j]["end_s"])
        text = " ".join(
            (s.get("text") or "").strip()
            for s in sentences[i : j + 1]
            if s.get("text")
        )

        start_hms = s_to_hms_ms(seg_start)
        end_hms = s_to_hms_ms(seg_end)
        seg_id = segment_uuid(parent_id, seg_start, seg_end)
        clip_url = f"{clip_base_url}&t={floor_s(seg_start)}s" if clip_base_url else None

        payload = {
            "node_type": "child",
            "segment_id": seg_id,
            "parent_id": parent_id,
            "document_type": parent["document_type"],
            "text": text,
            "start_s": seg_start,
            "end_s": seg_end,
            "start_hms": start_hms,
            "end_hms": end_hms,
            "clip_url": clip_url,
            "speaker": sentences[i].get("speaker", "S1"),
            "entities": [],
            "chapter": sentences[i].get("chapter"),
            "language": parent.get("language", "en"),
            "confidence_asr": None,
            "has_music": False,
            "flags": [],
            "rights": parent.get("rights", "public_reference_only"),
            "_ingest_version": 2,  # tolerate underscore if sanitized later
            "ingest_version": 2,
        }
        raw_bytes = json.dumps(
            {"text": text, "start": seg_start, "end": seg_end},
            sort_keys=True,
        ).encode("utf-8")
        payload["source_hash"] = sha1_hex(raw_bytes)
        children.append(payload)

        # Compute next window start with overlap
        next_start_time = seg_end - overlap_s
        k = j + 1
        while k < n and float(sentences[k]["end_s"]) <= next_start_time:
            k += 1
        i = k

    return children


def _make_parent_summary_child(
    sentences: List[Dict[str, Any]],
    parent: Dict[str, Any],
    char_cap: int = 900,
    time_cap_s: float = 60.0,
) -> Optional[Dict[str, Any]]:
    """
    Cheap extractive summary: gather earliest sentences until either time or char cap.
    Emits node_type='summary' (validated by the summary branch in validator).
    """
    if not sentences:
        return None

    acc: List[str] = []
    start_s = float(sentences[0].get("start_s", 0.0))
    end_s_limit = start_s + time_cap_s

    total_chars = 0
    for s in sentences:
        ss = float(s.get("start_s", 0.0))
        if ss > end_s_limit:
            break
        t = (s.get("text") or "").strip()
        if not t:
            continue
        acc.append(t)
        total_chars += len(t)
        if total_chars >= char_cap:
            break

    if not acc:
        return None

    text = " ".join(acc).strip()
    end_s = min(
        end_s_limit,
        float(sentences[-1].get("end_s", end_s_limit)),
    )

    # Deterministic-ish ID is not required here (not used for dedupe),
    # but leaving uuid4 keeps it distinct from child windows.
    node = {
        "node_type": "summary",
        "segment_id": str(uuid.uuid4()),
        "parent_id": parent["parent_id"],
        "document_type": parent["document_type"],
        "text": text,
        "start_s": start_s,
        "end_s": end_s,
        "speaker": None,
        "chapter": None,
        "entities": [],
        "clip_url": None,
        "language": parent.get("language", "en"),
        "ingest_version": 2,
        "rights": parent.get("rights", "public_reference_only"),
    }
    return node


# ────────────────────────────────────────────────────────────────────────────────
# Main entry
# ────────────────────────────────────────────────────────────────────────────────

def _canon_entity(e: str) -> str:
    e = (e or "").strip()
    if not e:
        return ""
    if e.startswith("$"):   return e.upper()
    if e.startswith("@"):   return e.lower()
    return e


def build_children_from_raw(parent: Dict[str, Any], raw: Dict[str, Any]) -> List[Dict[str, Any]]:
    """
    1) Build initial children via segmenter (15–60s, padded).
    2) Split any over-long child into sentence-aligned subchunks with small overlap.
    3) Add one per-parent summary node (node_type='summary').
    4) Extract entities using:
         - inline tickers/handles from text,
         - relabeled speaker name,
         - parent context entities,
         - AssemblyAI entities (no LLM).
    5) Validate each node, keep only OK.
    6) (Optional) Relabel speakers to resolved names from parent['speaker_map'].
    """
    from ..entities.postprocess import postprocess_aai_entities

    t0 = time.perf_counter()

    # Normalize to sentence list (used by segmenter fixes + summary)
    sentences = normalize_to_sentences(raw)
    if not sentences:
        logging.info(f"[children] parent_id={parent.get('parent_id')} no sentences after normalization")
        return []

    parent_id = parent["parent_id"]
    document_type = parent["document_type"]
    clip_base = f"https://www.youtube.com/watch?v={parent_id}" if document_type == "youtube_video" else None
    chapter_lookup = _chapters_lookup(parent.get("chapters"))

    # 1) Base segments
    children = build_segments(
        sentences=sentences,
        duration_s=parent["duration_s"],
        parent_id=parent_id,
        document_type=document_type,
        clip_base_url=clip_base,
        chapter_lookup=chapter_lookup,
        language=parent.get("language", "en"),
    )
    t1 = time.perf_counter()

    # 2) Split any over-long windows
    max_s = float(settings_v2.SEGMENT_MAX_S)
    pad_s = float(settings_v2.SEGMENT_PAD_S)
    tol_s = float(settings_v2.SEGMENT_TOLERANCE_S)
    overlap_s = float(settings_v2.SEGMENT_OVERLAP_S)
    upper_with_pad = max_s + (2.0 * pad_s) + tol_s

    fixed_children: List[Dict[str, Any]] = []
    for ch in children:
        seg_len = float(ch["end_s"]) - float(ch["start_s"])
        if seg_len <= upper_with_pad:
            fixed_children.append(ch)
            continue

        # Split this child into smaller children using sentence boundaries inside its window
        slice_sents = _sentences_in_range(sentences, ch["start_s"], ch["end_s"])
        subchildren = _split_by_sentences_with_overlap(
            sentences=slice_sents,
            max_window_s=max_s,
            overlap_s=overlap_s,
            parent=parent,
            clip_base_url=clip_base,
        )
        logging.info(
            f"[children] parent_id={parent_id} split_long segment_id={ch.get('segment_id')} "
            f"orig_len={seg_len:.3f}s -> {len(subchildren)} subchunks"
        )
        fixed_children.extend(subchildren)
    t2 = time.perf_counter()

    # 2.5) Optional relabel: map 'S1' -> resolved speaker name, keep original in 'speaker_label'
    speaker_map = (parent.get("speaker_map") or {}) if isinstance(parent.get("speaker_map"), dict) else {}
    mode = os.getenv("SPEAKER_NAME_MODE", "name").lower()  # 'label' | 'name' | 'both'
    if speaker_map and mode in ("name", "both"):
        def _map(spk: Optional[str]) -> Tuple[str, Optional[str]]:
            if not spk:
                return spk, None
            entry = speaker_map.get(spk) or {}
            name = entry.get("name")
            if not name:
                return spk, None
            if mode == "both":
                return f"{name} ({spk})", spk
            return name, spk

        relabeled = 0
        for ch in fixed_children:
            spk = ch.get("speaker")
            new_s, orig = _map(spk)
            if new_s and new_s != spk:
                ch["speaker_label"] = orig or spk
                ch["speaker"] = new_s
                entry = speaker_map.get(orig or spk) or speaker_map.get(spk) or {}
                if entry.get("role"):
                    ch["speaker_role"] = entry["role"]
                if entry.get("source"):
                    ch["speaker_source"] = entry["source"]
                if entry.get("confidence") is not None:
                    ch["speaker_confidence"] = float(entry["confidence"])
                relabeled += 1
        if relabeled:
            logging.info(f"[children/speakers] parent_id={parent_id} relabeled={relabeled}")

    # --- Global/context entities (AAI only; no LLM) ---
    aai_clean = set(postprocess_aai_entities((raw or {}).get("entities")))
    context_entities = set(_parent_context_entities(parent)) | aai_clean

    # 3) Entities + validation + keep/drop counters
    kept: List[Dict[str, Any]] = []
    dropped_counts = {"schema_error": 0, "timestamp_order_invalid": 0, "window_size_out_of_bounds": 0, "text_too_short": 0}

    for ch in fixed_children:
        base = set(extract_entities(ch.get("text", "")))
        # include the (re)labeled speaker as an entity
        spk_name = (ch.get("speaker") or "").strip()
        if spk_name:
            base.add(normalize_alias(spk_name))

        # merge + canon + cap
        ents = {_canon_entity(x) for x in (base | context_entities)}
        ents = {x for x in ents if x}  # drop empties
        ch["entities"] = sorted(ents, key=str.lower)[:64]

        ok, reason = validate_child_runtime(ch, duration_s=parent["duration_s"])
        if ok:
            kept.append(ch)
        else:
            dropped_counts[reason] = dropped_counts.get(reason, 0) + 1
    t3 = time.perf_counter()

    # 4) Add a higher-level summary node (best-effort)
    summary_node = _make_parent_summary_child(
        sentences, parent, char_cap=900, time_cap_s=min(60.0, parent.get("duration_s", 60.0))
    )
    if summary_node and len(summary_node.get("text", "")) >= 80:
        base = set(extract_entities(summary_node["text"]))
        ents = {_canon_entity(x) for x in (base | context_entities)}
        ents = {x for x in ents if x}
        summary_node["entities"] = sorted(ents, key=str.lower)[:64]

        ok, reason = validate_child_runtime(summary_node, duration_s=parent["duration_s"])
        if ok:
            kept.append(summary_node)
        else:
            dropped_counts[reason] = dropped_counts.get(reason, 0) + 1
    t4 = time.perf_counter()

    # Logging
    if any(dropped_counts.values()):
        logging.info(
            f"[children] parent_id={parent_id} kept={len(kept)}/{len(fixed_children)} "
            f"dropped={sum(dropped_counts.values())} by_reason="
            f"{ {k: v for k, v in dropped_counts.items() if v} }"
        )
    else:
        logging.info(f"[children] parent_id={parent_id} kept_all={len(kept)}")

    # Timing
    logging.info(
        "[timing/children] parent_id=%s normalize=%.3fs build=%.3fs split=%.3fs "
        "entities+validate=%.3fs summary=%.3fs total=%.3fs",
        parent_id, (t1 - t0), (t2 - t1), (t3 - t2), (t4 - t3), (time.perf_counter() - t4), (time.perf_counter() - t0)
    )

    return kept


def _parent_context_entities(parent: Dict[str, Any]) -> List[str]:
    """
    Stable entities to attach to every child:
    - resolved people from parent['speaker_map']
    - @handles and $TICKERS from parent['title']
    - parent['canonical_entities'] from the enrich step
    - channel if it looks like a person/handle
    """
    out: set[str] = set()

    # people from speaker_map
    spmap = parent.get("speaker_map") or {}
    people = []
    for _k, info in (spmap.items() if isinstance(spmap, dict) else []):
        nm = (info.get("name") or "").strip()
        if nm:
            people.append(nm)
    people = filter_to_people(people, host_names=[parent.get("channel_name") or ""])
    out.update(normalize_alias(p) for p in people)

    # handles/tickers from title
    title = parent.get("title") or ""
    if title:
        out.update(re.findall(r"@[A-Za-z0-9_]{3,}", title))
        out.update(re.findall(r"\$[A-Za-z]{2,6}", title))

    # canonical entities from enrich
    for c in (parent.get("canonical_entities") or []):
        if isinstance(c, str) and c.strip():
            out.add(c.strip())

    # channel if person-like
    ch = parent.get("channel_name") or ""
    if ch and looks_like_person(ch, host_names=[]):
        out.add(normalize_alias(ch))

    return sorted({x.strip() for x in out if isinstance(x, str) and x.strip()})[:32]


def _preview_text(sentences: List[Dict[str, Any]], cap_s: float = 5 * 60.0, char_cap: int = 1600) -> str:
    acc, total = [], 0
    for s in sentences:
        st = float(s.get("start_s") or s.get("start") or 0.0)
        if st > cap_s:
            break
        t = (s.get("text") or "").strip()
        if not t:
            continue
        if total + len(t) > char_cap:
            acc.append(t[: max(0, char_cap - total)])
            break
        acc.append(t)
        total += len(t)
    return " ".join(acc)


def _entities_from_title_and_preview(parent: Dict[str, Any], sentences: List[Dict[str, Any]]) -> List[str]:
    """
    ALWAYS call a tiny model to extract entities from (title + early transcript preview).
    Returns a deduped list of strings. Falls back to [] on error.
    """
    title = parent.get("title") or ""
    preview = _preview_text(sentences)
    model = os.getenv("ENTITIES_LLM_MODEL", "gpt-4o-mini")
    client = OpenAI()

    sys = (
        "Extract entities for video search. Return STRICT JSON only.\n"
        "Include: people (@handles or 'First Last'), orgs, protocols, products, tickers as $TICKER, and key topics/acronyms.\n"
        "Prefer canonical short forms (e.g., 'data availability sampling' → 'DAS'). De-duplicate."
    )
    user = {
        "title": title,
        "early_transcript": preview,
        "format": {
            "type": "object",
            "properties": {"entities": {"type": "array", "items": {"type": "string"}}},
            "required": ["entities"],
            "additionalProperties": False,
        },
    }

    try:
        resp = client.chat.completions.create(
            model=model,
            temperature=0.0,
            response_format={"type": "json_object"},
            messages=[
                {"role": "system", "content": sys},
                {"role": "user", "content": json.dumps(user, ensure_ascii=False)},
            ],
            timeout=45,
        )
        raw = resp.choices[0].message.content or "{}"
        data = json.loads(raw)
        ents = data.get("entities") or []
    except Exception as e:
        logging.warning("[children/entities_llm] failed: %s", e)
        return []

    # light cleanup + alias normalization for people/handles
    cleaned: List[str] = []
    seen = set()
    for x in ents:
        if not isinstance(x, str):
            continue
        s = x.strip()
        if not s:
            continue
        # normalize obvious person aliases/handles
        if s.startswith("@") or " " in s:
            s = normalize_alias(s)
        low = s.lower()
        if low in seen:
            continue
        seen.add(low)
        cleaned.append(s)
    return cleaned[:48]
