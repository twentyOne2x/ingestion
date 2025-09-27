# src/ingest_v2/pipelines/build_children.py
import json
import logging
import uuid
from typing import Any, Dict, List, Optional

from ..configs.settings import settings_v2
from ..entities.extract import extract_entities
from ..segmenter.segmenter import build_segments
from ..transcripts.normalize import normalize_to_sentences
from ..utils.ids import segment_uuid, sha1_hex
from ..utils.timefmt import floor_s, s_to_hms_ms
from ..validators.runtime import validate_child_runtime


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

def build_children_from_raw(parent: Dict[str, Any], raw: Dict[str, Any]) -> List[Dict[str, Any]]:
    """
    1) Build initial children via segmenter (15–60s, padded).
    2) Split any over-long child into sentence-aligned subchunks with small overlap.
    3) Add one per-parent summary node (node_type='summary').
    4) Extract entities, validate each node, keep only OK.
    """
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

    # 3) Entities + validation + keep/drop counters
    kept: List[Dict[str, Any]] = []
    dropped_counts = {"schema_error": 0, "timestamp_order_invalid": 0, "window_size_out_of_bounds": 0, "text_too_short": 0}

    for ch in fixed_children:
        ch["entities"] = extract_entities(ch.get("text", ""))
        ok, reason = validate_child_runtime(ch, duration_s=parent["duration_s"])
        if ok:
            kept.append(ch)
        else:
            dropped_counts[reason] = dropped_counts.get(reason, 0) + 1

    # 4) Add a higher-level summary node (best-effort)
    summary_node = _make_parent_summary_child(
        sentences, parent, char_cap=900, time_cap_s=min(60.0, parent.get("duration_s", 60.0))
    )
    if summary_node and len(summary_node.get("text", "")) >= 80:
        summary_node["entities"] = extract_entities(summary_node["text"])
        ok, reason = validate_child_runtime(summary_node, duration_s=parent["duration_s"])
        if ok:
            kept.append(summary_node)
        else:
            dropped_counts[reason] = dropped_counts.get(reason, 0) + 1

    # Logging
    if any(dropped_counts.values()):
        logging.info(
            f"[children] parent_id={parent_id} kept={len(kept)}/{len(fixed_children)} "
            f"dropped={sum(dropped_counts.values())} by_reason="
            f"{ {k: v for k, v in dropped_counts.items() if v} }"
        )
    else:
        logging.info(f"[children] parent_id={parent_id} kept_all={len(kept)}")

    return kept
