# src/ingest_v2/pipelines/build_children.py
import logging
import uuid
from typing import Dict, Any, List, Tuple, Optional

from ..configs.settings import settings_v2
from ..transcripts.normalize import normalize_to_sentences
from ..segmenter.segmenter import build_segments
from ..entities.extract import extract_entities
from ..validators.runtime import validate_child_runtime
from ..utils.timefmt import s_to_hms_ms, floor_s
from ..utils.ids import segment_uuid, sha1_hex
import json


def _chapters_lookup(chapters):
    if not chapters:
        return None
    chapters = sorted(chapters, key=lambda x: x["start_s"])
    def lookup(s: float):
        last = None
        for ch in chapters:
            if ch["start_s"] <= s:
                last = ch["title"]
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
    out = []
    for s in sentences:
        ss = float(s.get("start_s", 0.0))
        ee = float(s.get("end_s", ss))
        mid = (ss + ee) / 2.0
        if start_s - 1e-6 <= mid <= end_s + 1e-6:
            out.append(s)
    return out


def _split_by_sentences_with_overlap(sentences, max_window_s, overlap_s, parent, clip_base_url):
    children = []
    i = 0; n = len(sentences)
    while i < n:
        win_start = float(sentences[i]["start_s"])
        j = i
        while j < n:
            seg_start = float(sentences[i]["start_s"])
            seg_end = float(sentences[j]["end_s"])
            dur = seg_end - seg_start
            if dur > max_window_s and j > i:
                j -= 1; break
            if dur > max_window_s and j == i:
                cut_end = seg_start + max_window_s
                text = sentences[i]["text"].strip()
                start_hms = s_to_hms_ms(seg_start); end_hms = s_to_hms_ms(cut_end)
                seg_id = segment_uuid(parent["parent_id"], seg_start, cut_end)
                clip_url = f"{clip_base_url}&t={floor_s(seg_start)}s" if clip_base_url else None
                payload = {
                    "node_type": "child",
                    "segment_id": seg_id,
                    "parent_id": parent["parent_id"],
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
                raw_bytes = json.dumps({"text": text, "start": seg_start, "end": cut_end}, sort_keys=True).encode("utf-8")
                payload["source_hash"] = sha1_hex(raw_bytes)
                children.append(payload)
                i += 1
                break
            j += 1
        if j >= n: j = n - 1

        seg_start = float(sentences[i]["start_s"])
        seg_end = float(sentences[j]["end_s"])
        text = " ".join(s["text"].strip() for s in sentences[i:j+1] if s.get("text"))
        start_hms = s_to_hms_ms(seg_start); end_hms = s_to_hms_ms(seg_end)
        seg_id = segment_uuid(parent["parent_id"], seg_start, seg_end)
        clip_url = f"{clip_base_url}&t={floor_s(seg_start)}s" if clip_base_url else None
        payload = {
            "node_type": "child",
            "segment_id": seg_id,
            "parent_id": parent["parent_id"],
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
            "ingest_version": 2,
        }
        raw_bytes = json.dumps({"text": text, "start": seg_start, "end": seg_end}, sort_keys=True).encode("utf-8")
        payload["source_hash"] = sha1_hex(raw_bytes)
        children.append(payload)

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
    Cheap extractive summary: gather the earliest sentences until either
    time or char cap is reached. Mark as node_type='summary'.
    """
    if not sentences:
        return None

    acc: List[str] = []
    start_s = float(sentences[0].get("start_s", 0.0))
    end_s_limit = start_s + time_cap_s
    for s in sentences:
        ss = float(s.get("start_s", 0.0))
        ee = float(s.get("end_s", ss))
        if ss > end_s_limit:
            break
        acc.append(s.get("text", "").strip())
        if sum(len(x) for x in acc) >= char_cap:
            break

    if not acc:
        return None

    text = " ".join(acc).strip()
    # Keep timestamps within limits so standard validation can pass
    end_s = min(end_s_limit, float(sentences[-1].get("end_s", end_s_limit)))

    node = {
        "parent_id": parent["parent_id"],
        "document_type": parent["document_type"],
        "language": parent.get("language", "en"),
        "segment_id": str(uuid.uuid4()),
        "start_s": start_s,
        "end_s": end_s,
        "text": text,
        "speaker": None,
        "chapter": None,
        "entities": [],
        "clip_url": None,
        "node_type": "summary",
    }
    return node


def build_children_from_raw(parent: Dict[str, Any], raw: Dict[str, Any]) -> List[Dict[str, Any]]:
    """
    1) Build initial children via your existing segmenter.
    2) For any over-long child, split into sentence-aligned subchunks with small overlap.
    3) Add one higher-level per-parent summary node.
    """
    # Sentence list used for both (a) fixing long windows and (b) summary.
    sentences = normalize_to_sentences(raw)
    if not sentences:
        logging.info(f"[children] parent_id={parent.get('parent_id')} no sentences after normalization")
        return []

    parent_id = parent["parent_id"]
    document_type = parent["document_type"]
    clip_base = f"https://www.youtube.com/watch?v={parent_id}" if document_type == "youtube_video" else None
    chapter_lookup = _chapters_lookup([c if isinstance(c, dict) else c.dict() for c in (parent.get("chapters") or [])])

    # 1) Build with existing segmenter
    children = build_segments(
        sentences=sentences,
        duration_s=parent["duration_s"],
        parent_id=parent_id,
        document_type=document_type,
        clip_base_url=clip_base,
        chapter_lookup=chapter_lookup,
        language=parent.get("language", "en"),
    )

    # 2) Post-process: split those exceeding the cap
    max_s = float(getattr(settings_v2, "SEGMENT_MAX_S", 60.0))
    pad_s = float(getattr(settings_v2, "SEGMENT_PAD_S", 1.5))
    tol_s = float(getattr(settings_v2, "SEGMENT_TOLERANCE_S", 0.75))
    overlap_s = float(getattr(settings_v2, "SEGMENT_OVERLAP_S", 3.0))
    upper_with_pad = max_s + (2.0 * pad_s) + tol_s

    fixed_children: List[Dict[str, Any]] = []
    dropped_counts = {"timestamp_order_invalid": 0, "window_size_out_of_bounds": 0, "text_too_short": 0}

    for ch in children:
        seg_len = float(ch["end_s"]) - float(ch["start_s"])
        if seg_len <= upper_with_pad:
            fixed_children.append(ch)
            continue

        # Need to split this child into smaller children using sentence boundaries inside its window
        slice_sents = _sentences_in_range(sentences, ch["start_s"], ch["end_s"])
        parent_common = {
            "parent_id": parent_id,
            "document_type": document_type,
            "language": ch.get("language") or parent.get("language", "en"),
            "clip_url": ch.get("clip_url"),  # will be recomputed at validation time if you do that downstream
        }
        subchildren = subchildren = _split_by_sentences_with_overlap(slice_sents, max_s, overlap_s, parent, clip_base)
        logging.info(
            f"[children] parent_id={parent_id} split_long segment_id={ch.get('segment_id')} "
            f"orig_len={seg_len:.3f}s -> {len(subchildren)} subchunks"
        )
        fixed_children.extend(subchildren)

    # 3) Entities + validation + keep/drop counters
    kept: List[Dict[str, Any]] = []
    for ch in fixed_children:
        ch["entities"] = extract_entities(ch["text"])
        try:
            validate_child_runtime(ch, duration_s=parent["duration_s"])
            kept.append(ch)
        except AssertionError as e:
            reason = str(e)
            if "timestamp order invalid" in reason:
                dropped_counts["timestamp_order_invalid"] += 1
            elif "window size out of bounds" in reason:
                dropped_counts["window_size_out_of_bounds"] += 1
            elif "text too short" in reason:
                dropped_counts["text_too_short"] += 1
            else:
                dropped_counts["window_size_out_of_bounds"] += 1
        except Exception:
            # treat any other error as drop
            dropped_counts["window_size_out_of_bounds"] += 1

    # 4) Add a higher-level summary child (best-effort)
    summary_node = _make_parent_summary_child(sentences, parent, char_cap=900, time_cap_s=min(60.0, parent.get("duration_s", 60.0)))
    if summary_node and len(summary_node["text"]) >= 80:
        summary_node["entities"] = extract_entities(summary_node["text"])
        try:
            # Let it go through validator too (it fits <=60s)
            validate_child_runtime(summary_node, duration_s=parent["duration_s"])
            kept.append(summary_node)
        except Exception:
            # If you prefer to always include summaries regardless of timing, you can bypass validation here.
            pass

    # Logging
    if any(dropped_counts.values()):
        logging.info(
            f"[children] parent_id={parent_id} kept={len(kept)}/{len(fixed_children)} "
            f"dropped={sum(dropped_counts.values())} by_reason={{{k: v for k, v in dropped_counts.items() if v}}}"
        )
    else:
        logging.info(f"[children] parent_id={parent_id} kept_all={len(kept)}")

    return kept
