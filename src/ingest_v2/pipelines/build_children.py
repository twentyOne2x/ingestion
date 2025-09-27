# src/ingest_v2/pipelines/build_children.py

import logging
from typing import Dict, Any, List

from ..transcripts.normalize import normalize_to_sentences
from ..segmenter.segmenter import build_segments
from ..entities.extract import extract_entities
from ..validators.runtime import validate_child_runtime


def _chapters_lookup(chapters):
    if not chapters:
        return None
    # Ensure dicts
    chs = [(c if isinstance(c, dict) else c.dict()) for c in chapters]
    chs = [c for c in chs if "start_s" in c and "title" in c]
    if not chs:
        return None
    chs.sort(key=lambda x: x["start_s"])

    def lookup(s: float):
        last = None
        for ch in chs:
            if ch["start_s"] <= s:
                last = ch["title"]
            else:
                break
        return last

    return lookup


def build_children_from_raw(parent: Dict[str, Any], raw: Dict[str, Any]) -> List[Dict[str, Any]]:
    """
    Build child segments from raw transcript-ish input.
    Invalid children are DROPPED (non-fatal) with per-reason counts logged.
    """
    sentences = normalize_to_sentences(raw or {})
    if not sentences:
        logging.info(f"[children] parent_id={parent.get('parent_id')} no sentences after normalization")
        return []

    parent_id = parent["parent_id"]
    document_type = parent["document_type"]
    duration_s = float(parent.get("duration_s", 0.0) or 0.0)

    clip_base = None
    if document_type == "youtube_video":
        clip_base = f"https://www.youtube.com/watch?v={parent_id}"

    chapter_lookup = _chapters_lookup(parent.get("chapters"))

    # Build raw children (may include padding > max)
    children = build_segments(
        sentences=sentences,
        duration_s=duration_s,
        parent_id=parent_id,
        document_type=document_type,
        clip_base_url=clip_base,
        chapter_lookup=chapter_lookup,
        language=parent.get("language", "en"),
    )

    if not children:
        logging.info(f"[children] parent_id={parent_id} segmenter emitted 0 children")
        return []

    # Enrich + validate non-fatally
    valid_children: List[Dict[str, Any]] = []
    dropped_counts: Dict[str, int] = {}

    for ch in children:
        # lightweight enrichment first so validator can use text
        ch["entities"] = extract_entities(ch.get("text", "") or "")

        ok, reason = validate_child_runtime(ch, duration_s=duration_s)
        if not ok:
            dropped_counts[reason] = dropped_counts.get(reason, 0) + 1
            continue
        valid_children.append(ch)

    total = len(children)
    kept = len(valid_children)
    dropped_total = total - kept
    if dropped_total:
        logging.info(
            f"[children] parent_id={parent_id} kept={kept}/{total} "
            f"dropped={dropped_total} by_reason={dropped_counts}"
        )
    else:
        logging.info(f"[children] parent_id={parent_id} kept_all={kept}")

    return valid_children
