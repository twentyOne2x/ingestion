import logging
from typing import Dict, Any, List
from ..transcripts.normalize import normalize_to_sentences
from ..segmenter.segmenter import build_segments
from ..entities.extract import extract_entities
from ..validators.runtime import validate_child_runtime

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

def build_children_from_raw(parent: Dict[str, Any], raw: Dict[str, Any]) -> List[Dict[str, Any]]:
    sentences = normalize_to_sentences(raw)
    parent_id = parent["parent_id"]
    document_type = parent["document_type"]
    clip_base = f"https://www.youtube.com/watch?v={parent_id}" if document_type == "youtube_video" else None
    chapter_lookup = _chapters_lookup([c if isinstance(c, dict) else c.dict() for c in (parent.get("chapters") or [])])
    children = build_segments(
        sentences=sentences,
        duration_s=parent["duration_s"],
        parent_id=parent_id,
        document_type=document_type,
        clip_base_url=clip_base,
        chapter_lookup=chapter_lookup,
        language=parent.get("language", "en"),
    )
    for ch in children:
        ch["entities"] = extract_entities(ch["text"])
        validate_child_runtime(ch, duration_s=parent["duration_s"])
    return children
