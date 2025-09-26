from typing import List, Dict, Any, Optional
from ..configs.settings import settings_v2
from ..utils.timefmt import s_to_hms_ms, floor_s
from ..utils.ids import segment_uuid, sha1_hex
import json

def advance_by_stride(sentences: List[Dict[str, Any]], i: int, stride_s: float) -> int:
    base = sentences[i]["start_s"]
    target = base + stride_s
    j = i + 1
    while j < len(sentences) and sentences[j]["start_s"] < target:
        j += 1
    return j

def build_segments(
    sentences: List[Dict[str, Any]],
    duration_s: float,
    parent_id: str,
    document_type: str,
    clip_base_url: Optional[str] = None,
    chapter_lookup = None,
    language: str = "en",
) -> List[Dict[str, Any]]:
    i, segs = 0, []
    min_s, max_s, stride, pad = (
        settings_v2.SEGMENT_MIN_S,
        settings_v2.SEGMENT_MAX_S,
        settings_v2.SEGMENT_STRIDE_S,
        settings_v2.SEGMENT_PAD_S,
    )

    while i < len(sentences):
        win_start = sentences[i]["start_s"]
        j = i
        win_end = win_start
        buf = []
        speaker = sentences[i].get("speaker")

        while j < len(sentences) and (win_end - win_start) < max_s:
            s = sentences[j]
            buf.append(s["text"])
            win_end = s["end_s"]
            if (win_end - win_start) >= min_s and buf[-1].rstrip().endswith((".", "?", "!")):
                break
            j += 1

        if (win_end - win_start) >= min_s:
            start = max(0.0, win_start - pad)
            end = min(duration_s, win_end + pad)
            text = " ".join(x["text"] for x in sentences[i:j+1]).strip()

            if len(text) >= settings_v2.MIN_TEXT_CHARS or text.endswith("?"):
                start_hms = s_to_hms_ms(start)
                end_hms = s_to_hms_ms(end)
                seg_id = segment_uuid(parent_id, start, end)
                clip_url = f"{clip_base_url}&t={floor_s(start)}s" if clip_base_url else None

                payload = {
                    "node_type": "child",
                    "segment_id": seg_id,
                    "parent_id": parent_id,
                    "document_type": document_type,
                    "text": text,
                    "start_s": start,
                    "end_s": end,
                    "start_hms": start_hms,
                    "end_hms": end_hms,
                    "clip_url": clip_url,
                    "speaker": speaker,
                    "entities": [],
                    "chapter": None,
                    "language": language,
                    "confidence_asr": None,
                    "has_music": False,
                    "flags": [],
                    "rights": "public_reference_only",
                    "ingest_version": 2,
                }
                raw_bytes = json.dumps({"text": text, "start": start, "end": end}, sort_keys=True).encode("utf-8")
                payload["source_hash"] = sha1_hex(raw_bytes)

                if chapter_lookup:
                    payload["chapter"] = chapter_lookup(start)

                segs.append(payload)

        i = advance_by_stride(sentences, i, stride)

    return segs
