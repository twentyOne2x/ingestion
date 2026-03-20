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

    # Adaptive minimum window:
    # When callers tune SEGMENT_MIN_S high (e.g. 300s) for cost reasons, short videos
    # (or sparse transcripts) would otherwise emit 0 segments. For durations below
    # min_s, allow a smaller floor so we still index *something*.
    min_s_eff = float(min_s)
    try:
        if isinstance(duration_s, (int, float)) and duration_s > 0 and duration_s < min_s_eff:
            # Require at least ~half the clip, with a tiny absolute floor.
            #
            # NOTE: Many corpus clips are single-digit seconds long. A prior 10s floor
            # prevented indexing them entirely (no child segments), even when text was
            # non-trivial and summaries existed.
            min_s_eff = max(1.0, float(duration_s) * 0.5)
    except Exception:
        pass

    # Adaptive minimum text:
    # Many short-form videos (and some non-English clips) have concise transcripts.
    # If we require the default MIN_TEXT_CHARS=160, the segmenter can emit 0 segments,
    # which prevents the parent doc from ever being indexed.
    #
    # Keep this aligned with the runtime validator, which caps the effective minimum at 80
    # and allows operators to tune lower via env.
    min_chars_eff = min(int(getattr(settings_v2, "MIN_TEXT_CHARS", 160)), 80)
    try:
        if isinstance(duration_s, (int, float)) and duration_s > 0 and duration_s < 90:
            min_chars_eff = min(min_chars_eff, 40)
    except Exception:
        pass
    min_chars_eff = max(1, int(min_chars_eff))

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
            if (win_end - win_start) >= min_s_eff and buf[-1].rstrip().endswith((".", "?", "!")):
                break
            j += 1

        if (win_end - win_start) >= min_s_eff:
            start = max(0.0, win_start - pad)
            end = min(duration_s, win_end + pad)
            text = " ".join(x["text"] for x in sentences[i:j+1]).strip()

            # Match validator semantics: questions must still have a minimal length.
            if len(text) >= min_chars_eff or (text.endswith("?") and len(text) >= 20):
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
