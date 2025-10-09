from __future__ import annotations

from typing import Any, Dict, List, Optional

from .text import ms_to_seconds


def is_trivial_raw_segment(segment: Dict[str, Any]) -> bool:
    no_text = not (segment.get("text") or "").strip()
    no_words = not segment.get("words")
    start_none = segment.get("start") is None
    end_none = segment.get("end") is None
    return no_text and no_words and start_none and end_none


def looks_like_single_empty_file(obj: Any) -> bool:
    try:
        if not isinstance(obj, list) or len(obj) != 1:
            return False
        segment = obj[0]
        return (
            isinstance(segment, dict)
            and (segment.get("text") in ("", None))
            and segment.get("start") is None
            and segment.get("end") is None
            and isinstance(segment.get("words"), list)
            and len(segment["words"]) == 0
        )
    except Exception:
        return False


def convert_assemblyai_json_to_raw(obj: Any) -> Dict[str, Any]:
    """
    Normalize AssemblyAI-like payloads into:
        {"segments": [{"start": float|None, "end": float|None, "speaker": str, "text": str, "words": [...]?}, ...]}
    """
    segments_in: Optional[List[Dict[str, Any]]] = None

    if isinstance(obj, dict):
        if "segments" in obj and isinstance(obj["segments"], list):
            segments_in = obj["segments"]
        elif "utterances" in obj and isinstance(obj["utterances"], list):
            segments_in = []
            for utterance in obj["utterances"]:
                segments_in.append(
                    {
                        "start": utterance.get("start"),
                        "end": utterance.get("end"),
                        "speaker": utterance.get("speaker") or utterance.get("speaker_label"),
                        "text": utterance.get("text"),
                        "words": utterance.get("words") or [],
                    }
                )
    elif isinstance(obj, list):
        segments_in = obj

    if not isinstance(segments_in, list):
        raise ValueError("Expected top-level list or dict with 'segments'/'utterances'")

    normalized_segments: List[Dict[str, Any]] = []
    for segment in segments_in:
        if is_trivial_raw_segment(segment):
            continue

        words_in = segment.get("words")
        out_words = None
        if isinstance(words_in, list) and words_in:
            out_words = []
            for word in words_in:
                text = (word.get("text") or "").strip()
                if not text:
                    continue
                out_words.append(
                    {
                        "text": text,
                        "start": ms_to_seconds(word.get("start")),
                        "end": ms_to_seconds(word.get("end")),
                        "speaker": word.get("speaker") or segment.get("speaker") or "S1",
                    }
                )

        start_s = ms_to_seconds(segment.get("start"))
        end_s = ms_to_seconds(segment.get("end"))
        text = (segment.get("text") or "").strip()

        if start_s is None and end_s is None and not text and not out_words:
            continue

        normalized_segments.append(
            {
                "start": start_s,
                "end": end_s,
                "speaker": segment.get("speaker") or segment.get("speaker_label") or "S1",
                "text": text,
                **({"words": out_words} if out_words is not None else {}),
            }
        )

    return {"segments": normalized_segments}
