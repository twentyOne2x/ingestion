from typing import List, Dict, Any
from pathlib import Path
import json, re

from collections import Counter

_JUDO_RE = re.compile(r"\bJudo\b", re.IGNORECASE)

def apply_term_fixes(text: str) -> str:
    """
    Deterministic transcript normalization for common ASR confusions.

    Keep this small and explicit; this runs on every ingest and affects:
    - chunk text embeddings (retrieval)
    - router enrichment prompts (metadata generation)
    """
    s = (text or "").strip()
    if not s:
        return ""
    # Crypto-specific correction: ASR often hears "Jito" as "Judo".
    s = _JUDO_RE.sub("Jito", s)
    return s

def normalize_to_sentences(raw: Dict[str, Any], default_speaker: str = "S1") -> List[Dict[str, Any]]:
    segments = raw.get("segments") or raw.get("caption_lines") or []
    sentences: List[Dict[str, Any]] = []

    for seg in segments:
        start = float(seg.get("start", 0))
        end = float(seg.get("end", 0))
        text = apply_term_fixes(seg.get("text") or "")
        base_spk = seg.get("speaker") or seg.get("speaker_label") or default_speaker
        parts = re.split(r"(?<=[\.?!])\s+", text)
        if not parts:
            continue

        words = seg.get("words")
        if isinstance(words, list) and words:
            idx = 0
            for sent in parts:
                n = max(1, len(sent.split()))
                w_slice = words[idx: idx + n]
                s_start = float(w_slice[0]["start"]) if w_slice else start
                s_end = float(w_slice[-1]["end"]) if w_slice else min(end, s_start + 4.0)

                # majority speaker over the slice; fallback to seg speaker
                spks = [w.get("speaker") for w in w_slice if w.get("speaker")]
                spk = Counter(spks).most_common(1)[0][0] if spks else base_spk

                sentences.append({"start_s": s_start, "end_s": s_end, "text": sent, "speaker": spk})
                idx += n
        else:
            if len(parts) == 1:
                sentences.append({"start_s": start, "end_s": end, "text": parts[0], "speaker": base_spk})
            else:
                span = (end - start) / len(parts)
                for i, sent in enumerate(parts):
                    s_start = start + i * span
                    s_end = min(end, s_start + span)
                    sentences.append({"start_s": s_start, "end_s": s_end, "text": sent, "speaker": base_spk})

    sentences = [s for s in sentences if s["text"].strip()]
    sentences.sort(key=lambda x: (x["start_s"], x["end_s"]))
    return sentences

def write_normalized_jsonl(parent_id: str, sentences: List[Dict[str, Any]], out_dir: Path = Path("transcripts/normalized")) -> Path:
    out_dir.mkdir(parents=True, exist_ok=True)
    path = out_dir / f"{parent_id}.jsonl"
    with path.open("w", encoding="utf-8") as f:
        for s in sentences:
            f.write(json.dumps(s, ensure_ascii=False) + "\\n")
    return path
