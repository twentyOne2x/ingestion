import json, sys
from pathlib import Path
from typing import Any, Dict, List

def ms_to_s(x: Any) -> Any:
    try: return float(x) / 1000.0
    except Exception: return x

def convert(in_path: Path) -> Dict[str, Any]:
    data = json.loads(in_path.read_text(encoding="utf-8"))
    if isinstance(data, dict) and "segments" in data:
        segs = data["segments"]
    elif isinstance(data, list):
        segs = data
    else:
        raise ValueError("Input must be a list or an object with 'segments'")

    out: List[Dict[str, Any]] = []
    for seg in segs:
        words = seg.get("words")
        out_words = None
        if isinstance(words, list):
            out_words = [{"text": w.get("text",""),
                          "start": ms_to_s(w.get("start")),
                          "end": ms_to_s(w.get("end")),
                          "speaker": w.get("speaker") or seg.get("speaker") or "S1"} for w in words]
        out.append({
            "start": ms_to_s(seg.get("start")),
            "end": ms_to_s(seg.get("end")),
            "speaker": seg.get("speaker") or seg.get("speaker_label") or "S1",
            "text": seg.get("text","").strip(),
            **({"words": out_words} if out_words is not None else {}),
        })
    return {"segments": out}

def main():
    if len(sys.argv) < 3:
        print("usage: python -m src.ingest_v2.transcripts.assemblyai_json_ms_to_rawjson <in.json> <out.json>")
        sys.exit(1)
    inp = Path(sys.argv[1]); outp = Path(sys.argv[2])
    raw = convert(inp)
    outp.parent.mkdir(parents=True, exist_ok=True)
    outp.write_text(json.dumps(raw, ensure_ascii=False, indent=2), encoding="utf-8")
    print(f"wrote {outp} with {len(raw['segments'])} segments")

if __name__ == "__main__":
    main()
