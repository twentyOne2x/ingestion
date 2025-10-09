from __future__ import annotations
import argparse
import json
import logging
from pathlib import Path
from typing import Dict, List, Tuple, Optional

import numpy as np
import librosa
from resemblyzer import VoiceEncoder, preprocess_wav

from src.ingest_v2.configs.settings import settings_v2

DEFAULT_LIB_DIR = Path(settings_v2.VOICE_LIBRARY_DIR)
DEFAULT_LIB_PATH = DEFAULT_LIB_DIR / "library.json"

def _load_library(p: Path) -> Dict[str, List[float]]:
    if p.exists():
        return json.loads(p.read_text(encoding="utf-8"))
    return {}

def _save_library(p: Path, lib: Dict[str, List[float]]) -> None:
    p.parent.mkdir(parents=True, exist_ok=True)
    p.write_text(json.dumps(lib, ensure_ascii=False, indent=2), encoding="utf-8")

def _embed_spans(audio_path: Path, spans: List[Tuple[float, float]]) -> Optional[List[float]]:
    wav, sr = librosa.load(str(audio_path), sr=None, mono=True)  # mp3/m4a/wav via audioread/ffmpeg
    enc = VoiceEncoder()
    vecs = []
    for (st, et) in spans:
        st_i = max(0, int(st * sr))
        et_i = min(len(wav), int(et * sr))
        if et_i - st_i < int(0.4 * sr):  # skip too-short clips
            continue
        seg = wav[st_i:et_i]
        seg_pre = preprocess_wav(seg, source_sr=sr)
        if len(seg_pre) < int(0.4 * 16000):
            continue
        v = enc.embed_utterance(seg_pre).astype("float32")
        vecs.append(v)
    if not vecs:
        return None
    mean = np.mean(np.stack(vecs, axis=0), axis=0)
    mean = mean / (np.linalg.norm(mean) + 1e-12)
    return mean.astype("float32").tolist()

def enroll_from_file(name: str, audio: Path, spans: List[Tuple[float,float]], lib_path: Path = DEFAULT_LIB_PATH):
    lib = _load_library(lib_path)
    vec = _embed_spans(audio, spans)
    if not vec:
        raise SystemExit("No valid spans produced an embedding. Increase span length or pick different ranges.")
    lib[name] = vec
    _save_library(lib_path, lib)
    print(f"Enrolled '{name}' with {len(spans)} span(s) into {lib_path}")

def enroll_from_video(name: str, json_path: Path, audio: Optional[Path], speaker: str, min_seg: float, lib_path: Path = DEFAULT_LIB_PATH):
    obj = json.loads(json_path.read_text(encoding="utf-8"))
    # Accept both our normalized shape and raw AssemblyAI-like
    segs = obj["segments"] if isinstance(obj, dict) and "segments" in obj else obj
    spans = []
    for s in segs:
        spk = s.get("speaker") or s.get("speaker_label") or "S1"
        if spk != speaker:
            continue
        st = s.get("start"); et = s.get("end")
        if st is None or et is None:
            continue
        st = float(st) / (1000.0 if st > 1000 and et > 1000 else 1.0)  # tolerate ms inputs
        et = float(et) / (1000.0 if st > 1000 and et > 1000 else 1.0)
        if et - st >= min_seg:
            spans.append((st, et))
    if not spans:
        raise SystemExit(f"No spans >= {min_seg:.2f}s for speaker {speaker}")

    if not audio:
        # guess sibling audio by stem
        stem = json_path.stem.replace("_diarized_content", "")
        for ext in (".mp3", ".m4a", ".wav"):
            cand = json_path.with_name(f"{stem}{ext}")
            if cand.exists():
                audio = cand
                break
        if not audio:
            raise SystemExit("Audio file not found. Pass --audio explicitly.")

    enroll_from_file(name, audio, spans, lib_path)

def list_library(lib_path: Path = DEFAULT_LIB_PATH):
    lib = _load_library(lib_path)
    if not lib:
        print(f"(empty) — {lib_path}")
        return
    print(f"{len(lib)} voice(s) in {lib_path}:")
    for k, v in lib.items():
        print(f" - {k}  (dim={len(v)})")

def remove_name(name: str, lib_path: Path = DEFAULT_LIB_PATH):
    lib = _load_library(lib_path)
    if name in lib:
        del lib[name]
        _save_library(lib_path, lib)
        print(f"Removed '{name}' from {lib_path}")
    else:
        print(f"'{name}' not in library.")

def main():
    logging.basicConfig(level=logging.INFO)
    ap = argparse.ArgumentParser(prog="voice-lib")
    sub = ap.add_subparsers(dest="cmd", required=True)

    s1 = sub.add_parser("enroll-from-file", help="Enroll a name from an audio file and time spans")
    s1.add_argument("--name", required=True)
    s1.add_argument("--audio", required=True)
    s1.add_argument("--span", action="append", default=[], help="start:end in seconds (e.g. 12.5:42.0). May repeat.")
    s1.add_argument("--lib", default=str(DEFAULT_LIB_PATH))

    s2 = sub.add_parser("enroll-from-video", help="Enroll a name from a diarized JSON + sibling audio")
    s2.add_argument("--name", required=True)
    s2.add_argument("--json", required=True)
    s2.add_argument("--speaker", default="S1")
    s2.add_argument("--min-seg", type=float, default=0.8)
    s2.add_argument("--audio", default=None)
    s2.add_argument("--lib", default=str(DEFAULT_LIB_PATH))

    s3 = sub.add_parser("list", help="List names in the library")
    s3.add_argument("--lib", default=str(DEFAULT_LIB_PATH))

    s4 = sub.add_parser("remove", help="Remove a name from the library")
    s4.add_argument("--name", required=True)
    s4.add_argument("--lib", default=str(DEFAULT_LIB_PATH))

    args = ap.parse_args()

    if args.cmd == "enroll-from-file":
        spans = []
        for sp in args.span:
            try:
                a, b = sp.split(":")
                spans.append((float(a), float(b)))
            except Exception:
                raise SystemExit(f"Bad --span '{sp}', expected start:end (seconds)")
        enroll_from_file(args.name, Path(args.audio), spans, Path(args.lib))
    elif args.cmd == "enroll-from-video":
        enroll_from_video(args.name, Path(args.json), Path(args.audio) if args.audio else None, args.speaker, args.min_seg, Path(args.lib))
    elif args.cmd == "list":
        list_library(Path(args.lib))
    elif args.cmd == "remove":
        remove_name(args.name, Path(args.lib))

if __name__ == "__main__":
    main()
