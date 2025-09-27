from __future__ import annotations
import hashlib
import json
import logging
import os
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import numpy as np

# Runtime deps (install):
#   pip install resemblyzer==0.1.3a1 librosa==0.10.2.post1 soundfile==0.12.1 numpy
import soundfile as sf
import librosa
from resemblyzer import VoiceEncoder, preprocess_wav


# ────────────────────────────────────────────────────────────────────────────────
# Public API
# ────────────────────────────────────────────────────────────────────────────────

def embed_speakers_from_audio(
    audio_path: str,
    raw: Dict[str, Any],
    meta: Optional[Dict[str, Any]] = None,
) -> Dict[str, List[float]]:
    """
    Returns { "S1": [f32,...], "S2": [...], ... } averaged per speaker across all their segments.

    - Uses diarized `raw["segments"]` (start/end in seconds, speaker label like "S1")
    - Loads the single audio file at `audio_path`
    - Computes per-segment speaker embeddings with Resemblyzer, then mean-pools per speaker
    - Caches results to disk so we don't re-encode
    """

    audio_p = Path(audio_path)
    if not audio_p.exists():
        logging.warning("[tier2/aai_backend] audio missing at %s", audio_p)
        return {}

    # Cache dir & key
    cache_dir = Path(os.getenv("SPEAKER_EMBED_CACHE_DIR", "pipeline_storage_v2/speaker_embeds"))
    cache_dir.mkdir(parents=True, exist_ok=True)

    vid = (meta or {}).get("video_id") or _fallback_video_key(audio_p, raw)
    cache_file = cache_dir / f"{vid}.json"

    # Short-circuit if cache exists
    cached = _try_read_json(cache_file)
    if cached:
        logging.info("[tier2/aai_backend] cache hit for vid=%s (%s)", vid, cache_file)
        # validate vector shape minimally
        if _looks_like_embed_map(cached):
            return cached

    # Build time ranges per speaker from your diarized segments
    segs = (raw or {}).get("segments") or []
    if not segs:
        logging.info("[tier2/aai_backend] no segments in raw; nothing to embed")
        return {}

    spans_by_spk: Dict[str, List[Tuple[float, float]]] = {}
    for s in segs:
        spk = s.get("speaker") or "S1"
        st = _to_f(s.get("start"))
        et = _to_f(s.get("end"))
        if st is None or et is None or et <= st:
            continue
        spans_by_spk.setdefault(spk, []).append((st, et))

    if not spans_by_spk:
        logging.info("[tier2/aai_backend] no valid (start,end) spans; nothing to embed")
        return {}

    # Load full waveform once (mono float32)
    # soundfile gives (samples, channels) or (samples,)
    wav, sr = librosa.load(str(audio_p), sr=None, mono=True)  # handles mp3/m4a via audioread/ffmpeg
    if wav.ndim == 2:
        # mixdown to mono
        wav = wav.mean(axis=1)

    # Init encoder once
    encoder = VoiceEncoder()

    # Encode per span, then mean-pool per speaker
    out: Dict[str, List[float]] = {}
    for spk, spans in spans_by_spk.items():
        vecs: List[np.ndarray] = []
        for (st, et) in _merge_small_gaps(spans):
            start_i = max(0, int(st * sr))
            end_i = min(len(wav), int(et * sr))
            if end_i - start_i < int(0.4 * sr):
                # too short for a good embedding, skip
                continue

            clip = wav[start_i:end_i]
            # Resemblyzer expects either a file path or a preprocessed wav;
            # we'll preprocess the raw mono segment
            try:
                # preprocess_wav can accept a floating array + sr via librosa interface,
                # but the simplest here: assume 16k is fine (Resemblyzer does resample internally)
                # Convert to embedding directly:
                #   NOTE: VoiceEncoder.embed_utterance expects 16k mono np.ndarray
                #   preprocess_wav will handle loudness normalization & VAD by default.
                seg_pre = preprocess_wav(clip, source_sr=sr)
                if len(seg_pre) < int(0.4 * 16000):
                    continue
                emb = encoder.embed_utterance(seg_pre)
                vecs.append(emb.astype("float32"))
            except Exception as e:
                logging.warning("[tier2/aai_backend] embedding failed for %s span (%.2f, %.2f): %s", spk, st, et, e)

        if vecs:
            mean_vec = np.mean(np.stack(vecs, axis=0), axis=0)
            # L2 normalize
            norm = np.linalg.norm(mean_vec) + 1e-12
            out[spk] = (mean_vec / norm).astype("float32").tolist()

    # Cache to disk
    try:
        if out:
            cache_file.write_text(json.dumps(out), encoding="utf-8")
            logging.info("[tier2/aai_backend] wrote cache for vid=%s at %s", vid, cache_file)
    except Exception as e:
        logging.warning("[tier2/aai_backend] failed writing cache: %s", e)

    return out


# ────────────────────────────────────────────────────────────────────────────────
# Helpers
# ────────────────────────────────────────────────────────────────────────────────

def _try_read_json(p: Path) -> Optional[Dict[str, Any]]:
    try:
        if p.exists():
            obj = json.loads(p.read_text(encoding="utf-8"))
            return obj
    except Exception as e:
        logging.warning("[tier2/aai_backend] failed to read cache %s: %s", p, e)
    return None

def _looks_like_embed_map(obj: Any) -> bool:
    if not isinstance(obj, dict):
        return False
    for k, v in obj.items():
        if not isinstance(k, str):
            return False
        if not (isinstance(v, list) and all(isinstance(x, (int, float)) for x in v)):
            return False
    return True

def _to_f(x: Any) -> Optional[float]:
    try:
        if x is None:
            return None
        return float(x)
    except Exception:
        return None

def _merge_small_gaps(spans: List[Tuple[float, float]], gap_s: float = 0.25) -> List[Tuple[float, float]]:
    """
    Merge tiny gaps between spans to form fewer, longer chunks (better embeddings).
    Assumes spans are in arbitrary order.
    """
    if not spans:
        return []
    spans = sorted(spans)
    out = [list(spans[0])]
    for (st, et) in spans[1:]:
        prev = out[-1]
        if st - prev[1] <= gap_s:
            prev[1] = max(prev[1], et)
        else:
            out.append([st, et])
    return [(a, b) for a, b in out]

def _fallback_video_key(audio_p: Path, raw: Dict[str, Any]) -> str:
    # Stable cache key if meta["video_id"] is unavailable:
    # hash of path + first/last timestamps so moving files doesn't explode the cache.
    segs = (raw or {}).get("segments") or []
    first = str(segs[0].get("start")) if segs else "na"
    last = str(segs[-1].get("end")) if segs else "na"
    s = f"{audio_p.as_posix()}::{first}::{last}"
    return hashlib.sha1(s.encode("utf-8")).hexdigest()[:16]
