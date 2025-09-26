#!/bin/bash

# Helper: write a file safely (create dirs, overwrite by default).
# Supports either:
#   write path 'literal content'
# or:
#   write path <<'EOF'
#   multiline content
#   EOF
write() {
  local path="$1"; shift
  mkdir -p "$(dirname "$path")"
  if [ $# -gt 0 ]; then
    # Content passed as a single argument (with newlines allowed).
    # Preserve bytes exactly and ensure trailing newline.
    printf "%s" "$1" > "$path"
    # add a newline if missing
    tail -c1 "$path" 2>/dev/null | read -r _ || echo >> "$path"
  else
    # Content provided via stdin (heredoc).
    cat > "$path"
  fi
}

# Backward-compat alias (same semantics as write)
w() {
  write "$@"
}


# # ────────────────────────────────────────────────────────────────────────────────
# Directory skeleton
# # ────────────────────────────────────────────────────────────────────────────────
mkdir -p \
  src/ingest_v2 \
  src/ingest_v2/configs \
  src/ingest_v2/entities \
  src/ingest_v2/pipelines \
  src/ingest_v2/schemas \
  src/ingest_v2/segmenter \
  src/ingest_v2/sources \
  src/ingest_v2/tests/fixtures \
  src/ingest_v2/transcripts \
  src/ingest_v2/utils \
  src/ingest_v2/validators \
  scripts

# # ────────────────────────────────────────────────────────────────────────────────
# Top-level files
# # ────────────────────────────────────────────────────────────────────────────────
write src/ingest_v2/__init__.py \
'__all__ = []'

write src/ingest_v2/README_v2.md \
'# Ingestion v2 (Parent/Child, time-based segments)

- Emits **Parent** (1 per asset) + **Child** segments (15–60s, stride=5s, padded) with timestamped clip URLs.
- Upserts children into Pinecone under namespaces:
  - `videos` (YouTube)
  - `streams` (Pump.fun)
- Reuses v1 Pinecone index but isolates via namespaces.

## Quickstart
```bash
export PINECONE_API_KEY=...
export PINECONE_INDEX_NAME=icmfyi
export OPENAI_API_KEY=...
python -m src.ingest_v2.pipelines.run_all --doc-type youtube_video --backfill-days 60
```'

# ────────────────────────────────────────────────────────────────────────────────
# configs/settings.py
# ────────────────────────────────────────────────────────────────────────────────
write src/ingest_v2/configs/__init__.py ''

write src/ingest_v2/configs/settings.py \
'import os
from dataclasses import dataclass

@dataclass(frozen=True)
class SettingsV2:
    SEGMENT_MIN_S: float = float(os.getenv("SEGMENT_MIN_S", 15))
    SEGMENT_MAX_S: float = float(os.getenv("SEGMENT_MAX_S", 60))
    SEGMENT_STRIDE_S: float = float(os.getenv("SEGMENT_STRIDE_S", 5))
    SEGMENT_PAD_S: float = float(os.getenv("SEGMENT_PAD_S", 1.5))
    MIN_TEXT_CHARS: int = int(os.getenv("MIN_TEXT_CHARS", 160))

    EMBED_MODEL: str = os.getenv("EMBED_MODEL", "text-embedding-3-small")
    EMBED_DIM: int = int(os.getenv("EMBED_DIM", "1536"))
    EMBED_PROVIDER: str = os.getenv("EMBED_PROVIDER", "openai")  # openai|sentence-transformers

    PINECONE_INDEX_NAME: str = os.getenv("PINECONE_INDEX_NAME", "icmfyi")
    NAMESPACE_VIDEOS: str = os.getenv("PINECONE_NAMESPACE_VIDEOS", "videos")
    NAMESPACE_STREAMS: str = os.getenv("PINECONE_NAMESPACE_STREAMS", "streams")

    MAX_METADATA_BYTES: int = int(os.getenv("MAX_METADATA_BYTES", 12000))
    RIGHTS_DEFAULT: str = os.getenv("RIGHTS_DEFAULT", "public_reference_only")

    BACKFILL_DAYS: int = int(os.getenv("BACKFILL_DAYS", 60))

settings_v2 = SettingsV2()'

# ────────────────────────────────────────────────────────────────────────────────
# schemas
# ────────────────────────────────────────────────────────────────────────────────
write src/ingest_v2/schemas/__init__.py ''

write src/ingest_v2/schemas/parent.py \
'from pydantic import BaseModel, Field, HttpUrl
from typing import List, Optional, Literal

DocType = Literal["youtube_video", "stream"]

class Chapter(BaseModel):
    title: str
    start_s: float = Field(ge=0)

class ParentNode(BaseModel):
    node_type: Literal["parent"] = "parent"
    parent_id: str
    document_type: DocType
    title: str
    description: Optional[str] = None
    channel_name: Optional[str] = None
    speaker_primary: Optional[str] = None
    published_at: Optional[str] = Field(default=None, description="YYYY-MM-DD")
    start_ts: Optional[str] = None
    end_ts: Optional[str] = None
    duration_s: float = 0
    url: HttpUrl
    thumbnail_url: Optional[HttpUrl] = None
    language: Optional[str] = "en"
    entities: List[str] = []
    chapters: Optional[List[Chapter]] = None
    rights: str = "public_reference_only"
    ingest_version: int = 2
    source: Literal["youtube", "pumpfun"] = "youtube"
    source_hash: str

    class Config:
        extra = "ignore"'

write src/ingest_v2/schemas/child.py \
'from pydantic import BaseModel, Field, HttpUrl
from typing import List, Optional, Literal

DocType = Literal["youtube_video", "stream"]

class ChildNode(BaseModel):
    node_type: Literal["child"] = "child"
    segment_id: str
    parent_id: str
    document_type: DocType
    text: str
    start_s: float = Field(ge=0)
    end_s: float = Field(gt=0)
    start_hms: str
    end_hms: str
    clip_url: Optional[HttpUrl] = None
    speaker: Optional[str] = None
    entities: List[str] = []
    chapter: Optional[str] = None
    language: Optional[str] = "en"
    confidence_asr: Optional[float] = Field(default=None, ge=0, le=1)
    has_music: Optional[bool] = False
    flags: List[str] = []
    rights: str = "public_reference_only"
    ingest_version: int = 2
    source_hash: str

    class Config:
        extra = "ignore"'

write src/ingest_v2/schemas/json_schemas.py \
'from .parent import ParentNode
from .child import ChildNode

PARENT_JSON_SCHEMA = ParentNode.model_json_schema()
CHILD_JSON_SCHEMA = ChildNode.model_json_schema()'

# ────────────────────────────────────────────────────────────────────────────────
# utils
# ────────────────────────────────────────────────────────────────────────────────
write src/ingest_v2/utils/__init__.py ''

write src/ingest_v2/utils/ids.py \
'import uuid
from hashlib import sha1

NAMESPACE = uuid.NAMESPACE_URL

def segment_uuid(parent_id: str, start_s: float, end_s: float) -> str:
    key = f"{parent_id}:{start_s:.3f}:{end_s:.3f}"
    return str(uuid.uuid5(NAMESPACE, key))

def sha1_hex(payload: bytes) -> str:
    return sha1(payload).hexdigest()'

write src/ingest_v2/utils/timefmt.py \
'import math

def s_to_hms_ms(s: float) -> str:
    ms = int(round((s - int(s)) * 1000))
    h = int(s) // 3600
    m = (int(s) % 3600) // 60
    sec = int(s) % 60
    return f"{h:02d}:{m:02d}:{sec:02d}.{ms:03d}"

def floor_s(s: float) -> int:
    return int(math.floor(s))'

write src/ingest_v2/utils/hashing.py \
'from datasketch import MinHash
import mmh3

def minhash_signature(text: str, n_perm: int = 64) -> MinHash:
    mh = MinHash(num_perm=n_perm)
    for token in text.split():
        mh.update(token.encode("utf-8"))
    return mh

def simhash_int(text: str, bits: int = 64) -> int:
    return mmh3.hash128(text, signed=False) & ((1 << bits) - 1)'

write src/ingest_v2/utils/batching.py \
'from typing import Iterable, List, TypeVar
T = TypeVar("T")

def chunked(items: Iterable[T], size: int) -> Iterable[List[T]]:
    buf = []
    for x in items:
        buf.append(x)
        if len(buf) >= size:
            yield buf
            buf = []
    if buf:
        yield buf'

write src/ingest_v2/utils/backoff.py \
'import time, random

def expo_backoff(attempt: int, base: float = 0.5, cap: float = 20.0):
    delay = min(cap, base * (2 ** attempt) + random.random())
    time.sleep(delay)'

write src/ingest_v2/utils/pinecone_client.py \
'import os
from typing import List, Dict, Any
from pinecone import Pinecone, ServerlessSpec

def get_index(index_name: str, dimension: int):
    pc = Pinecone(api_key=os.environ.get("PINECONE_API_KEY"))
    names = [x["name"] for x in pc.list_indexes().get("indexes", [])]
    if index_name not in names:
        pc.create_index(
            name=index_name,
            dimension=dimension,
            metric="cosine",
            spec=ServerlessSpec(cloud="aws", region="us-east-1"),
        )
    return pc.Index(index_name)

def upsert_vectors(index, namespace: str, vectors: List[Dict[str, Any]], batch_size: int = 100):
    from .batching import chunked
    for batch in chunked(vectors, batch_size):
        index.upsert(vectors=batch, namespace=namespace)'

write src/ingest_v2/utils/logging.py \
'import logging
from datetime import datetime
from pathlib import Path

def setup_logger(prefix: str = "ingest_v2"):
    logs_dir = Path("logs/txt")
    logs_dir.mkdir(parents=True, exist_ok=True)
    ts = datetime.now().strftime("%Y-%m-%d_%H-%M")
    log_path = logs_dir / f"{ts}_{prefix}.log"

    root = logging.getLogger()
    if root.hasHandlers():
        root.handlers.clear()
    root.setLevel(logging.INFO)

    fh = logging.FileHandler(log_path)
    ch = logging.StreamHandler()
    fmt = logging.Formatter("%(asctime)s - %(levelname)s - %(message)s")
    fh.setFormatter(fmt)
    ch.setFormatter(fmt)
    root.addHandler(fh)
    root.addHandler(ch)
    logging.info("********* ingest_v2 logging started *********")'

# ────────────────────────────────────────────────────────────────────────────────
# sources
# ────────────────────────────────────────────────────────────────────────────────
write src/ingest_v2/sources/__init__.py ''

write src/ingest_v2/sources/youtube.py \
'from typing import Dict, Any, Optional
import json
from pathlib import Path
from ..utils.ids import sha1_hex
from ..schemas.parent import ParentNode
from ..configs.settings import settings_v2

def _safe_get(d: Dict, *keys, default=None):
    for k in keys:
        if d is None:
            return default
        d = d.get(k)
    return d if d is not None else default

def build_parent_from_metadata(meta: Dict[str, Any]) -> ParentNode:
    raw_bytes = json.dumps(meta, sort_keys=True).encode("utf-8")
    source_hash = sha1_hex(raw_bytes)
    parent = ParentNode(
        parent_id=meta["video_id"],
        document_type="youtube_video",
        title=meta.get("title", ""),
        description=meta.get("description", ""),
        channel_name=meta.get("channel_name"),
        speaker_primary=_safe_get(meta, "speaker_primary"),
        published_at=_safe_get(meta, "published_at"),
        start_ts=None,
        end_ts=None,
        duration_s=float(meta.get("duration_s", meta.get("duration", 0)) or 0),
        url=meta["url"],
        thumbnail_url=meta.get("thumbnail_url"),
        language=meta.get("language", "en"),
        entities=meta.get("entities", []),
        chapters=meta.get("chapters"),
        rights=settings_v2.RIGHTS_DEFAULT,
        ingest_version=2,
        source="youtube",
        source_hash=source_hash,
    )
    return parent

def load_existing_youtube_raw(parent_id: str) -> Optional[Path]:
    p = Path("transcripts/raw") / f"{parent_id}_raw.json"
    return p if p.exists() else None'

write src/ingest_v2/sources/pumpfun.py \
'# Placeholder for Pump.fun stream metadata loader'

write src/ingest_v2/sources/rss.py \
'# Placeholder for RSS/feeds loader'

# ────────────────────────────────────────────────────────────────────────────────
# transcripts normalization
# ────────────────────────────────────────────────────────────────────────────────
write src/ingest_v2/transcripts/__init__.py ''

write src/ingest_v2/transcripts/normalize.py \
'from typing import List, Dict, Any
from pathlib import Path
import json, re

def normalize_to_sentences(raw: Dict[str, Any], default_speaker: str = "S1") -> List[Dict[str, Any]]:
    segments = raw.get("segments") or raw.get("caption_lines") or []
    sentences: List[Dict[str, Any]] = []

    for seg in segments:
        start = float(seg.get("start", 0))
        end = float(seg.get("end", 0))
        text = seg.get("text", "").strip()
        parts = re.split(r"(?<=[\.\\?\\!])\\s+", text)
        if not parts:
            continue

        if seg.get("words"):
            words = seg["words"]
            idx = 0
            for sent in parts:
                n = len(sent.split())
                w_slice = words[idx: idx + max(1, n)]
                s_start = float(w_slice[0]["start"]) if w_slice else start
                s_end = float(w_slice[-1]["end"]) if w_slice else min(end, s_start + 4.0)
                sentences.append({"start_s": s_start, "end_s": s_end, "text": sent, "speaker": default_speaker})
                idx += n
        else:
            if len(parts) == 1:
                sentences.append({"start_s": start, "end_s": end, "text": parts[0], "speaker": default_speaker})
            else:
                span = (end - start) / len(parts)
                for i, sent in enumerate(parts):
                    s_start = start + i * span
                    s_end = min(end, s_start + span)
                    sentences.append({"start_s": s_start, "end_s": s_end, "text": sent, "speaker": default_speaker})

    sentences = [s for s in sentences if s["text"].strip()]
    sentences.sort(key=lambda x: (x["start_s"], x["end_s"]))
    return sentences

def write_normalized_jsonl(parent_id: str, sentences: List[Dict[str, Any]], out_dir: Path = Path("transcripts/normalized")) -> Path:
    out_dir.mkdir(parents=True, exist_ok=True)
    path = out_dir / f"{parent_id}.jsonl"
    with path.open("w", encoding="utf-8") as f:
        for s in sentences:
            f.write(json.dumps(s, ensure_ascii=False) + "\\n")
    return path'

# ────────────────────────────────────────────────────────────────────────────────
# segmenter
# ────────────────────────────────────────────────────────────────────────────────
write src/ingest_v2/segmenter/__init__.py ''

write src/ingest_v2/segmenter/segmenter.py \
'from typing import List, Dict, Any, Optional
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

    return segs'

# ────────────────────────────────────────────────────────────────────────────────
# entities
# ────────────────────────────────────────────────────────────────────────────────
write src/ingest_v2/entities/__init__.py ''

write src/ingest_v2/entities/extract.py \
'import re
from typing import List

TICKER_RE = re.compile(r"\\B\\$[A-Z]{2,6}\\b")
HANDLE_RE = re.compile(r"@[\\w\\d_]{2,30}")

def extract_entities(text: str) -> List[str]:
    ents = set()
    for m in TICKER_RE.findall(text):
        ents.add(m.upper())
    for m in HANDLE_RE.findall(text):
        ents.add(m)
    return sorted(ents)[:32]'

# ────────────────────────────────────────────────────────────────────────────────
# validators
# ────────────────────────────────────────────────────────────────────────────────
write src/ingest_v2/validators/__init__.py ''

write src/ingest_v2/validators/runtime.py \
'from ..schemas.child import ChildNode
from ..schemas.parent import ParentNode
import requests

def validate_child_runtime(x: dict, duration_s: float) -> None:
    c = ChildNode(**x)
    assert 0 <= c.start_s < c.end_s <= duration_s, "timestamp order invalid"
    assert 15 - 1e-6 <= (c.end_s - c.start_s) <= 60 + 1e-6, "window size out of bounds"
    assert len(c.text) >= 80 or c.text.endswith("?"), "text too short"
    if c.clip_url:
        try:
            resp = requests.head(str(c.clip_url), allow_redirects=True, timeout=4)
            assert resp.status_code in (200, 301, 302, 303, 307, 308)
        except Exception:
            pass

def validate_parent_runtime(p: dict) -> None:
    ParentNode(**p)'

# ────────────────────────────────────────────────────────────────────────────────
# pipelines
# ────────────────────────────────────────────────────────────────────────────────
write src/ingest_v2/pipelines/__init__.py ''

write src/ingest_v2/pipelines/build_parents.py \
'import logging
from typing import Dict, Any, List
from ..schemas.parent import ParentNode
from ..sources.youtube import build_parent_from_metadata
from ..validators.runtime import validate_parent_runtime
from ..utils.logging import setup_logger

def build_parent(meta: Dict[str, Any]) -> ParentNode:
    p = build_parent_from_metadata(meta)
    validate_parent_runtime(p.dict())
    return p

def run_build_parents(metas: List[Dict[str, Any]]) -> List[ParentNode]:
    logging.info(f"[parents] building parents for {len(metas)} assets")
    out = [build_parent(m) for m in metas]
    logging.info(f"[parents] done {len(out)}")
    return out'

write src/ingest_v2/pipelines/build_children.py \
'import logging
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
    return children'

write src/ingest_v2/pipelines/upsert_pinecone.py \
'import logging
from typing import List, Dict, Any
from ..configs.settings import settings_v2
from ..utils.pinecone_client import get_index, upsert_vectors
from ..utils.backoff import expo_backoff

def _embedder():
    if settings_v2.EMBED_PROVIDER == "openai":
        from openai import OpenAI
        client = OpenAI()
        model = settings_v2.EMBED_MODEL
        def embed_texts(texts: List[str]) -> List[List[float]]:
            vecs = []
            for i in range(0, len(texts), 128):
                chunk = texts[i:i+128]
                resp = client.embeddings.create(model=model, input=chunk)
                vecs.extend([d.embedding for d in resp.data])
            return vecs
        return embed_texts
    else:
        from sentence_transformers import SentenceTransformer
        model = SentenceTransformer(settings_v2.EMBED_MODEL)
        def embed_texts(texts: List[str]) -> List[List[float]]:
            return model.encode(texts, batch_size=64, normalize_embeddings=True).tolist()
        return embed_texts

def choose_namespace(document_type: str) -> str:
    return settings_v2.NAMESPACE_VIDEOS if document_type == "youtube_video" else settings_v2.NAMESPACE_STREAMS

def trim_metadata(meta: Dict[str, Any]) -> Dict[str, Any]:
    import json
    s = json.dumps(meta, ensure_ascii=False)
    if len(s.encode("utf-8")) <= settings_v2.MAX_METADATA_BYTES:
        return meta
    m = dict(meta)
    t = m.get("text", "")
    if t and len(t) > 1200:
        m["text"] = t[:1200] + "…"
    return m

def upsert_children(children: List[Dict[str, Any]]):
    if not children:
        return
    index = get_index(settings_v2.PINECONE_INDEX_NAME, settings_v2.EMBED_DIM)
    embed = _embedder()
    texts = [c["text"] for c in children]
    attempt = 0
    while True:
        try:
            embs = embed(texts)
            break
        except Exception as e:
            attempt += 1
            logging.warning(f"[embed] retry {attempt} due to {e}")
            expo_backoff(attempt)

    vectors = [{
        "id": c["segment_id"],
        "values": vec,
        "metadata": trim_metadata(c),
    } for c, vec in zip(children, embs)]

    ns = choose_namespace(children[0]["document_type"])
    upsert_vectors(index=index, namespace=ns, vectors=vectors, batch_size=100)
    logging.info(f"[upsert] upserted {len(vectors)} vectors into namespace={ns}")'

write src/ingest_v2/pipelines/run_all.py \
'import argparse, logging
from pathlib import Path
from ..configs.settings import settings_v2
from ..pipelines.build_parents import run_build_parents
from ..pipelines.build_children import build_children_from_raw
from ..pipelines.upsert_pinecone import upsert_children
from ..utils.logging import setup_logger
from src.Llama_index_sandbox.utils.utils import root_directory
from src.Llama_index_sandbox.data_ingestion_youtube.load.load import load_video_transcripts

def _iter_youtube_assets(add_new_transcripts: bool = True, num_files=None, files_window=None):
    docs = load_video_transcripts(
        directory_path=Path(root_directory()) / "datasets/evaluation_data/diarized_youtube_content_2023-10-06/",
        add_new_transcripts=add_new_transcripts,
        num_files=num_files,
        files_window=files_window,
        overwrite=False
    )
    for d in docs:
        meta = dict(d.metadata)
        raw = {
            "segments": meta.get("segments") or [],
            "caption_lines": meta.get("caption_lines") or [],
            "diarization": meta.get("diarization") or [],
        }
        meta.setdefault("video_id", meta.get("video_id") or meta.get("id") or meta.get("youtube_video_id"))
        meta.setdefault("duration_s", meta.get("duration_s") or meta.get("duration") or 0)
        meta.setdefault("url", meta.get("url") or f"https://www.youtube.com/watch?v={meta['"'"'video_id'"'"']}")
        yield meta, raw

def main():
    setup_logger("ingest_v2")
    ap = argparse.ArgumentParser()
    ap.add_argument("--doc-type", default="youtube_video", choices=["youtube_video", "stream"])
    ap.add_argument("--backfill-days", type=int, default=settings_v2.BACKFILL_DAYS)
    args = ap.parse_args()

    if args.doc_type == "youtube_video":
        metas_raw = list(_iter_youtube_assets(add_new_transcripts=True))
        metas = [m for m, _ in metas_raw]
    else:
        metas_raw = []
        metas = []

    parents = run_build_parents(metas=[{
        "video_id": m["video_id"],
        "title": m.get("title", ""),
        "description": m.get("description", ""),
        "channel_name": m.get("channel_name"),
        "speaker_primary": m.get("speaker_primary"),
        "published_at": m.get("published_at"),
        "duration_s": m.get("duration_s", 0),
        "url": m.get("url"),
        "thumbnail_url": m.get("thumbnail_url"),
        "language": m.get("language", "en"),
        "entities": m.get("entities", []),
        "chapters": m.get("chapters"),
    } for m in metas])

    parents_map = {p.parent_id: p.dict() for p in parents}
    total_children = 0
    for (meta, raw) in metas_raw:
        parent = parents_map[meta["video_id"]]
        children = build_children_from_raw(parent, raw)
        upsert_children(children)
        total_children += len(children)

    logging.info(f"[ingest_v2] finished upserting {total_children} child segments.")

if __name__ == "__main__":
    main()'

# ────────────────────────────────────────────────────────────────────────────────
#   tests
# ────────────────────────────────────────────────────────────────────────────────
write src/ingest_v2/tests/__init__.py ''

write src/ingest_v2/tests/test_ids.py \
'from src.ingest_v2.utils.ids import segment_uuid

def test_deterministic_uuid():
    a = segment_uuid("vid123", 12.3456, 27.89)
    b = segment_uuid("vid123", 12.346, 27.8901)
    assert a == b'

write src/ingest_v2/tests/test_segmenter.py \
'from src.ingest_v2.segmenter.segmenter import build_segments

def test_segmenter_basic():
    sentences = [
        {"start_s": 0.0, "end_s": 2.0, "text": "Hello world.", "speaker": "S1"},
        {"start_s": 2.0, "end_s": 8.0, "text": "This is a longer sentence that should help reach the minimum window.", "speaker": "S1"},
        {"start_s": 8.0, "end_s": 18.0, "text": "Ending now.", "speaker": "S1"},
    ]
    segs = build_segments(
        sentences,
        duration_s=120.0,
        parent_id="vid",
        document_type="youtube_video",
        clip_base_url="https://www.youtube.com/watch?v=vid"
    )
    assert segs
    for s in segs:
        assert 15 <= (s["end_s"] - s["start_s"]) <= 60'

write src/ingest_v2/tests/test_entities.py \
'from src.ingest_v2.entities.extract import extract_entities

def test_entities():
    t = "We talked to @anatoly about $SOL and @builder_xyz."
    ents = extract_entities(t)
    assert "$SOL" in ents and "@anatoly" in ents'

write src/ingest_v2/tests/test_validators.py \
'from src.ingest_v2.validators.runtime import validate_child_runtime

def test_child_validation():
    child = {
        "node_type": "child",
        "segment_id": "abc",
        "parent_id": "p",
        "document_type": "youtube_video",
        "text": "X" * 100,
        "start_s": 10.0,
        "end_s": 30.0,
        "start_hms": "00:00:10.000",
        "end_hms": "00:00:30.000",
        "clip_url": "https://www.youtube.com/watch?v=dummy&t=10s",
        "speaker": "S1",
        "entities": [],
        "chapter": None,
        "language": "en",
        "confidence_asr": 0.9,
        "has_music": False,
        "flags": [],
        "rights": "public_reference_only",
        "ingest_version": 2,
        "source_hash": "deadbeef",
    }
    validate_child_runtime(child, duration_s=3600.0)'

write src/ingest_v2/tests/fixtures/__init__.py ''

write src/ingest_v2/tests/fixtures/golden_children_reference.jsonl \
'{"start_s": 10.0, "end_s": 28.0, "text": "Example sentence one. Example sentence two.", "speaker": "S1"}'

write src/ingest_v2/tests/fixtures/example_normalized.jsonl \
'{"start_s": 0.0, "end_s": 2.0, "text": "Hello world.", "speaker": "S1"}'