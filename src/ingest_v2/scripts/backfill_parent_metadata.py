#!/usr/bin/env python3
"""
Backfill Qdrant parent payloads with enriched metadata (router fields + speakers).

Why:
  - `/catalog/search` is metadata-only and relies on parent payload fields.
  - Older ingests often missed GPT-generated router fields and speaker classification.

Source of truth:
  - Router sidecars:  ROUTER_CACHE_DIR/<parent_id>.json
  - Speaker sidecars: SPEAKER_MAP_DIR/<parent_id>.json

Optionally, this script can generate missing router sidecars using OpenAI by
building a short transcript preview from existing child/summary points.

Examples:
  - Dry-run a small batch:
      python -m src.ingest_v2.scripts.backfill_parent_metadata --limit 50 --dry-run
  - Apply updates (sidecars only):
      python -m src.ingest_v2.scripts.backfill_parent_metadata --limit 500
  - Generate missing router fields (bounded):
      python -m src.ingest_v2.scripts.backfill_parent_metadata --generate-missing --max-generate 25
"""

from __future__ import annotations

import argparse
import json
import logging
import os
import sys
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

from qdrant_client.http import models as qm

from src.ingest_v2.configs.settings import settings_v2
from src.ingest_v2.utils.vector_store import qdrant_client, qdrant_collection_name, vector_store_backend

LOG = logging.getLogger("backfill_parent_metadata")

# Optional dotenv for running outside docker.
try:  # pragma: no cover
    from dotenv import find_dotenv, load_dotenv

    load_dotenv(find_dotenv(usecwd=True), override=False)
    load_dotenv(".env", override=False)
    load_dotenv(".env.local", override=False)
except Exception:  # pragma: no cover
    pass


def _nonempty_str(v: Any) -> bool:
    if v is None:
        return False
    return bool(str(v).strip())


def _nonempty_list(v: Any) -> bool:
    if not isinstance(v, list):
        return False
    return any(_nonempty_str(x) for x in v)


def _read_json(path: Path) -> Optional[Dict[str, Any]]:
    if not path.exists():
        return None
    try:
        obj = json.loads(path.read_text(encoding="utf-8"))
    except Exception:
        return None
    return obj if isinstance(obj, dict) else None


def _router_sidecar(parent_id: str, router_dir: Path) -> Optional[Dict[str, Any]]:
    return _read_json(router_dir / f"{parent_id}.json")


def _speaker_sidecar(parent_id: str, speaker_dir: Path) -> Optional[Dict[str, Any]]:
    return _read_json(speaker_dir / f"{parent_id}.json")


def _speaker_names_from_map(speaker_map: Any) -> Optional[List[str]]:
    if not isinstance(speaker_map, dict):
        return None
    out: List[str] = []
    for info in speaker_map.values():
        if not isinstance(info, dict):
            continue
        name = str(info.get("name") or "").strip()
        if name:
            out.append(name)
    # de-dupe preserve order
    seen = set()
    uniq: List[str] = []
    for n in out:
        k = n.lower()
        if k in seen:
            continue
        seen.add(k)
        uniq.append(n)
    return uniq or None


def _preview_segments_from_children(
    *,
    parent_id: str,
    child_collection: str,
    max_points: int,
) -> List[Dict[str, Any]]:
    """
    Fetch a small transcript preview for `parent_id` from child/summary points.
    Prefer summary nodes; fallback to a few child chunks.
    """
    client = qdrant_client()

    def _scroll(node_type: str, limit: int) -> List[Dict[str, Any]]:
        flt = qm.Filter(
            must=[
                qm.FieldCondition(key="parent_id", match=qm.MatchValue(value=parent_id)),
                qm.FieldCondition(key="node_type", match=qm.MatchValue(value=node_type)),
            ]
        )
        points, _ = client.scroll(
            collection_name=child_collection,
            scroll_filter=flt,
            limit=limit,
            with_payload=True,
            with_vectors=False,
        )
        out: List[Dict[str, Any]] = []
        for p in points or []:
            payload = dict(getattr(p, "payload", None) or {})
            text = str(payload.get("text") or "").strip()
            if not text:
                continue
            start = payload.get("start_s")
            if start is None:
                start = payload.get("start_seconds")
            try:
                start_s = float(start) if start is not None else 0.0
            except Exception:
                start_s = 0.0
            end = payload.get("end_s")
            try:
                end_s = float(end) if end is not None else (start_s + 10.0)
            except Exception:
                end_s = start_s + 10.0
            out.append({"start": start_s, "end": end_s, "text": text})
        return out

    segs = _scroll("summary", limit=min(3, max(1, max_points)))
    if segs:
        return segs[:max_points]

    segs = _scroll("child", limit=max(1, max_points))
    return segs[:max_points]


def _generate_router_sidecar(
    *,
    parent_id: str,
    meta: Dict[str, Any],
    child_collection: str,
    max_points: int,
) -> Optional[Dict[str, Any]]:
    """
    Generate router fields (GPT) from a transcript preview already indexed in Qdrant.
    """
    segs = _preview_segments_from_children(
        parent_id=parent_id,
        child_collection=child_collection,
        max_points=max_points,
    )
    if not segs:
        return None

    try:
        from src.ingest_v2.transcripts.normalize import normalize_to_sentences
        from src.ingest_v2.router.enrich_parent import enrich_parent_router_fields
    except Exception:
        return None

    raw = {"segments": segs}
    sentences = normalize_to_sentences(raw)
    if not sentences:
        # fall back to the segment text as a single "sentence"
        sentences = [{"start_s": float(segs[0].get("start") or 0.0), "end_s": float(segs[0].get("end") or 0.0), "text": str(segs[0].get("text") or "").strip()}]

    return enrich_parent_router_fields(meta, sentences)


def parse_args(argv: Optional[List[str]] = None) -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Backfill parent payload metadata in Qdrant (router fields + speakers).")
    p.add_argument("--namespace", default=os.getenv("PINECONE_NAMESPACE_VIDEOS", "videos"))
    p.add_argument("--dry-run", action="store_true", help="Log intended updates without writing to Qdrant.")
    p.add_argument("--limit", type=int, default=None, help="Optional cap on number of parents to scan.")
    p.add_argument("--force", action="store_true", help="Overwrite existing fields even if non-empty.")
    p.add_argument("--router-dir", default=settings_v2.ROUTER_CACHE_DIR, help="Directory containing router sidecars.")
    p.add_argument("--speaker-dir", default=settings_v2.SPEAKER_MAP_DIR, help="Directory containing speaker sidecars.")
    p.add_argument("--generate-missing", action="store_true", help="Use OpenAI to generate router fields for parents missing sidecars.")
    p.add_argument("--max-generate", type=int, default=0, help="Cap number of OpenAI generations (0 = unlimited when enabled).")
    p.add_argument("--preview-points", type=int, default=6, help="How many child/summary points to use for GPT preview.")
    return p.parse_args(argv)


def main(argv: Optional[List[str]] = None) -> int:
    logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s")
    # qdrant-client uses httpx; keep request logs from flooding output.
    logging.getLogger("httpx").setLevel(logging.WARNING)
    logging.getLogger("qdrant_client").setLevel(logging.WARNING)
    args = parse_args(argv)

    if vector_store_backend() != "qdrant":
        LOG.error("VECTOR_STORE must be 'qdrant' for this script (got %r).", os.getenv("VECTOR_STORE"))
        return 2

    router_dir = Path(str(args.router_dir)).expanduser().resolve()
    speaker_dir = Path(str(args.speaker_dir)).expanduser().resolve()
    router_dir.mkdir(parents=True, exist_ok=True)
    speaker_dir.mkdir(parents=True, exist_ok=True)

    parent_collection = qdrant_collection_name(args.namespace)
    streams_collection = qdrant_collection_name(settings_v2.NAMESPACE_STREAMS)

    client = qdrant_client()

    flt = qm.Filter(must=[qm.FieldCondition(key="node_type", match=qm.MatchValue(value="parent"))])

    scanned = 0
    patched = 0
    patched_router = 0
    patched_speakers = 0
    generated = 0
    missing_router_sidecar = 0
    missing_speaker_sidecar = 0
    offset = None

    max_generate = int(args.max_generate or 0)
    remaining_generate = max_generate if max_generate > 0 else None

    while True:
        points, offset = client.scroll(
            collection_name=parent_collection,
            scroll_filter=flt,
            limit=128,
            with_payload=True,
            with_vectors=False,
            offset=offset,
        )
        if not points:
            break

        for pt in points:
            scanned += 1
            payload = dict(getattr(pt, "payload", None) or {})
            parent_id = str(payload.get("parent_id") or payload.get("video_id") or "").strip()
            if not parent_id:
                continue

            # Decide which child collection to pull a transcript preview from if we need GPT.
            doc_type = str(payload.get("document_type") or "").strip().lower()
            child_collection = parent_collection if doc_type == "youtube_video" else streams_collection

            patch: Dict[str, Any] = {}

            # Router fields
            router_need = args.force or (
                (not _nonempty_str(payload.get("topic_summary")))
                or (not _nonempty_list(payload.get("router_tags")))
                or (not _nonempty_list(payload.get("aliases")))
                or (not _nonempty_list(payload.get("canonical_entities")))
            )
            router_sidecar = None
            if router_need:
                router_sidecar = _router_sidecar(parent_id, router_dir)
                if router_sidecar is None:
                    missing_router_sidecar += 1
                    if args.generate_missing and (remaining_generate is None or remaining_generate > 0):
                        try:
                            meta = {
                                "video_id": parent_id,
                                "title": payload.get("title") or parent_id,
                                "description": payload.get("description") or "",
                                "channel_name": payload.get("channel_name") or "",
                                "published_at": payload.get("published_at") or "",
                                "entities": payload.get("entities") or [],
                            }
                            router_sidecar = _generate_router_sidecar(
                                parent_id=parent_id,
                                meta=meta,
                                child_collection=child_collection,
                                max_points=max(1, int(args.preview_points)),
                            )
                            if isinstance(router_sidecar, dict):
                                generated += 1
                                try:
                                    (router_dir / f"{parent_id}.json").write_text(
                                        json.dumps(router_sidecar, ensure_ascii=False, indent=2), encoding="utf-8"
                                    )
                                except Exception:
                                    pass
                            if remaining_generate is not None and remaining_generate > 0:
                                remaining_generate -= 1
                        except Exception as exc:
                            LOG.info("[router/gen] failed parent=%s err=%s", parent_id, exc)
                            router_sidecar = None

                if isinstance(router_sidecar, dict):
                    for key in (
                        "description",
                        "topic_summary",
                        "router_tags",
                        "aliases",
                        "canonical_entities",
                        "is_explainer",
                        "router_boost",
                    ):
                        if key in router_sidecar and router_sidecar[key] is not None:
                            patch[key] = router_sidecar[key]

            # Speakers
            speakers_need = args.force or (not isinstance(payload.get("speaker_map"), dict)) or (not _nonempty_list(payload.get("speaker_names")))
            speaker_sidecar = None
            if speakers_need:
                speaker_sidecar = _speaker_sidecar(parent_id, speaker_dir)
                if speaker_sidecar is None:
                    missing_speaker_sidecar += 1
                if isinstance(speaker_sidecar, dict):
                    if speaker_sidecar.get("speaker_primary") is not None:
                        patch["speaker_primary"] = speaker_sidecar.get("speaker_primary")
                    if speaker_sidecar.get("speaker_map") is not None:
                        patch["speaker_map"] = speaker_sidecar.get("speaker_map")
                        names = _speaker_names_from_map(speaker_sidecar.get("speaker_map"))
                        if names:
                            patch["speaker_names"] = names
                    if speaker_sidecar.get("speaker_names") is not None and _nonempty_list(speaker_sidecar.get("speaker_names")):
                        patch["speaker_names"] = speaker_sidecar.get("speaker_names")

            if not patch:
                if args.limit and scanned >= int(args.limit):
                    break
                continue

            patched += 1
            if any(k in patch for k in ("topic_summary", "router_tags", "aliases", "canonical_entities", "router_boost", "is_explainer")):
                patched_router += 1
            if any(k in patch for k in ("speaker_primary", "speaker_map", "speaker_names")):
                patched_speakers += 1

            if args.dry_run:
                if patched <= 5:
                    LOG.info("[dry-run] parent=%s patch_keys=%s", parent_id, sorted(patch.keys()))
            else:
                client.set_payload(collection_name=parent_collection, payload=patch, points=[pt.id], wait=True)

            if args.limit and scanned >= int(args.limit):
                break

        if args.limit and scanned >= int(args.limit):
            break
        if offset is None:
            break

    LOG.info(
        "Done. scanned=%d patched=%d patched_router=%d patched_speakers=%d generated_router=%d missing_router_sidecar=%d missing_speaker_sidecar=%d",
        scanned,
        patched,
        patched_router,
        patched_speakers,
        generated,
        missing_router_sidecar,
        missing_speaker_sidecar,
    )
    return 0


if __name__ == "__main__":
    sys.exit(main())
