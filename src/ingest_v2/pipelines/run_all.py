import argparse
import logging
import os
import time
from pathlib import Path
from typing import Dict, List, Any

from src.ingest_v2.configs.settings import settings_v2
from src.ingest_v2.pipelines.build_children import build_children_from_raw
from src.ingest_v2.pipelines.build_parents import run_build_parents
from src.ingest_v2.pipelines.run_all_components.assets import iter_youtube_assets_from_fs
from src.ingest_v2.pipelines.run_all_components.dedupe import (
    get_ingested_parent_ids,
    get_ingested_parent_ids_qdrant,
)
from src.ingest_v2.pipelines.run_all_components.enrich import enrich_assets
from src.ingest_v2.pipelines.run_all_components.namespace import (
    load_namespace_channels,
    parse_list_env,
)
from src.ingest_v2.pipelines.run_all_components.prioritize import Asset, prioritize_assets
from src.ingest_v2.pipelines.run_all_components.speakers_stage import resolve_speakers_for_assets
from src.ingest_v2.pipelines.run_all_components.text import ascii_safe
from src.ingest_v2.pipelines.upsert_parents import upsert_parents
from src.ingest_v2.pipelines.upsert_pinecone import upsert_children
from src.ingest_v2.utils.vector_store import vector_store_backend
from src.ingest_v2.utils.logging import setup_logger
from src.ingest_v2.utils.progress import progress


def _default_youtube_root() -> str:
    """
    v2 ingestion should not depend on v1 sandbox config (which can require unrelated env vars).
    Prefer an explicit env override, else fall back to the repo dataset path if present,
    else use the docker-compose mount path.
    """
    env = (os.getenv("YT_DIARIZED_ROOT") or os.getenv("INGEST_YT_ROOT") or "").strip()
    if env:
        return env

    repo_root = Path(__file__).resolve().parents[4]
    candidate = repo_root / "youtube-transcript-pipeline/datasets/evaluation_data/diarized_youtube_content_2023-10-06"
    if candidate.exists():
        return str(candidate)

    return "/datasets/diarized_youtube"


def _ingest_assets(enriched_metas: List[Dict[str, Any]]) -> Dict[str, Dict[str, object]]:
    parents = run_build_parents(
        metas=[
            {
                "video_id": meta["video_id"],
                "title": meta.get("title", ""),
                "description": meta.get("description", ""),
                "channel_name": meta.get("channel_name"),
                "speaker_primary": meta.get("speaker_primary"),
                "published_at": meta.get("published_at"),
                "duration_s": meta.get("duration_s", 0.0),
                "url": meta.get("url"),
                "thumbnail_url": meta.get("thumbnail_url"),
                "language": meta.get("language", "en"),
                "entities": meta.get("entities", []),
                "chapters": meta.get("chapters"),
                "document_type": "youtube_video",
                "source": "youtube",
                "router_tags": meta.get("router_tags"),
                "aliases": meta.get("aliases"),
                "canonical_entities": meta.get("canonical_entities"),
                "is_explainer": meta.get("is_explainer"),
                "router_boost": meta.get("router_boost"),
                "topic_summary": meta.get("topic_summary"),
                "speaker_map": meta.get("speaker_map"),
            }
            for meta in enriched_metas
        ]
    )

    progress.set_parents_total(len(parents))
    parents_payload = [dict(parent.model_dump(mode="json"), parent_id=parent.parent_id) for parent in parents]
    upsert_parents(parents_payload)
    return {parent.parent_id: parent.model_dump(mode="json") for parent in parents}


def main():
    setup_logger("ingest_v2")

    parser = argparse.ArgumentParser()
    parser.add_argument("--doc-type", default="youtube_video", choices=["youtube_video", "stream"])
    parser.add_argument(
        "--namespace",
        default=os.getenv("PINECONE_NAMESPACE", "videos"),
        help="Namespace to write to (defaults to env or 'videos').",
    )
    parser.add_argument(
        "--include-channels",
        nargs="*",
        default=None,
        help="Only ingest these channel folders (e.g. @BinanceYoutube @notthreadguy).",
    )
    parser.add_argument(
        "--all-channels",
        action="store_true",
        help="Ingest ALL channel folders found under --root (ignores YT_NAMESPACE_CONFIG).",
    )
    parser.add_argument("--backfill-days", type=int, default=settings_v2.BACKFILL_DAYS)
    parser.add_argument("--root", default=_default_youtube_root(), help="Root dir with diarized JSONs")
    parser.add_argument("--prune-empty", action="store_true", help="Delete obviously empty AssemblyAI files")
    parser.add_argument("--concurrency", type=int, default=int(os.getenv("ROUTER_GEN_CONCURRENCY", "6")))
    parser.add_argument(
        "--speakers-workers",
        type=int,
        default=int(os.getenv("SPEAKERS_WORKERS", "0")),
        help="Workers for speaker resolution across videos (default: ~70% of CPUs when 0)",
    )
    parser.add_argument("--skip-speakers", action="store_true", help="Skip cross-video speaker resolution/enroll stages")
    parser.add_argument("--skip-dedupe", action="store_true", help="Skip Pinecone deduplication check")
    args = parser.parse_args()

    index_name = os.getenv("PINECONE_INDEX_NAME", "icmfyi-v2")
    namespace = args.namespace
    backend = vector_store_backend()
    logging.info("[cfg] vector_store=%s index=%s namespace=%s", backend, index_name, namespace)

    root = Path(args.root).expanduser().resolve()

    channels = [channel for channel in (args.include_channels or []) if channel and channel.strip()]
    if not channels and args.all_channels:
        channels = sorted(
            [
                p.name
                for p in root.iterdir()
                if p.is_dir() and p.name and not p.name.startswith(".")
            ]
        )
        logging.info("[cfg] discovered %d channel folder(s) under %s", len(channels), root)
    if not channels and not args.all_channels:
        channels = load_namespace_channels(namespace)
        if channels:
            logging.info("[cfg] loaded %d channel(s) from namespace '%s'", len(channels), namespace)
    if not channels and not args.all_channels:
        legacy = parse_list_env("INGEST_INCLUDE_CHANNELS")
        if legacy:
            logging.warning(
                "INGEST_INCLUDE_CHANNELS is DEPRECATED. Move to YT_NAMESPACE_CONFIG "
                "or YT_NAMESPACE_CONFIG_JSON keyed by namespace='%s'.",
                namespace,
            )
            channels = legacy
    if not channels:
        parser.error(
            "No channels resolved. Provide --include-channels, or configure YT_NAMESPACE_CONFIG "
            "(JSON/YAML) or YT_NAMESPACE_CONFIG_JSON for namespace='%s', or pass --all-channels." % namespace
        )

    logging.info("[cfg] namespace=%s channels=%s", namespace, channels)
    args.include_channels = channels

    os.environ.setdefault("VOICE_EMBED_BACKEND", "off")
    os.environ["PINECONE_NAMESPACE"] = namespace
    os.environ.setdefault("PINECONE_STREAMS_NS", f"{namespace}_streams")

    if args.doc_type != "youtube_video":
        logging.info("[v2] 'stream' not wired yet in run_all; nothing to do.")
        return

    logging.info("[v2] scanning for AssemblyAI JSON under: %s (prune_empty=%s)", root, args.prune_empty)

    assets = list(
        iter_youtube_assets_from_fs(
            root,
            prune_empty=args.prune_empty,
            allowed_channels=args.include_channels,
        )
    )
    if not assets:
        logging.info("[v2] no youtube assets found; exiting.")
        return

    if args.include_channels:
        allowed = {channel.strip() for channel in args.include_channels if channel.strip()}
        before = len(assets)
        assets = [item for item in assets if (item[0].get("channel_name") or "").strip() in allowed]
        logging.info("[v2/filter] channels=%s kept=%d/%d", sorted(allowed), len(assets), before)
        if not assets:
            logging.info("[v2] no assets match include filter; exiting.")
            return

    if args.skip_dedupe:
        logging.info("[v2/dedupe] skipping deduplication check (--skip-dedupe)")
    else:
        if vector_store_backend() == "qdrant":
            logging.info("[v2/dedupe] checking which parents already have children in Qdrant...")
            ingested = get_ingested_parent_ids_qdrant(namespace=namespace)
            before = len(assets)
            assets = [item for item in assets if item[0]["video_id"] not in ingested]
            after = len(assets)
            logging.info(
                "[v2/dedupe] parent filter: %d discovered, %d already ingested, %d remaining to process",
                before,
                len(ingested),
                after,
            )
            if not assets:
                logging.info("[v2] all parents already ingested in Qdrant. Nothing to do.")
                return
        else:
            logging.info("[v2/dedupe] checking which parents are already in Pinecone...")
            ingested = get_ingested_parent_ids(index_name=index_name, namespace=namespace)
            before = len(assets)
            assets = [item for item in assets if item[0]["video_id"] not in ingested]
            after = len(assets)
            logging.info(
                "[v2/dedupe] parent filter: %d discovered, %d already ingested, %d remaining to process",
                before,
                len(ingested),
                after,
            )
            if not assets:
                logging.info("[v2] all parents already in Pinecone. Nothing to do.")
                return

    assets = prioritize_assets(
        assets,
        deprioritize_channels=["@SolanaFndn", "@timroughgardenlectures1861"],
    )

    if args.skip_speakers:
        logging.info("[v2/speakers] skipping speaker resolution (--skip-speakers)")
        processed_assets = list(assets)
        speakers_total = 0.0
    else:
        workers = args.speakers_workers if args.speakers_workers > 0 else None
        processed_assets, speakers_total = resolve_speakers_for_assets(assets, workers=workers)

    enriched_metas, enrich_stats = enrich_assets(processed_assets, args.concurrency)
    logging.info(
        "[v2/router] total=%d cached_hits=%d fresh_enriched=%d failed=%d",
        len(enriched_metas),
        enrich_stats["cached_hits"],
        enrich_stats["fresh"],
        enrich_stats["failed"],
    )

    parents_map = _ingest_assets(enriched_metas)

    total_children = 0
    missing_parents = 0
    skipped_empty = 0
    children_total = 0.0

    for meta, raw, _path in processed_assets:
        parent = parents_map.get(meta["video_id"])
        if not parent:
            logging.warning("[v2] missing parent for video_id=%s; skipping", meta["video_id"])
            missing_parents += 1
            continue

        start = time.perf_counter()
        children = build_children_from_raw(parent, raw)
        if not children:
            logging.info("[v2] no children emitted for %s (%s)", meta["video_id"], ascii_safe(meta.get("title") or ""))
            skipped_empty += 1
            continue

        stats = upsert_children(children)
        elapsed = time.perf_counter() - start
        children_total += elapsed
        total_children += len(children)
        logging.info(
            "[v2/children] vid=%s upserted=%d in %.2fs (embed=%.2fs upsert=%.2fs embed_reqs=%d pinecone_batches=%d)",
            meta["video_id"],
            len(children),
            elapsed,
            stats["t_embed"],
            stats["t_upsert"],
            stats["embed_reqs"],
            stats["pinecone_batches"],
        )

    logging.info(
        "[ingest_v2] finished upserting %d child segments from %d parents "
        "(missing_parents=%d, skipped_empty=%d, router_cached=%d, router_enriched=%d, router_failed=%d).",
        total_children,
        len(parents_map),
        missing_parents,
        skipped_empty,
        enrich_stats["cached_hits"],
        enrich_stats["fresh"],
        enrich_stats["failed"],
    )

    avg_speakers = speakers_total / max(1, len(processed_assets))
    logging.info(
        "[timing] speakers_total=%.2fs (~%.2fs/video), enrich_total=%.2fs, children_total=%.2fs",
        speakers_total,
        avg_speakers,
        enrich_stats["total_time"],
        children_total,
    )


if __name__ == "__main__":
    main()
