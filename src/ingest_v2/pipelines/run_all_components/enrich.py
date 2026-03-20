from __future__ import annotations

import asyncio
import logging
import os
import time
from typing import Any, Dict, Iterable, List, Tuple

from src.ingest_v2.router.cache import load as router_cache_load
from src.ingest_v2.router.cache import save as router_cache_save
from src.ingest_v2.router.enrich_parent import enrich_parent_router_fields_async
from src.ingest_v2.transcripts.normalize import normalize_to_sentences

from .prioritize import Asset
from .text import ascii_safe, snip_text


async def _enrich_async(meta: Dict[str, Any], raw: Dict[str, Any]) -> Dict[str, Any]:
    try:
        sentences = normalize_to_sentences(raw)
    except Exception:
        sentences = []
    return await enrich_parent_router_fields_async(meta, sentences)


def _merge_meta(meta: Dict[str, Any], enrich: Dict[str, Any]) -> Dict[str, Any]:
    merged = dict(meta)
    merged.update(
        {
            "description": enrich.get("description", merged.get("description", "")),
            "topic_summary": enrich.get("topic_summary"),
            "router_tags": enrich.get("router_tags"),
            "aliases": enrich.get("aliases"),
            "canonical_entities": enrich.get("canonical_entities"),
            "is_explainer": enrich.get("is_explainer"),
            "router_boost": enrich.get("router_boost"),
        }
    )
    return merged


def _log_cache_hit(video_id: str, enrich: Dict[str, Any]) -> None:
    n = int(os.getenv("ROUTER_LOG_DESC_CHARS", "240"))
    logging.info(
        "[router/cache] vid=%s boost=%s tags=%s desc_preview=%r topic_preview=%r",
        video_id,
        enrich.get("router_boost"),
        (enrich.get("router_tags") or [])[:6],
        ascii_safe(snip_text(enrich.get("description") or "", n)),
        ascii_safe(snip_text(enrich.get("topic_summary") or "", max(80, n // 2))),
    )


async def _gather_enrichment(pairs: List[Tuple[Dict[str, Any], Dict[str, Any]]], concurrency: int):
    semaphore = asyncio.Semaphore(concurrency)

    async def _one(meta: Dict[str, Any], raw: Dict[str, Any]):
        video_id = meta["video_id"]
        try:
            start = time.perf_counter()
            async with semaphore:
                enrich = await _enrich_async(meta, raw)
            elapsed = time.perf_counter() - start
            logging.info("[v2/router] vid=%s enriched in %.2fs", video_id, elapsed)
            try:
                router_cache_save(video_id, enrich)
            except Exception as exc:
                logging.warning("[v2/router/cache] save failed for %s: %s", video_id, ascii_safe(str(exc)))
            return meta, enrich, None, elapsed
        except Exception as exc:
            return meta, None, exc, 0.0

    tasks = [asyncio.create_task(_one(meta, raw)) for (meta, raw) in pairs]
    return await asyncio.gather(*tasks)


def enrich_assets(
    assets: Iterable[Asset],
    concurrency: int,
) -> Tuple[List[Dict[str, Any]], Dict[str, Any]]:
    if os.getenv("ROUTER_ENRICH", "1").strip().lower() in ("0", "false", "no", "off"):
        # Bulk local ingest should not require (or pay for) router enrichment.
        metas = [meta for (meta, _raw, _path) in assets]
        return metas, {
            "cached_hits": 0,
            "fresh": 0,
            "failed": 0,
            "total_time": 0.0,
            "skipped": len(metas),
        }

    enriched_metas: List[Dict[str, Any]] = []
    cached_hits = 0
    fresh = 0
    failed = 0
    total_time = 0.0

    to_enrich: List[Tuple[Dict[str, Any], Dict[str, Any]]] = []
    for meta, raw, _ in assets:
        video_id = meta["video_id"]
        enrich = None
        try:
            enrich = router_cache_load(video_id)
        except Exception as exc:
            logging.warning("[v2/router/cache] read failed for %s: %s", video_id, ascii_safe(str(exc)))

        if enrich is not None:
            cached_hits += 1
            _log_cache_hit(video_id, enrich)
            enriched_metas.append(_merge_meta(meta, enrich))
        else:
            to_enrich.append((meta, raw))

    if to_enrich:
        results = asyncio.run(_gather_enrichment(to_enrich, concurrency))
        for meta, enrich, err, elapsed in results:
            video_id = meta["video_id"]
            if err or enrich is None:
                logging.warning("[v2/router] enrichment failed for %s: %s", video_id, ascii_safe(str(err)))
                failed += 1
                enrich = {
                    "description": meta.get("description") or "",
                    "topic_summary": "",
                    "router_tags": [],
                    "aliases": [],
                    "canonical_entities": [],
                    "is_explainer": False,
                    "router_boost": 1.0,
                }
            else:
                fresh += 1
                total_time += elapsed
            enriched_metas.append(_merge_meta(meta, enrich))

    stats = {
        "cached_hits": cached_hits,
        "fresh": fresh,
        "failed": failed,
        "total_time": total_time,
    }
    return enriched_metas, stats
