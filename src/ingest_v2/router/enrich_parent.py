# File: src/ingest_v2/router/enrich_parent.py
from __future__ import annotations

import asyncio
import json
import logging
import os
from typing import Any, Dict, List

from ..configs.settings import settings_v2
from ..utils.backoff import expo_backoff  # used by sync path
from openai import OpenAI, AsyncOpenAI


def _snip(s: str, n: int) -> str:
    s = (s or "").strip()
    return s if len(s) <= n else (s[: max(0, n - 1)] + "…")


def _summarize_sentences_for_prompt(
    sentences: List[Dict[str, Any]],
    cap: int,
) -> List[Dict[str, Any]]:
    if not sentences:
        return []
    # Take earliest sentences (cap) to keep token usage predictable
    return sentences[:cap]


def _build_messages(meta: Dict[str, Any], sentences: List[Dict[str, Any]]) -> List[Dict[str, str]]:
    title = meta.get("title") or ""
    channel = meta.get("channel_name") or ""
    published = meta.get("published_at") or ""
    ents = meta.get("entities") or []
    base_desc = meta.get("description") or ""

    # Convert sentences to a compact JSON-like block (timestamps + text)
    s_lines: List[Dict[str, Any]] = []
    for s in sentences:
        try:
            s_lines.append(
                {
                    "start_s": float(s.get("start_s", 0.0)),
                    "end_s": float(s.get("end_s", 0.0)),
                    "text": (s.get("text") or "").strip(),
                }
            )
        except Exception:
            continue

    sys = (
        "You are a data enrichment worker for a video search router.\n"
        "Return STRICT JSON. No commentary. No markdown.\n"
        "Use inline timestamp references like [t=MM:SS] in description/topic_summary "
        "where helpful (derive from start_s/end_s). Keep language concise and neutral.\n"
        "router_boost should be a float between 1.0 and 3.0 (default 1.0). "
        "aliases should include reasonable capitalization variants. "
        "canonical_entities should contain normalized tickers/handles eg. ['$SOL','@anatoly'] "
        "based only on provided text.\n"
    )

    user = {
        "title": title,
        "channel_name": channel,
        "published_at": published,
        "base_description": base_desc,
        "entities_seed": ents,
        "transcript_preview": s_lines,
        "want_fields": [
            "description",
            "topic_summary",
            "router_tags",
            "aliases",
            "canonical_entities",
            "is_explainer",
            "router_boost",
        ],
        "format": {
            "type": "object",
            "properties": {
                "description": {"type": "string"},
                "topic_summary": {"type": "string"},
                "router_tags": {"type": "array", "items": {"type": "string"}},
                "aliases": {"type": "array", "items": {"type": "string"}},
                "canonical_entities": {"type": "array", "items": {"type": "string"}},
                "is_explainer": {"type": "boolean"},
                "router_boost": {"type": "number"},
            },
            "required": [
                "description",
                "topic_summary",
                "router_tags",
                "aliases",
                "canonical_entities",
                "is_explainer",
                "router_boost",
            ],
            "additionalProperties": False,
        },
    }

    return [
        {"role": "system", "content": sys},
        {"role": "user", "content": json.dumps(user, ensure_ascii=False)},
    ]


def _parse_json(s: str) -> Dict[str, Any]:
    try:
        return json.loads(s)
    except Exception:
        # Try to salvage a JSON object if model added noise
        start = s.find("{")
        end = s.rfind("}")
        if start >= 0 and end > start:
            try:
                return json.loads(s[start : end + 1])
            except Exception:
                pass
        raise


def enrich_parent_router_fields(meta: Dict[str, Any], sentences: List[Dict[str, Any]]) -> Dict[str, Any]:
    """
    Sync version.
    Returns dict of fields to merge into meta:
      description, topic_summary, router_tags, aliases, canonical_entities, is_explainer, router_boost
    """
    preview = _summarize_sentences_for_prompt(sentences, settings_v2.ROUTER_GEN_MAX_SENTENCES)
    messages = _build_messages(meta, preview)

    client = OpenAI()
    model = settings_v2.ROUTER_GEN_MODEL
    last_err = None

    for attempt in range(settings_v2.ROUTER_GEN_RETRIES + 1):
        try:
            resp = client.chat.completions.create(
                model=model,
                messages=messages,
                temperature=0.2,
                response_format={"type": "json_object"},
                timeout=45,  # seconds (prevents a hung request from stalling the run)
            )
            raw = resp.choices[0].message.content or "{}"
            data = _parse_json(raw)

            # Light schema guardrails + clamp router_boost to [1.0, 3.0]
            rb = float(data.get("router_boost") or 1.0)
            rb = max(1.0, min(3.0, rb))

            out = {
                "description": data.get("description") or meta.get("description") or "",
                "topic_summary": data.get("topic_summary") or "",
                "router_tags": [str(x) for x in (data.get("router_tags") or [])][:24],
                "aliases": [str(x) for x in (data.get("aliases") or [])][:24],
                "canonical_entities": [str(x) for x in (data.get("canonical_entities") or [])][:24],
                "is_explainer": bool(data.get("is_explainer")),
                "router_boost": rb,
            }

            # Observability preview
            n = int(os.getenv("ROUTER_LOG_DESC_CHARS", "240"))
            vid = meta.get("video_id") or "unknown"
            logging.info(
                "[router/enrich] vid=%s boost=%.2f tags=%s desc_preview=%r topic_preview=%r",
                vid,
                rb,
                out["router_tags"][:6],
                _snip(out["description"], n),
                _snip(out["topic_summary"], max(80, n // 2)),
            )
            return out
        except Exception as e:
            last_err = e
            logging.warning(f"[router/enrich] attempt={attempt + 1} err={e}")
            expo_backoff(attempt)

    logging.error(f"[router/enrich] failed after retries: {last_err}")
    # Fallback: minimal fields so we keep pipeline flowing
    return {
        "description": meta.get("description") or "",
        "topic_summary": "",
        "router_tags": [],
        "aliases": [],
        "canonical_entities": [],
        "is_explainer": False,
        "router_boost": 1.0,
    }


async def enrich_parent_router_fields_async(
    meta: Dict[str, Any], sentences: List[Dict[str, Any]]
) -> Dict[str, Any]:
    """
    Async version (for concurrent use). Same fields/behavior as sync.
    """
    preview = _summarize_sentences_for_prompt(sentences, settings_v2.ROUTER_GEN_MAX_SENTENCES)
    messages = _build_messages(meta, preview)

    client = AsyncOpenAI()
    model = settings_v2.ROUTER_GEN_MODEL
    last_err = None

    for attempt in range(settings_v2.ROUTER_GEN_RETRIES + 1):
        try:
            resp = await client.chat.completions.create(
                model=model,
                messages=messages,
                temperature=0.2,
                response_format={"type": "json_object"},
                timeout=45,  # seconds
            )
            raw = resp.choices[0].message.content or "{}"
            data = _parse_json(raw)

            rb = float(data.get("router_boost") or 1.0)
            rb = max(1.0, min(3.0, rb))

            out = {
                "description": data.get("description") or meta.get("description") or "",
                "topic_summary": data.get("topic_summary") or "",
                "router_tags": [str(x) for x in (data.get("router_tags") or [])][:24],
                "aliases": [str(x) for x in (data.get("aliases") or [])][:24],
                "canonical_entities": [str(x) for x in (data.get("canonical_entities") or [])][:24],
                "is_explainer": bool(data.get("is_explainer")),
                "router_boost": rb,
            }

            n = int(os.getenv("ROUTER_LOG_DESC_CHARS", "240"))
            vid = meta.get("video_id") or "unknown"
            logging.info(
                "[router/enrich/async] vid=%s boost=%.2f tags=%s desc_preview=%r topic_preview=%r",
                vid,
                rb,
                out["router_tags"][:6],
                _snip(out["description"], n),
                _snip(out["topic_summary"], max(80, n // 2)),
            )
            return out
        except Exception as e:
            last_err = e
            logging.warning(f"[router/enrich/async] attempt={attempt + 1} err={e}")
            # simple async backoff
            await asyncio.sleep(min(20.0, 0.5 * (2 ** attempt)))

    logging.error(f"[router/enrich/async] failed after retries: {last_err}")
    return {
        "description": meta.get("description") or "",
        "topic_summary": "",
        "router_tags": [],
        "aliases": [],
        "canonical_entities": [],
        "is_explainer": False,
        "router_boost": 1.0,
    }
