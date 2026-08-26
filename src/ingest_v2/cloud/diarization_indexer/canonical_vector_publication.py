from __future__ import annotations

import hashlib
import json
import math
from collections.abc import Sequence
from typing import Any

from ...configs.settings import settings_v2
from ...pipelines.build_children import build_children_from_raw
from ...pipelines.upsert_pinecone import _prep_metadata_for_upsert, upsert_children
from ...transcripts.normalize import normalize_to_sentences
from ...utils.ids import segment_uuid, sha1_hex
from ...utils.timefmt import s_to_hms_ms
from ...utils.vector_store import (
    fetch_qdrant_points,
    qdrant_collection_exists,
    qdrant_collection_name,
    vector_store_backend,
)
from .channel_service_config import canonical_namespace, embedding_contract
from .public_acquisition import PublicItemDescriptor
from .transcription_runtime import TranscriptResult

PUBLIC_VECTOR_PUBLICATION_SCHEMA = "icmfyi.canonical-qdrant-publication.v1"
_DOCUMENT_TYPES = {
    "youtube": "youtube_video",
    "twitch": "twitch_vod",
    "pumpfun": "pumpfun_clip",
    "x": "media",
}
_READBACK_FIELDS = (
    "parent_id",
    "media_id",
    "transcript_revision_id",
    "platform",
    "provider_video_id",
    "document_type",
    "embedding_provider",
    "embedding_model",
    "embedding_model_revision",
    "embedding_dimension",
    "source_hash",
    "text",
)


class CanonicalVectorPublicationError(RuntimeError):
    """Canonical transcript vectors are not authoritatively readable in Qdrant."""


def publish_canonical_transcript_vectors(
    *,
    item: PublicItemDescriptor,
    transcript: TranscriptResult,
    media_id: str,
    transcript_revision_id: str,
    language: str,
) -> dict[str, Any]:
    """Idempotently publish and read back one canonical transcript revision."""
    if vector_store_backend() != "qdrant":
        raise CanonicalVectorPublicationError(
            "generic public ingestion requires VECTOR_STORE=qdrant"
        )
    platform = str(item.platform or "").strip().lower()
    document_type = _DOCUMENT_TYPES.get(platform)
    if document_type is None:
        raise CanonicalVectorPublicationError(
            f"unsupported canonical vector platform: {platform or 'missing'}"
        )
    media_id = _required(media_id, "media_id")
    transcript_revision_id = _required(transcript_revision_id, "transcript_revision_id")
    namespace = canonical_namespace()
    collection = qdrant_collection_name(namespace)
    contract = embedding_contract()
    dimension = int(contract["dimension"])
    if dimension != int(settings_v2.EMBED_DIM):
        raise CanonicalVectorPublicationError(
            "embedding contract dimension differs from the vector pipeline"
        )

    children = _canonical_children(
        item=item,
        transcript=transcript,
        media_id=media_id,
        transcript_revision_id=transcript_revision_id,
        language=language,
        document_type=document_type,
        embedding=contract,
    )
    point_ids = [str(child["segment_id"]) for child in children]
    if len(set(point_ids)) != len(point_ids):
        raise CanonicalVectorPublicationError(
            "canonical transcript produced duplicate point identities"
        )
    prepared_payloads = {
        str(child["segment_id"]): _prep_metadata_for_upsert(child) for child in children
    }
    missing_readback_fields = {
        point_id: [field for field in _READBACK_FIELDS if field not in prepared_payload]
        for point_id, prepared_payload in prepared_payloads.items()
        if any(field not in prepared_payload for field in _READBACK_FIELDS)
    }
    if missing_readback_fields:
        raise CanonicalVectorPublicationError(
            "canonical vector metadata limits remove required readback identity"
        )
    expected = {
        point_id: {field: prepared_payload[field] for field in _READBACK_FIELDS}
        for point_id, prepared_payload in prepared_payloads.items()
    }
    point_ids = sorted(expected)

    before = (
        fetch_qdrant_points(collection_name=collection, ids=point_ids)
        if qdrant_collection_exists(collection)
        else {}
    )
    matching_before = {
        point_id
        for point_id, expected_payload in expected.items()
        if _point_matches(before.get(point_id), expected_payload, dimension=dimension)
    }
    pending = [
        child for child in children if str(child["segment_id"]) not in matching_before
    ]
    if pending:
        upsert_children(
            pending,
            qdrant_namespace=namespace,
            qdrant_wait=True,
        )

    readback = fetch_qdrant_points(collection_name=collection, ids=point_ids)
    incomplete = [
        point_id
        for point_id, expected_payload in expected.items()
        if not _point_matches(
            readback.get(point_id), expected_payload, dimension=dimension
        )
    ]
    if incomplete:
        raise CanonicalVectorPublicationError(
            "canonical Qdrant readback is incomplete for "
            f"{len(incomplete)} of {len(point_ids)} transcript points"
        )

    readback_rows = []
    for point_id in point_ids:
        point = readback[point_id]
        payload = point["payload"]
        vector = list(point["vector"])
        readback_rows.append(
            {
                "id": point_id,
                "payload": {field: payload[field] for field in _READBACK_FIELDS},
                "vector_sha256": hashlib.sha256(
                    json.dumps(
                        vector,
                        allow_nan=False,
                        ensure_ascii=True,
                        separators=(",", ":"),
                    ).encode("ascii")
                ).hexdigest(),
            }
        )
    readback_sha256 = hashlib.sha256(
        json.dumps(
            readback_rows,
            ensure_ascii=True,
            separators=(",", ":"),
            sort_keys=True,
        ).encode("ascii")
    ).hexdigest()
    return {
        "schema": PUBLIC_VECTOR_PUBLICATION_SCHEMA,
        "collection": collection,
        "media_id": media_id,
        "transcript_revision_id": transcript_revision_id,
        "point_count": len(point_ids),
        "reused_point_count": len(matching_before),
        "published_point_count": len(pending),
        "readback_sha256": readback_sha256,
        "embedding": contract,
    }


def _canonical_children(
    *,
    item: PublicItemDescriptor,
    transcript: TranscriptResult,
    media_id: str,
    transcript_revision_id: str,
    language: str,
    document_type: str,
    embedding: dict[str, str | int],
) -> list[dict[str, Any]]:
    raw_segments = [_raw_segment(segment) for segment in transcript.segments]
    if not raw_segments:
        raise CanonicalVectorPublicationError(
            "canonical transcript has no segments to publish"
        )
    duration_s = max(
        float(item.duration_ms or 0) / 1000.0,
        max(float(segment["end"]) for segment in raw_segments),
    )
    if duration_s <= 0:
        raise CanonicalVectorPublicationError(
            "canonical transcript duration must be positive"
        )
    parent = {
        "parent_id": media_id,
        "media_id": media_id,
        "document_type": document_type,
        "title": item.title or item.external_id,
        "description": item.description or "",
        "channel_name": item.channel_handle,
        "channel_id": item.channel_external_id,
        "published_at": item.published_at,
        "duration_s": duration_s,
        "url": item.canonical_url,
        "language": language,
        "rights": "public_reference_only",
        "entities": [],
        "canonical_entities": [],
    }
    raw = {"segments": raw_segments}
    children = build_children_from_raw(parent, raw)
    canonical_text = " ".join(
        sentence["text"] for sentence in normalize_to_sentences(raw)
    ).strip()
    if not canonical_text:
        raise CanonicalVectorPublicationError(
            "canonical transcript has no non-empty text to publish"
        )
    if not any(
        str(child.get("text") or "").strip() in canonical_text
        for child in children
        if str(child.get("text") or "").strip()
    ):
        children = [
            _fallback_transcript_child(
                parent=parent,
                raw_segments=raw_segments,
                text=canonical_text,
            )
        ]
    for child in children:
        child.setdefault("source_hash", _source_hash(child))
        child.update(
            {
                "media_id": media_id,
                "transcript_revision_id": transcript_revision_id,
                "platform": item.platform,
                "provider_video_id": item.external_id,
                "video_id": item.external_id,
                "channel_id": item.channel_external_id,
                "channel_name": item.channel_handle,
                "published_at": item.published_at,
                "duration_s": duration_s,
                "url": item.canonical_url,
                "title": item.title,
                "description": item.description,
                "transcript_provider": transcript.provider,
                "source": item.platform,
                "embedding_provider": embedding["provider"],
                "embedding_model": embedding["model"],
                "embedding_model_revision": embedding["revision"],
                "embedding_dimension": embedding["dimension"],
            }
        )
    return children


def _source_hash(child: dict[str, Any]) -> str:
    try:
        start_s = float(child["start_s"])
        end_s = float(child["end_s"])
    except (KeyError, TypeError, ValueError) as exc:
        raise CanonicalVectorPublicationError(
            "canonical transcript point timestamps are invalid"
        ) from exc
    raw_bytes = json.dumps(
        {
            "text": str(child.get("text") or ""),
            "start": start_s,
            "end": end_s,
        },
        sort_keys=True,
    ).encode("utf-8")
    return sha1_hex(raw_bytes)


def _raw_segment(segment: dict[str, Any]) -> dict[str, Any]:
    text = str(segment.get("text") or "").strip()
    if not text:
        raise CanonicalVectorPublicationError(
            "canonical transcript segment text is empty"
        )
    try:
        start_ms = int(segment["start_ms"])
        end_ms = int(segment["end_ms"])
    except (KeyError, TypeError, ValueError) as exc:
        raise CanonicalVectorPublicationError(
            "canonical transcript segment timestamps are invalid"
        ) from exc
    if start_ms < 0 or end_ms <= start_ms:
        raise CanonicalVectorPublicationError(
            "canonical transcript segment timestamps are invalid"
        )
    return {
        "start": start_ms / 1000.0,
        "end": end_ms / 1000.0,
        "speaker": segment.get("speaker_label"),
        "text": text,
    }


def _fallback_transcript_child(
    *,
    parent: dict[str, Any],
    raw_segments: list[dict[str, Any]],
    text: str,
) -> dict[str, Any]:
    start_s = min(float(segment["start"]) for segment in raw_segments)
    end_s = max(float(segment["end"]) for segment in raw_segments)
    raw_bytes = json.dumps(
        {"text": text, "start": start_s, "end": end_s},
        sort_keys=True,
    ).encode("utf-8")
    return {
        "node_type": "child",
        "segment_id": segment_uuid(parent["parent_id"], start_s, end_s),
        "parent_id": parent["parent_id"],
        "document_type": parent["document_type"],
        "text": text,
        "start_s": start_s,
        "end_s": end_s,
        "start_hms": s_to_hms_ms(start_s),
        "end_hms": s_to_hms_ms(end_s),
        "clip_url": None,
        "speaker": raw_segments[0].get("speaker"),
        "entities": [],
        "chapter": None,
        "language": parent["language"],
        "confidence_asr": None,
        "has_music": False,
        "flags": [],
        "rights": parent["rights"],
        "ingest_version": 2,
        "source_hash": sha1_hex(raw_bytes),
    }


def _point_matches(
    point: dict[str, Any] | None,
    expected_payload: dict[str, Any],
    *,
    dimension: int,
) -> bool:
    if not isinstance(point, dict):
        return False
    payload = point.get("payload")
    vector = point.get("vector")
    if not isinstance(payload, dict):
        return False
    if any(payload.get(field) != value for field, value in expected_payload.items()):
        return False
    if (
        isinstance(vector, (str, bytes, bytearray))
        or not isinstance(vector, Sequence)
        or len(vector) != dimension
    ):
        return False
    return all(
        isinstance(value, (int, float))
        and not isinstance(value, bool)
        and math.isfinite(float(value))
        for value in vector
    )


def _required(value: Any, field_name: str) -> str:
    normalized = str(value or "").strip()
    if not normalized:
        raise CanonicalVectorPublicationError(f"{field_name} is required")
    return normalized
