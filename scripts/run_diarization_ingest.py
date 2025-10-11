#!/usr/bin/env python3
"""
Replay a diarization-ready event through the ingestion pipeline.

Usage:
    python scripts/run_diarization_ingest.py --namespace videos --event-file event.json

The event JSON should contain the fields expected by ``DiarizationReadyEvent``. For
local testing you may use ``file://`` URIs that point to JSON artefacts on disk.
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

from src.ingest_v2.cloud.diarization_indexer.ingest import create_ingest_service
from src.ingest_v2.cloud.diarization_indexer.schemas import DiarizationReadyEvent
from src.ingest_v2.pipelines.run_all_components.namespace import load_namespace_channels


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Replay a diarization-ready event")
    parser.add_argument("--namespace", default="videos", help="Ingestion namespace")
    parser.add_argument(
        "--event-file",
        type=Path,
        required=True,
        help="Path to JSON file containing diarization-ready payload",
    )
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    if not args.event_file.exists():
        print(f"Event file {args.event_file} not found", file=sys.stderr)
        return 1

    payload = json.loads(args.event_file.read_text(encoding="utf-8"))
    event = DiarizationReadyEvent.from_payload(payload)

    channels = load_namespace_channels(args.namespace)
    if not channels:
        print(f"No channels configured for namespace '{args.namespace}'", file=sys.stderr)
        return 1

    service = create_ingest_service(args.namespace, channels)
    service.handle_event(event)
    print(f"Ingested diarization event for video {event.diarized_uri}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
