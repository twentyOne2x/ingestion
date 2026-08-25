#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
from pathlib import Path

from src.ingest_v2.cloud.diarization_indexer.archive_catalog_loader import (
    load_archive_catalog,
    write_archive_catalog_receipt,
)
from src.ingest_v2.cloud.diarization_indexer.channel_service_store import session_scope


def main() -> int:
    parser = argparse.ArgumentParser(
        description="Load one immutable icmfyi.archive-catalog-import.v1 JSONL packet"
    )
    parser.add_argument("jsonl", type=Path)
    parser.add_argument("sidecar", type=Path)
    parser.add_argument("--expect-jsonl-sha256", required=True)
    parser.add_argument("--receipt-dir", type=Path, required=True)
    arguments = parser.parse_args()

    with session_scope() as session:
        receipt = load_archive_catalog(
            session,
            jsonl_path=arguments.jsonl,
            sidecar_path=arguments.sidecar,
            expected_jsonl_sha256=arguments.expect_jsonl_sha256,
        )
    receipt_path, receipt_sha256 = write_archive_catalog_receipt(
        receipt,
        receipt_dir=arguments.receipt_dir,
    )
    print(
        json.dumps(
            {
                "ok": True,
                "receipt_path": str(receipt_path),
                "receipt_sha256": receipt_sha256,
                "counts": receipt["counts"],
            },
            sort_keys=True,
        )
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
