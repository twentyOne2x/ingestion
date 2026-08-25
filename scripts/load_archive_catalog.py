#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import os
import sys
from pathlib import Path

REPOSITORY_ROOT = Path(__file__).resolve().parents[1]
if str(REPOSITORY_ROOT) not in sys.path:
    sys.path.insert(0, str(REPOSITORY_ROOT))

from src.ingest_v2.cloud.diarization_indexer.archive_admin import (
    claim_archive_sources,
    register_hot_media_hydration,
)
from src.ingest_v2.cloud.diarization_indexer.archive_catalog_loader import (
    apply_archive_catalog,
)
from src.ingest_v2.cloud.diarization_indexer.channel_service_store import session_scope


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Crash-safe internal archive catalog administration"
    )
    commands = parser.add_subparsers(dest="command", required=True)

    apply_parser = commands.add_parser(
        "apply", help="apply one immutable icmfyi.archive-catalog-import.v1 packet"
    )
    apply_parser.add_argument("jsonl", type=Path)
    apply_parser.add_argument("sidecar", type=Path)
    apply_parser.add_argument("--expect-jsonl-sha256", required=True)
    apply_parser.add_argument("--receipt-dir", type=Path, required=True)

    claim_parser = commands.add_parser(
        "claim", help="grant selected imported source channels to an exact tenant"
    )
    claim_parser.add_argument("--catalog-jsonl-sha256", required=True)
    claim_parser.add_argument("--tenant-id", required=True)
    claim_parser.add_argument("--admin-user-id", required=True)
    claim_parser.add_argument("--idempotency-key", required=True)
    claim_parser.add_argument("--source-key", action="append", required=True)
    claim_parser.add_argument("--receipt-dir", type=Path, required=True)

    hydration_parser = commands.add_parser(
        "register-hydration",
        help="register one independently verified archive video in the hot-media CAS",
    )
    hydration_parser.add_argument("source_receipt", type=Path)
    hydration_parser.add_argument("--expect-source-receipt-sha256", required=True)
    hydration_parser.add_argument("--hot-media-root", type=Path, required=True)
    hydration_parser.add_argument("--receipt-dir", type=Path, required=True)
    hydration_parser.add_argument(
        "--ffprobe-bin",
        type=Path,
        default=Path(
            os.getenv("CHANNEL_SERVICE_FFPROBE_BIN") or "/usr/local/bin/ffprobe"
        ),
    )
    return parser


def main(argv: list[str] | None = None) -> int:
    arguments_list = list(sys.argv[1:] if argv is None else argv)
    # Preserve the original positional invocation while making `apply` explicit.
    if (
        arguments_list
        and arguments_list[0]
        not in {
            "apply",
            "claim",
            "register-hydration",
        }
        and not arguments_list[0].startswith("-")
    ):
        arguments_list.insert(0, "apply")
    arguments = _parser().parse_args(arguments_list)

    with session_scope() as session:
        if arguments.command == "apply":
            result = apply_archive_catalog(
                session,
                jsonl_path=arguments.jsonl,
                sidecar_path=arguments.sidecar,
                expected_jsonl_sha256=arguments.expect_jsonl_sha256,
                receipt_dir=arguments.receipt_dir,
            )
        elif arguments.command == "claim":
            result = claim_archive_sources(
                session,
                catalog_jsonl_sha256=arguments.catalog_jsonl_sha256,
                tenant_id=arguments.tenant_id,
                admin_user_id=arguments.admin_user_id,
                idempotency_key=arguments.idempotency_key,
                source_keys=arguments.source_key,
                receipt_dir=arguments.receipt_dir,
            )
        else:
            result = register_hot_media_hydration(
                session,
                source_receipt_path=arguments.source_receipt,
                expected_source_receipt_sha256=arguments.expect_source_receipt_sha256,
                hot_media_root=arguments.hot_media_root,
                receipt_dir=arguments.receipt_dir,
                ffprobe_bin=arguments.ffprobe_bin,
            )

    print(
        json.dumps(
            {
                "ok": True,
                "operation": arguments.command,
                "receipt_path": str(result.receipt_path),
                "receipt_sha256": result.receipt_sha256,
                "reconciled": result.reconciled,
                "counts": result.receipt.get("counts", {}),
            },
            sort_keys=True,
        )
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
