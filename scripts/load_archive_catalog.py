#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import os
import re
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
from src.ingest_v2.cloud.diarization_indexer.archive_receipts import (
    ArchiveProtocolError,
    read_pinned_regular_file,
)
from src.ingest_v2.cloud.diarization_indexer.channel_service_store import session_scope

_SHA256 = re.compile(r"[0-9a-f]{64}\Z")
_HYDRATION_BATCH_SCHEMA = "icmfyi.archive-hot-hydration.v1"


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

    batch_parser = commands.add_parser(
        "register-hydration-batch",
        help="idempotently register every per-media receipt from one pinned hydration receipt",
    )
    batch_parser.add_argument("hydration_receipt", type=Path)
    batch_parser.add_argument("--expect-hydration-receipt-sha256", required=True)
    batch_parser.add_argument("--hot-media-root", type=Path, required=True)
    batch_parser.add_argument("--receipt-dir", type=Path, required=True)
    batch_parser.add_argument(
        "--ffprobe-bin",
        type=Path,
        default=Path(
            os.getenv("CHANNEL_SERVICE_FFPROBE_BIN") or "/usr/local/bin/ffprobe"
        ),
    )
    return parser


def _hydration_batch(
    path: Path, *, expected_sha256: str, hot_media_root: Path
) -> list[tuple[Path, str]]:
    if not _SHA256.fullmatch(expected_sha256):
        raise ArchiveProtocolError("hydration batch SHA-256 is invalid")
    validated = read_pinned_regular_file(
        path, expected_sha256=expected_sha256, max_bytes=256 * 1024 * 1024
    )
    try:
        payload = json.loads(validated.payload.decode("ascii"))
    except (UnicodeDecodeError, json.JSONDecodeError) as exc:
        raise ArchiveProtocolError("hydration batch must be ASCII JSON") from exc
    if (
        not isinstance(payload, dict)
        or payload.get("schema") != _HYDRATION_BATCH_SCHEMA
        or payload.get("complete") is not True
        or payload.get("storagebox_read_only") is not True
        or payload.get("remote_delete_used") is not False
    ):
        raise ArchiveProtocolError("hydration batch schema/state is invalid")
    items = payload.get("items")
    if not isinstance(items, list) or not items or len(items) > 100_000:
        raise ArchiveProtocolError("hydration batch item cardinality is invalid")
    if payload.get("items_count") != len(items):
        raise ArchiveProtocolError("hydration batch item count is inconsistent")
    root = Path(os.path.abspath(os.fspath(hot_media_root.expanduser())))
    receipts_root = root / "receipts"
    result: list[tuple[Path, str]] = []
    seen_media: set[str] = set()
    seen_receipts: set[Path] = set()
    byte_total = 0
    for index, item in enumerate(items):
        if not isinstance(item, dict):
            raise ArchiveProtocolError(f"hydration batch item {index} is invalid")
        media_sha256 = item.get("media_sha256")
        receipt_sha256 = item.get("source_receipt_sha256")
        receipt_text = item.get("source_receipt_path")
        size_bytes = item.get("size_bytes")
        if (
            not isinstance(media_sha256, str)
            or not _SHA256.fullmatch(media_sha256)
            or media_sha256 in seen_media
            or not isinstance(receipt_sha256, str)
            or not _SHA256.fullmatch(receipt_sha256)
            or not isinstance(receipt_text, str)
            or not receipt_text.startswith("/")
            or isinstance(size_bytes, bool)
            or not isinstance(size_bytes, int)
            or size_bytes <= 0
            or item.get("local_sha256_verified") is not True
            or item.get("local_size_verified") is not True
            or item.get("remote_delete_used") is not False
        ):
            raise ArchiveProtocolError(f"hydration batch item {index} identity is invalid")
        receipt_path = Path(os.path.abspath(receipt_text))
        try:
            relative = receipt_path.relative_to(receipts_root)
        except ValueError as exc:
            raise ArchiveProtocolError(
                f"hydration batch item {index} receipt escapes hot-media receipts"
            ) from exc
        if (
            len(relative.parts) != 1
            or receipt_path.name != f"hydration-source-{media_sha256}.json"
            or receipt_path in seen_receipts
        ):
            raise ArchiveProtocolError(
                f"hydration batch item {index} receipt path is not canonical"
            )
        seen_media.add(media_sha256)
        seen_receipts.add(receipt_path)
        byte_total += size_bytes
        result.append((receipt_path, receipt_sha256))
    if payload.get("bytes_total") != byte_total:
        raise ArchiveProtocolError("hydration batch byte count is inconsistent")
    return result


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
            "register-hydration-batch",
        }
        and not arguments_list[0].startswith("-")
    ):
        arguments_list.insert(0, "apply")
    arguments = _parser().parse_args(arguments_list)

    if arguments.command == "register-hydration-batch":
        items = _hydration_batch(
            arguments.hydration_receipt,
            expected_sha256=arguments.expect_hydration_receipt_sha256,
            hot_media_root=arguments.hot_media_root,
        )
        reconciled = 0
        for source_receipt_path, source_receipt_sha256 in items:
            with session_scope() as session:
                result = register_hot_media_hydration(
                    session,
                    source_receipt_path=source_receipt_path,
                    expected_source_receipt_sha256=source_receipt_sha256,
                    hot_media_root=arguments.hot_media_root,
                    receipt_dir=arguments.receipt_dir,
                    ffprobe_bin=arguments.ffprobe_bin,
                )
            reconciled += int(result.reconciled)
        print(
            json.dumps(
                {
                    "ok": True,
                    "operation": arguments.command,
                    "items": len(items),
                    "created": len(items) - reconciled,
                    "reconciled": reconciled,
                    "hydration_receipt_sha256": arguments.expect_hydration_receipt_sha256,
                },
                sort_keys=True,
            )
        )
        return 0

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
