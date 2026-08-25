from __future__ import annotations

import hashlib
import json
import os
import stat
import tempfile
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from sqlalchemy import text
from sqlalchemy.orm import Session


class ArchiveProtocolError(RuntimeError):
    """An immutable archive input, receipt, or transaction lock is unsafe."""


@dataclass(frozen=True)
class ValidatedFile:
    path: Path
    payload: bytes
    sha256: str


@dataclass(frozen=True)
class ValidatedLargeFile:
    path: Path
    sha256: str
    size_bytes: int
    device: int
    inode: int
    mode: int


def canonical_json_bytes(payload: Any) -> bytes:
    return (
        json.dumps(payload, ensure_ascii=True, separators=(",", ":"), sort_keys=True)
        + "\n"
    ).encode("ascii")


def sha256_json(payload: Any) -> str:
    return hashlib.sha256(canonical_json_bytes(payload)).hexdigest()


def safe_absolute_path(path: Path) -> Path:
    expanded = Path(path).expanduser()
    return Path(os.path.abspath(os.fspath(expanded)))


def assert_no_symlink_components(path: Path, *, include_leaf: bool = True) -> None:
    absolute = safe_absolute_path(path)
    components = absolute.parts
    current = Path(components[0])
    last_index = len(components) - (0 if include_leaf else 1)
    for component in components[1:last_index]:
        current /= component
        try:
            metadata = current.lstat()
        except FileNotFoundError:
            continue
        if stat.S_ISLNK(metadata.st_mode):
            raise ArchiveProtocolError(
                f"symlink path component is forbidden: {current}"
            )


def read_pinned_regular_file(
    path: Path,
    *,
    expected_sha256: str | None = None,
    max_bytes: int = 64 * 1024 * 1024,
) -> ValidatedFile:
    absolute = safe_absolute_path(path)
    assert_no_symlink_components(absolute)
    flags = os.O_RDONLY | getattr(os, "O_CLOEXEC", 0) | getattr(os, "O_NOFOLLOW", 0)
    try:
        descriptor = os.open(absolute, flags)
    except OSError as exc:
        raise ArchiveProtocolError(
            f"unable to open immutable regular file: {absolute}"
        ) from exc
    try:
        before = os.fstat(descriptor)
        if not stat.S_ISREG(before.st_mode):
            raise ArchiveProtocolError(f"path is not a regular file: {absolute}")
        if before.st_size > max_bytes:
            raise ArchiveProtocolError(
                f"file exceeds the {max_bytes}-byte limit: {absolute}"
            )
        chunks: list[bytes] = []
        total = 0
        while True:
            chunk = os.read(descriptor, min(1024 * 1024, max_bytes + 1 - total))
            if not chunk:
                break
            chunks.append(chunk)
            total += len(chunk)
            if total > max_bytes:
                raise ArchiveProtocolError(
                    f"file exceeds the {max_bytes}-byte limit: {absolute}"
                )
        after = os.fstat(descriptor)
        if (
            before.st_dev,
            before.st_ino,
            before.st_size,
            before.st_mtime_ns,
        ) != (
            after.st_dev,
            after.st_ino,
            after.st_size,
            after.st_mtime_ns,
        ):
            raise ArchiveProtocolError(f"file changed while being read: {absolute}")
    finally:
        os.close(descriptor)
    payload = b"".join(chunks)
    digest = hashlib.sha256(payload).hexdigest()
    if expected_sha256 is not None and digest != expected_sha256:
        raise ArchiveProtocolError(
            "file SHA-256 does not match the caller-pinned digest"
        )
    return ValidatedFile(path=absolute, payload=payload, sha256=digest)


def hash_pinned_regular_file(
    path: Path,
    *,
    expected_sha256: str,
    expected_size_bytes: int,
) -> ValidatedLargeFile:
    absolute = safe_absolute_path(path)
    assert_no_symlink_components(absolute)
    flags = os.O_RDONLY | getattr(os, "O_CLOEXEC", 0) | getattr(os, "O_NOFOLLOW", 0)
    try:
        descriptor = os.open(absolute, flags)
    except OSError as exc:
        raise ArchiveProtocolError(
            f"unable to open immutable regular file: {absolute}"
        ) from exc
    try:
        before = os.fstat(descriptor)
        if not stat.S_ISREG(before.st_mode):
            raise ArchiveProtocolError(f"path is not a regular file: {absolute}")
        if before.st_size != expected_size_bytes:
            raise ArchiveProtocolError("retained file size does not match the receipt")
        digest = hashlib.sha256()
        for chunk in iter(lambda: os.read(descriptor, 4 * 1024 * 1024), b""):
            digest.update(chunk)
        after = os.fstat(descriptor)
        if (
            before.st_dev,
            before.st_ino,
            before.st_size,
            before.st_mtime_ns,
        ) != (
            after.st_dev,
            after.st_ino,
            after.st_size,
            after.st_mtime_ns,
        ):
            raise ArchiveProtocolError(f"file changed while being hashed: {absolute}")
        actual_sha256 = digest.hexdigest()
        if actual_sha256 != expected_sha256:
            raise ArchiveProtocolError(
                "retained file SHA-256 does not match the receipt"
            )
        return ValidatedLargeFile(
            path=absolute,
            sha256=actual_sha256,
            size_bytes=before.st_size,
            device=before.st_dev,
            inode=before.st_ino,
            mode=before.st_mode,
        )
    finally:
        os.close(descriptor)


def prepare_immutable_receipt_dir(receipt_dir: Path) -> Path:
    absolute = safe_absolute_path(receipt_dir)
    assert_no_symlink_components(absolute, include_leaf=False)
    absolute.mkdir(parents=True, exist_ok=True, mode=0o750)
    assert_no_symlink_components(absolute)
    metadata = absolute.lstat()
    if not stat.S_ISDIR(metadata.st_mode):
        raise ArchiveProtocolError("receipt destination must be a directory")
    return absolute


def write_immutable_json_receipt(
    receipt: dict[str, Any], *, receipt_dir: Path, schema: str
) -> tuple[Path, str]:
    directory = prepare_immutable_receipt_dir(receipt_dir)
    payload = canonical_json_bytes(receipt)
    payload_sha256 = hashlib.sha256(payload).hexdigest()
    destination = directory / f"{schema}-{payload_sha256}.json"
    descriptor, temporary_name = tempfile.mkstemp(
        prefix=".archive-receipt-", dir=directory
    )
    temporary_path = Path(temporary_name)
    try:
        with os.fdopen(descriptor, "wb") as handle:
            handle.write(payload)
            handle.flush()
            os.fsync(handle.fileno())
        os.chmod(temporary_path, 0o444)
        try:
            os.link(temporary_path, destination)
            _fsync_directory(directory)
        except FileExistsError:
            existing = read_pinned_regular_file(
                destination,
                expected_sha256=payload_sha256,
                max_bytes=max(len(payload), 1),
            )
            if existing.payload != payload:
                raise ArchiveProtocolError("immutable receipt collision")
        if destination.lstat().st_mode & 0o222:
            raise ArchiveProtocolError("immutable receipt is unexpectedly writable")
    finally:
        temporary_path.unlink(missing_ok=True)
    return destination, payload_sha256


def acquire_transaction_lock(session: Session, lock_identity: str) -> None:
    """Serialize one admin apply boundary before its first database read."""
    if session.in_transaction():
        raise ArchiveProtocolError(
            "archive admin operation requires a fresh transaction"
        )
    digest = hashlib.sha256(lock_identity.encode("utf-8")).digest()
    dialect = session.get_bind().dialect.name
    if dialect == "postgresql":
        lock_value = int.from_bytes(digest[:8], "big", signed=True)
        session.execute(
            text("SELECT pg_advisory_xact_lock(:lock_value)"),
            {"lock_value": lock_value},
        )
    elif dialect == "sqlite":
        session.connection().exec_driver_sql("BEGIN IMMEDIATE")


def _fsync_directory(path: Path) -> None:
    descriptor = os.open(path, os.O_RDONLY | getattr(os, "O_CLOEXEC", 0))
    try:
        os.fsync(descriptor)
    finally:
        os.close(descriptor)
