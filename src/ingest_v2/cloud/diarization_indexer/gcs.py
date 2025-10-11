from __future__ import annotations

import json
import subprocess
import tempfile
from pathlib import Path
from typing import Tuple

try:  # pragma: no cover - optional dependency resolved at runtime
    from google.auth.exceptions import DefaultCredentialsError
    from google.cloud import storage  # type: ignore
except Exception:  # pragma: no cover - fallback when SDK missing
    DefaultCredentialsError = Exception  # type: ignore
    storage = None  # type: ignore


def read_json_from_gcs(uri: str) -> dict:
    if uri.startswith("file://"):
        return json.loads(Path(uri[7:]).read_text(encoding="utf-8"))
    if not uri.startswith("gs://"):
        return json.loads(Path(uri).read_text(encoding="utf-8"))
    bucket, blob = _split_gs_uri(uri)
    if storage is not None:
        try:
            client = storage.Client()
            data = client.bucket(bucket).blob(blob).download_as_bytes()
            return json.loads(data.decode("utf-8"))
        except DefaultCredentialsError:
            pass
    data = _read_with_gsutil(uri)
    return json.loads(data.decode("utf-8"))


def download_to_temp(uri: str, suffix: str = "") -> Path:
    if uri.startswith("file://"):
        src = Path(uri[7:])
        tmp = Path(tempfile.mkstemp(suffix=suffix)[1])
        tmp.write_bytes(src.read_bytes())
        return tmp
    if not uri.startswith("gs://"):
        src = Path(uri)
        tmp = Path(tempfile.mkstemp(suffix=suffix)[1])
        tmp.write_bytes(src.read_bytes())
        return tmp
    bucket, blob = _split_gs_uri(uri)
    with tempfile.NamedTemporaryFile(suffix=suffix, delete=False) as tmp_file:
        tmp_path = Path(tmp_file.name)
    if storage is not None:
        try:
            client = storage.Client()
            client.bucket(bucket).blob(blob).download_to_filename(tmp_path.as_posix())
            return tmp_path
        except DefaultCredentialsError:
            pass
    subprocess.run(["gsutil", "cp", uri, tmp_path.as_posix()], check=True)
    return tmp_path


def _read_with_gsutil(uri: str) -> bytes:
    result = subprocess.run(["gsutil", "cat", uri], check=True, capture_output=True)
    return result.stdout


def _split_gs_uri(uri: str) -> Tuple[str, str]:
    if not uri.startswith("gs://"):
        raise ValueError(f"Invalid GCS URI: {uri}")
    without = uri[len("gs://") :]
    bucket, _, blob = without.partition("/")
    if not bucket or not blob:
        raise ValueError(f"Invalid GCS URI: {uri}")
    return bucket, blob
