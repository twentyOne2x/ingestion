from __future__ import annotations

import hashlib
import json
import os
import re
import stat
import subprocess
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Callable


class TranscriptionConfigurationError(RuntimeError):
    """The selected transcription backend is not safely configured."""


class TranscriptionError(RuntimeError):
    """A transcription attempt failed before an authoritative result."""


class AmbiguousTranscriptionError(TranscriptionError):
    """A paid provider request may have executed and must not be retried blindly."""


@dataclass(frozen=True)
class TranscriptionContract:
    mode: str
    model_id: str
    model_revision: str | None

    def as_payload(self) -> dict[str, str | None]:
        return {
            "mode": self.mode,
            "model_id": self.model_id,
            "model_revision": self.model_revision,
        }


@dataclass(frozen=True)
class TranscriptResult:
    provider: str
    provider_request_id: str | None
    segments: tuple[dict[str, Any], ...]

    @property
    def sha256(self) -> str:
        payload = json.dumps(
            list(self.segments),
            ensure_ascii=True,
            separators=(",", ":"),
            sort_keys=True,
        ).encode("ascii")
        return hashlib.sha256(payload).hexdigest()


DEFAULT_LOCAL_TRANSCRIPTION_MODEL = "openai/whisper-small"
DEFAULT_LOCAL_TRANSCRIPTION_REVISION = "973afd24965f72e36ca33b3055d56a652f456b4d"


def resolve_transcription_contract(requested_mode: str) -> TranscriptionContract:
    mode = str(requested_mode or "").strip().lower()
    if mode == "auto":
        mode = (
            (os.getenv("CHANNEL_SERVICE_TRANSCRIPTION_MODE") or "local_cpu")
            .strip()
            .lower()
        )
    if mode == "openai":
        model = (
            os.getenv("CHANNEL_SERVICE_OPENAI_TRANSCRIPTION_MODEL")
            or "gpt-4o-mini-transcribe"
        ).strip()
        if not model or len(model) > 255:
            raise TranscriptionConfigurationError(
                "OpenAI transcription model is invalid"
            )
        return TranscriptionContract(mode=mode, model_id=model, model_revision=None)
    if mode == "local_cpu":
        model = (
            os.getenv("CHANNEL_SERVICE_LOCAL_TRANSCRIPTION_MODEL")
            or DEFAULT_LOCAL_TRANSCRIPTION_MODEL
        ).strip()
        revision = (
            os.getenv("CHANNEL_SERVICE_LOCAL_TRANSCRIPTION_REVISION")
            or DEFAULT_LOCAL_TRANSCRIPTION_REVISION
        ).strip()
        if not model or len(model) > 255:
            raise TranscriptionConfigurationError(
                "local CPU transcription model is invalid"
            )
        if (os.getenv("CHANNEL_SERVICE_ENV") or "").strip().lower() == "production":
            if not re.fullmatch(r"[0-9a-f]{40}", revision):
                raise TranscriptionConfigurationError(
                    "production local CPU transcription requires a 40-hex model revision"
                )
        return TranscriptionContract(
            mode=mode, model_id=model, model_revision=revision or None
        )
    raise TranscriptionConfigurationError(
        "transcription_mode must be auto, openai, or local_cpu"
    )


def transcription_temp_path(*, job_id: str, attempt_number: int) -> Path:
    if not re.fullmatch(r"job_[0-9a-f]{40}", str(job_id or "")):
        raise TranscriptionConfigurationError("job identity is unsafe")
    if isinstance(attempt_number, bool) or attempt_number < 1 or attempt_number > 100:
        raise TranscriptionConfigurationError("attempt number is out of bounds")
    root = Path(
        (
            os.getenv("CHANNEL_SERVICE_TRANSCRIPTION_TMP_ROOT")
            or "/data/transcription-tmp"
        ).strip()
    ).expanduser()
    if not root.is_absolute():
        raise TranscriptionConfigurationError(
            "CHANNEL_SERVICE_TRANSCRIPTION_TMP_ROOT must be absolute"
        )
    return root / job_id / f"attempt-{attempt_number}.flac"


def extract_temporary_audio(*, video_path: Path, audio_path: Path) -> tuple[str, int]:
    video = Path(video_path).resolve(strict=True)
    target = _contained_audio_path(audio_path)
    target.parent.mkdir(mode=0o700, parents=True, exist_ok=True)
    _reject_symlink_path(target.parent)
    if target.exists() or target.is_symlink():
        raise TranscriptionError("temporary audio target already exists")
    ffmpeg = (
        os.getenv("CHANNEL_SERVICE_FFMPEG_BIN") or "/usr/local/bin/ffmpeg"
    ).strip()
    if not Path(ffmpeg).is_absolute():
        raise TranscriptionConfigurationError(
            "CHANNEL_SERVICE_FFMPEG_BIN must be absolute"
        )
    timeout = _bounded_int(
        "CHANNEL_SERVICE_TRANSCRIPTION_FFMPEG_TIMEOUT_SECONDS", 3600, 30, 21_600
    )
    completed = subprocess.run(
        [
            ffmpeg,
            "-nostdin",
            "-v",
            "error",
            "-i",
            str(video),
            "-vn",
            "-ac",
            "1",
            "-ar",
            "16000",
            "-c:a",
            "flac",
            str(target),
        ],
        stdin=subprocess.DEVNULL,
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
        text=True,
        timeout=timeout,
        check=False,
    )
    if completed.returncode != 0:
        target.unlink(missing_ok=True)
        raise TranscriptionError(
            f"ffmpeg audio extraction failed: {completed.stderr[-1000:]}"
        )
    fd = os.open(target, os.O_RDONLY | getattr(os, "O_NOFOLLOW", 0))
    try:
        info = os.fstat(fd)
        if not stat.S_ISREG(info.st_mode) or info.st_nlink != 1 or info.st_size <= 0:
            raise TranscriptionError("temporary audio is not a private regular file")
        os.fchmod(fd, 0o600)
        digest = _sha256_fd(fd)
        size = int(info.st_size)
    finally:
        os.close(fd)
    return digest, size


def delete_temporary_audio(path: Path) -> None:
    """Unlink one private temporary audio file without following links.

    This is lifecycle deletion, not a promise of forensic erasure on SSD media.
    """
    target = _contained_audio_path(path)
    try:
        fd = os.open(target, os.O_RDONLY | getattr(os, "O_NOFOLLOW", 0))
    except FileNotFoundError:
        return
    try:
        info = os.fstat(fd)
        if not stat.S_ISREG(info.st_mode) or info.st_nlink != 1:
            raise TranscriptionError(
                "refusing to delete a non-private temporary audio path"
            )
    finally:
        os.close(fd)
    target.unlink()


def transcribe_audio(
    *,
    audio_path: Path,
    contract: TranscriptionContract,
    language: str,
    openai_client_factory: Callable[[], Any] | None = None,
    local_pipeline_factory: Callable[[TranscriptionContract], Any] | None = None,
) -> TranscriptResult:
    path = _contained_audio_path(audio_path).resolve(strict=True)
    normalized_language = str(language or "").strip().lower()
    if not re.fullmatch(r"[a-z]{2,3}(?:-[a-z0-9]{2,8})?", normalized_language):
        raise TranscriptionError("transcription language is invalid")
    if contract.mode == "openai":
        return _transcribe_openai(
            path=path,
            contract=contract,
            language=normalized_language,
            client_factory=openai_client_factory,
        )
    if contract.mode == "local_cpu":
        return _transcribe_local_cpu(
            path=path,
            contract=contract,
            language=normalized_language,
            pipeline_factory=local_pipeline_factory,
        )
    raise TranscriptionConfigurationError("unsupported transcription mode")


def _transcribe_openai(
    *,
    path: Path,
    contract: TranscriptionContract,
    language: str,
    client_factory: Callable[[], Any] | None,
) -> TranscriptResult:
    enabled = (
        (os.getenv("CHANNEL_SERVICE_OPENAI_TRANSCRIPTION_ENABLED") or "")
        .strip()
        .lower()
    )
    if enabled not in {"1", "true", "yes", "on"}:
        raise TranscriptionConfigurationError(
            "OpenAI transcription is disabled by the explicit budget gate"
        )
    if path.stat().st_size > _bounded_int(
        "CHANNEL_SERVICE_OPENAI_TRANSCRIPTION_MAX_BYTES", 24_000_000, 1, 25_000_000
    ):
        raise TranscriptionConfigurationError(
            "temporary audio exceeds the configured paid transcription byte budget"
        )
    duration_s = _audio_duration_seconds(path)
    if duration_s > _bounded_int(
        "CHANNEL_SERVICE_OPENAI_TRANSCRIPTION_MAX_AUDIO_SECONDS", 1200, 1, 14_400
    ):
        raise TranscriptionConfigurationError(
            "temporary audio exceeds the configured paid transcription duration budget"
        )
    if not (os.getenv("OPENAI_API_KEY") or "").strip() and client_factory is None:
        raise TranscriptionConfigurationError("OPENAI_API_KEY is unavailable")
    if client_factory is None:
        from openai import OpenAI

        client_factory = OpenAI
    try:
        with path.open("rb") as handle:
            response = client_factory().audio.transcriptions.create(
                model=contract.model_id,
                file=handle,
                language=language.split("-", 1)[0],
                response_format="verbose_json",
                timestamp_granularities=["segment"],
                timeout=_bounded_int(
                    "CHANNEL_SERVICE_OPENAI_TRANSCRIPTION_TIMEOUT_SECONDS",
                    900,
                    30,
                    1800,
                ),
            )
    except (TimeoutError, ConnectionError) as exc:
        raise AmbiguousTranscriptionError(
            "paid transcription response was ambiguous; reconciliation is required"
        ) from exc
    except Exception as exc:
        # OpenAI SDK connection/timeout exceptions are provider-specific subclasses.
        name = exc.__class__.__name__.lower()
        status_code = getattr(exc, "status_code", None)
        if (
            "timeout" in name
            or "connection" in name
            or status_code in {408, 500, 502, 503, 504}
        ):
            raise AmbiguousTranscriptionError(
                "paid transcription response was ambiguous; reconciliation is required"
            ) from exc
        raise TranscriptionError(f"OpenAI transcription failed: {exc}") from exc
    segments = _normalize_provider_segments(getattr(response, "segments", None))
    request_id = (
        str(getattr(response, "_request_id", "") or getattr(response, "id", "")).strip()
        or None
    )
    return TranscriptResult(
        provider=_provider_label("openai", contract),
        provider_request_id=request_id,
        segments=segments,
    )


def _transcribe_local_cpu(
    *,
    path: Path,
    contract: TranscriptionContract,
    language: str,
    pipeline_factory: Callable[[TranscriptionContract], Any] | None,
) -> TranscriptResult:
    if pipeline_factory is None:
        from transformers import pipeline

        def pipeline_factory(selected: TranscriptionContract):
            return pipeline(
                "automatic-speech-recognition",
                model=selected.model_id,
                revision=selected.model_revision,
                device=-1,
            )

    try:
        output = pipeline_factory(contract)(
            str(path),
            chunk_length_s=30,
            stride_length_s=5,
            return_timestamps=True,
            generate_kwargs={"language": language.split("-", 1)[0]},
        )
    except Exception as exc:
        raise TranscriptionError(f"local CPU transcription failed: {exc}") from exc
    if not isinstance(output, dict):
        raise TranscriptionError("local CPU transcription returned an invalid payload")
    chunks = output.get("chunks")
    if not isinstance(chunks, list):
        text = str(output.get("text") or "").strip()
        if not text:
            raise TranscriptionError("local CPU transcription returned no timed text")
        chunks = [{"timestamp": (0.0, 0.001), "text": text}]
    segments = _normalize_local_chunks(chunks)
    return TranscriptResult(
        provider=_provider_label("local_cpu", contract),
        provider_request_id=None,
        segments=segments,
    )


def _normalize_provider_segments(raw: Any) -> tuple[dict[str, Any], ...]:
    if not isinstance(raw, (list, tuple)):
        raise TranscriptionError(
            "transcription provider returned no timestamped segments"
        )
    rows = []
    for index, value in enumerate(raw):
        getter = (
            value.get
            if isinstance(value, dict)
            else lambda key, default=None: getattr(value, key, default)
        )
        text = str(getter("text", "") or "").strip()
        start = getter("start")
        end = getter("end")
        if (
            not text
            or not isinstance(start, (int, float))
            or not isinstance(end, (int, float))
        ):
            raise TranscriptionError("transcription provider segment is invalid")
        rows.append(_segment(index, float(start), float(end), text))
    if not rows:
        raise TranscriptionError("transcription provider returned an empty transcript")
    return tuple(rows)


def _normalize_local_chunks(raw: list[Any]) -> tuple[dict[str, Any], ...]:
    rows = []
    for index, value in enumerate(raw):
        if not isinstance(value, dict):
            raise TranscriptionError("local CPU transcript chunk is invalid")
        stamp = value.get("timestamp")
        text = str(value.get("text") or "").strip()
        if (
            not isinstance(stamp, (list, tuple))
            or len(stamp) != 2
            or not isinstance(stamp[0], (int, float))
            or not isinstance(stamp[1], (int, float))
            or not text
        ):
            raise TranscriptionError("local CPU transcript lacks exact timestamps")
        rows.append(_segment(index, float(stamp[0]), float(stamp[1]), text))
    if not rows:
        raise TranscriptionError("local CPU transcription returned an empty transcript")
    return tuple(rows)


def _segment(index: int, start_s: float, end_s: float, text: str) -> dict[str, Any]:
    start_ms = max(0, int(round(start_s * 1000)))
    end_ms = int(round(end_s * 1000))
    if end_ms <= start_ms:
        end_ms = start_ms + 1
    return {
        "ordinal": index,
        "start_ms": start_ms,
        "end_ms": end_ms,
        "speaker_label": None,
        "text": text,
    }


def _contained_audio_path(path: Path) -> Path:
    target = Path(path).expanduser()
    if not target.is_absolute():
        raise TranscriptionConfigurationError("temporary audio path must be absolute")
    root = Path(
        (
            os.getenv("CHANNEL_SERVICE_TRANSCRIPTION_TMP_ROOT")
            or "/data/transcription-tmp"
        ).strip()
    ).expanduser()
    if not root.is_absolute():
        raise TranscriptionConfigurationError(
            "CHANNEL_SERVICE_TRANSCRIPTION_TMP_ROOT must be absolute"
        )
    root = root.resolve()
    parent = target.parent.resolve()
    try:
        parent.relative_to(root)
    except ValueError as exc:
        raise TranscriptionConfigurationError(
            "temporary audio path escapes its root"
        ) from exc
    return parent / target.name


def _reject_symlink_path(path: Path) -> None:
    current = path
    while current != current.parent:
        if current.is_symlink():
            raise TranscriptionError(
                "temporary audio directory must not contain symlinks"
            )
        current = current.parent


def _sha256_fd(fd: int) -> str:
    digest = hashlib.sha256()
    os.lseek(fd, 0, os.SEEK_SET)
    while True:
        chunk = os.read(fd, 1024 * 1024)
        if not chunk:
            break
        digest.update(chunk)
    return digest.hexdigest()


def _bounded_int(name: str, default: int, minimum: int, maximum: int) -> int:
    raw = (os.getenv(name) or "").strip()
    try:
        value = int(raw) if raw else default
    except ValueError as exc:
        raise TranscriptionConfigurationError(f"{name} must be an integer") from exc
    if value < minimum or value > maximum:
        raise TranscriptionConfigurationError(
            f"{name} must be between {minimum} and {maximum}"
        )
    return value


def _provider_label(prefix: str, contract: TranscriptionContract) -> str:
    identity = f"{contract.model_id}@{contract.model_revision or 'provider-versioned'}"
    return f"{prefix}:{hashlib.sha256(identity.encode('utf-8')).hexdigest()[:24]}"


def _audio_duration_seconds(path: Path) -> float:
    executable = (
        os.getenv("CHANNEL_SERVICE_FFPROBE_BIN") or "/usr/local/bin/ffprobe"
    ).strip()
    if not Path(executable).is_absolute():
        raise TranscriptionConfigurationError(
            "CHANNEL_SERVICE_FFPROBE_BIN must be an absolute path"
        )
    try:
        completed = subprocess.run(
            [
                executable,
                "-v",
                "error",
                "-show_entries",
                "format=duration",
                "-of",
                "default=noprint_wrappers=1:nokey=1",
                str(path),
            ],
            stdin=subprocess.DEVNULL,
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            text=True,
            timeout=30,
            check=False,
        )
    except (OSError, subprocess.TimeoutExpired) as exc:
        raise TranscriptionConfigurationError(
            "ffprobe could not enforce the paid transcription duration budget"
        ) from exc
    if completed.returncode != 0:
        raise TranscriptionConfigurationError(
            "ffprobe could not enforce the paid transcription duration budget"
        )
    try:
        duration = float(completed.stdout.strip())
    except ValueError as exc:
        raise TranscriptionConfigurationError(
            "ffprobe returned an invalid paid transcription duration"
        ) from exc
    if duration <= 0 or duration > 7 * 24 * 3600:
        raise TranscriptionConfigurationError("paid transcription duration is invalid")
    return duration
