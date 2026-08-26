from __future__ import annotations

import copy
import json
import math
import os
import random
from collections.abc import Mapping, Sequence
from importlib import metadata
from pathlib import Path
from typing import Any

YTDLP_VERSION = "2026.8.19"
YTDLP_EJS_VERSION = "0.8.0"
BGUTIL_PROVIDER_VERSION = "1.3.2"
YOUTUBE_MP4_FORMAT = "bestvideo*[ext=mp4]+bestaudio[ext=m4a]/best[ext=mp4]"

_DEFAULT_DENO_BIN = "/usr/local/bin/deno"
_DEFAULT_COOKIE_REFERENCES = (
    Path("/run/secrets/youtube_cookies"),
    Path("/cookies/youtube.txt"),
)
_DISABLED_PROVIDER_VALUES = {"", "0", "disabled", "false", "none", "off"}
_BGUTIL_PROVIDER_VALUES = {"bgutil-script", "bgutil_script"}
_DISABLED_HTTP_PROVIDER_URL = "http://127.0.0.1:0"
_DISABLED_SCRIPT_PROVIDER_HOME = "/dev/null"


class _SensitiveOutputSuppressingLogger:
    """Keep yt-dlp and provider output from emitting cookie or token material."""

    def debug(self, _message: str) -> None:
        return None

    def warning(self, _message: str) -> None:
        return None

    def error(self, _message: str) -> None:
        return None


_SENSITIVE_OUTPUT_SUPPRESSING_LOGGER = _SensitiveOutputSuppressingLogger()


def _first_env(*names: str) -> str | None:
    for name in names:
        value = (os.environ.get(name) or "").strip()
        if value:
            return value
    return None


def _bounded_float(
    names: Sequence[str],
    *,
    default: float,
    minimum: float,
    maximum: float,
) -> float:
    raw = _first_env(*names)
    if raw is None:
        return default
    try:
        value = float(raw)
    except ValueError as exc:
        raise RuntimeError(f"{names[0]} must be a bounded number") from exc
    if not math.isfinite(value) or value < minimum or value > maximum:
        raise RuntimeError(f"{names[0]} must be between {minimum:g} and {maximum:g}")
    return value


def _deno_reference() -> Path:
    raw = _first_env("YTDLP_DENO_BIN") or _DEFAULT_DENO_BIN
    deno = Path(raw).expanduser()
    if not deno.is_absolute():
        raise RuntimeError("YTDLP_DENO_BIN must reference an absolute path")
    return deno


def _cookie_reference(override: str | Path | None = None) -> str | None:
    configured = (
        str(override).strip()
        if override is not None
        else _first_env("YTDLP_COOKIES_FILE", "YTDLP_COOKIES_PATH")
    )
    if configured:
        candidate = Path(configured).expanduser()
        if not candidate.is_absolute():
            raise RuntimeError(
                "configured YouTube cookie reference must be an absolute path"
            )
        if not candidate.is_file():
            raise RuntimeError("configured YouTube cookie reference is unavailable")
        return str(candidate)
    for candidate in _DEFAULT_COOKIE_REFERENCES:
        if candidate.is_file():
            return str(candidate)
    return None


def _provider_mode() -> str | None:
    if _first_env("YTDLP_PO_TOKEN") is not None:
        raise RuntimeError(
            "YTDLP_PO_TOKEN is unsupported; use the automated provider reference"
        )
    raw = (os.environ.get("YTDLP_PO_TOKEN_PROVIDER") or "disabled").strip().lower()
    if raw in _DISABLED_PROVIDER_VALUES:
        return None
    if raw not in _BGUTIL_PROVIDER_VALUES:
        raise RuntimeError("YTDLP_PO_TOKEN_PROVIDER must be disabled or bgutil-script")
    return "bgutil-script"


def configured_youtube_player_clients() -> list[str]:
    mode = _provider_mode()
    raw = _first_env("YTDLP_PLAYER_CLIENTS")
    clients = [part.strip() for part in (raw or "").split(",") if part.strip()]
    if not clients:
        clients = ["mweb"] if mode else ["default"]
    if any(
        not client.replace("_", "").replace("-", "").isalnum() for client in clients
    ):
        raise RuntimeError("YTDLP_PLAYER_CLIENTS contains an invalid client name")
    if mode and clients != ["mweb"]:
        raise RuntimeError(
            "bgutil-script requires YTDLP_PLAYER_CLIENTS to contain only mweb"
        )
    return clients


def configured_youtube_proxies() -> list[str | None]:
    raw_pool = _first_env("YTDLP_PROXIES", "YTDLP_PROXY_POOL") or ""
    values = [value.strip() for value in raw_pool.split(",") if value.strip()]
    single = _first_env("YTDLP_PROXY")
    if single:
        values.insert(0, single)
    unique = list(dict.fromkeys(values))
    if not unique:
        return [None]
    if (
        os.environ.get("YTDLP_PROXY_STRATEGY") or "round_robin"
    ).strip().lower() == "random":
        random.shuffle(unique)
    forced = (os.environ.get("YTDLP_PROXY_FORCE") or "").strip().lower() in {
        "1",
        "true",
        "yes",
        "y",
        "on",
    }
    return unique if forced else [None, *unique]


def _provider_extractor_args(deno: Path) -> dict[str, dict[str, list[str]]]:
    # The plugin registers HTTP and script transports automatically. Point both
    # at impossible local targets unless the validated script lane is enabled.
    args = {
        "youtubepot-bgutilhttp": {"base_url": [_DISABLED_HTTP_PROVIDER_URL]},
        "youtubepot-bgutilscript": {
            "server_home": [_DISABLED_SCRIPT_PROVIDER_HOME]
        },
    }
    if _provider_mode() is None:
        return args
    raw_home = _first_env("YTDLP_PO_TOKEN_PROVIDER_DIR")
    if raw_home is None:
        raise RuntimeError("YTDLP_PO_TOKEN_PROVIDER_DIR is required for bgutil-script")
    provider_home = Path(raw_home).expanduser()
    if not provider_home.is_absolute():
        raise RuntimeError("YTDLP_PO_TOKEN_PROVIDER_DIR must be an absolute path")
    if not (
        provider_home.is_dir()
        and (provider_home / "src" / "generate_once.ts").is_file()
        and (provider_home / "node_modules").is_dir()
        and (provider_home / "package.json").is_file()
    ):
        raise RuntimeError("configured same-host PO-token provider is unavailable")
    manifest_path = provider_home / "package.json"
    try:
        if manifest_path.stat().st_size > 64 * 1024:
            raise ValueError("provider manifest is oversized")
        manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    except (OSError, UnicodeError, ValueError) as exc:
        raise RuntimeError("configured same-host PO-token provider is invalid") from exc
    if (
        not isinstance(manifest, dict)
        or manifest.get("name") != "bgutil-ytdlp-pot-provider"
        or manifest.get("version") != BGUTIL_PROVIDER_VERSION
    ):
        raise RuntimeError("the same-host PO-token provider source version is unpinned")
    if not deno.is_file() or not os.access(deno, os.X_OK):
        raise RuntimeError("configured same-host Deno runtime is unavailable")
    try:
        installed = metadata.version("bgutil-ytdlp-pot-provider")
    except metadata.PackageNotFoundError as exc:
        raise RuntimeError(
            "the pinned same-host PO-token provider plugin is unavailable"
        ) from exc
    if installed != BGUTIL_PROVIDER_VERSION:
        raise RuntimeError("the same-host PO-token provider plugin version is unpinned")
    args["youtubepot-bgutilscript"] = {"server_home": [str(provider_home)]}
    return args


def build_youtube_ytdlp_options(
    base_options: Mapping[str, Any] | None = None,
    *,
    media: bool = False,
    player_clients: Sequence[str] | None = None,
    proxy: str | None = None,
    cookie_reference: str | Path | None = None,
) -> dict[str, Any]:
    """Build the single fail-closed yt-dlp policy used by every YouTube path."""

    if _first_env("YTDLP_REMOTE_COMPONENTS") is not None:
        raise RuntimeError(
            "YTDLP_REMOTE_COMPONENTS is unsupported; EJS must be bundled and pinned"
        )
    options: dict[str, Any] = copy.deepcopy(dict(base_options or {}))
    if options.get("remote_components"):
        raise RuntimeError("remote yt-dlp components are not allowed")
    deno = _deno_reference()
    minimum_sleep = _bounded_float(
        (
            "YTDLP_SLEEP_INTERVAL",
            "YTDLP_SLEEP_INTERVAL_S",
            "YTDLP_MIN_SLEEP_INTERVAL",
            "YTDLP_MIN_SLEEP_INTERVAL_S",
        ),
        default=15.0,
        minimum=5.0,
        maximum=120.0,
    )
    maximum_sleep = _bounded_float(
        ("YTDLP_MAX_SLEEP_INTERVAL", "YTDLP_MAX_SLEEP_INTERVAL_S"),
        default=30.0,
        minimum=5.0,
        maximum=180.0,
    )
    if maximum_sleep < minimum_sleep:
        raise RuntimeError(
            "YTDLP_MAX_SLEEP_INTERVAL must not be lower than YTDLP_SLEEP_INTERVAL"
        )

    selected_clients = list(player_clients or configured_youtube_player_clients())
    if not selected_clients or any(
        not isinstance(client, str)
        or not client
        or not client.replace("_", "").replace("-", "").isalnum()
        for client in selected_clients
    ):
        raise RuntimeError("YouTube player clients are invalid")
    mode = _provider_mode()
    if mode and selected_clients != ["mweb"]:
        raise RuntimeError("the automated PO-token provider requires the mweb client")
    extractor_args = copy.deepcopy(options.get("extractor_args") or {})
    youtube_args = copy.deepcopy(extractor_args.get("youtube") or {})
    if "po_token" in youtube_args:
        raise RuntimeError("static PO-token extractor arguments are unsupported")
    youtube_args["player_client"] = selected_clients
    extractor_args["youtube"] = youtube_args
    extractor_args.update(_provider_extractor_args(deno))

    options.update(
        {
            "quiet": True,
            "no_warnings": True,
            "logger": _SENSITIVE_OUTPUT_SUPPRESSING_LOGGER,
            "js_runtimes": {"deno": {"path": str(deno)}},
            "remote_components": set(),
            "sleep_requests": _bounded_float(
                ("YTDLP_SLEEP_REQUESTS", "YTDLP_SLEEP_REQUESTS_S"),
                default=1.0,
                minimum=0.25,
                maximum=10.0,
            ),
            "sleep_interval": minimum_sleep,
            "max_sleep_interval": maximum_sleep,
            "sleep_subtitles": _bounded_float(
                ("YTDLP_SLEEP_SUBTITLES", "YTDLP_SLEEP_SUBTITLES_S"),
                default=5.0,
                minimum=1.0,
                maximum=60.0,
            ),
            "extractor_args": extractor_args,
        }
    )
    reference = _cookie_reference(cookie_reference)
    if reference:
        options["cookiefile"] = reference
    else:
        options.pop("cookiefile", None)
    user_agent = _first_env("YTDLP_USER_AGENT")
    if user_agent:
        options["user_agent"] = user_agent
    if proxy:
        options["proxy"] = proxy
    else:
        options.pop("proxy", None)
    if media:
        configured_format = _first_env("CHANNEL_SERVICE_YTDLP_VIDEO_FORMAT")
        if configured_format and configured_format != YOUTUBE_MP4_FORMAT:
            raise RuntimeError(
                "CHANNEL_SERVICE_YTDLP_VIDEO_FORMAT conflicts with the pinned "
                "MP4 policy"
            )
        options["format"] = YOUTUBE_MP4_FORMAT
        options["merge_output_format"] = "mp4"
    return options


def safe_ytdlp_error_message(exc: BaseException) -> str:
    """Classify an extractor failure without returning secret-bearing provider text."""

    message = str(exc).lower()
    if any(
        marker in message
        for marker in (
            "http error 429",
            "too many requests",
            "sign in to confirm",
            "not a bot",
            "rate-limited by youtube",
        )
    ):
        return "YouTube acquisition was rate-limited or challenged"
    if any(
        marker in message
        for marker in ("private video", "members-only", "login required")
    ):
        return "YouTube item is unavailable or requires authorization"
    return f"YouTube extractor failed ({exc.__class__.__name__})"
