from __future__ import annotations

import json
import os
from contextlib import contextmanager
from typing import Iterator, List, Optional

import redis


LANE_ORDER: List[str] = [
    "starter_paid",
    "expansion_paid",
    "quote_starter_probe",
    "quote_deferred_probe",
    "cache_refresh",
]


def _env_bool(name: str, default: bool) -> bool:
    raw = (os.getenv(name) or "").strip().lower()
    if not raw:
        return default
    return raw in {"1", "true", "yes", "y", "on"}


def _env_int(name: str, default: int) -> int:
    raw = (os.getenv(name) or "").strip()
    if not raw:
        return default
    try:
        return int(raw)
    except ValueError:
        return default


def _env_json(name: str) -> object:
    raw = (os.getenv(name) or "").strip()
    if not raw:
        return None
    try:
        return json.loads(raw)
    except Exception:
        return None


def _string_or_none(value) -> Optional[str]:
    if value is None:
        return None
    out = str(value).strip()
    return out or None


def _to_bool(value, default: bool) -> bool:
    if value is None:
        return default
    if isinstance(value, bool):
        return value
    raw = str(value).strip().lower()
    if not raw:
        return default
    return raw in {"1", "true", "yes", "y", "on"}


def _to_int(value, default: int) -> int:
    if value is None:
        return default
    try:
        return int(value)
    except Exception:
        return default


def _string_list(value, *, default: Optional[List[str]] = None) -> List[str]:
    if value is None:
        return list(default or [])
    if isinstance(value, list):
        values = value
    else:
        values = [part.strip() for part in str(value).split(",")]
    out = []
    for item in values:
        if item is None:
            continue
        text = str(item).strip()
        if text:
            out.append(text)
    if out:
        return out
    return list(default or [])


def redis_url() -> str:
    return (os.getenv("REDIS_URL") or "redis://redis:6379/0").strip()


def redis_client() -> redis.Redis:
    return redis.Redis.from_url(redis_url(), decode_responses=True)


def scheduler_enabled() -> bool:
    return _env_bool("CHANNEL_SERVICE_USE_REDIS_SCHEDULER", True)


def scheduler_queue_name(lane: str) -> str:
    return f"channel_service:ready:{lane}"


def scheduler_queue_names() -> List[str]:
    return [scheduler_queue_name(lane) for lane in LANE_ORDER]


def scheduler_enqueued_tokens_key() -> str:
    return "channel_service:dispatch:tokens"


def scheduler_summary_key() -> str:
    return "channel_service:dispatch:summary"


def dispatch_token(job_id: str, pool_id: str) -> str:
    return f"{job_id}:{pool_id}"


def dispatch_payload(*, job_id: str, probe_key: str, pool_id: str, lane: str, token: str) -> str:
    return json.dumps(
        {
            "job_id": job_id,
            "probe_key": probe_key,
            "pool_id": pool_id,
            "lane": lane,
            "token": token,
        },
        sort_keys=True,
    )


def enqueue_dispatch(*, client: redis.Redis, lane: str, payload: str, token: str) -> bool:
    if client.sadd(scheduler_enqueued_tokens_key(), token) != 1:
        return False
    client.rpush(scheduler_queue_name(lane), payload)
    return True


def pop_dispatch(*, client: redis.Redis, timeout_s: int) -> Optional[dict]:
    result = client.blpop(scheduler_queue_names(), timeout=max(1, int(timeout_s)))
    if not result:
        return None
    _, raw_payload = result
    try:
        payload = json.loads(raw_payload)
    except Exception:
        return None
    token = str(payload.get("token") or "").strip()
    if token:
        client.srem(scheduler_enqueued_tokens_key(), token)
    return payload


def lane_priority(lane: str) -> int:
    try:
        return LANE_ORDER.index(lane)
    except ValueError:
        return len(LANE_ORDER)


def global_dispatch_limit() -> int:
    return max(1, _env_int("CHANNEL_SERVICE_SCHEDULER_GLOBAL_LIMIT", 40))


def per_channel_dispatch_limit() -> int:
    return max(1, _env_int("CHANNEL_SERVICE_SCHEDULER_PER_CHANNEL_LIMIT", 5))


def dispatch_ttl_s() -> int:
    return max(30, _env_int("CHANNEL_SERVICE_SCHEDULER_DISPATCH_TTL_S", 120))


def default_player_clients() -> List[str]:
    raw = (os.getenv("CHANNEL_SERVICE_ACQUIRE_PLAYER_CLIENTS") or "").strip()
    if not raw:
        return ["android", "ios"]
    values = [part.strip() for part in raw.split(",") if part.strip()]
    return values or ["android", "ios"]


def default_canary_language() -> str:
    return _string_or_none(os.getenv("CHANNEL_SERVICE_POOL_CANARY_LANGUAGE")) or "en"


def default_canary_video_url() -> Optional[str]:
    return _string_or_none(os.getenv("CHANNEL_SERVICE_POOL_CANARY_VIDEO_URL"))


def direct_pool_enabled() -> bool:
    return _env_bool("CHANNEL_SERVICE_POOL_DIRECT_ENABLED", True)


def proxy_pool_enabled() -> bool:
    if _env_bool("CHANNEL_SERVICE_POOL_PROXY_ENABLED", False):
        return True
    return bool((os.getenv("CHANNEL_SERVICE_ACQUIRE_PROXIES") or "").strip())


def _normalize_pool_profile(raw: dict) -> Optional[dict]:
    pool_id = _string_or_none(raw.get("id") or raw.get("pool_id"))
    if not pool_id:
        return None
    if not _to_bool(raw.get("enabled"), True):
        return None
    proxy_value = _string_or_none(raw.get("proxy_value") or raw.get("proxy") or raw.get("proxies"))
    pool_kind = _string_or_none(raw.get("pool_kind")) or ("proxy" if proxy_value else "direct")
    profile = {
        "id": pool_id,
        "display_name": _string_or_none(raw.get("display_name")) or pool_id,
        "pool_kind": pool_kind,
        "concurrency_limit": max(1, _to_int(raw.get("concurrency_limit"), 2)),
        "player_clients": _string_list(raw.get("player_clients"), default=default_player_clients()),
        "allow_transcript_api": _to_bool(raw.get("allow_transcript_api"), False),
        "use_proxy_pool": _to_bool(raw.get("use_proxy_pool"), bool(proxy_value)),
        "proxy_value": proxy_value,
        "proxy_force": _to_bool(raw.get("proxy_force"), False),
        "cookie_file": _string_or_none(raw.get("cookie_file")),
        "health_group": _string_or_none(raw.get("health_group")) or pool_id,
        "canary_video_url": _string_or_none(raw.get("canary_video_url")) or default_canary_video_url(),
        "canary_language": _string_or_none(raw.get("canary_language")) or default_canary_language(),
    }
    return profile


def _configured_pool_profiles() -> List[dict]:
    data = _env_json("CHANNEL_SERVICE_EGRESS_POOLS_JSON")
    if isinstance(data, dict):
        data = data.get("pools")
    if not isinstance(data, list):
        return []
    profiles: List[dict] = []
    for item in data:
        if not isinstance(item, dict):
            continue
        profile = _normalize_pool_profile(item)
        if profile is not None:
            profiles.append(profile)
    deduped: List[dict] = []
    seen = set()
    for profile in profiles:
        if profile["id"] in seen:
            continue
        seen.add(profile["id"])
        deduped.append(profile)
    return deduped


def _legacy_pool_profiles() -> List[dict]:
    profiles: List[dict] = []
    if direct_pool_enabled():
        profiles.append(
            {
                "id": "direct",
                "display_name": "Direct",
                "pool_kind": "direct",
                "concurrency_limit": max(1, _env_int("CHANNEL_SERVICE_POOL_DIRECT_CONCURRENCY", 2)),
                "player_clients": default_player_clients(),
                "allow_transcript_api": _env_bool("CHANNEL_SERVICE_POOL_DIRECT_ALLOW_TRANSCRIPT_API", False),
                "use_proxy_pool": False,
                "proxy_value": None,
                "proxy_force": False,
                "cookie_file": None,
                "health_group": "direct",
                "canary_video_url": default_canary_video_url(),
                "canary_language": default_canary_language(),
            }
        )

    proxies = _string_or_none(os.getenv("CHANNEL_SERVICE_ACQUIRE_PROXIES"))
    if proxies and proxy_pool_enabled():
        profiles.append(
            {
                "id": "configured_proxy",
                "display_name": "Configured Proxy",
                "pool_kind": "proxy",
                "concurrency_limit": max(1, _env_int("CHANNEL_SERVICE_POOL_PROXY_CONCURRENCY", 2)),
                "player_clients": default_player_clients(),
                "allow_transcript_api": _env_bool("CHANNEL_SERVICE_POOL_PROXY_ALLOW_TRANSCRIPT_API", False),
                "use_proxy_pool": True,
                "proxy_value": proxies,
                "proxy_force": _env_bool("CHANNEL_SERVICE_ACQUIRE_PROXY_FORCE", False),
                "cookie_file": None,
                "health_group": "configured_proxy",
                "canary_video_url": default_canary_video_url(),
                "canary_language": default_canary_language(),
            }
        )
    return profiles


def all_pool_profiles() -> List[dict]:
    configured = _configured_pool_profiles()
    if configured:
        return configured
    return _legacy_pool_profiles()


def pool_profile(pool_id: str) -> Optional[dict]:
    for profile in all_pool_profiles():
        if profile["id"] == pool_id:
            return profile
    return None


def distinct_health_group_count() -> int:
    groups = {_string_or_none(profile.get("health_group")) or profile["id"] for profile in all_pool_profiles()}
    return len(groups)


@contextmanager
def pool_execution_env(profile: dict) -> Iterator[None]:
    saved = {
        "YTDLP_PROXIES": os.environ.get("YTDLP_PROXIES"),
        "YTDLP_PROXY_FORCE": os.environ.get("YTDLP_PROXY_FORCE"),
        "YTDLP_COOKIES_FILE": os.environ.get("YTDLP_COOKIES_FILE"),
        "YTDLP_COOKIES_PATH": os.environ.get("YTDLP_COOKIES_PATH"),
    }
    try:
        proxy_value = profile.get("proxy_value")
        if proxy_value:
            os.environ["YTDLP_PROXIES"] = str(proxy_value)
        else:
            os.environ.pop("YTDLP_PROXIES", None)

        if profile.get("proxy_force"):
            os.environ["YTDLP_PROXY_FORCE"] = "1"
        else:
            os.environ.pop("YTDLP_PROXY_FORCE", None)

        cookie_file = _string_or_none(profile.get("cookie_file"))
        if cookie_file:
            os.environ["YTDLP_COOKIES_FILE"] = cookie_file
            os.environ["YTDLP_COOKIES_PATH"] = cookie_file
        yield
    finally:
        for key, value in saved.items():
            if value is None:
                os.environ.pop(key, None)
            else:
                os.environ[key] = value
