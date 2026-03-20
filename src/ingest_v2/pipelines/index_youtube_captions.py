from __future__ import annotations

import logging
import os
import random
import re
import tempfile
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple
from urllib.parse import parse_qs, urlparse

import glob
import requests

from ..configs.settings import settings_v2
from ..utils.ids import segment_uuid, sha1_hex
from .upsert_parents import upsert_parents
from .upsert_pinecone import upsert_children

try:
    from yt_dlp import YoutubeDL  # type: ignore
except Exception:  # pragma: no cover
    YoutubeDL = None  # type: ignore

try:  # pragma: no cover - optional dependency for transcript-only fallback
    from youtube_transcript_api import YouTubeTranscriptApi  # type: ignore
except Exception:  # pragma: no cover
    YouTubeTranscriptApi = None  # type: ignore

LOG = logging.getLogger(__name__)


@dataclass(frozen=True)
class CaptionCue:
    start_s: float
    end_s: float
    text: str


_TIME_RE = re.compile(
    r"^(?P<a>\d{2}:\d{2}:\d{2}\.\d{3})\s*-->\s*(?P<b>\d{2}:\d{2}:\d{2}\.\d{3})"
)


def _hms_ms_to_seconds(hms: str) -> float:
    hh, mm, rest = hms.split(":", 2)
    ss, ms = rest.split(".", 1)
    return int(hh) * 3600 + int(mm) * 60 + int(ss) + (int(ms) / 1000.0)


def _seconds_to_hms(seconds: float) -> str:
    if seconds < 0:
        seconds = 0.0
    total_ms = int(round(seconds * 1000))
    hh = total_ms // 3_600_000
    rem = total_ms % 3_600_000
    mm = rem // 60_000
    rem = rem % 60_000
    ss = rem // 1000
    ms = rem % 1000
    return f"{hh:02d}:{mm:02d}:{ss:02d}.{ms:03d}"


def _strip_vtt_tags(line: str) -> str:
    # remove simple HTML tags seen in auto-subs
    return re.sub(r"<[^>]+>", "", line or "").strip()


_YT_VIDEO_ID_RE = re.compile(r"^[A-Za-z0-9_-]{11}$")


def _parse_youtube_video_id(video_url: str) -> Optional[str]:
    """
    Extract an 11-char YouTube video id from common URL forms.
    """
    raw = (video_url or "").strip()
    if not raw:
        return None
    if _YT_VIDEO_ID_RE.match(raw):
        return raw
    try:
        parsed = urlparse(raw)
    except Exception:
        return None
    host = (parsed.hostname or "").lower()
    if host.endswith("youtu.be"):
        slug = parsed.path.lstrip("/").split("/", 1)[0]
        return slug if _YT_VIDEO_ID_RE.match(slug or "") else None
    if "youtube.com" in host:
        qs = parse_qs(parsed.query or "")
        vid = (qs.get("v") or [None])[0]
        if vid and _YT_VIDEO_ID_RE.match(vid):
            return vid
        parts = [p for p in (parsed.path or "").split("/") if p]
        for key in ("shorts", "embed"):
            if key in parts:
                idx = parts.index(key)
                if idx + 1 < len(parts):
                    cand = parts[idx + 1]
                    if _YT_VIDEO_ID_RE.match(cand):
                        return cand
    return None


def _fetch_transcript_api_cues(*, video_id: str, language: str, prefer_auto: bool) -> Optional[List[CaptionCue]]:
    """
    Best-effort transcript fetch using `youtube-transcript-api`.

    This is a useful local fallback when yt-dlp is temporarily rate-limited / blocked.
    """
    if YouTubeTranscriptApi is None:
        return None
    vid = (video_id or "").strip()
    if not vid:
        return None

    langs: List[str] = []
    lang_raw = (language or "").strip()
    if lang_raw:
        langs.append(lang_raw)
        for sep in ("-", "_"):
            if sep in lang_raw:
                langs.append(lang_raw.split(sep, 1)[0])
                break
    if not langs:
        langs = ["en"]
    # de-dupe preserve order
    seen = set()
    langs = [l for l in langs if l and not (l in seen or seen.add(l))]

    try:
        api = YouTubeTranscriptApi()
        transcripts = api.list(vid)
        selected = None
        if prefer_auto:
            try:
                selected = transcripts.find_generated_transcript(langs)
            except Exception:
                selected = None
            if selected is None:
                try:
                    selected = transcripts.find_manually_created_transcript(langs)
                except Exception:
                    selected = None
        else:
            try:
                selected = transcripts.find_manually_created_transcript(langs)
            except Exception:
                selected = None
            if selected is None:
                try:
                    selected = transcripts.find_generated_transcript(langs)
                except Exception:
                    selected = None
        if selected is None:
            try:
                selected = transcripts.find_transcript(langs)
            except Exception:
                selected = None
        if selected is None:
            return None
        fetched = selected.fetch()
        rows = fetched.to_raw_data() if hasattr(fetched, "to_raw_data") else list(fetched)
    except Exception as exc:
        LOG.debug("[yt-transcript-api] fetch failed video_id=%s err=%s", vid, exc)
        return None

    cues: List[CaptionCue] = []
    for row in rows or []:
        try:
            start_s = float(row.get("start") or 0.0)
            dur_s = float(row.get("duration") or 0.0)
        except Exception:
            continue
        end_s = start_s + dur_s if dur_s > 0 else start_s
        text = str(row.get("text") or "").replace("\n", " ").strip()
        if not text:
            continue
        if end_s <= start_s:
            continue
        cues.append(CaptionCue(start_s=start_s, end_s=end_s, text=text))

    return cues or None


def parse_vtt(path: str) -> List[CaptionCue]:
    cues: List[CaptionCue] = []
    with open(path, "r", encoding="utf-8", errors="replace") as f:
        lines = [ln.rstrip("\n") for ln in f]

    i = 0
    # Skip header
    while i < len(lines) and (not lines[i].strip() or lines[i].strip().startswith("WEBVTT")):
        i += 1

    while i < len(lines):
        # Skip blank lines
        while i < len(lines) and not lines[i].strip():
            i += 1
        if i >= len(lines):
            break

        # Optional cue id line: if next line is time range, treat current as id and advance.
        if i + 1 < len(lines) and _TIME_RE.match(lines[i + 1].strip()):
            i += 1

        m = _TIME_RE.match(lines[i].strip())
        if not m:
            i += 1
            continue

        start_s = _hms_ms_to_seconds(m.group("a"))
        end_s = _hms_ms_to_seconds(m.group("b"))
        i += 1

        text_lines: List[str] = []
        while i < len(lines) and lines[i].strip():
            cleaned = _strip_vtt_tags(lines[i])
            if cleaned:
                text_lines.append(cleaned)
            i += 1

        text = " ".join(text_lines).strip()
        if not text:
            continue
        if end_s <= start_s:
            continue
        cues.append(CaptionCue(start_s=start_s, end_s=end_s, text=text))

    return cues


def _segment_windows(
    cues: List[CaptionCue],
    *,
    segment_min_s: float,
    segment_max_s: float,
    segment_stride_s: float,
    min_text_chars: int,
) -> List[Tuple[float, float, str]]:
    if not cues:
        return []

    start0 = max(0.0, cues[0].start_s)
    endN = max(c.end_s for c in cues)
    if endN <= start0:
        return []

    segs: List[Tuple[float, float, str]] = []
    t = start0
    stride = max(1.0, float(segment_stride_s))
    seg_min = max(1.0, float(segment_min_s))
    seg_max = max(seg_min, float(segment_max_s))

    # If the video is shorter than the configured minimum window, emit a single segment
    # spanning the whole transcript (keeps short uploads indexable even with coarse defaults).
    if (endN - start0) < seg_min:
        text = " ".join(c.text for c in cues).strip()
        if len(text) >= int(min_text_chars):
            return [(start0, endN, text)]
        return []

    # Sliding windows: [t, t+seg_max], advance by stride. This intentionally overlaps.
    while t + seg_min <= endN:
        end_t = min(endN, t + seg_max)
        parts = [c.text for c in cues if c.end_s > t and c.start_s < end_t]
        text = " ".join(parts).strip()
        if len(text) >= int(min_text_chars):
            segs.append((t, end_t, text))
        t += stride

    # De-dupe identical segment text (common with sparse cues).
    deduped: List[Tuple[float, float, str]] = []
    seen_hashes = set()
    for a, b, txt in segs:
        h = sha1_hex(txt.encode("utf-8"))
        if h in seen_hashes:
            continue
        seen_hashes.add(h)
        deduped.append((a, b, txt))
    return deduped


def _yt_watch_url(video_id: str) -> str:
    return f"https://www.youtube.com/watch?v={video_id}"


def _yt_clip_url(video_id: str, start_s: float) -> str:
    return f"{_yt_watch_url(video_id)}&t={max(0, int(start_s))}s"


def _is_youtube_watch_url(url: str) -> bool:
    u = (url or "").strip()
    return ("youtube.com/watch" in u) or ("youtu.be/" in u)


def _coerce_watch_url(value: str) -> Optional[str]:
    """
    Convert a yt-dlp "entry" value (id/url/webpage_url) into a canonical watch URL.

    yt-dlp may return:
    - a bare video id (e.g. "dQw4w9WgXcQ")
    - a watch URL ("https://www.youtube.com/watch?v=...")
    - a path ("/watch?v=...") when using extract_flat
    """
    v = (value or "").strip()
    if not v:
        return None
    if v.startswith("http://") or v.startswith("https://"):
        return v
    if v.startswith("/watch?v="):
        return f"https://www.youtube.com{v}"
    if "watch?v=" in v:
        return f"https://www.youtube.com/{v.lstrip('/')}"
    # Assume bare ID
    return _yt_watch_url(v)


def _normalize_channel_url(ch: str) -> str:
    """
    Best-effort normalization so yt-dlp returns a channel "playlist" with entries.

    Passing a bare channel root (e.g. https://www.youtube.com/@veritasium) often returns
    a channel metadata object with no video entries. The /videos tab yields entries.
    """
    c = (ch or "").strip()
    if not c:
        return c
    if c.startswith("@"):
        return f"https://www.youtube.com/{c}/videos"
    if c.startswith("UC") and len(c) > 5:
        return f"https://www.youtube.com/channel/{c}/videos"
    if c.startswith("http://") or c.startswith("https://"):
        # If it's already a watch URL, treat it as a single video target.
        if _is_youtube_watch_url(c):
            return c
        # If it's already pointing at a tab, don't override.
        if any(seg in c for seg in ("/videos", "/streams", "/shorts", "/playlists")):
            return c
        return c.rstrip("/") + "/videos"
    # best-effort: treat as path segment
    return f"https://www.youtube.com/{c}/videos"


def _require_ytdlp() -> Any:
    if YoutubeDL is None:
        raise RuntimeError("yt-dlp is required for caption indexing (missing dependency)")
    return YoutubeDL


def _ytdlp_extra_opts() -> Dict[str, Any]:
    """
    Optional yt-dlp runtime knobs (primarily for YouTube anti-bot challenges).

    These are passed via env so we don't accept secrets (cookies/proxies) in API request bodies.
    """
    opts: Dict[str, Any] = {}

    def _env_float(*names: str) -> Optional[float]:
        for name in names:
            raw = (os.environ.get(name) or "").strip()
            if not raw:
                continue
            try:
                return float(raw)
            except ValueError:
                continue
        return None
    cookiefile = os.environ.get("YTDLP_COOKIES_FILE") or os.environ.get("YTDLP_COOKIES_PATH")
    # Local-friendly default: if the compose-mounted cookie jar exists, use it automatically.
    if not cookiefile:
        try:
            default_cookie = Path("/cookies/youtube.txt")
            if default_cookie.exists():
                cookiefile = str(default_cookie)
        except Exception:
            pass
    if cookiefile:
        opts["cookiefile"] = cookiefile
    # Proxy is applied per-attempt (supports proxy pools). Keep base opts proxy-free.
    user_agent = os.environ.get("YTDLP_USER_AGENT")
    if user_agent:
        opts["user_agent"] = user_agent
    remote_components = (os.environ.get("YTDLP_REMOTE_COMPONENTS") or "").strip()
    if remote_components:
        parts = [p.strip() for p in remote_components.split(",") if p.strip()]
        if parts:
            # Enables yt-dlp's EJS/JS challenge solver distribution downloads.
            # Common value: "ejs:github" (recommended by yt-dlp).
            opts["remote_components"] = parts

    # Throttling can reduce the likelihood of anti-bot triggers / rate limits.
    # yt-dlp supports: sleep_requests, sleep_interval(+max_sleep_interval), sleep_subtitles.
    sleep_requests = _env_float("YTDLP_SLEEP_REQUESTS", "YTDLP_SLEEP_REQUESTS_S")
    if sleep_requests is not None:
        opts["sleep_requests"] = sleep_requests

    sleep_interval = _env_float(
        "YTDLP_SLEEP_INTERVAL",
        "YTDLP_SLEEP_INTERVAL_S",
        "YTDLP_MIN_SLEEP_INTERVAL",
        "YTDLP_MIN_SLEEP_INTERVAL_S",
    )
    if sleep_interval is not None:
        opts["sleep_interval"] = sleep_interval

    max_sleep_interval = _env_float("YTDLP_MAX_SLEEP_INTERVAL", "YTDLP_MAX_SLEEP_INTERVAL_S")
    if max_sleep_interval is not None:
        opts["max_sleep_interval"] = max_sleep_interval

    sleep_subtitles = _env_float("YTDLP_SLEEP_SUBTITLES", "YTDLP_SLEEP_SUBTITLES_S")
    if sleep_subtitles is not None:
        opts["sleep_subtitles"] = sleep_subtitles
    return opts


def _maybe_wrap_ytdlp_error(exc: Exception) -> RuntimeError:
    msg = str(exc) or exc.__class__.__name__
    lowered = msg.lower()
    if (
        "rate-limited by youtube" in lowered
        or "current session has been rate-limited" in lowered
        or "this content isn't available, try again later" in lowered
    ):
        msg = (
            f"{msg}\n\n"
            "Tip: YouTube rate-limited this session. Add throttling to reduce request volume.\n"
            "Recommended local defaults (yt-dlp alias `-t sleep`):\n"
            "  YTDLP_SLEEP_SUBTITLES=5\n"
            "  YTDLP_SLEEP_REQUESTS=0.75\n"
            "  YTDLP_SLEEP_INTERVAL=10\n"
            "  YTDLP_MAX_SLEEP_INTERVAL=20\n"
        )
    if ("not a bot" in lowered or "sign in to confirm" in lowered or "confirm you’re not a bot" in lowered) and not (
        os.environ.get("YTDLP_COOKIES_FILE")
        or os.environ.get("YTDLP_COOKIES_PATH")
        or Path("/cookies/youtube.txt").exists()
    ):
        msg = (
            f"{msg}\n\n"
            "Tip: YouTube sometimes blocks anonymous scraping. Export browser cookies to a Netscape cookies.txt file,\n"
            "put it at /Users/user/PycharmProjects/icmfyi/.local-data/cookies/youtube.txt (mounted as /cookies/youtube.txt).\n"
            "Ingestion will auto-use it; you can also set YTDLP_COOKIES_FILE=/cookies/youtube.txt to override.\n"
            "Local helper: run ./scripts/seed_youtube_cookies.sh (supports COOKIE_HEADER=... for non-interactive runs)."
        )
    return RuntimeError(msg)


def _is_youtube_bot_check(msg: str) -> bool:
    s = (msg or "").lower()
    return (
        ("http error 429" in s)
        or ("too many requests" in s)
        or ("sign in to confirm" in s and "not a bot" in s)
        or ("confirm you" in s and "not a bot" in s)
        or ("this helps protect our community" in s)
        or ("rate-limited by youtube" in s)
        or ("current session has been rate-limited" in s)
        or ("this content isn't available, try again later" in s)
    )


def _classify_transcript_fetch_error(msg: Optional[str]) -> str:
    s = (msg or "").lower()
    if not s:
        return "fetch_failed"
    if _is_youtube_bot_check(s):
        return "rate_limited"
    if (
        "requested format is not available" in s
        or "no .vtt subtitles downloaded" in s
        or "no subtitles" in s
        or "no captions" in s
        or "subtitles are not available" in s
    ):
        return "transcript_unavailable"
    if "private video" in s or "members-only" in s or "login required" in s:
        return "video_unavailable"
    return "fetch_failed"


def _proxy_pool() -> List[Optional[str]]:
    """
    Return a list of proxies to try in order.

    - Supports a pool via `YTDLP_PROXIES` (comma-separated).
    - Supports a single proxy via `YTDLP_PROXY`.
    - By default includes `None` first (no proxy), unless `YTDLP_PROXY_FORCE=1`.
    """
    raw_pool = (os.environ.get("YTDLP_PROXIES") or os.environ.get("YTDLP_PROXY_POOL") or "").strip()
    items: List[str] = []
    if raw_pool:
        items.extend([p.strip() for p in raw_pool.split(",") if p.strip()])

    single = (os.environ.get("YTDLP_PROXY") or "").strip()
    if single:
        items.insert(0, single)

    # de-dupe while preserving order
    seen = set()
    uniq: List[str] = []
    for p in items:
        if p in seen:
            continue
        seen.add(p)
        uniq.append(p)

    if not uniq:
        return [None]

    strategy = (os.environ.get("YTDLP_PROXY_STRATEGY") or "round_robin").strip().lower()
    if strategy == "random":
        random.shuffle(uniq)

    force = (os.environ.get("YTDLP_PROXY_FORCE") or "").strip().lower() in ("1", "true", "yes", "y", "on")
    if force:
        return list(uniq)
    return [None] + list(uniq)


def _apply_proxy(ydl_opts: Dict[str, Any], proxy: Optional[str]) -> None:
    if proxy:
        ydl_opts["proxy"] = proxy
    else:
        ydl_opts.pop("proxy", None)


def _parse_player_clients() -> List[str]:
    """
    Return a list of yt-dlp YouTube player clients to try in order.

    This can help bypass some YouTube anti-bot challenges.
    """
    raw = (os.environ.get("YTDLP_PLAYER_CLIENTS") or "").strip()
    if raw:
        out: List[str] = []
        for part in raw.split(","):
            p = part.strip()
            if p:
                out.append(p)
        if out:
            return out

    # Default: try non-web clients first even when cookies are available.
    #
    # Rationale: web clients increasingly require PO tokens for subtitles; android/ios/tv
    # are often less restrictive for subtitle downloads.
    return ["android", "ios", "tv", "web_embedded"]


def _apply_player_client(ydl_opts: Dict[str, Any], client: str) -> None:
    y = ydl_opts.setdefault("extractor_args", {}).setdefault("youtube", {})
    y["player_client"] = [client]

    # Optional PO token (YouTube sometimes requires this for subtitles in experiments).
    # Expected format matches yt-dlp docs, e.g. "web.subs+<TOKEN>".
    po_token = (os.environ.get("YTDLP_PO_TOKEN") or "").strip()
    if po_token:
        y["po_token"] = [po_token]

    # By default, keep cookies enabled for all clients. Some YouTube challenges can be
    # solved by switching clients, and dropping cookies can reduce success rates for
    # login-gated videos. If you want the old behavior, set:
    #   YTDLP_DROP_COOKIES_FOR_NON_WEB=1
    drop = (os.environ.get("YTDLP_DROP_COOKIES_FOR_NON_WEB") or "").strip().lower() in ("1", "true", "yes", "y", "on")
    if drop and not client.startswith("web"):
        ydl_opts.pop("cookiefile", None)
        ydl_opts.pop("cookiesfrombrowser", None)


def _fetch_ytdlp_cues(
    *,
    video_url: str,
    language: str,
    prefer_auto: bool,
    use_proxy_pool: bool = True,
    player_clients: Optional[List[str]] = None,
) -> Tuple[Optional[str], List[CaptionCue], Optional[Dict[str, Any]]]:
    YDL = _require_ytdlp()
    with tempfile.TemporaryDirectory(prefix="yt-captions-") as tmp:
        outtmpl = os.path.join(tmp, "%(id)s.%(ext)s")
        base_opts: Dict[str, Any] = {
            "quiet": True,
            "no_warnings": True,
            "skip_download": True,
            "writesubtitles": True,
            "writeautomaticsub": bool(prefer_auto),
            "subtitleslangs": [language],
            "subtitlesformat": "vtt",
            "outtmpl": outtmpl,
            "noplaylist": True,
        }
        extra_opts = _ytdlp_extra_opts()

        info: Optional[Dict[str, Any]] = None
        last_exc: Optional[Exception] = None
        preferred_exc: Optional[Exception] = None
        proxies = _proxy_pool() if use_proxy_pool else [None]
        clients = player_clients or _parse_player_clients()
        for proxy in proxies:
            for client in clients:
                ydl_opts = dict(base_opts)
                ydl_opts.update(extra_opts)
                _apply_proxy(ydl_opts, proxy)
                _apply_player_client(ydl_opts, client)
                with YDL(ydl_opts) as ydl:
                    try:
                        info = ydl.extract_info(video_url, download=True)
                        break
                    except Exception as exc:
                        last_exc = exc
                        if _classify_transcript_fetch_error(str(exc)) == "rate_limited":
                            preferred_exc = exc
                        if _is_youtube_bot_check(str(exc)) and len(proxies) > 1:
                            break
                        continue
            if info is not None:
                break

        if info is None:
            chosen_exc = preferred_exc or last_exc or RuntimeError("yt-dlp failed")
            raise _maybe_wrap_ytdlp_error(chosen_exc) from chosen_exc

        vid = str(info.get("id") or "").strip()
        if not vid:
            raise RuntimeError("yt-dlp did not return a video id")

        candidates = glob.glob(os.path.join(tmp, f"{vid}*.vtt"))
        if not candidates:
            raise FileNotFoundError(f"no .vtt subtitles downloaded for video_id={vid} lang={language}")
        candidates.sort(key=lambda p: os.path.getsize(p), reverse=True)
        vtt_path = candidates[0]
        return vid, parse_vtt(vtt_path), info


def fetch_transcript_cues(
    *,
    video_url: str,
    video_id: Optional[str],
    language: str,
    prefer_auto: bool,
    allow_transcript_api: bool = True,
    use_proxy_pool: bool = True,
    player_clients: Optional[List[str]] = None,
) -> Tuple[Optional[List[CaptionCue]], Optional[str], Optional[Dict[str, Any]], Optional[str], Optional[str]]:
    vid = (video_id or "").strip() or _parse_youtube_video_id(video_url)
    if vid and allow_transcript_api:
        maybe = _fetch_transcript_api_cues(video_id=vid, language=language, prefer_auto=prefer_auto)
        if maybe:
            return maybe, "yt_transcript_api", None, vid, None

    try:
        resolved_vid, cues, info = _fetch_ytdlp_cues(
            video_url=video_url,
            language=language,
            prefer_auto=prefer_auto,
            use_proxy_pool=use_proxy_pool,
            player_clients=player_clients,
        )
    except Exception as exc:
        LOG.debug("[yt-cues] fetch failed video_url=%s video_id=%s err=%s", video_url, vid, exc)
        return None, None, None, vid, str(exc)

    if not cues:
        return None, None, info, resolved_vid or vid, "transcript_unavailable"
    return cues, "yt_captions", info, resolved_vid or vid, None


def index_youtube_video_captions(
    *,
    video_url: str,
    namespace: str = "videos",
    language: str = "en",
    prefer_auto: bool = True,
    segment_min_s: Optional[float] = None,
    segment_max_s: Optional[float] = None,
    segment_stride_s: Optional[float] = None,
    min_text_chars: Optional[int] = None,
    metadata_override: Optional[Dict[str, Any]] = None,
) -> Dict[str, Any]:
    """
    Index a single YouTube video's captions into the existing v2 vector store schema.

    Local-first: uses yt-dlp to fetch subtitles (no YouTube Data API key required),
    then embeds and upserts into the configured vector store.
    """
    # Namespace support is currently "videos"/"streams" via env; keep this explicit for now.
    if namespace != "videos":
        raise ValueError("only namespace=videos is supported for caption indexing right now")

    seg_min = float(segment_min_s if segment_min_s is not None else settings_v2.SEGMENT_MIN_S)
    seg_max = float(segment_max_s if segment_max_s is not None else settings_v2.SEGMENT_MAX_S)
    seg_stride = float(segment_stride_s if segment_stride_s is not None else settings_v2.SEGMENT_STRIDE_S)
    min_chars = int(min_text_chars if min_text_chars is not None else settings_v2.MIN_TEXT_CHARS)

    meta = metadata_override or {}

    def _meta_str(key: str) -> Optional[str]:
        value = meta.get(key)
        if value is None:
            return None
        s = str(value).strip()
        return s if s else None

    def _meta_float(key: str) -> Optional[float]:
        value = meta.get(key)
        if value is None:
            return None
        try:
            f = float(value)
        except Exception:
            return None
        return f if f >= 0 else None

    vid = _meta_str("video_id") or _parse_youtube_video_id(video_url)
    cues, source_label, info, resolved_vid, error_detail = fetch_transcript_cues(
        video_url=video_url,
        video_id=vid,
        language=language,
        prefer_auto=prefer_auto,
    )
    vid = resolved_vid or vid
    cues = cues or []

    if not cues:
        suffix = f" ({error_detail})" if error_detail else ""
        raise RuntimeError(f"parsed 0 caption cues for video_id={vid}{suffix}")

    # Deterministic transcript normalization (shared with diarized indexing).
    # This affects both stored chunk text and router enrichment prompts.
    try:
        from src.ingest_v2.transcripts.normalize import apply_term_fixes

        cues = [CaptionCue(start_s=c.start_s, end_s=c.end_s, text=apply_term_fixes(c.text)) for c in cues]
    except Exception:
        pass

    segments = _segment_windows(
        cues,
        segment_min_s=seg_min,
        segment_max_s=seg_max,
        segment_stride_s=seg_stride,
        min_text_chars=min_chars,
    )
    if not segments:
        raise RuntimeError(f"no caption segments produced for video_id={vid} (min_chars={min_chars})")

    # Metadata (prefer caller-provided overrides, then yt-dlp, then YouTube Data API).
    title = _meta_str("title") or (info.get("title") if info else None)
    channel_name = _meta_str("channel_name") or (info.get("channel") if info else None) or (info.get("uploader") if info else None) or (info.get("uploader_id") if info else None)
    channel_id = _meta_str("channel_id") or (str(info.get("channel_id")).strip() if info and info.get("channel_id") else None)
    published_at = _meta_str("published_at") or (str(info.get("upload_date")).strip() if info and info.get("upload_date") else None)  # yt-dlp: YYYYMMDD
    duration_s = _meta_float("duration_s")
    if duration_s is None and info and info.get("duration") is not None:
        try:
            duration_s = float(info.get("duration"))
        except Exception:
            duration_s = None
    thumbnail_url = _meta_str("thumbnail_url") or (str(info.get("thumbnail")).strip() if info and info.get("thumbnail") else None)
    if not thumbnail_url and vid:
        thumbnail_url = f"https://i.ytimg.com/vi/{vid}/hqdefault.jpg"

    if vid and (not title or not channel_name or not published_at or duration_s is None or not thumbnail_url):
        yt_key = (os.getenv("YOUTUBE_API_KEY") or "").strip()
        if yt_key:
            try:
                from src.ingest_v2.cloud.diarization_indexer.youtube import YouTubeClient

                vm = YouTubeClient(api_key=yt_key).fetch_video_metadata(vid)
                title = title or vm.title or vid
                channel_name = channel_name or (vm.channel_title or vm.channel_handle) or channel_name
                channel_id = channel_id or (vm.channel_id or "")
                published_at = published_at or (vm.published_at or None)
                if duration_s is None and vm.duration_seconds:
                    duration_s = float(vm.duration_seconds)
                thumbnail_url = thumbnail_url or vm.thumbnail_url
            except Exception as exc:
                LOG.debug("[yt-meta] youtube api enrich failed video_id=%s err=%s", vid, exc)

    # Prefer raw YouTube description, but allow caller override. Router enrichment may
    # replace this with a short, topic-focused summary for better catalog search.
    description = _meta_str("description") or (str(info.get("description")).strip() if info and info.get("description") else None) or ""

    # Optional: enrich parent metadata (GPT router fields) for fast metadata-based search.
    router_enabled = (os.getenv("ROUTER_ENRICH", "1") or "1").strip().lower() not in ("0", "false", "no", "off")
    topic_summary = None
    router_tags = None
    aliases = None
    canonical_entities = None
    is_explainer = None
    router_boost = None

    if router_enabled and vid:
        try:
            from src.ingest_v2.router.cache import load as router_cache_load
            from src.ingest_v2.router.cache import save as router_cache_save
            from src.ingest_v2.router.enrich_parent import enrich_parent_router_fields
            from src.ingest_v2.transcripts.normalize import normalize_to_sentences

            enrich = router_cache_load(vid)
            if enrich is None:
                raw_for_router = {
                    "caption_lines": [
                        {"start": c.start_s, "end": c.end_s, "text": c.text}
                        for c in cues
                    ]
                }
                sentences = normalize_to_sentences(raw_for_router)
                enrich = enrich_parent_router_fields(
                    {
                        "video_id": vid,
                        "title": title,
                        "description": description,
                        "channel_name": channel_name,
                        "published_at": published_at,
                        "entities": [],
                    },
                    sentences,
                )
                try:
                    router_cache_save(vid, enrich)
                except Exception:
                    pass

            if isinstance(enrich, dict):
                description = enrich.get("description") or description
                topic_summary = enrich.get("topic_summary") or ""
                router_tags = enrich.get("router_tags") or []
                aliases = enrich.get("aliases") or []
                canonical_entities = enrich.get("canonical_entities") or []
                is_explainer = bool(enrich.get("is_explainer"))
                router_boost = float(enrich.get("router_boost") or 1.0)
        except Exception as exc:
            LOG.debug("[yt-index] router enrich failed video_id=%s err=%s", vid, exc)

    watch_url = _yt_watch_url(vid)

    parent = {
        "parent_id": vid,
        "video_id": vid,
        "title": title,
        "description": description,
        "channel_name": channel_name,
        "channel_id": channel_id,
        "published_at": published_at,
        "duration_s": duration_s,
        "url": watch_url,
        "thumbnail_url": thumbnail_url,
        "document_type": "youtube_video",
        "node_type": "parent",
        "source": source_label,
        "topic_summary": topic_summary,
        "router_tags": router_tags,
        "aliases": aliases,
        "canonical_entities": canonical_entities,
        "is_explainer": is_explainer,
        "router_boost": router_boost,
    }

    children: List[Dict[str, Any]] = []
    for start_s, end_s, text in segments:
        sid = segment_uuid(vid, start_s, end_s)
        source_hash = sha1_hex(f"{vid}:{start_s:.3f}:{end_s:.3f}:{text}".encode("utf-8"))
        children.append(
            {
                "segment_id": sid,
                "parent_id": vid,
                "video_id": vid,
                "document_type": "youtube_video",
                "node_type": "child",
                "title": title,
                "channel_name": channel_name,
                "channel_id": channel_id,
                "published_at": published_at,
                "duration_s": duration_s,
                "url": watch_url,
                "clip_url": _yt_clip_url(vid, start_s),
                "thumbnail_url": thumbnail_url,
                "start_seconds": float(start_s),
                "start_hms": _seconds_to_hms(start_s),
                "end_hms": _seconds_to_hms(end_s),
                "text": text,
                "source": source_label,
                "source_hash": source_hash,
            }
        )

    upsert_parents([parent])
    stats = upsert_children(children)
    return {
        "video_id": vid,
        "title": title,
        "channel_name": channel_name,
        "segments": len(children),
        "upsert_stats": stats,
    }


def discover_channel_video_urls(*, channel: str, max_videos: int = 10) -> List[str]:
    """
    Resolve a handle/channel-id/url into a list of watch URLs.

    Prefer yt-dlp first so channel discovery does not consume YouTube Data API
    quota by default. Fall back to the API only if yt-dlp discovery fails.
    """
    raw = (channel or "").strip()
    if not raw:
        return []

    YDL = _require_ytdlp()
    channel_url = _normalize_channel_url(raw)
    # If user gave a watch URL, treat as a single "discovered" item.
    if _is_youtube_watch_url(channel_url):
        coerced = _coerce_watch_url(channel_url)
        return [coerced] if coerced else []

    ydl_opts: Dict[str, Any] = {
        "quiet": True,
        "no_warnings": True,
        "extract_flat": True,
        "skip_download": True,
        "playlistend": int(max(1, max_videos)),
    }
    ydl_opts.update(_ytdlp_extra_opts())
    with YDL(ydl_opts) as ydl:
        try:
            info = ydl.extract_info(channel_url, download=False)
        except Exception as exc:
            raise _maybe_wrap_ytdlp_error(exc) from exc
    entries = info.get("entries") or []
    urls: List[str] = []
    for e in entries:
        value = None
        if isinstance(e, dict):
            value = e.get("webpage_url") or e.get("url") or e.get("id")
        if not value:
            continue
        watch = _coerce_watch_url(str(value))
        if not watch:
            continue
        urls.append(watch)
        if len(urls) >= int(max_videos):
            break
    if urls:
        return urls

    api_key = (os.environ.get("YOUTUBE_API_KEY") or "").strip()
    if api_key:
        try:
            urls = _discover_channel_video_urls_via_api(channel=raw, max_videos=max_videos, api_key=api_key)
            if urls:
                return urls
        except Exception:
            pass
    return urls


def index_youtube_channel_captions(
    *,
    channel: str,
    max_videos: int = 10,
    namespace: str = "videos",
    language: str = "en",
    prefer_auto: bool = True,
    segment_min_s: Optional[float] = None,
    segment_max_s: Optional[float] = None,
    segment_stride_s: Optional[float] = None,
    min_text_chars: Optional[int] = None,
) -> Dict[str, Any]:
    urls = discover_channel_video_urls(channel=channel, max_videos=max_videos)
    if not urls:
        raise RuntimeError(f"no videos discovered for channel={channel!r}")
    out: Dict[str, Any] = {"channel": channel, "max_videos": max_videos, "indexed": [], "failed": []}
    for url in urls:
        try:
            res = index_youtube_video_captions(
                video_url=url,
                namespace=namespace,
                language=language,
                prefer_auto=prefer_auto,
                segment_min_s=segment_min_s,
                segment_max_s=segment_max_s,
                segment_stride_s=segment_stride_s,
                min_text_chars=min_text_chars,
            )
            out["indexed"].append(res)
        except Exception as exc:
            LOG.warning("[yt-index] failed url=%s err=%s", url, exc)
            out["failed"].append({"url": url, "error": str(exc)})
    return out


def discover_query_video_urls(*, query: str, max_videos: int = 10) -> List[str]:
    """
    Resolve a free-text query into a list of watch URLs using yt-dlp.

    Uses ytsearch first so query discovery does not consume YouTube Data API
    quota by default. Falls back to the API only if yt-dlp search fails.
    """
    q = (query or "").strip()
    if not q:
        return []

    YDL = _require_ytdlp()
    ydl_opts: Dict[str, Any] = {
        "quiet": True,
        "no_warnings": True,
        "extract_flat": True,
        "skip_download": True,
    }
    ydl_opts.update(_ytdlp_extra_opts())
    with YDL(ydl_opts) as ydl:
        try:
            info = ydl.extract_info(f"ytsearch{int(max(1, max_videos))}:{q}", download=False)
        except Exception as exc:
            raise _maybe_wrap_ytdlp_error(exc) from exc

    entries = info.get("entries") or []
    urls: List[str] = []
    for e in entries:
        value = None
        if isinstance(e, dict):
            value = e.get("webpage_url") or e.get("url") or e.get("id")
        if not value:
            continue
        watch = _coerce_watch_url(str(value))
        if not watch:
            continue
        urls.append(watch)
        if len(urls) >= int(max_videos):
            break
    if urls:
        return urls

    api_key = (os.environ.get("YOUTUBE_API_KEY") or "").strip()
    if api_key:
        try:
            urls = _discover_query_video_urls_via_api(query=q, max_videos=max_videos, api_key=api_key)
            if urls:
                return urls
        except Exception:
            pass
    return urls


def index_youtube_query_captions(
    *,
    query: str,
    max_videos: int = 10,
    namespace: str = "videos",
    language: str = "en",
    prefer_auto: bool = True,
    segment_min_s: Optional[float] = None,
    segment_max_s: Optional[float] = None,
    segment_stride_s: Optional[float] = None,
    min_text_chars: Optional[int] = None,
) -> Dict[str, Any]:
    urls = discover_query_video_urls(query=query, max_videos=max_videos)
    if not urls:
        raise RuntimeError(f"no videos discovered for query={query!r}")
    out: Dict[str, Any] = {"query": query, "max_videos": max_videos, "indexed": [], "failed": []}
    for url in urls:
        try:
            res = index_youtube_video_captions(
                video_url=url,
                namespace=namespace,
                language=language,
                prefer_auto=prefer_auto,
                segment_min_s=segment_min_s,
                segment_max_s=segment_max_s,
                segment_stride_s=segment_stride_s,
                min_text_chars=min_text_chars,
            )
            out["indexed"].append(res)
        except Exception as exc:
            LOG.warning("[yt-index] failed url=%s err=%s", url, exc)
            out["failed"].append({"url": url, "error": str(exc)})
    return out


_YOUTUBE_API_BASE = "https://www.googleapis.com/youtube/v3"


def _ytapi_get(path: str, *, api_key: str, params: Dict[str, Any]) -> Dict[str, Any]:
    qp = dict(params)
    qp["key"] = api_key
    url = f"{_YOUTUBE_API_BASE}/{path.lstrip('/')}"
    with requests.Session() as session:
        session.trust_env = False
        resp = session.get(url, params=qp, timeout=20)
    resp.raise_for_status()
    data = resp.json()
    if not isinstance(data, dict):
        raise ValueError("unexpected YouTube API response type")
    return data


def _resolve_channel_id_via_api(identifier: str, *, api_key: str) -> Optional[str]:
    ident = (identifier or "").strip()
    if not ident:
        return None
    if ident.startswith("UC") and len(ident) >= 16:
        return ident
    m = re.search(r"youtube\\.com/channel/(?P<id>UC[a-zA-Z0-9_-]{22})", ident)
    if m:
        return m.group("id")
    handle = None
    if ident.startswith("@"):
        handle = ident[1:]
    else:
        mh = re.search(r"youtube\\.com/@(?P<h>[A-Za-z0-9._-]{3,})", ident)
        if mh:
            handle = mh.group("h")
    if handle:
        resp = _ytapi_get("channels", api_key=api_key, params={"part": "id", "forHandle": handle, "maxResults": 1})
        items = resp.get("items") or []
        if items:
            cid = items[0].get("id")
            if cid:
                return str(cid)
    # fallback: search
    resp = _ytapi_get(
        "search",
        api_key=api_key,
        params={"part": "id", "q": ident, "type": "channel", "maxResults": 1},
    )
    items = resp.get("items") or []
    if items:
        cid = (items[0].get("id") or {}).get("channelId")
        if cid:
            return str(cid)
    return None


def _uploads_playlist_id(channel_id: str, *, api_key: str) -> Optional[str]:
    resp = _ytapi_get(
        "channels",
        api_key=api_key,
        params={"part": "contentDetails", "id": channel_id, "maxResults": 1},
    )
    items = resp.get("items") or []
    if not items:
        return None
    cd = items[0].get("contentDetails") or {}
    rel = cd.get("relatedPlaylists") or {}
    return rel.get("uploads")


def _discover_channel_video_urls_via_api(*, channel: str, max_videos: int, api_key: str) -> List[str]:
    cid = _resolve_channel_id_via_api(channel, api_key=api_key)
    if not cid:
        return []
    uploads = _uploads_playlist_id(cid, api_key=api_key)
    if not uploads:
        return []
    resp = _ytapi_get(
        "playlistItems",
        api_key=api_key,
        params={"part": "snippet", "playlistId": uploads, "maxResults": max(1, min(int(max_videos), 50))},
    )
    urls: List[str] = []
    for item in resp.get("items") or []:
        sn = item.get("snippet") or {}
        rid = (sn.get("resourceId") or {}).get("videoId")
        if not rid:
            continue
        urls.append(_yt_watch_url(str(rid)))
        if len(urls) >= int(max_videos):
            break
    return urls


def _discover_query_video_urls_via_api(*, query: str, max_videos: int, api_key: str) -> List[str]:
    q = (query or "").strip()
    if not q:
        return []

    target = max(1, int(max_videos))
    urls: List[str] = []
    page_token: Optional[str] = None
    while len(urls) < target:
        remaining = target - len(urls)
        params: Dict[str, Any] = {
            "part": "id",
            "q": q,
            "type": "video",
            "maxResults": max(1, min(int(remaining), 50)),
        }
        if page_token:
            params["pageToken"] = str(page_token)
        resp = _ytapi_get("search", api_key=api_key, params=params)
        items = resp.get("items") or []
        if not items:
            break
        for item in items:
            vid = (item.get("id") or {}).get("videoId")
            if not vid:
                continue
            urls.append(_yt_watch_url(str(vid)))
            if len(urls) >= target:
                break
        page_token = resp.get("nextPageToken")
        if not page_token:
            break
    return urls
