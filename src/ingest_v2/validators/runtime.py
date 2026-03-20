# src/ingest_v2/validators/runtime.py

import os
import logging
import re
from typing import Optional, Tuple

import requests

from ..schemas.child import ChildNode
from ..schemas.parent import ParentNode
from ..configs.settings import settings_v2

_ASCII_RE = re.compile(r"[^\x00-\x7F]+")

def _safe(s: Optional[str], max_len: int = 180) -> str:
    """ASCII-safe preview for logs to avoid latin-1 console crashes."""
    s = s or ""
    if len(s) > max_len:
        s = s[:max_len] + "..."
    return _ASCII_RE.sub("?", s)

def _ctx_str(prefix: str, c: "ChildNode", duration_s: float, extra: Optional[str] = None) -> str:
    seg_len = (c.end_s - c.start_s) if (c.start_s is not None and c.end_s is not None) else -1.0
    # guard formatting if any is None (ChildNode usually enforces floats, but be safe)
    try:
        start = f"{c.start_s:.3f}"
        end = f"{c.end_s:.3f}"
        seg_len_str = f"{seg_len:.3f}" if seg_len is not None else "NA"
        dur = f"{duration_s:.3f}"
    except Exception:
        start = str(c.start_s)
        end = str(c.end_s)
        seg_len_str = str(seg_len)
        dur = str(duration_s)

    snippet = _safe(getattr(c, "text", ""))
    base = (
        f"{prefix} parent_id={_safe(str(c.parent_id))} seg_id={_safe(str(c.segment_id))} "
        f"start={start} end={end} len={seg_len_str} "
        f"dur={dur} lang={_safe(str(getattr(c, 'language', '')))} "
        f"speaker={_safe(str(getattr(c, 'speaker', '')))} "
        f"text_len={len(getattr(c, 'text', ''))} preview={snippet!r}"
    )
    return base + (f" | {extra}" if extra else "")

def validate_child_runtime(x: dict, duration_s: float) -> Tuple[bool, str]:
    """
    Returns (ok, reason). Never raises.
    Accepts padding: allowed_max = max_s + 2*pad + tol.
    Optional HEAD check gated by VALIDATE_URLS.
    """
    # Keep validator thresholds aligned with the segmenter. For local backfills we
    # often lower MIN_TEXT_CHARS to ingest short/sparse transcripts; the validator
    # must not keep a higher fixed minimum or it will drop most segments.
    min_chars = min(int(getattr(settings_v2, "MIN_TEXT_CHARS", 160)), 80)
    try:
        override = (os.getenv("VALIDATE_MIN_TEXT_CHARS") or "").strip()
        if override:
            min_chars = int(override)
    except Exception:
        pass
    min_chars = max(1, int(min_chars))

    EPS = 1e-6
    min_s = getattr(settings_v2, "SEGMENT_MIN_S", 15.0)
    max_s = getattr(settings_v2, "SEGMENT_MAX_S", 60.0)
    pad_s = getattr(settings_v2, "SEGMENT_PAD_S", 1.5)
    tol_s = getattr(settings_v2, "SEGMENT_TOLERANCE_S", 0.75)

    upper_with_pad = max_s + (2.0 * pad_s) + tol_s
    # Mirror the segmenter's adaptive minimum window behavior:
    # when SEGMENT_MIN_S is tuned high (e.g. 300s) for cost reasons, short videos
    # should still be able to index at least one shorter segment.
    min_s_eff = float(min_s)
    try:
        if isinstance(duration_s, (int, float)) and duration_s > 0 and duration_s < min_s_eff:
            # Keep this floor tiny: many corpus clips are <10s and should still be indexed.
            min_s_eff = max(1.0, float(duration_s) * 0.5)
    except Exception:
        pass

    lower_with_tol = min_s_eff - tol_s

    if (x or {}).get("node_type") == "summary":
        try:
            start = float(x.get("start_s", 0.0)); end = float(x.get("end_s", 0.0))
            if not (0 <= start < end <= duration_s + 1e-6): return False, "timestamp_order_invalid"
            txt = (x.get("text") or "")
            if not (len(txt) >= min_chars or (txt.endswith("?") and len(txt) >= 20)): return False, "text_too_short"
            return True, "ok"
        except Exception:
            return False, "schema_error"

    # Build schema object
    try:
        c = ChildNode(**x)
    except Exception as e:
        pid = x.get("parent_id"); sid = x.get("segment_id")
        logging.warning(f"[validate_child] schema_error parent_id={_safe(str(pid))} "
                        f"seg_id={_safe(str(sid))} err={_safe(str(e))}")
        return False, "schema_error"

    is_summary = (getattr(c, "node_type", None) == "summary")

    # Timestamp order
    if not (0 <= c.start_s < c.end_s <= duration_s + EPS):
        logging.info(_ctx_str("[validate_child] bad_timestamps", c, duration_s))
        return False, "timestamp_order_invalid"

    # Text sufficiency (summary has the same minimal guard)
    txt = getattr(c, "text", "") or ""
    if not (len(txt) >= min_chars or (txt.endswith("?") and len(txt) >= 20)):
        tag = "summary_text_too_short" if is_summary else "text_too_short"
        logging.info(_ctx_str(f"[validate_child] {tag}", c, duration_s))
        return False, "text_too_short"

    # Window length check (skip for summaries)
    if not is_summary:
        seg_len = c.end_s - c.start_s
        if not (lower_with_tol <= seg_len <= upper_with_pad):
            extra = (f"allowed=[{lower_with_tol:.3f},{upper_with_pad:.3f}] "
                     f"min={min_s} min_eff={min_s_eff:.3f} max={max_s} pad={pad_s} tol={tol_s}")
            logging.info(_ctx_str("[validate_child] window_oob", c, duration_s, extra))
            return False, "window_size_out_of_bounds"

    # Optional URL HEAD check (non-fatal)
    if os.getenv("VALIDATE_URLS", "0").lower() in ("1", "true", "yes"):
        clip = getattr(c, "clip_url", None)
        if clip:
            try:
                resp = requests.head(str(clip), allow_redirects=True, timeout=4)
                if resp.status_code not in (200, 301, 302, 303, 307, 308):
                    logging.info(
                        f"[validate_child] clip_url_status parent_id={_safe(str(c.parent_id))} "
                        f"seg_id={_safe(str(c.segment_id))} status={resp.status_code} "
                        f"url={_safe(str(clip), 240)}"
                    )
            except Exception as e:
                logging.info(
                    f"[validate_child] clip_url_head_failed parent_id={_safe(str(c.parent_id))} "
                    f"seg_id={_safe(str(c.segment_id))} err={_safe(str(e))} "
                    f"url={_safe(str(clip), 240)}"
                )

    return True, "ok"

def validate_parent_runtime(p: dict) -> Tuple[bool, str]:
    try:
        parent = ParentNode(**p)
    except Exception as e:
        pid = p.get("parent_id")
        logging.warning(f"[validate_parent] schema_error parent_id={_safe(str(pid))} err={_safe(str(e))}")
        return False, "schema_error"

    if parent.duration_s is None or parent.duration_s < 0:
        logging.info(f"[validate_parent] non_positive_duration parent_id={_safe(str(parent.parent_id))} "
                     f"duration_s={parent.duration_s}")
        # Not fatal
    return True, "ok"
