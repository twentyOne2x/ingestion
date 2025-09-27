# File: src/ingest_v2/utils/logging.py
import logging
import os
import sys
from datetime import datetime
from pathlib import Path

# ANSI color codes (no external deps). Only used for console.
_LEVEL_COLORS = {
    "DEBUG": "\033[36m",   # cyan
    "INFO": "\033[32m",    # green
    "WARNING": "\033[33m", # yellow
    "ERROR": "\033[31m",   # red
    "CRITICAL": "\033[41m" # red background
}
_RESET = "\033[0m"


class _EncodingSafeFilter(logging.Filter):
    """Defensive: ensure record.msg is safe to emit even if the stream encoding is odd."""
    def filter(self, record: logging.LogRecord) -> bool:
        try:
            if isinstance(record.msg, str):
                record.msg = record.msg.encode("utf-8", "replace").decode("utf-8", "replace")
        except Exception:
            pass
        return True


class _ColorFormatter(logging.Formatter):
    """Colorize only the levelname and message for console output."""
    def __init__(self, fmt: str, datefmt: str | None = None, use_color: bool = True):
        super().__init__(fmt, datefmt)
        self.use_color = use_color and self._stream_supports_color()

    @staticmethod
    def _stream_supports_color() -> bool:
        # Respect NO_COLOR (https://no-color.org/)
        if os.environ.get("NO_COLOR"):
            return False
        try:
            return sys.stdout.isatty()
        except Exception:
            return False

    def format(self, record: logging.LogRecord) -> str:
        s = super().format(record)
        if not self.use_color:
            return s
        levelname = record.levelname
        color = _LEVEL_COLORS.get(levelname, "")
        if not color:
            return s
        # Colorize level name and message portion (best-effort)
        # Example original: "2025-09-27 13:13:57,701 - INFO - message"
        parts = s.split(" - ", 2)
        if len(parts) == 3:
            parts[1] = f"{color}{parts[1]}{_RESET}"
            parts[2] = f"{color}{parts[2]}{_RESET}"
            return " - ".join(parts)
        return s


def _utf8_stream(stream):
    """
    Wrap a TTY text stream in UTF-8 if possible.
    If this fails, we just return the original stream.
    """
    try:
        enc = getattr(stream, "encoding", None)
        if enc and enc.lower().replace("-", "") == "utf8":
            return stream
        return open(stream.fileno(), mode="w", encoding="utf-8", buffering=1, closefd=False)
    except Exception:
        return stream


def setup_logger(prefix: str = "ingest_v2"):
    logs_dir = Path("logs/txt")
    logs_dir.mkdir(parents=True, exist_ok=True)
    ts = datetime.now().strftime("%Y-%m-%d_%H-%M")
    log_path = logs_dir / f"{ts}_{prefix}.log"

    root = logging.getLogger()
    if root.hasHandlers():
        root.handlers.clear()
    root.setLevel(logging.INFO)

    # File handler — force UTF-8, no colors
    fh = logging.FileHandler(log_path, encoding="utf-8")
    file_fmt = logging.Formatter("%(asctime)s - %(levelname)s - %(message)s")

    # Console handler — UTF-8 + colors when possible
    ch = logging.StreamHandler(_utf8_stream(sys.stdout))
    console_fmt = _ColorFormatter("%(asctime)s - %(levelname)s - %(message)s")

    fh.setFormatter(file_fmt)
    ch.setFormatter(console_fmt)

    safe_filter = _EncodingSafeFilter()
    fh.addFilter(safe_filter)
    ch.addFilter(safe_filter)

    root.addHandler(fh)
    root.addHandler(ch)
    logging.info("********* ingest_v2 logging started (UTF-8, colored console) *********")
