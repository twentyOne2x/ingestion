"""
Cross-process concurrency guard backed by a shared state file.

The guard keeps the combined number of "slots" (typically threads) reserved
across multiple Python processes under a configurable global limit. Each
reservation acquires slots and releases them when leaving the context manager
or when `release` is called explicitly.

Default behaviour:
  • Global limit defaults to the host CPU count (minimum 2). Override via the
    `GLOBAL_THREAD_LIMIT` environment variable.
  • Separate "scopes" can be created with `GLOBAL_THREAD_SCOPE`. When unset,
    the scope is derived automatically from the nearest project directory
    (falling back to `"default"`).
  • State files live under `~/.cache/global_thread_guard/` by default and can be
    relocated via `GLOBAL_THREAD_STATE_DIR`.

Typical usage:
    from src.utils.global_thread_guard import get_global_thread_limiter

    limiter = get_global_thread_limiter()
    with limiter.claim(4, label="ingest-workers"):
        with ThreadPoolExecutor(max_workers=4) as pool:
            ...

Async code can use `claim_async` or `global_thread_slots_async`.
"""

from __future__ import annotations

import asyncio
import hashlib
import json
import logging
import os
import time
import uuid
from contextlib import asynccontextmanager, contextmanager
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, Optional

LOCK_BYTES = 1  # advisory lock byte count

if os.name == "nt":  # pragma: no cover - Windows support retained for completeness
    import msvcrt

    def _lock_file(handle):
        handle.seek(0)
        msvcrt.locking(handle.fileno(), msvcrt.LK_LOCK, LOCK_BYTES)

    def _unlock_file(handle):
        handle.seek(0)
        try:
            msvcrt.locking(handle.fileno(), msvcrt.LK_UNLCK, LOCK_BYTES)
        except OSError:
            pass
else:  # POSIX platforms
    import fcntl

    def _lock_file(handle):
        fcntl.flock(handle, fcntl.LOCK_EX)

    def _unlock_file(handle):
        try:
            fcntl.flock(handle, fcntl.LOCK_UN)
        except OSError:
            pass


def _pid_alive(pid: int) -> bool:
    """Best-effort check whether a PID is still alive."""
    if pid <= 0:
        return False
    if pid == os.getpid():
        return True
    try:
        os.kill(pid, 0)
    except ProcessLookupError:
        return False
    except PermissionError:
        return True
    else:
        return True


def _default_limit_from_cpu() -> int:
    """Derive a conservative default limit from the host CPU count."""
    cpu_cnt = os.cpu_count() or 4
    return max(2, cpu_cnt)


def _derive_default_scope() -> str:
    """
    Attempt to derive a stable scope identifier from the current project root.
    Falls back to 'default' if we cannot infer anything sensible.
    """
    candidates = []
    try:
        cwd = Path.cwd().resolve()
        candidates.append(cwd)
        candidates.extend(parent for parent in cwd.parents)
    except Exception:
        candidates = []

    for path in candidates:
        try:
            if (path / ".git").exists() or (path / "pyproject.toml").exists() or (path / "setup.cfg").exists():
                digest = hashlib.sha1(str(path).encode("utf-8")).hexdigest()[:12]
                name = path.name.strip() or "project"
                safe = "".join(ch if ch.isalnum() or ch in ("-", "_") else "_" for ch in name)
                return f"{safe}_{digest}"
        except Exception:
            continue

    return "default"


@dataclass
class _Token:
    token_id: str
    slots: int
    label: Optional[str]


class GlobalThreadLimiter:
    """Coordinates slot reservations across processes using a JSON state file."""

    def __init__(
        self,
        *,
        limit: Optional[int] = None,
        scope: Optional[str] = None,
        state_dir: Optional[str | Path] = None,
        poll_interval: float = 0.5,
        stale_timeout: float = 120.0,
        limit_env: str = "GLOBAL_THREAD_LIMIT",
        scope_env: str = "GLOBAL_THREAD_SCOPE",
    ) -> None:
        self._configured_limit = max(1, int(limit)) if limit else None
        self._limit_env = limit_env
        raw_scope = scope if isinstance(scope, str) else None
        env_scope = os.getenv(scope_env, "")
        scope_value = (raw_scope or env_scope).strip()
        if not scope_value:
            scope_value = _derive_default_scope()
        self._scope = scope_value
        state_dir = Path(state_dir or os.getenv("GLOBAL_THREAD_STATE_DIR") or Path.home() / ".cache" / "global_thread_guard")
        state_dir.mkdir(parents=True, exist_ok=True)
        self._state_path = state_dir / f"{self._scope}.json"
        self._poll_interval = max(0.05, poll_interval)
        self._stale_timeout = max(30.0, stale_timeout)
        self._logger = logging.getLogger(__name__)

    # ------------------------------------------------------------------ helpers
    def _resolve_limit(self) -> int:
        env_override = os.getenv(self._limit_env)
        if env_override:
            try:
                value = int(env_override)
                if value > 0:
                    return value
            except ValueError:
                self._logger.warning("Invalid %s=%s; falling back to configured limit.", self._limit_env, env_override)
        if self._configured_limit:
            return self._configured_limit
        return _default_limit_from_cpu()

    def _open_state_file(self):
        self._state_path.touch(exist_ok=True)
        return self._state_path.open("r+")

    def _load_state(self, handle) -> Dict:
        handle.seek(0)
        raw = handle.read()
        if not raw:
            return {"limit": self._resolve_limit(), "processes": {}}
        try:
            data = json.loads(raw)
        except json.JSONDecodeError:
            self._logger.warning("Corrupted thread guard state file; recreating.")
            return {"limit": self._resolve_limit(), "processes": {}}

        limit = data.get("limit") or self._resolve_limit()
        processes: Dict[str, Dict] = {}
        for pid_str, info in data.get("processes", {}).items():
            try:
                int(pid_str)
            except (TypeError, ValueError):
                continue
            if not isinstance(info, dict):
                continue
            tokens = {}
            for token_id, token_info in info.get("tokens", {}).items():
                if not isinstance(token_info, dict):
                    continue
                slots = int(token_info.get("slots", 0))
                if slots <= 0:
                    continue
                tokens[str(token_id)] = {
                    "slots": slots,
                    "label": token_info.get("label"),
                    "timestamp": float(token_info.get("timestamp", 0.0) or 0.0),
                }
            slots_total = sum(tok["slots"] for tok in tokens.values())
            if slots_total <= 0:
                continue
            processes[str(pid_str)] = {
                "slots": slots_total,
                "tokens": tokens,
                "timestamp": float(info.get("timestamp", time.time())),
            }
        return {"limit": int(limit), "processes": processes}

    def _write_state(self, handle, data: Dict) -> None:
        handle.seek(0)
        serialized = {
            "limit": int(data.get("limit", self._resolve_limit())),
            "processes": {},
            "version": 1,
            "updated_at": time.time(),
        }
        for pid_str, info in data.get("processes", {}).items():
            tokens = {}
            for token_id, token_info in info.get("tokens", {}).items():
                tokens[token_id] = {
                    "slots": int(token_info.get("slots", 0)),
                    "label": token_info.get("label"),
                    "timestamp": float(token_info.get("timestamp", 0.0)),
                }
            serialized["processes"][pid_str] = {
                "slots": int(info.get("slots", 0)),
                "tokens": tokens,
                "timestamp": float(info.get("timestamp", time.time())),
            }
        handle.write(json.dumps(serialized, separators=(",", ":")))
        handle.truncate()
        handle.flush()
        os.fsync(handle.fileno())

    def _prune_stale(self, state: Dict) -> bool:
        now = time.time()
        changed = False
        for pid_str in list(state["processes"].keys()):
            try:
                pid_int = int(pid_str)
            except (TypeError, ValueError):
                state["processes"].pop(pid_str, None)
                changed = True
                continue
            info = state["processes"][pid_str]
            tokens = info.get("tokens", {})
            if not tokens:
                state["processes"].pop(pid_str, None)
                changed = True
                continue
            if pid_int == os.getpid():
                continue
            if _pid_alive(pid_int):
                if now - info.get("timestamp", now) > self._stale_timeout:
                    info["timestamp"] = now
                    changed = True
                continue
            state["processes"].pop(pid_str, None)
            changed = True
        return changed

    def _current_usage(self, state: Dict) -> int:
        return sum(int(info.get("slots", 0)) for info in state.get("processes", {}).values())

    # ---------------------------------------------------------------- interface
    def acquire(self, slots: int, *, label: Optional[str] = None, timeout: Optional[float] = None) -> _Token:
        """Block until `slots` are available, then reserve them and return a token."""
        if slots <= 0:
            raise ValueError("slots must be >= 1")

        limit = self._resolve_limit()
        if slots > limit:
            raise ValueError(
                f"Requested {slots} slot(s) but the global limit is {limit}. "
                f"Increase GLOBAL_THREAD_LIMIT or request fewer slots."
            )

        deadline = time.time() + timeout if timeout else None
        token_id = f"{os.getpid()}-{uuid.uuid4().hex}"

        while True:
            with self._open_state_file() as handle:
                _lock_file(handle)
                try:
                    state = self._load_state(handle)
                    state["limit"] = limit
                    changed = self._prune_stale(state)
                    in_use = self._current_usage(state)
                    available = max(0, limit - in_use)

                    if slots <= available:
                        pid_key = str(os.getpid())
                        proc_entry = state["processes"].setdefault(
                            pid_key, {"slots": 0, "tokens": {}, "timestamp": time.time()}
                        )
                        proc_entry["slots"] = int(proc_entry.get("slots", 0)) + slots
                        tokens = proc_entry.setdefault("tokens", {})
                        tokens[token_id] = {
                            "slots": slots,
                            "label": label,
                            "timestamp": time.time(),
                        }
                        proc_entry["timestamp"] = time.time()
                        state["processes"][pid_key] = proc_entry
                        self._write_state(handle, state)
                        return _Token(token_id=token_id, slots=slots, label=label)

                    if changed:
                        self._write_state(handle, state)
                finally:
                    _unlock_file(handle)

            if deadline and time.time() > deadline:
                raise TimeoutError(
                    f"Timed out waiting for {slots} slot(s). "
                    f"limit={limit} currently in use={in_use} available={available}"
                )

            time.sleep(self._poll_interval)

    def release(self, token: _Token) -> None:
        """Release slots associated with the provided token."""
        with self._open_state_file() as handle:
            _lock_file(handle)
            try:
                state = self._load_state(handle)
                pid_key = str(os.getpid())
                proc_entry = state["processes"].get(pid_key)
                if not proc_entry:
                    self._write_state(handle, state)
                    return

                tokens = proc_entry.get("tokens", {})
                reservation = tokens.pop(token.token_id, None)
                if reservation is None:
                    self._write_state(handle, state)
                    return

                slots = int(reservation.get("slots", token.slots))
                proc_entry["slots"] = max(0, int(proc_entry.get("slots", 0)) - slots)
                if proc_entry["slots"] <= 0 or not tokens:
                    state["processes"].pop(pid_key, None)
                else:
                    proc_entry["tokens"] = tokens
                    proc_entry["timestamp"] = time.time()
                    state["processes"][pid_key] = proc_entry

                self._write_state(handle, state)
            finally:
                _unlock_file(handle)

    def status(self) -> Dict[str, int]:
        """Return a snapshot of current allocation per PID."""
        with self._open_state_file() as handle:
            _lock_file(handle)
            try:
                state = self._load_state(handle)
            finally:
                _unlock_file(handle)
        summary = {
            "limit": int(state.get("limit", self._resolve_limit())),
            "in_use": self._current_usage(state),
            "processes": {pid: info.get("slots", 0) for pid, info in state.get("processes", {}).items()},
        }
        summary["available"] = max(0, summary["limit"] - summary["in_use"])
        return summary

    # ------------------------------------------------------------ context utils
    @contextmanager
    def claim(self, slots: int, *, label: Optional[str] = None, timeout: Optional[float] = None):
        token = self.acquire(slots, label=label, timeout=timeout)
        try:
            yield token
        finally:
            self.release(token)

    @asynccontextmanager
    async def claim_async(self, slots: int, *, label: Optional[str] = None, timeout: Optional[float] = None):
        token = await asyncio.to_thread(self.acquire, slots, label=label, timeout=timeout)
        try:
            yield token
        finally:
            await asyncio.to_thread(self.release, token)


_DEFAULT_LIMITER: Optional[GlobalThreadLimiter] = None


def get_global_thread_limiter(
    *,
    limit: Optional[int] = None,
    scope: Optional[str] = None,
    state_dir: Optional[str | Path] = None,
) -> GlobalThreadLimiter:
    """Return the process-wide singleton limiter."""
    global _DEFAULT_LIMITER
    if _DEFAULT_LIMITER is None:
        _DEFAULT_LIMITER = GlobalThreadLimiter(limit=limit, scope=scope, state_dir=state_dir)
    else:
        if limit and limit != _DEFAULT_LIMITER._configured_limit:
            _DEFAULT_LIMITER._configured_limit = max(1, int(limit))
        if scope and scope != _DEFAULT_LIMITER._scope:
            _DEFAULT_LIMITER = GlobalThreadLimiter(limit=limit, scope=scope, state_dir=state_dir)
    return _DEFAULT_LIMITER


@contextmanager
def global_thread_slots(slots: int, *, label: Optional[str] = None, timeout: Optional[float] = None):
    """Convenience sync context manager using the default limiter."""
    limiter = get_global_thread_limiter()
    with limiter.claim(slots, label=label, timeout=timeout):
        yield


@asynccontextmanager
async def global_thread_slots_async(slots: int, *, label: Optional[str] = None, timeout: Optional[float] = None):
    """Convenience async context manager using the default limiter."""
    limiter = get_global_thread_limiter()
    async with limiter.claim_async(slots, label=label, timeout=timeout):
        yield
