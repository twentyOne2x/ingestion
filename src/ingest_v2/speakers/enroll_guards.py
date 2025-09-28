# src/ingest_v2/speakers/enroll_guard.py
import os, json, tempfile
from contextlib import contextmanager
from .name_filters import looks_like_person

# POSIX file lock; fine on Linux (your logs show Linux). If you need cross-platform, swap for portalocker.
@contextmanager
def _lock(path: str):
    lock_path = f"{path}.lock"
    os.makedirs(os.path.dirname(lock_path), exist_ok=True)
    fd = os.open(lock_path, os.O_CREAT | os.O_RDWR, 0o644)
    try:
        import fcntl
        fcntl.flock(fd, fcntl.LOCK_EX)
        yield
    finally:
        try:
            import fcntl
            fcntl.flock(fd, fcntl.LOCK_UN)
        finally:
            os.close(fd)

def _safe_load_json(path: str):
    try:
        with open(path, "r", encoding="utf-8") as f:
            return json.load(f)
    except Exception:
        return {}

def _atomic_write_json(path: str, data: dict):
    os.makedirs(os.path.dirname(path), exist_ok=True)
    with tempfile.NamedTemporaryFile("w", delete=False, dir=os.path.dirname(path), encoding="utf-8") as tmp:
        json.dump(data, tmp, ensure_ascii=False)
        tmp.flush()
        os.fsync(tmp.fileno())
        tmp_name = tmp.name
    os.replace(tmp_name, path)

# src/ingest_v2/speakers/enroll_guards.py
def safe_auto_enroll(voice_lib_path: str, label: str, embedding: list[float], host_names=()) -> bool:
    allow = looks_like_person(label, host_names=host_names)
    if not allow:
        return False

    with _lock(voice_lib_path):
        lib = _safe_load_json(voice_lib_path)
        # normalize any legacy nested entries to flat lists
        for k, v in list(lib.items()):
            if isinstance(v, dict) and isinstance(v.get("embedding"), list):
                lib[k] = v["embedding"]
        if label in lib:
            return False
        lib[label] = embedding  # <-- write flat list
        _atomic_write_json(voice_lib_path, lib)
        return True
