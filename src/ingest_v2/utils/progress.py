# utils/progress.py
import threading

class _Progress:
    def __init__(self):
        self._lock = threading.Lock()
        self.parents_total = 0
        self.parents_done = 0
        self.planned = 0      # total vectors we will upsert (new_or_changed)
        self.done = 0         # successfully upserted vectors

    def set_parents_total(self, n: int):
        with self._lock:
            self.parents_total = n

    def add_planned(self, n: int):
        if n <= 0: return
        with self._lock:
            self.planned += n

    def add_done(self, n: int):
        if n <= 0: return
        with self._lock:
            self.done += n

    def parent_done(self):
        with self._lock:
            self.parents_done += 1

    def snapshot(self):
        with self._lock:
            planned, done = self.planned, self.done
            pt, pd = self.parents_total, self.parents_done
        remaining = max(0, planned - done)
        pct = (done / planned * 100.0) if planned else 100.0
        return {
            "planned": planned, "done": done, "remaining": remaining, "pct": pct,
            "parents_total": pt, "parents_done": pd,
        }

    def fmt(self) -> str:
        s = self.snapshot()
        return (f"done={s['done']}/{s['planned']} ({s['pct']:.1f}%) "
                f"remaining={s['remaining']} parents={s['parents_done']}/{s['parents_total']}")

progress = _Progress()
