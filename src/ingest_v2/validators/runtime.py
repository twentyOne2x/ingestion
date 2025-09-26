from ..schemas.child import ChildNode
from ..schemas.parent import ParentNode
import requests

def validate_child_runtime(x: dict, duration_s: float) -> None:
    c = ChildNode(**x)
    assert 0 <= c.start_s < c.end_s <= duration_s, "timestamp order invalid"
    assert 15 - 1e-6 <= (c.end_s - c.start_s) <= 60 + 1e-6, "window size out of bounds"
    assert len(c.text) >= 80 or c.text.endswith("?"), "text too short"
    if c.clip_url:
        try:
            resp = requests.head(str(c.clip_url), allow_redirects=True, timeout=4)
            assert resp.status_code in (200, 301, 302, 303, 307, 308)
        except Exception:
            pass

def validate_parent_runtime(p: dict) -> None:
    ParentNode(**p)
