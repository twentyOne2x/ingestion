import math

def s_to_hms_ms(s: float) -> str:
    ms = int(round((s - int(s)) * 1000))
    h = int(s) // 3600
    m = (int(s) % 3600) // 60
    sec = int(s) % 60
    return f"{h:02d}:{m:02d}:{sec:02d}.{ms:03d}"

def floor_s(s: float) -> int:
    return int(math.floor(s))
