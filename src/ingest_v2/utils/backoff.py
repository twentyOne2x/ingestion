import time, random

def expo_backoff(attempt: int, base: float = 0.5, cap: float = 20.0):
    delay = min(cap, base * (2 ** attempt) + random.random())
    time.sleep(delay)
