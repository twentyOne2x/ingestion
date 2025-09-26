import logging
from datetime import datetime
from pathlib import Path

def setup_logger(prefix: str = "ingest_v2"):
    logs_dir = Path("logs/txt")
    logs_dir.mkdir(parents=True, exist_ok=True)
    ts = datetime.now().strftime("%Y-%m-%d_%H-%M")
    log_path = logs_dir / f"{ts}_{prefix}.log"

    root = logging.getLogger()
    if root.hasHandlers():
        root.handlers.clear()
    root.setLevel(logging.INFO)

    fh = logging.FileHandler(log_path)
    ch = logging.StreamHandler()
    fmt = logging.Formatter("%(asctime)s - %(levelname)s - %(message)s")
    fh.setFormatter(fmt)
    ch.setFormatter(fmt)
    root.addHandler(fh)
    root.addHandler(ch)
    logging.info("********* ingest_v2 logging started *********")
