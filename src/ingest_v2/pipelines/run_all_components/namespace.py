from __future__ import annotations

import json
import os
from pathlib import Path
from typing import List


def parse_list_env(var: str) -> List[str]:
    raw = os.getenv(var, "")
    if not raw:
        return []
    import re

    return [item.strip() for item in re.split(r"[,\s]+", raw) if item.strip()]


def load_namespace_channels(namespace: str) -> List[str]:
    """
    Resolve channels for a given namespace.
    Priority:
      1) YT_NAMESPACE_CONFIG_JSON  (inline JSON string)
      2) YT_NAMESPACE_CONFIG       (path to JSON/YAML file)
      3) Empty fallback
    """
    ns = (namespace or "").strip() or "default"

    inline = os.getenv("YT_NAMESPACE_CONFIG_JSON")
    if inline:
        try:
            cfg = json.loads(inline)
            channels = (cfg.get("namespaces", {}).get(ns, {}) or {}).get("channels", [])
            if channels:
                return [channel.strip() for channel in channels if channel and channel.strip()]
        except Exception:
            pass

    config_path = os.getenv("YT_NAMESPACE_CONFIG")
    if config_path:
        path = Path(config_path).expanduser().resolve()
        if path.exists():
            text = path.read_text(encoding="utf-8")
            try:
                if path.suffix.lower() in (".yaml", ".yml"):
                    import yaml  # type: ignore

                    cfg = yaml.safe_load(text) or {}
                else:
                    cfg = json.loads(text)
                channels = (cfg.get("namespaces", {}).get(ns, {}) or {}).get("channels", [])
                if channels:
                    return [channel.strip() for channel in channels if channel and channel.strip()]
            except Exception:
                pass

    return []
