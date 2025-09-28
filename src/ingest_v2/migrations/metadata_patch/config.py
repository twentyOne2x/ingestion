from __future__ import annotations
import os
from typing import Dict, Any

# Default fields to enforce if PATCH_FIELDS env isn't set.
# You can change these defaults freely; they're only used when a key is missing.
DEFAULT_FIELDS: Dict[str, Any] = {
    "router_tags": [],
    "aliases": [],
    "canonical_entities": [],
    "is_explainer": False,
    "router_boost": 1.0,
    "topic_summary": "",
    # Example child-safe defaults (uncomment if you decide to enforce these too)
    # "rights": "public_reference_only",
    # "ingest_version": 2,
}


def load_target_fields() -> Dict[str, Any]:
    """
    Build the target field->default map from env and DEFAULT_FIELDS.
    You can override any default via env: PATCH_DEFAULT_<UPPERCASE_FIELD>
    Example: PATCH_DEFAULT_ROUTER_BOOST=1.25
    Lists should be JSON-like strings e.g. '["a","b"]' or '[]'
    """
    import json

    fields_env = os.getenv("PATCH_FIELDS", "").strip()
    if fields_env:
        keys = [k.strip() for k in fields_env.split(",") if k.strip()]
        # Only keep specified keys; fall back to DEFAULT_FIELDS for defaults
        base = {k: DEFAULT_FIELDS.get(k, None) for k in keys}
    else:
        base = dict(DEFAULT_FIELDS)

    out: Dict[str, Any] = {}
    for k, v in base.items():
        override = os.getenv(f"PATCH_DEFAULT_{k.upper()}")
        if override is None:
            out[k] = v
        else:
            # Try JSON first, then fall back to raw strings/numbers/bools
            try:
                out[k] = json.loads(override)
            except Exception:
                # Best-effort cast for common types
                if override.lower() in ("true", "false"):
                    out[k] = (override.lower() == "true")
                else:
                    try:
                        out[k] = float(override) if "." in override else int(override)
                    except Exception:
                        out[k] = override
    return out
