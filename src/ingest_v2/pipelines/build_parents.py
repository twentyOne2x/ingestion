import logging
from typing import Dict, Any, List
from ..schemas.parent import ParentNode
from ..sources.youtube import build_parent_from_metadata
from ..validators.runtime import validate_parent_runtime
from ..utils.logging import setup_logger

def build_parent(meta: Dict[str, Any]) -> ParentNode:
    p = build_parent_from_metadata(meta)
    validate_parent_runtime(p.dict())
    return p

def run_build_parents(metas: List[Dict[str, Any]]) -> List[ParentNode]:
    logging.info(f"[parents] building parents for {len(metas)} assets")
    out = [build_parent(m) for m in metas]
    logging.info(f"[parents] done {len(out)}")
    return out
