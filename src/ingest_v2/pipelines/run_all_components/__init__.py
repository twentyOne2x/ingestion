"""
Support modules for the `run_all` ingestion pipeline.

The submodules expose focused helpers so `run_all.py` can remain readable while
still reusing the previous logic.
"""

from . import text, entities, assemblyai, extractors, assets, prioritize, speakers_stage, enrich, dedupe, namespace  # noqa: F401

__all__ = [
    "text",
    "entities",
    "assemblyai",
    "extractors",
    "assets",
    "prioritize",
    "speakers_stage",
    "enrich",
    "dedupe",
    "namespace",
]
