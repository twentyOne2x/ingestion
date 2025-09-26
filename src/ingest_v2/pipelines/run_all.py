import argparse, logging
from pathlib import Path
from ..configs.settings import settings_v2
from ..pipelines.build_parents import run_build_parents
from ..pipelines.build_children import build_children_from_raw
from ..pipelines.upsert_pinecone import upsert_children
from ..utils.logging import setup_logger
from src.Llama_index_sandbox.utils.utils import root_directory
from src.Llama_index_sandbox.data_ingestion_youtube.load.load import load_video_transcripts

def _iter_youtube_assets(add_new_transcripts: bool = True, num_files=None, files_window=None):
    docs = load_video_transcripts(
        directory_path=Path(root_directory()) / "datasets/evaluation_data/diarized_youtube_content_2023-10-06/",
        add_new_transcripts=add_new_transcripts,
        num_files=num_files,
        files_window=files_window,
        overwrite=False
    )
    for d in docs:
        meta = dict(d.metadata)
        raw = {
            "segments": meta.get("segments") or [],
            "caption_lines": meta.get("caption_lines") or [],
            "diarization": meta.get("diarization") or [],
        }
        meta.setdefault("video_id", meta.get("video_id") or meta.get("id") or meta.get("youtube_video_id"))
        meta.setdefault("duration_s", meta.get("duration_s") or meta.get("duration") or 0)
        meta.setdefault("url", meta.get("url") or f"https://www.youtube.com/watch?v={meta['video_id']}")
        yield meta, raw

def main():
    setup_logger("ingest_v2")
    ap = argparse.ArgumentParser()
    ap.add_argument("--doc-type", default="youtube_video", choices=["youtube_video", "stream"])
    ap.add_argument("--backfill-days", type=int, default=settings_v2.BACKFILL_DAYS)
    args = ap.parse_args()

    if args.doc_type == "youtube_video":
        metas_raw = list(_iter_youtube_assets(add_new_transcripts=True))
        metas = [m for m, _ in metas_raw]
    else:
        metas_raw = []
        metas = []

    parents = run_build_parents(metas=[{
        "video_id": m["video_id"],
        "title": m.get("title", ""),
        "description": m.get("description", ""),
        "channel_name": m.get("channel_name"),
        "speaker_primary": m.get("speaker_primary"),
        "published_at": m.get("published_at"),
        "duration_s": m.get("duration_s", 0),
        "url": m.get("url"),
        "thumbnail_url": m.get("thumbnail_url"),
        "language": m.get("language", "en"),
        "entities": m.get("entities", []),
        "chapters": m.get("chapters"),
    } for m in metas])

    parents_map = {p.parent_id: p.dict() for p in parents}
    total_children = 0
    for (meta, raw) in metas_raw:
        parent = parents_map[meta["video_id"]]
        children = build_children_from_raw(parent, raw)
        upsert_children(children)
        total_children += len(children)

    logging.info(f"[ingest_v2] finished upserting {total_children} child segments.")

if __name__ == "__main__":
    main()
