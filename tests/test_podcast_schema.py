from src.ingest_v2.pipelines.upsert_pinecone import choose_namespace
from src.ingest_v2.schemas.child import ChildNode
from src.ingest_v2.schemas.parent import ParentNode


def test_podcast_parent_and_child_validate_in_video_namespace() -> None:
    parent = ParentNode(
        parent_id="boxes-lines-canary",
        document_type="podcast_episode",
        title="Boxes and Lines canary",
        duration_s=10.0,
        url="https://www.iex.io/podcast/canary",
        source="podcast",
        source_hash="parent-hash",
    )
    child = ChildNode(
        segment_id="boxes-lines-canary:0001",
        parent_id=parent.parent_id,
        document_type="podcast_episode",
        text="Market structure.",
        start_s=0.0,
        end_s=10.0,
        start_hms="00:00:00",
        end_hms="00:00:10",
        source_hash="child-hash",
    )

    assert parent.source == "podcast"
    assert child.document_type == "podcast_episode"
    assert choose_namespace(parent.document_type) == choose_namespace("youtube_video")
