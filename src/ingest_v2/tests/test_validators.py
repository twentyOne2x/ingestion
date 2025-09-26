from src.ingest_v2.validators.runtime import validate_child_runtime

def test_child_validation():
    child = {
        "node_type": "child",
        "segment_id": "abc",
        "parent_id": "p",
        "document_type": "youtube_video",
        "text": "X" * 100,
        "start_s": 10.0,
        "end_s": 30.0,
        "start_hms": "00:00:10.000",
        "end_hms": "00:00:30.000",
        "clip_url": "https://www.youtube.com/watch?v=dummy&t=10s",
        "speaker": "S1",
        "entities": [],
        "chapter": None,
        "language": "en",
        "confidence_asr": 0.9,
        "has_music": False,
        "flags": [],
        "rights": "public_reference_only",
        "ingest_version": 2,
        "source_hash": "deadbeef",
    }
    validate_child_runtime(child, duration_s=3600.0)
