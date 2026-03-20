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
    ok, _reason = validate_child_runtime(child, duration_s=3600.0)
    assert ok


def test_child_validation_respects_override_min_chars():
    import os

    child = {
        "node_type": "child",
        "segment_id": "abc",
        "parent_id": "p",
        "document_type": "youtube_video",
        "text": "X" * 60,
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

    prev = os.environ.get("VALIDATE_MIN_TEXT_CHARS")
    os.environ["VALIDATE_MIN_TEXT_CHARS"] = "40"
    try:
        ok, _reason = validate_child_runtime(child, duration_s=3600.0)
        assert ok
    finally:
        if prev is None:
            os.environ.pop("VALIDATE_MIN_TEXT_CHARS", None)
        else:
            os.environ["VALIDATE_MIN_TEXT_CHARS"] = prev
