from src.ingest_v2.segmenter.segmenter import build_segments

def test_segmenter_basic():
    sentences = [
        {"start_s": 0.0, "end_s": 2.0, "text": "Hello world.", "speaker": "S1"},
        {"start_s": 2.0, "end_s": 8.0, "text": "This is a longer sentence that should help reach the minimum window.", "speaker": "S1"},
        {"start_s": 8.0, "end_s": 18.0, "text": "Ending now.", "speaker": "S1"},
    ]
    segs = build_segments(
        sentences,
        duration_s=120.0,
        parent_id="vid",
        document_type="youtube_video",
        clip_base_url="https://www.youtube.com/watch?v=vid"
    )
    assert segs
    for s in segs:
        assert 15 <= (s["end_s"] - s["start_s"]) <= 60
