from src.ingest_v2.segmenter.segmenter import build_segments

def test_segmenter_basic():
    sentences = [
        {"start_s": 0.0, "end_s": 2.0, "text": "Hello world.", "speaker": "S1"},
        {
            "start_s": 2.0,
            "end_s": 8.0,
            # Ensure we exceed the default MIN_TEXT_CHARS=160 so the segmenter emits at least one chunk.
            "text": (
                "This is a longer sentence that should help reach the minimum window. "
                "We add additional filler text so the combined segment passes the default minimum text length "
                "threshold used by the v2 segmenter during ingestion."
            ),
            "speaker": "S1",
        },
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


def test_segmenter_short_video_emits_segment_with_default_min_text_chars():
    # With the default MIN_TEXT_CHARS=160, short-form videos often have transcripts below
    # that threshold. We still want to index at least one segment so the parent doc exists.
    sentences = [
        {"start_s": 0.0, "end_s": 10.0, "text": "Short transcript, but still meaningful.", "speaker": "S1"},
        {"start_s": 10.0, "end_s": 25.0, "text": "Another sentence to exceed the minimum window.", "speaker": "S1"},
        {"start_s": 25.0, "end_s": 30.0, "text": "Done.", "speaker": "S1"},
    ]
    segs = build_segments(
        sentences,
        duration_s=30.0,
        parent_id="shortvid",
        document_type="youtube_video",
        clip_base_url="https://www.youtube.com/watch?v=shortvid"
    )
    assert segs, "expected at least one segment for short videos"


def test_segmenter_ultra_short_video_still_indexes_one_segment():
    # Some corpus items are single-digit seconds long. We still want a single chunk
    # so the clip is searchable (summary alone is often insufficient).
    sentences = [
        {
            "start_s": 0.08,
            "end_s": 9.92,
            "text": "If a transcript is non-trivial, even a <10s clip should still be indexed.",
            "speaker": "S1",
        },
    ]
    segs = build_segments(
        sentences,
        duration_s=9.92,
        parent_id="tinyvid",
        document_type="youtube_video",
        clip_base_url="https://www.youtube.com/watch?v=tinyvid",
    )
    assert segs, "expected at least one segment for ultra-short videos"
    assert segs[0]["end_s"] <= 9.92 + 1e-6
