from src.ingest_v2.transcripts.normalize import normalize_to_sentences


def test_normalize_tolerates_none_segment_timestamps():
    raw = {
        "segments": [
            {
                "start": None,
                "end": None,
                "speaker": "S1",
                "text": "Hello world. Another sentence.",
                "words": [],
            }
        ]
    }

    out = normalize_to_sentences(raw)

    assert out
    assert out[0]["start_s"] == 0.0
    assert out[0]["end_s"] == 0.0
    assert out[0]["text"] == "Hello world."
