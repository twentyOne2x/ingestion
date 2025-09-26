from src.ingest_v2.utils.ids import segment_uuid

def test_deterministic_uuid():
    a = segment_uuid("vid123", 12.3456, 27.89)
    b = segment_uuid("vid123", 12.346, 27.8901)
    assert a == b
