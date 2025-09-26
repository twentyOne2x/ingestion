from src.ingest_v2.entities.extract import extract_entities

def test_entities():
    t = "We talked to @anatoly about $SOL and @builder_xyz."
    ents = extract_entities(t)
    assert "$SOL" in ents and "@anatoly" in ents
