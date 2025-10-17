from src.ingest_v2.scripts.backfill_pumpfun_metadata import plan_updates


def test_plan_updates_applies_pumpfun_metadata():
    vectors = {
        "vid123": {
            "metadata": {
                "router_tags": ["pumpfun", "pumpfun_token:rasmr"],
                "pumpfun_coin": {"name": "RASMR", "symbol": "RASMR"},
                "pumpfun_clip": {"roomName": "RasMr", "clipId": "clip123", "startTime": "2025-10-12T00:00:00Z"},
            }
        }
    }

    plan = plan_updates(vectors)
    assert plan.vector_ids == ["vid123"]
    updated = plan.updates["vid123"]
    assert updated["channel_name"] == "RASMR (Pumpfun)"
    assert updated["channel_handle"] is None
    assert updated["pumpfun_coin_symbol"] == "RASMR"
    assert updated["pumpfun_clip_id"] == "clip123"
    assert not plan.skipped


def test_plan_updates_skips_non_pumpfun():
    vectors = {
        "vid999": {
            "metadata": {
                "router_tags": ["youtube"],
                "channel_name": "SomeChannel",
            }
        }
    }

    plan = plan_updates(vectors)
    assert plan.vector_ids == []
    assert plan.skipped == ["vid999"]
