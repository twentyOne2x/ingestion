# Ingestion v2 (Parent/Child, time-based segments)

- Emits **Parent** (1 per asset) + **Child** segments (15–60s, stride=5s, padded) with timestamped clip URLs.
- Upserts children into Pinecone under namespaces:
  - `videos` (YouTube)
  - `streams` (Pump.fun)
- Reuses v1 Pinecone index but isolates via namespaces.

## Quickstart
```bash
export PINECONE_API_KEY=...
export PINECONE_INDEX_NAME=icmfyi
export OPENAI_API_KEY=...
python -m src.ingest_v2.pipelines.run_all --doc-type youtube_video --backfill-days 60
```
