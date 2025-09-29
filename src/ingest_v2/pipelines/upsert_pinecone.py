import logging
from typing import List, Dict, Any, Callable, Optional

from ..configs.settings import settings_v2
from ..utils.pinecone_client import get_index, upsert_vectors, sanitize_metadata, trim_metadata_utf8
from ..utils.backoff import expo_backoff
from time import perf_counter
from math import ceil
from concurrent.futures import ThreadPoolExecutor, as_completed

from ..utils.batching import chunked

def _fetch_existing_meta(index, namespace: str, ids: List[str]) -> Dict[str, Dict]:
    out: Dict[str, Dict] = {}
    for chunk in chunked(ids, 1000):
        resp = index.fetch(ids=chunk, namespace=namespace, include_values=False) or {}
        for vid, v in (resp.get("vectors") or {}).items():
            out[vid] = v.get("metadata") or {}
    return out


def _embedder() -> Callable[[List[str]], List[List[float]]]:
    if settings_v2.EMBED_PROVIDER == "openai":
        from openai import OpenAI
        client = OpenAI()
        model = settings_v2.EMBED_MODEL
        bs = settings_v2.EMBED_BATCH_SIZE
        conc = settings_v2.EMBED_CONCURRENCY

        def _embed_chunk(chunk: List[str]) -> List[List[float]]:
            attempt = 0
            while True:
                try:
                    resp = client.embeddings.create(model=model, input=chunk)
                    return [d.embedding for d in resp.data]
                except Exception as e:
                    attempt += 1
                    logging.warning(f"[embed/openai] retry {attempt} size={len(chunk)} err={e}")
                    expo_backoff(attempt)

        def embed_texts(texts: List[str]) -> List[List[float]]:
            chunks = [texts[i:i+bs] for i in range(0, len(texts), bs)]
            out_slots: List[Optional[List[List[float]]]] = [None]*len(chunks)
            with ThreadPoolExecutor(max_workers=max(1, conc)) as ex:
                futs = {ex.submit(_embed_chunk, ch): idx for idx, ch in enumerate(chunks)}
                for fut in as_completed(futs):
                    idx = futs[fut]
                    out_slots[idx] = fut.result()
            vecs: List[List[float]] = []
            for slot in out_slots:
                vecs.extend(slot or [])
            return vecs

        # attach helpers for stats
        embed_texts._batch_size = bs           # type: ignore[attr-defined]
        embed_texts._concurrency = conc        # type: ignore[attr-defined]
        return embed_texts

    # sentence-transformers path
    from sentence_transformers import SentenceTransformer
    model = SentenceTransformer(settings_v2.EMBED_MODEL)

    def embed_texts(texts: List[str]) -> List[List[float]]:
        # normalize to unit vectors for cosine (optional but consistent)
        return model.encode(texts, batch_size=64, normalize_embeddings=True).tolist()

    return embed_texts


def choose_namespace(document_type: str) -> str:
    if document_type == "youtube_video":
        return settings_v2.NAMESPACE_VIDEOS
    return settings_v2.NAMESPACE_STREAMS


def _prep_metadata_for_upsert(c: Dict[str, Any]) -> Dict[str, Any]:
    """
    Sanitize + trim metadata once (if caller bypasses upsert_vectors' internal sanitize).
    """
    md = sanitize_metadata(c)
    md = trim_metadata_utf8(md, settings_v2.MAX_METADATA_BYTES)
    return md


def upsert_children(children: List[Dict[str, Any]]) -> Dict[str, float]:
    if not children:
        return {"t_embed": 0.0, "t_upsert": 0.0, "embed_reqs": 0, "pinecone_batches": 0}

    ns = choose_namespace(children[0]["document_type"])
    index = get_index(settings_v2.PINECONE_INDEX_NAME, settings_v2.EMBED_DIM)

    # --- DIFF: check what's already there
    ids = [c["segment_id"] for c in children]
    existing = _fetch_existing_meta(index, ns, ids)

    to_embed: List[Dict[str, Any]] = []
    to_update_meta: List[Dict[str, Any]] = []

    for c in children:
        seg_id = c["segment_id"]
        prev = existing.get(seg_id)
        if not prev:
            to_embed.append(c)  # new id
            continue
        if str(prev.get("source_hash") or "") != str(c.get("source_hash") or ""):
            to_embed.append(c)  # text (or time window) changed ⇒ re-embed
            continue
        # text same; maybe metadata changed — update without embedding
        md_new = _prep_metadata_for_upsert(c)
        # cheap compare; if different, push to update
        if {k: v for k, v in md_new.items() if k != "text"} != {k: v for k, v in prev.items() if k != "text"}:
            to_update_meta.append({"id": seg_id, "metadata": md_new})

    # --- fast path: only metadata updates
    for item in to_update_meta:
        index.update(id=item["id"], namespace=ns, set_metadata=item["metadata"])

    # --- embed + upsert only diffs
    if not to_embed:
        return {"t_embed": 0.0, "t_upsert": 0.0, "embed_reqs": 0, "pinecone_batches": 0}

    embed = _embedder()
    texts = [c["text"] for c in to_embed]

    t0 = perf_counter()
    embs = embed(texts)
    t1 = perf_counter()

    if len(embs) != len(to_embed):
        logging.error(f"[upsert] embedding/count mismatch: {len(embs)} != {len(to_embed)}; dropping batch")
        return {"t_embed": t1 - t0, "t_upsert": 0.0, "embed_reqs": 0, "pinecone_batches": 0}

    vectors = []
    for c, vec in zip(to_embed, embs):
        md = _prep_metadata_for_upsert(c)
        vectors.append({"id": c["segment_id"], "values": vec, "metadata": md})

    pc_bs = settings_v2.PINECONE_UPSERT_BATCH
    t2 = perf_counter()
    upsert_vectors(index=index, namespace=ns, vectors=vectors, batch_size=pc_bs)
    t3 = perf_counter()

    embed_bs = getattr(embed, "_batch_size", 128)
    embed_reqs = ceil(len(texts) / max(1, int(embed_bs)))
    pinecone_batches = ceil(len(vectors) / max(1, int(pc_bs)))
    logging.info("[upsert/diff] new_or_changed=%d meta_only=%d skipped=%d",
                 len(to_embed), len(to_update_meta), len(children) - len(to_embed) - len(to_update_meta))
    return {"t_embed": (t1 - t0), "t_upsert": (t3 - t2), "embed_reqs": embed_reqs, "pinecone_batches": pinecone_batches}