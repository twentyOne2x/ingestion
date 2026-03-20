import logging
from typing import List, Dict, Any, Callable, Optional
import os, json, statistics, re

from ..configs.settings import settings_v2
from ..utils.pinecone_client import get_index, upsert_vectors, sanitize_metadata, trim_metadata_utf8
from ..utils.vector_store import (
    qdrant_collection_name,
    upsert_qdrant_vectors,
    vector_store_backend,
)
from ..utils.backoff import expo_backoff
from time import perf_counter
from math import ceil
from concurrent.futures import ThreadPoolExecutor, as_completed

from ..utils.batching import chunked
from ..utils.progress import progress
from src.utils.global_thread_guard import get_global_thread_limiter

# ────────────────────────────────────────────────────────────────────────────────
# Fast JSON size estimation helpers (for bytes-aware batching)
# ────────────────────────────────────────────────────────────────────────────────

try:
    import orjson as _fj  # type: ignore

    def _dumps_bytes(x) -> bytes:
        return _fj.dumps(x)
except Exception:
    def _dumps_bytes(x) -> bytes:
        # compact separators to better approximate wire size
        return json.dumps(x, ensure_ascii=False, separators=(",", ":")).encode("utf-8")


def _estimate_vector_bytes(v: Dict[str, Any]) -> int:
    """
    Estimate serialized byte size of a Pinecone vector payload:
      {"id": "...", "values": [...], "metadata": {...}}
    """
    try:
        return len(_dumps_bytes(v))
    except Exception:
        return 0


def _chunk_vectors_by_bytes(vectors: List[Dict[str, Any]], max_bytes: int, safety_overhead: int = 2048):
    """
    Yield batches (list, est_size_bytes) whose serialized JSON size is <= max_bytes.
    We keep a small safety overhead for HTTP/JSON framing.
    """
    batch: List[Dict[str, Any]] = []
    size = 2  # for surrounding "[]"
    for v in vectors:
        vsz = _estimate_vector_bytes(v)
        if batch and size + vsz + safety_overhead > max_bytes:
            yield batch, size
            batch = []
            size = 2
        batch.append(v)
        size += vsz + 1  # + comma
    if batch:
        yield batch, size


# ────────────────────────────────────────────────────────────────────────────────
# Fetch existing vectors' metadata (supports Pinecone v2/v3 SDK shapes)
# ────────────────────────────────────────────────────────────────────────────────

def _fetch_existing_meta(index, namespace: str, ids: List[str]) -> Dict[str, Dict]:
    """
    Safely fetch existing vectors' metadata without tripping 414 (Request-URI Too Large).
    Uses adaptive chunking: starts at PINECONE_FETCH_IDS_PER_REQ (default 200) and halves on 414/413.
    """
    out: Dict[str, Dict] = {}
    if not ids:
        return out

    # Start conservative; allow overrides
    max_per = int(os.getenv("PINECONE_FETCH_IDS_PER_REQ", "200"))
    max_per = max(1, max_per)

    i = 0
    logging.info("[fetch/meta] ids=%d start_batch=%d", len(ids), max_per)

    while i < len(ids):
        chunk = ids[i:i + max_per]

        attempt = 0
        while True:
            try:
                resp = index.fetch(ids=chunk, namespace=namespace) or {}

                # Support both SDKs:
                vectors = getattr(resp, "vectors", None)  # v3: dict[str, Vector]
                if vectors is None and isinstance(resp, dict):  # v2: dict style
                    vectors = resp.get("vectors")

                if vectors:
                    sample = next(iter(vectors.values()))
                    if isinstance(sample, dict):
                        for vid, v in vectors.items():
                            out[vid] = (v.get("metadata") or {})
                    else:
                        for vid, v in vectors.items():
                            out[vid] = (getattr(v, "metadata", {}) or {})

                # success → advance window
                i += len(chunk)
                break

            except Exception as e:
                status = getattr(e, "status", None)
                msg = str(e)
                # Too-large query (GET) or payload (if SDK switches to POST): shrink
                if status in (413, 414) or "414" in msg or "Request-URI Too Large" in msg or "Payload Too Large" in msg:
                    if max_per == 1 and len(chunk) == 1:
                        logging.error("[fetch/meta] cannot shrink further; re-raising (id=%s)", chunk[0])
                        raise
                    old = max_per
                    max_per = max(1, max_per // 2)
                    logging.warning("[fetch/meta] %s; reducing ids per request %d → %d and retrying",
                                    f"{status or ''}".strip(), old, max_per)
                    # retry same window with smaller chunk
                    chunk = ids[i:i + max_per]
                    continue

                # transient → backoff & retry
                attempt += 1
                logging.warning("[fetch/meta] fetch failed size=%d attempt=%d err=%s", len(chunk), attempt, e)
                expo_backoff(attempt)

    return out


# ────────────────────────────────────────────────────────────────────────────────
# Embedding provider wrapper with logging
# ────────────────────────────────────────────────────────────────────────────────

def _embedder() -> Callable[[List[str]], List[List[float]]]:
    if settings_v2.EMBED_PROVIDER == "openai":
        from openai import OpenAI
        try:
            import tiktoken  # type: ignore
        except Exception:  # pragma: no cover
            tiktoken = None  # type: ignore

        client = OpenAI()
        model = settings_v2.EMBED_MODEL
        bs = settings_v2.EMBED_BATCH_SIZE
        conc = settings_v2.EMBED_CONCURRENCY
        max_attempts = int(os.getenv("OPENAI_EMBED_MAX_ATTEMPTS", "6"))
        max_tokens = int(os.getenv("OPENAI_EMBED_MAX_TOKENS", "8000"))
        logging.info(
            "[embed/openai] model=%s batch_size=%d concurrency=%d max_attempts=%d max_tokens=%d",
            model,
            bs,
            conc,
            max_attempts,
            max_tokens,
        )

        def _status_code(exc: Exception) -> Optional[int]:
            for attr in ("status_code", "status"):
                v = getattr(exc, attr, None)
                if isinstance(v, int):
                    return v
                if isinstance(v, str) and v.isdigit():
                    try:
                        return int(v)
                    except Exception:
                        pass
            m = re.search(r"Error code:\\s*(\\d{3})", str(exc))
            if m:
                try:
                    return int(m.group(1))
                except Exception:
                    return None
            return None

        def _is_context_length_error(exc: Exception) -> bool:
            s = (str(exc) or "").lower()
            return "maximum context length" in s and "tokens" in s and "requested" in s

        def _is_insufficient_quota(exc: Exception) -> bool:
            s = (str(exc) or "").lower()
            return "insufficient_quota" in s or "insufficient quota" in s

        # Best-effort tokenizer for the embedding model. If this fails, we still rely on
        # metadata trimming to keep inputs reasonably sized.
        _enc = None
        if tiktoken is not None and max_tokens > 0:
            try:
                _enc = tiktoken.encoding_for_model(model)
            except Exception:
                try:
                    _enc = tiktoken.get_encoding("cl100k_base")
                except Exception:
                    _enc = None

        def _trim_to_max_tokens(s: str) -> str:
            if not s:
                return ""
            if _enc is None or max_tokens <= 0:
                return s
            try:
                toks = _enc.encode(s)
                if len(toks) <= max_tokens:
                    return s
                return _enc.decode(toks[:max_tokens])
            except Exception:
                # If tokenization fails, keep the original string; OpenAI will reject if too large.
                return s

        def _embed_chunk(chunk: List[str]) -> List[List[float]]:
            last_exc: Optional[Exception] = None
            for attempt in range(1, max(1, max_attempts) + 1):
                try:
                    t0 = perf_counter()
                    resp = client.embeddings.create(model=model, input=chunk)
                    dt = perf_counter() - t0
                    logging.info("[embed/openai] ok size=%d dt=%.2fs", len(chunk), dt)
                    return [d.embedding for d in resp.data]
                except Exception as e:
                    last_exc = e
                    status = _status_code(e)
                    lowered = (str(e) or "").lower()
                    non_retriable_4xx = status is not None and 400 <= status < 500 and status != 429
                    if non_retriable_4xx or _is_context_length_error(e) or _is_insufficient_quota(e):
                        # These won't succeed with retries; surface quickly.
                        logging.warning("[embed/openai] non-retriable status=%s size=%d err=%s", status, len(chunk), e)
                        raise
                    if attempt >= max(1, max_attempts):
                        logging.warning("[embed/openai] giving up attempts=%d size=%d err=%s", attempt, len(chunk), e)
                        raise
                    logging.warning("[embed/openai] retry %d/%d size=%d err=%s", attempt, max_attempts, len(chunk), e)
                    expo_backoff(attempt)

            # Should be unreachable, but keep mypy happy.
            raise RuntimeError(f"OpenAI embedding failed: {last_exc}") from last_exc

        limiter = get_global_thread_limiter()

        def embed_texts(texts: List[str]) -> List[List[float]]:
            # Normalize and token-trim before batching so a single oversized input doesn't
            # fail the whole request with a 400 context-length error.
            normed = [_trim_to_max_tokens(t if isinstance(t, str) else str(t or "")) for t in (texts or [])]
            chunks = [normed[i:i + bs] for i in range(0, len(normed), bs)]
            out_slots: List[Optional[List[List[float]]]] = [None] * len(chunks)
            worker_count = max(1, conc)
            with limiter.claim(worker_count, label="embed-openai"):
                with ThreadPoolExecutor(max_workers=worker_count) as ex:
                    futs = {ex.submit(_embed_chunk, ch): idx for idx, ch in enumerate(chunks)}
                    for fut in as_completed(futs):
                        idx = futs[fut]
                        out_slots[idx] = fut.result()
            vecs: List[List[float]] = []
            for slot in out_slots:
                vecs.extend(slot or [])
            return vecs

        # attach helpers for stats
        embed_texts._batch_size = bs            # type: ignore[attr-defined]
        embed_texts._concurrency = conc         # type: ignore[attr-defined]
        return embed_texts

    # sentence-transformers path
    from sentence_transformers import SentenceTransformer
    model = SentenceTransformer(settings_v2.EMBED_MODEL)

    def embed_texts(texts: List[str]) -> List[List[float]]:
        # normalize to unit vectors for cosine (optional but consistent)
        return model.encode(texts, batch_size=64, normalize_embeddings=True).tolist()

    return embed_texts


# ────────────────────────────────────────────────────────────────────────────────
# Namespace + metadata prep
# ────────────────────────────────────────────────────────────────────────────────

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


# ────────────────────────────────────────────────────────────────────────────────
# Parallel, retrying Pinecone upserts (drop-in)
# Knobs:
#   PINECONE_UPSERT_CONCURRENCY: default 3
#   PINECONE_UPSERT_RETRIES:     default 5
# ────────────────────────────────────────────────────────────────────────────────

def _retry_upsert_once(index, namespace: str, vectors: List[Dict[str, Any]]):
    # Force single HTTP call inside upsert_vectors by setting batch_size=len(vectors)
    upsert_vectors(index=index, namespace=namespace, vectors=vectors, batch_size=len(vectors))

def _retry_upsert(index, namespace: str, vectors: List[Dict[str, Any]], max_retries: int):
    attempt = 0
    while True:
        try:
            _retry_upsert_once(index, namespace, vectors)
            return
        except Exception as e:
            attempt += 1
            if attempt > max_retries:
                logging.exception("[upsert/send] giving up size=%d err=%s", len(vectors), e)
                raise
            logging.warning("[upsert/send] retry=%d size=%d err=%s", attempt, len(vectors), e)
            expo_backoff(attempt)

def _send_upserts_parallel(
    index,
    namespace: str,
    batches: List[tuple],
    pc_bs: int,
    parent_id: str,
) -> int:
    max_workers = int(os.getenv("PINECONE_UPSERT_CONCURRENCY", "3"))
    max_retries = int(os.getenv("PINECONE_UPSERT_RETRIES", "5"))

    futs = {}
    sent = 0
    limiter = get_global_thread_limiter()
    with limiter.claim(max_workers, label="pinecone-upsert"):
        with ThreadPoolExecutor(max_workers=max_workers) as ex:
            for i, (vec_batch, _est_bytes) in enumerate(batches, 1):
                # Respect count-based batch size within each bytes-batch
                for j, k in enumerate(range(0, len(vec_batch), pc_bs), 1):
                    sub = vec_batch[k:k + pc_bs]
                    est_sub = sum(_estimate_vector_bytes(v) for v in sub) + 2 + 1024
                    logging.info("[upsert/send] parent=%s batch=%d.%d size=%d est_bytes=%d",
                                 parent_id, i, j, len(sub), est_sub)
                    fut = ex.submit(_retry_upsert, index, namespace, sub, max_retries)
                    futs[fut] = len(sub)

            for f in as_completed(futs):
                # propagate any failure
                f.result()
                # progress only after a successful upsert of that sub-batch
                progress.add_done(futs[f])
                logging.info("[progress] %s", progress.fmt())
                sent += 1

    return sent

# ────────────────────────────────────────────────────────────────────────────────
# Main upsert
# ────────────────────────────────────────────────────────────────────────────────

def _upsert_children_qdrant(children: List[Dict[str, Any]]) -> Dict[str, float]:
    if not children:
        return {"t_embed": 0.0, "t_upsert": 0.0, "embed_reqs": 0, "pinecone_batches": 0}

    ns = choose_namespace(children[0]["document_type"])
    collection = qdrant_collection_name(ns)

    embed = _embedder()
    # Embed the exact text we store (sanitized + bytes-trimmed), not the raw child payload.
    # This prevents OpenAI 400 context-length errors and keeps vector content consistent.
    metas = [_prep_metadata_for_upsert(c) for c in children]
    texts = [str((md or {}).get("text") or "") for md in metas]
    t0 = perf_counter()
    embs = embed(texts)
    t1 = perf_counter()

    if len(embs) != len(children):
        logging.error("[upsert/qdrant] embedding/count mismatch: %d != %d; dropping batch", len(embs), len(children))
        return {"t_embed": t1 - t0, "t_upsert": 0.0, "embed_reqs": 0, "pinecone_batches": 0}

    vectors: List[Dict[str, Any]] = []
    for c, vec, md in zip(children, embs, metas):
        vectors.append({"id": c["segment_id"], "values": vec, "metadata": md})

    if not vectors:
        return {"t_embed": t1 - t0, "t_upsert": 0.0, "embed_reqs": 0, "pinecone_batches": 0}

    batch_size = int(os.getenv("QDRANT_UPSERT_BATCH", str(settings_v2.PINECONE_UPSERT_BATCH)))
    t2 = perf_counter()
    sent_batches = upsert_qdrant_vectors(
        collection_name=collection,
        vectors=vectors,
        dimension=settings_v2.EMBED_DIM,
        batch_size=batch_size,
    )
    t3 = perf_counter()

    embed_bs = getattr(embed, "_batch_size", len(texts) or 1)
    embed_reqs = ceil(len(texts) / max(1, int(embed_bs)))
    logging.info("[upsert/qdrant] namespace=%s collection=%s vectors=%d batches=%d", ns, collection, len(vectors), sent_batches)
    return {
        "t_embed": (t1 - t0),
        "t_upsert": (t3 - t2),
        "embed_reqs": embed_reqs,
        "pinecone_batches": sent_batches,
    }


def upsert_children(children: List[Dict[str, Any]]) -> Dict[str, float]:
    if not children:
        return {"t_embed": 0.0, "t_upsert": 0.0, "embed_reqs": 0, "pinecone_batches": 0}

    if vector_store_backend() == "qdrant":
        return _upsert_children_qdrant(children)

    # local import so this is truly drop-in
    from ..utils.progress import progress

    ns = choose_namespace(children[0]["document_type"])
    index = get_index(settings_v2.PINECONE_INDEX_NAME, settings_v2.EMBED_DIM)

    # --- DIFF: check what's already there (chunked to avoid 414s)
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
            to_embed.append(c)  # text/timewindow changed ⇒ re-embed
            continue

        # text same; maybe metadata changed — update without embedding
        md_new = _prep_metadata_for_upsert(c)
        if {k: v for k, v in md_new.items() if k != "text"} != {k: v for k, v in prev.items() if k != "text"}:
            to_update_meta.append({"id": seg_id, "metadata": md_new})

    # plan the work for this parent
    progress.add_planned(len(to_embed))
    logging.info("[progress] %s", progress.fmt())

    # --- fast path: only metadata updates (best-effort)
    for item in to_update_meta:
        try:
            index.update(id=item["id"], namespace=ns, set_metadata=item["metadata"])
        except Exception as e:
            logging.warning("[upsert/meta] update failed id=%s err=%s", item["id"], e)

    # --- embed + upsert only diffs
    if not to_embed:
        logging.info("[upsert/diff] new_or_changed=0 meta_only=%d skipped=%d",
                     len(to_update_meta), len(children) - len(to_update_meta))
        # parent finished (nothing to embed)
        progress.parent_done()
        logging.info("[progress] %s", progress.fmt())
        return {"t_embed": 0.0, "t_upsert": 0.0, "embed_reqs": 0, "pinecone_batches": 0}

    embed = _embedder()
    metas = [_prep_metadata_for_upsert(c) for c in to_embed]
    texts = [str((md or {}).get("text") or "") for md in metas]

    t0 = perf_counter()
    embs = embed(texts)
    t1 = perf_counter()

    if len(embs) != len(to_embed):
        logging.error("[upsert] embedding/count mismatch: %d != %d; dropping batch", len(embs), len(to_embed))
        # parent finished (failed to proceed)
        progress.parent_done()
        logging.info("[progress] %s", progress.fmt())
        return {"t_embed": t1 - t0, "t_upsert": 0.0, "embed_reqs": 0, "pinecone_batches": 0}

    # Build Pinecone vectors
    vectors: List[Dict[str, Any]] = []
    for c, vec, md in zip(to_embed, embs, metas):
        vectors.append({"id": c["segment_id"], "values": vec, "metadata": md})

    if not vectors:
        # parent finished (nothing to send)
        progress.parent_done()
        logging.info("[progress] %s", progress.fmt())
        return {"t_embed": (t1 - t0), "t_upsert": 0.0, "embed_reqs": 0, "pinecone_batches": 0}

    # ---- Debug stats before upsert
    dims = len(embs[0]) if embs else 0
    md_bytes = [len(_dumps_bytes(v["metadata"])) for v in vectors]
    text_lens = [len(str((md or {}).get("text") or "")) for md in metas]
    avg_text = (sum(text_lens) // max(1, len(text_lens))) if text_lens else 0
    p95_md = 0
    if md_bytes:
        if len(md_bytes) >= 20:
            try:
                p95_md = int(statistics.quantiles(md_bytes, n=20)[18])
            except Exception:
                p95_md = sorted(md_bytes)[int(0.95 * (len(md_bytes) - 1))]
        else:
            p95_md = max(md_bytes)
    logging.info(
        "[upsert/debug] vectors=%d dims=%d avg_text=%d max_text=%d avg_md_bytes=%d p95_md_bytes=%d max_md_bytes=%d",
        len(vectors), dims, avg_text, (max(text_lens) if text_lens else 0),
        int(statistics.mean(md_bytes)) if md_bytes else 0,
        p95_md,
        (max(md_bytes) if md_bytes else 0),
    )

    # ---- Bytes-aware batching to avoid Pinecone 2MB cap (leave headroom)
    pc_bs = settings_v2.PINECONE_UPSERT_BATCH
    max_req = int(os.getenv("PINECONE_MAX_REQ_BYTES", "1800000"))  # under 2MB
    parent_id = (to_embed[0].get("parent_id") if to_embed else children[0].get("parent_id")) or "unknown"

    batches = list(_chunk_vectors_by_bytes(vectors, max_req))
    logging.info(
        "[upsert/batching] parent=%s total_vectors=%d bytes_batches=%d (pc_bs=%d max_req=%d)",
        parent_id, len(vectors), len(batches), pc_bs, max_req
    )

    # ---- Parallel, retried upserts (this should call progress.add_done per sub-batch)
    t2 = perf_counter()
    sent_batches = _send_upserts_parallel(
        index=index,
        namespace=ns,
        batches=batches,
        pc_bs=pc_bs,
        parent_id=parent_id,
    )
    t3 = perf_counter()

    # parent finished (all sub-batches accounted for)
    progress.parent_done()
    logging.info("[progress] %s", progress.fmt())

    embed_bs = getattr(embed, "_batch_size", 128)
    embed_reqs = ceil(len(texts) / max(1, int(embed_bs)))
    pinecone_batches = sent_batches

    logging.info(
        "[upsert/diff] new_or_changed=%d meta_only=%d skipped=%d",
        len(to_embed), len(to_update_meta), len(children) - len(to_embed) - len(to_update_meta)
    )

    return {
        "t_embed": (t1 - t0),
        "t_upsert": (t3 - t2),
        "embed_reqs": embed_reqs,
        "pinecone_batches": pinecone_batches,
    }
