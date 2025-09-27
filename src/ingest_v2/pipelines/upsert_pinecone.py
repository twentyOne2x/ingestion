import logging
from typing import List, Dict, Any, Callable

from ..configs.settings import settings_v2
from ..utils.pinecone_client import get_index, upsert_vectors, sanitize_metadata, trim_metadata_utf8
from ..utils.backoff import expo_backoff


def _embedder() -> Callable[[List[str]], List[List[float]]]:
    """
    Return a function that embeds a list of texts -> list of vectors.
    Supports OpenAI or sentence-transformers based on settings.
    """
    if settings_v2.EMBED_PROVIDER == "openai":
        from openai import OpenAI
        client = OpenAI()
        model = settings_v2.EMBED_MODEL

        def embed_texts(texts: List[str]) -> List[List[float]]:
            vecs: List[List[float]] = []
            # conservative batch size for OpenAI
            bs = 128
            for i in range(0, len(texts), bs):
                chunk = texts[i : i + bs]
                attempt = 0
                while True:
                    try:
                        resp = client.embeddings.create(model=model, input=chunk)
                        vecs.extend([d.embedding for d in resp.data])
                        break
                    except Exception as e:
                        attempt += 1
                        logging.warning(f"[embed/openai] retry {attempt} size={len(chunk)} err={e}")
                        expo_backoff(attempt)
            return vecs

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


def upsert_children(children: List[Dict[str, Any]]) -> None:
    """
    - Embeds child texts
    - Upserts to Pinecone with namespace routing
    - Retries on transient errors
    """
    if not children:
        return

    # All children are from same document_type within one parent batch
    ns = choose_namespace(children[0]["document_type"])
    index = get_index(settings_v2.PINECONE_INDEX_NAME, settings_v2.EMBED_DIM)

    # Build embeddings with retries inside
    embed = _embedder()
    texts = [c["text"] for c in children]

    attempt = 0
    while True:
        try:
            embs = embed(texts)
            break
        except Exception as e:
            attempt += 1
            logging.warning(f"[embed] retry {attempt} err={e}")
            expo_backoff(attempt)

    if len(embs) != len(children):
        logging.error(f"[upsert] embedding/count mismatch: {len(embs)} != {len(children)}; dropping batch")
        return

    vectors = [
        {
            "id": c["segment_id"],
            "values": vec,
            "metadata": _prep_metadata_for_upsert(c),
        }
        for c, vec in zip(children, embs)
    ]

    # Delegate batching + retries + final sanitize to helper
    upsert_vectors(index=index, namespace=ns, vectors=vectors, batch_size=100)
    logging.info(f"[upsert] upserted {len(vectors)} vectors into namespace={ns}")
