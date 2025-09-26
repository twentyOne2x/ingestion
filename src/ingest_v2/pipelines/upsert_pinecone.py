import logging
from typing import List, Dict, Any
from ..configs.settings import settings_v2
from ..utils.pinecone_client import get_index, upsert_vectors
from ..utils.backoff import expo_backoff

def _embedder():
    if settings_v2.EMBED_PROVIDER == "openai":
        from openai import OpenAI
        client = OpenAI()
        model = settings_v2.EMBED_MODEL
        def embed_texts(texts: List[str]) -> List[List[float]]:
            vecs = []
            for i in range(0, len(texts), 128):
                chunk = texts[i:i+128]
                resp = client.embeddings.create(model=model, input=chunk)
                vecs.extend([d.embedding for d in resp.data])
            return vecs
        return embed_texts
    else:
        from sentence_transformers import SentenceTransformer
        model = SentenceTransformer(settings_v2.EMBED_MODEL)
        def embed_texts(texts: List[str]) -> List[List[float]]:
            return model.encode(texts, batch_size=64, normalize_embeddings=True).tolist()
        return embed_texts

def choose_namespace(document_type: str) -> str:
    return settings_v2.NAMESPACE_VIDEOS if document_type == "youtube_video" else settings_v2.NAMESPACE_STREAMS

def trim_metadata(meta: Dict[str, Any]) -> Dict[str, Any]:
    import json
    s = json.dumps(meta, ensure_ascii=False)
    if len(s.encode("utf-8")) <= settings_v2.MAX_METADATA_BYTES:
        return meta
    m = dict(meta)
    t = m.get("text", "")
    if t and len(t) > 1200:
        m["text"] = t[:1200] + "…"
    return m

def upsert_children(children: List[Dict[str, Any]]):
    if not children:
        return
    index = get_index(settings_v2.PINECONE_INDEX_NAME, settings_v2.EMBED_DIM)
    embed = _embedder()
    texts = [c["text"] for c in children]
    attempt = 0
    while True:
        try:
            embs = embed(texts)
            break
        except Exception as e:
            attempt += 1
            logging.warning(f"[embed] retry {attempt} due to {e}")
            expo_backoff(attempt)

    vectors = [{
        "id": c["segment_id"],
        "values": vec,
        "metadata": trim_metadata(c),
    } for c, vec in zip(children, embs)]

    ns = choose_namespace(children[0]["document_type"])
    upsert_vectors(index=index, namespace=ns, vectors=vectors, batch_size=100)
    logging.info(f"[upsert] upserted {len(vectors)} vectors into namespace={ns}")
