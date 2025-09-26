import os
from typing import List, Dict, Any
from pinecone import Pinecone, ServerlessSpec

def get_index(index_name: str, dimension: int):
    pc = Pinecone(api_key=os.environ.get("PINECONE_API_KEY"))
    names = [x["name"] for x in pc.list_indexes().get("indexes", [])]
    if index_name not in names:
        pc.create_index(
            name=index_name,
            dimension=dimension,
            metric="cosine",
            spec=ServerlessSpec(cloud="aws", region="us-east-1"),
        )
    return pc.Index(index_name)

def upsert_vectors(index, namespace: str, vectors: List[Dict[str, Any]], batch_size: int = 100):
    from .batching import chunked
    for batch in chunked(vectors, batch_size):
        index.upsert(vectors=batch, namespace=namespace)
