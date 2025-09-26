from datasketch import MinHash
import mmh3

def minhash_signature(text: str, n_perm: int = 64) -> MinHash:
    mh = MinHash(num_perm=n_perm)
    for token in text.split():
        mh.update(token.encode("utf-8"))
    return mh

def simhash_int(text: str, bits: int = 64) -> int:
    return mmh3.hash128(text, signed=False) & ((1 << bits) - 1)
