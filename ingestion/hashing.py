"""
Deterministic ID generation via SHA-512, adopted from GraphRAG.

Why not UUIDs?
- UUIDs are random: re-ingesting the same PDF creates duplicate chunks.
- SHA-512 of content is deterministic: same text → same ID every time.
- This enables idempotent re-ingestion and incremental updates.
"""
from hashlib import sha512
from typing import Any


def gen_sha512_hash(data: dict[str, Any], keys: list[str]) -> str:
    """
    Generate a SHA-512 hash from specific fields of a dict.
    
    Follows GraphRAG's pattern exactly:
      gen_sha512_hash({"text": "hello world"}, ["text"])
      → SHA-512 of "hello world"
    
    Args:
        data: Dictionary containing the values to hash.
        keys: Which keys to include in the hash.
    
    Returns:
        Hex string of the SHA-512 hash.
    """
    combined = "".join(str(data[key]) for key in keys)
    return sha512(combined.encode("utf-8"), usedforsecurity=False).hexdigest()