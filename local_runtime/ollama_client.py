"""Shared, explicit Ollama configuration and embedding contract."""

import ipaddress
import math
import os
import socket
from urllib.parse import urlparse

import httpx

DEFAULT_URL = "http://172.19.120.10:11434"
DEFAULT_MODEL = "qwen3.5:35b"
DEFAULT_EMBED_MODEL = "bge-m3:latest"


def local_url(value: str) -> str:
    """Keep plant data on loopback or the private plant network."""
    parsed = urlparse(value)
    if (
        parsed.scheme not in {"http", "https"} or not parsed.hostname
        or parsed.username or parsed.password or parsed.query or parsed.fragment
        or parsed.path not in {"", "/"}
    ):
        raise ValueError("Ollama URL must be an HTTP(S) origin without credentials or a path.")
    addresses = socket.getaddrinfo(parsed.hostname, parsed.port or 80)
    if not addresses or any(
        not (ipaddress.ip_address(item[4][0]).is_private
             or ipaddress.ip_address(item[4][0]).is_loopback)
        or ipaddress.ip_address(item[4][0]).is_unspecified
        for item in addresses
    ):
        raise ValueError("Ollama must resolve exclusively to private or loopback addresses.")
    return value.rstrip("/")


def ollama_url() -> str:
    return local_url(os.environ.get("OLLAMA_URL", DEFAULT_URL))


def embedding_model() -> str:
    return os.environ.get("OLLAMA_EMBED_MODEL", DEFAULT_EMBED_MODEL)


def embed_texts(texts: list[str], timeout: float = 300) -> list[list[float]]:
    if not texts:
        raise ValueError("At least one text is required for embedding.")
    with httpx.Client(timeout=timeout, trust_env=False, follow_redirects=False) as client:
        response = client.post(
            f"{ollama_url()}/api/embed",
            json={"model": embedding_model(), "input": texts, "truncate": False},
        )
        response.raise_for_status()
        payload = response.json()
        if not isinstance(payload, dict):
            raise ValueError("Ollama embedding response must be a JSON object.")
        vectors = payload.get("embeddings")
    if not isinstance(vectors, list) or len(vectors) != len(texts):
        raise ValueError("Ollama returned an incorrect embedding count.")
    dimension = None
    for vector in vectors:
        if not isinstance(vector, list) or not vector:
            raise ValueError("Ollama returned an empty or invalid embedding.")
        if dimension is None:
            dimension = len(vector)
        if len(vector) != dimension or any(
            isinstance(value, bool) or not isinstance(value, (int, float))
            or not math.isfinite(value) for value in vector
        ):
            raise ValueError("Ollama returned inconsistent or non-finite embeddings.")
    return vectors


def validate_index(collection) -> None:
    """Reject model changes rather than silently mixing embedding spaces."""
    metadata = collection.metadata or {}
    indexed_model = metadata.get("embedding_model")
    if indexed_model is not None and indexed_model != embedding_model():
        raise ValueError(
            f"Index uses {indexed_model}, not {embedding_model()}. "
            "Run build_docs_db.py --rebuild before changing embedding models."
        )
    if metadata.get("query_prefix", "") != "":
        raise ValueError("Index query prefix differs; run build_docs_db.py --rebuild.")
