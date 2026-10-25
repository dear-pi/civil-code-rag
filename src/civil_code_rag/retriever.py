"""Vector retrievers: exact numpy cosine and Chroma persistent store.

Both implement the same minimal interface: ``add(chunks, vectors)`` /
``search(query_vec, k) -> [(chunk, score)]`` so evaluation code does not care
which backend is active.
"""

from __future__ import annotations

from collections.abc import Sequence
from pathlib import Path

import numpy as np

from .chunking import Chunk


class NumpyRetriever:
    """Exact cosine search over an in-memory matrix."""

    def __init__(self) -> None:
        self._matrix: np.ndarray | None = None
        self._chunks: list[Chunk] = []

    def add(self, chunks: Sequence[Chunk], vectors: np.ndarray) -> None:
        v = np.asarray(vectors, dtype=np.float32)
        if v.ndim != 2:
            raise ValueError("vectors must be 2-D")
        self._matrix = v if self._matrix is None else np.vstack([self._matrix, v])
        self._chunks.extend(chunks)

    def __len__(self) -> int:
        return len(self._chunks)

    def search(self, query_vec: np.ndarray, k: int) -> list[tuple[Chunk, float]]:
        if self._matrix is None or not len(self._chunks):
            return []
        q = np.asarray(query_vec, dtype=np.float32)
        q = q / (np.linalg.norm(q) or 1.0)
        sims = self._matrix @ q
        order = np.argsort(-sims)[:k]
        return [(self._chunks[i], float(sims[i])) for i in order]


class ChromaRetriever:
    """Persistent Chroma collection; documents store chunk text + metadata."""

    def __init__(self, path: str | Path, collection: str) -> None:
        import chromadb

        self._client = chromadb.PersistentClient(path=str(path))
        self._col = self._client.get_or_create_collection(
            name=collection, metadata={"hnsw:space": "cosine"}
        )
        self._chunks: dict[str, Chunk] = {}

    def add(self, chunks: Sequence[Chunk], vectors: np.ndarray) -> None:
        if not chunks:
            return
        import json

        self._col.add(
            ids=[c.chunk_id for c in chunks],
            documents=[c.text for c in chunks],
            metadatas=[
                {"strategy": c.strategy, "articles": json.dumps(sorted(c.article_numbers))}
                for c in chunks
            ],
            embeddings=[v.tolist() for v in np.asarray(vectors, dtype=np.float32)],
        )
        for ch in chunks:
            self._chunks[ch.chunk_id] = ch

    def __len__(self) -> int:
        return self._col.count()

    def search(self, query_vec: np.ndarray, k: int) -> list[tuple[Chunk, float]]:
        if self._col.count() == 0:
            return []
        res = self._col.query(
            query_embeddings=[np.asarray(query_vec, dtype=np.float32).tolist()],
            n_results=min(k, self._col.count()),
        )
        out: list[tuple[Chunk, float]] = []
        ids = res["ids"][0]
        dists = res["distances"][0] if res.get("distances") else [0.0] * len(ids)
        for cid, dist in zip(ids, dists, strict=True):
            chunk = self._chunks.get(cid)
            if chunk is not None:
                out.append((chunk, 1.0 - float(dist)))  # cosine distance -> similarity
        return out


def build_retriever(backend: str, path: str | Path | None = None, collection: str = "default"):
    if backend == "numpy":
        return NumpyRetriever()
    if backend == "chroma":
        if path is None:
            raise ValueError("chroma backend needs a persist path")
        return ChromaRetriever(path, collection)
    raise ValueError(f"unknown backend: {backend}")
