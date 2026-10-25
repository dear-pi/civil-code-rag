"""Text embedders.

``LocalBgeEmbedder`` wraps BAAI/bge-small-zh-v1.5 via sentence-transformers
(lazy import: heavy deps only load when actually used). Following the model
card, retrieval queries get the official Chinese query instruction prefix
while documents are embedded as-is.

``CharNgramEmbedder`` is a dependency-free test double: hashed character
2-gram TF vectors, L2-normalized. Similar texts get similar vectors, which
is enough to exercise the retrieval stack in unit tests.
"""

from __future__ import annotations

from collections import Counter
from collections.abc import Sequence

import numpy as np

QUERY_PREFIX = "为这个句子生成表示以用于检索相关文章："


class LocalBgeEmbedder:
    def __init__(
        self,
        model_name: str = "BAAI/bge-small-zh-v1.5",
        device: str = "cpu",
        cache_dir: str = "data/models",
    ) -> None:
        self.model_name = model_name
        self.dim = 512
        self._device = device
        self._cache_dir = cache_dir
        self._model = None

    def _load(self):
        if self._model is None:
            from sentence_transformers import SentenceTransformer

            self._model = SentenceTransformer(
                self.model_name, device=self._device, cache_folder=self._cache_dir
            )
        return self._model

    def encode_documents(self, texts: Sequence[str]) -> np.ndarray:
        vecs = self._load().encode(list(texts), normalize_embeddings=True)
        return np.asarray(vecs, dtype=np.float32)

    def encode_query(self, text: str) -> np.ndarray:
        vec = self._load().encode([QUERY_PREFIX + text], normalize_embeddings=True)
        return np.asarray(vec, dtype=np.float32)[0]

    # uniform interface for chunk embedding
    def encode(self, texts: Sequence[str]) -> np.ndarray:
        return self.encode_documents(texts)


class CharNgramEmbedder:
    """Hashed char-2-gram TF embedder for tests; dim fixed at construction."""

    def __init__(self, dim: int = 512) -> None:
        self.dim = dim

    def _vec(self, texts: Sequence[str]) -> np.ndarray:
        import zlib

        out = np.zeros((len(texts), self.dim), dtype=np.float32)
        for i, t in enumerate(texts):
            counts = Counter(t[j : j + 2] for j in range(max(0, len(t) - 1)))
            for gram, c in counts.items():
                out[i, zlib.crc32(gram.encode("utf-8")) % self.dim] += c
            n = np.linalg.norm(out[i])
            if n > 0:
                out[i] /= n
        return out

    def encode_documents(self, texts: Sequence[str]) -> np.ndarray:
        return self._vec(texts)

    def encode_query(self, text: str) -> np.ndarray:
        return self._vec([text])[0]

    def encode(self, texts: Sequence[str]) -> np.ndarray:
        return self._vec(texts)
