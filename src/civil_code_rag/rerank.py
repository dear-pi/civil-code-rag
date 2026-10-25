"""Cross-encoder reranking with BAAI/bge-reranker-base (lazy import)."""

from __future__ import annotations

import os
from collections.abc import Sequence
from pathlib import Path

from .chunking import Chunk

DEFAULT_MODEL = "BAAI/bge-reranker-base"


def _resolve_model(model_name: str, cache_dir: str) -> str:
    """Prefer a fully-downloaded local copy (weights present) over the hub id.

    The hub download path can stall behind some middleboxes; the same files
    can be fetched with plain curl into ``<cache_dir>/<model-name>/``.
    """
    local = Path(cache_dir) / Path(model_name).name
    if (local / "model.safetensors").is_file():
        return str(local)
    return os.environ.get("CCR_RERANKER_PATH", model_name)


class CrossEncoderReranker:
    def __init__(
        self,
        model_name: str = DEFAULT_MODEL,
        device: str = "cpu",
        cache_dir: str = "data/models",
    ) -> None:
        self.model_name = model_name
        self._device = device
        self._cache_dir = cache_dir
        self._model = None

    def _load(self):
        if self._model is None:
            from sentence_transformers import CrossEncoder

            src = _resolve_model(self.model_name, self._cache_dir)
            self._model = CrossEncoder(src, device=self._device)
        return self._model

    def rerank(
        self, query: str, candidates: Sequence[Chunk], top_k: int
    ) -> list[tuple[Chunk, float]]:
        if not candidates:
            return []
        model = self._load()
        pairs = [(query, c.text) for c in candidates]
        scores = model.predict(pairs)
        ranked = sorted(
            zip(candidates, scores, strict=True), key=lambda x: -float(x[1])
        )
        return [(c, float(s)) for c, s in ranked[:top_k]]
