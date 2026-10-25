"""Retrieval metrics computed at the article level.

Golden answers are article numbers; a retrieved chunk "hits" when any of its
overlapping articles is golden. Keeping metrics article-level (not
chunk-level) makes them comparable across chunking strategies, whose chunk
granularities differ.
"""

from __future__ import annotations

from collections.abc import Sequence
from typing import TypeVar

T = TypeVar("T")  # element type of the golden set


def rank_of_first_hit(
    ranked_article_sets: Sequence[frozenset[int]], golden: set[int]
) -> int | None:
    """1-based rank of the first ranked chunk that overlaps a golden article."""
    for rank, arts in enumerate(ranked_article_sets, start=1):
        if arts & golden:
            return rank
    return None


def evaluate_runs(
    runs: Sequence[tuple[Sequence[frozenset[int]], set[int]]], k: int
) -> dict[str, float]:
    """Compute hit@k, MRR@k and recall@k over (ranked article sets, golden) pairs."""
    n = len(runs)
    if n == 0:
        return {"hit@k": 0.0, "mrr@k": 0.0, "recall@k": 0.0, "k": float(k), "n": 0.0}
    hits, rr, recalls = 0, 0.0, 0.0
    for ranked, golden in runs:
        top_k = list(ranked[:k])
        rank = rank_of_first_hit(top_k, golden)
        if rank is not None:
            hits += 1
            rr += 1.0 / rank
        covered: set[int] = set()
        for arts in top_k:
            covered |= arts
        recalls += len(covered & golden) / len(golden) if golden else 0.0
    return {
        "hit@k": hits / n,
        "mrr@k": rr / n,
        "recall@k": recalls / n,
        "k": float(k),
        "n": float(n),
    }
