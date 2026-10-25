"""Chunking strategies under comparison.

Three strategies spanning a structure-blind baseline to a structure-aware
one, plus a learned middle ground:

- ``fixed``: sliding window over the concatenated code, ignoring all
  structure — the naive "paste the PDF into a splitter" baseline.
- ``article``: one chunk per legal article, prefixed with its hierarchy
  heading — the natural citation unit of a legal code.
- ``semantic``: sentence embeddings; adjacent sentences stay together while
  their cosine similarity is high. Learns boundaries without using the
  explicit article markers.

Every chunk records which article numbers it overlaps, so retrieval metrics
can be computed at the article level (the golden answers are articles).
"""

from __future__ import annotations

from collections.abc import Callable, Iterator, Sequence
from dataclasses import dataclass

from .corpus import Article

EmbedFn = Callable[[Sequence[str]], "list"]  # returns (n, dim) array-like, normalized


@dataclass(frozen=True)
class Chunk:
    chunk_id: str
    strategy: str
    text: str
    article_numbers: frozenset[int]

    @property
    def citation(self) -> str:
        nums = sorted(self.article_numbers)
        if len(nums) == 1:
            return f"第{nums[0]}条"
        return f"第{nums[0]}–{nums[-1]}条（{len(nums)}条）"


def fixed_size(articles: Sequence[Article], size: int = 256, overlap: int = 32) -> list[Chunk]:
    """Structure-blind sliding window over the whole code."""
    full = "".join(f"{a.label}{a.text}" for a in articles)
    # char offset -> article index map
    bounds: list[tuple[int, int, int]] = []  # (start, end_exclusive, article_number)
    pos = 0
    for a in articles:
        n = len(a.label) + len(a.text)
        bounds.append((pos, pos + n, a.number))
        pos += n
    assert pos == len(full)

    chunks: list[Chunk] = []
    step = max(1, size - overlap)
    i = 0
    idx = 0
    while i < len(full):
        window = full[i : i + size]
        nums = frozenset(
            num for start, end, num in bounds if start < i + len(window) and end > i
        )
        chunks.append(Chunk(f"fixed-{idx:05d}", "fixed", window, nums))
        idx += 1
        if i + size >= len(full):
            break
        i += step
    return chunks


def by_article(articles: Sequence[Article], max_len: int = 512) -> list[Chunk]:
    """One chunk per article; over-long articles are split, keeping the label."""
    chunks: list[Chunk] = []
    for a in articles:
        base = f"{a.label}（{a.heading}）"
        text = f"{base}{a.text}"
        if len(text) <= max_len:
            chunks.append(Chunk(f"art-{a.number:04d}", "article", text, frozenset({a.number})))
            continue
        # split the body of a single over-long article on sentence boundaries
        body = a.text
        pieces: list[str] = []
        buf = base
        for sent in _iter_sentences(body):
            if len(buf) + len(sent) > max_len and buf != base:
                pieces.append(buf)
                buf = base
            buf += sent
        if buf != base:
            pieces.append(buf)
        for j, piece in enumerate(pieces):
            chunks.append(
                Chunk(f"art-{a.number:04d}-{j}", "article", piece, frozenset({a.number}))
            )
    return chunks


def _iter_sentences(text: str) -> Iterator[str]:
    import re

    for sent in re.split(r"(?<=[。！？；])", text):
        if sent:
            yield sent


def semantic(
    articles: Sequence[Article],
    embed: EmbedFn,
    threshold: float = 0.5,
    min_size: int = 96,
    max_size: int = 384,
) -> list[Chunk]:
    """Embedding-threshold segmentation.

    Sentences of the whole code are embedded once; a segment break is placed
    wherever adjacent-sentence cosine similarity drops below ``threshold``.
    Segments shorter than ``min_size`` are merged with the more similar
    neighbour; segments longer than ``max_size`` are split at their least
    similar internal boundary. Chunk text carries the heading of its first
    article so terminology stays retrievable.
    """
    import numpy as np

    if embed is None:
        raise ValueError("semantic chunking needs an embed function")
    encode = embed if callable(embed) else embed.encode  # accept fn or embedder

    sentences: list[tuple[int, str]] = []  # (article_number, sentence)
    for a in articles:
        for sent in _iter_sentences(f"{a.label}：{a.text}"):
            sentences.append((a.number, sent))

    vecs = np.asarray(encode([s for _, s in sentences]), dtype=np.float32)
    norms = np.linalg.norm(vecs, axis=1, keepdims=True)
    norms[norms == 0] = 1.0
    vecs = vecs / norms
    adj = np.sum(vecs[:-1] * vecs[1:], axis=1) if len(vecs) > 1 else np.array([])

    # segment starts: 0 plus every index where similarity to previous < threshold
    starts = [0]
    starts.extend(i + 1 for i, s in enumerate(adj) if s < threshold)
    starts.append(len(sentences))
    segments = [list(range(starts[i], starts[i + 1])) for i in range(len(starts) - 1)]
    segments = [seg for seg in segments if seg]

    def make_chunk(seg: list[int]) -> Chunk:
        nums = frozenset(sentences[i][0] for i in seg)
        first_no = sentences[seg[0]][0]
        art = articles_by_no[first_no]
        text = f"【{art.heading}】" + "".join(sentences[i][1] for i in seg)
        return Chunk(f"sem-{seg[0]:05d}", "semantic", text, nums)

    articles_by_no = {a.number: a for a in articles}

    # merge tiny segments into the previous neighbour while under min_size
    merged: list[list[int]] = []
    for seg in segments:
        if merged and (_size(merged[-1], sentences) + _size(seg, sentences)) < min_size:
            merged[-1].extend(seg)
        else:
            merged.append(list(seg))
    # enforce max_size by hard split
    final: list[list[int]] = []
    for seg in merged:
        acc = _size(seg, sentences)
        if acc <= max_size:
            final.append(seg)
            continue
        cur: list[int] = []
        for i in seg:
            cur.append(i)
            if _size(cur, sentences) >= max_size:
                final.append(cur)
                cur = []
        if cur:
            final.append(cur)
    return [make_chunk(seg) for seg in final]


def _size(seg: list[int], sentences: list[tuple[int, str]]) -> int:
    return sum(len(sentences[i][1]) for i in seg)


def build_chunks(
    strategy: str,
    articles: Sequence[Article],
    embed: EmbedFn | None = None,
    threshold: float = 0.5,
) -> list[Chunk]:
    if strategy == "fixed":
        return fixed_size(articles)
    if strategy == "article":
        return by_article(articles)
    if strategy == "semantic":
        if embed is None:
            raise ValueError("semantic strategy needs an embed function")
        return semantic(articles, embed, threshold=threshold)
    raise ValueError(f"unknown strategy: {strategy}")
