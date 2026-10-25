import pytest

from civil_code_rag.chunking import by_article, fixed_size, semantic
from civil_code_rag.corpus import Article
from civil_code_rag.embeddings import CharNgramEmbedder


def make_articles():
    body = "民事主体从事民事活动，应当遵循诚信原则，秉持诚实，恪守承诺。"
    return [
        Article(1, "第一条", "第一编总则", None, "第一章基本规定", None, body),
        Article(2, "第二条", "第一编总则", None, "第一章基本规定", None, body + "民事主体不得违反法律。"),  # noqa: E501
        Article(3, "第三条", "第一编总则", None, "第一章基本规定", None, "短条文。"),
    ]


def test_fixed_size_covers_every_article_and_char():
    arts = make_articles()
    chunks = fixed_size(arts, size=16, overlap=4)
    assert chunks, "no chunks produced"
    covered = set().union(*[c.article_numbers for c in chunks])
    assert covered == {1, 2, 3}
    # structure-blind: windows are all <= size
    assert all(len(c.text) <= 16 for c in chunks)
    assert all(c.strategy == "fixed" for c in chunks)


def test_by_article_one_chunk_per_article():
    arts = make_articles()
    chunks = by_article(arts, max_len=512)
    assert [c.article_numbers for c in chunks] == [frozenset({1}), frozenset({2}), frozenset({3})]
    assert "第一编总则" in chunks[0].text  # heading prefix included


def test_by_article_splits_overlong():
    long_body = "这是一个很长的条文。" * 30
    arts = [Article(9, "第九条", "第一编总则", None, None, None, long_body)]
    chunks = by_article(arts, max_len=64)
    assert len(chunks) > 1
    assert all(len(c.text) <= 64 + 40 for c in chunks)  # heading prefix allowed
    assert all(c.article_numbers == frozenset({9}) for c in chunks)


def test_semantic_covers_all_articles():
    arts = make_articles()
    emb = CharNgramEmbedder(128)
    chunks = semantic(arts, emb, threshold=0.5, min_size=4, max_size=400)
    covered = set().union(*[c.article_numbers for c in chunks])
    assert covered == {1, 2, 3}
    assert all(c.strategy == "semantic" for c in chunks)


def test_semantic_requires_embedder():
    with pytest.raises(ValueError):
        semantic(make_articles(), None)
