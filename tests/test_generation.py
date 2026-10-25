import pytest  # noqa: F401  (kept for fixture parity across test modules)

from civil_code_rag.chunking import Chunk
from civil_code_rag.generation import (
    Answer,
    Generator,
    build_messages,
    extract_citations,
    should_refuse,
    unfaithful_citations,
)


def test_should_refuse_boundary():
    assert should_refuse(0.31, 0.32) is True
    assert should_refuse(0.32, 0.32) is False
    assert should_refuse(0.9, 0.32) is False


def test_extract_citations_handles_compound_numerals():
    text = "依据第一千零七十七条和第一百八十八条，以及第五条的规定。"
    assert extract_citations(text) == [1077, 188, 5]


def test_unfaithful_citations():
    assert unfaithful_citations([188, 1077, 12], [188, 1077, 209]) == [12]
    assert unfaithful_citations([188], [188]) == []


def test_build_messages_contains_all_chunks():
    chunks = [
        Chunk("a", "article", "第一条 内容一", frozenset({1})),
        Chunk("b", "article", "第二条 内容二", frozenset({2})),
    ]
    msgs = build_messages("问题", chunks)
    assert msgs[0]["role"] == "system"
    assert "第一条 内容一" in msgs[1]["content"]
    assert "第二条 内容二" in msgs[1]["content"]
    assert "问题" in msgs[1]["content"]


def test_generator_refuses_without_llm_call(monkeypatch):
    gen = Generator(api_key="fake")
    called = []

    def boom(*a, **kw):
        called.append(1)
        raise AssertionError("LLM must not be called when refusing")

    monkeypatch.setattr(gen, "_load", boom)
    chunks = [Chunk("a", "article", "某条文", frozenset({1}))]
    ans = gen.answer("问题", chunks, top_score=0.1, refusal_threshold=0.5)
    assert ans.refused is True
    assert called == []
    assert ans.refusal_reason and "threshold" in ans.refusal_reason


def test_generator_unavailable_without_key(monkeypatch):
    monkeypatch.delenv("CCR_LLM_API_KEY", raising=False)
    assert Generator().available is False


def test_answer_dataclass_defaults():
    a = Answer("q", refused=True)
    assert a.text is None
    assert a.cited_articles == []
    assert a.unfaithful_citations == []
