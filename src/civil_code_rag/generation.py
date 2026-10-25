"""Citation-grounded answer generation with low-confidence refusal.

Flow: retrieval already produced ranked chunks; the generator (a) refuses
before calling any LLM when the best chunk score is below a threshold,
(b) prompts the model to answer only from the provided articles and cite
article labels, (c) post-checks that every citation in the answer really was
provided, flagging unfaithful citations.

The refusal gate and citation check are pure functions — fully tested without
an API key. The LLM call itself is optional (OpenAI-compatible endpoint).
"""

from __future__ import annotations

import os
import re
from collections.abc import Sequence
from dataclasses import dataclass, field

from .chunking import Chunk
from .cnnum import cn2int

CITATION_RE = re.compile(r"第([一二三四五六七八九十百千零]+)条")

SYSTEM_PROMPT = (
    "你是法律检索问答助手。只能依据用户提供的《中华人民共和国民法典》条文回答问题。"
    "回答时必须引用所依据的条文编号（如：第一千零七十七条）。"
    "如果提供的条文不足以回答问题，必须直接回答：所提供条文无法回答该问题，不得编造。"
)


@dataclass
class Answer:
    question: str
    refused: bool
    refusal_reason: str | None = None
    text: str | None = None
    cited_articles: list[int] = field(default_factory=list)
    provided_articles: list[int] = field(default_factory=list)
    unfaithful_citations: list[int] = field(default_factory=list)
    model: str | None = None


def should_refuse(top_score: float, threshold: float) -> bool:
    """Refuse when even the best retrieved chunk scores below ``threshold``."""
    return top_score < threshold


def extract_citations(text: str) -> list[int]:
    """Article numbers cited in an answer text, deduplicated, order kept."""
    seen: list[int] = []
    for m in CITATION_RE.finditer(text):
        try:
            n = cn2int(m.group(1))
        except ValueError:
            continue
        if n not in seen:
            seen.append(n)
    return seen


def unfaithful_citations(cited: Sequence[int], provided: Sequence[int]) -> list[int]:
    """Citations that do not correspond to any provided chunk's articles."""
    p = set(provided)
    return [c for c in cited if c not in p]


def build_messages(question: str, chunks: Sequence[Chunk]) -> list[dict]:
    context = "\n\n".join(f"{c.text}" for c in chunks)
    return [
        {"role": "system", "content": SYSTEM_PROMPT},
        {
            "role": "user",
            "content": f"已知条文：\n{context}\n\n问题：{question}",
        },
    ]


class Generator:
    """Thin OpenAI-compatible chat client. Configure via environment:

    - ``CCR_LLM_API_KEY``  (required for generation; absence raises)
    - ``CCR_LLM_BASE_URL`` (default: https://api.deepseek.com)
    - ``CCR_LLM_MODEL``    (default: deepseek-chat)
    """

    def __init__(self, api_key: str | None = None, base_url: str | None = None,
                 model: str | None = None) -> None:
        self.api_key = api_key or os.environ.get("CCR_LLM_API_KEY", "")
        self.base_url = base_url or os.environ.get("CCR_LLM_BASE_URL", "https://api.deepseek.com")
        self.model = model or os.environ.get("CCR_LLM_MODEL", "deepseek-chat")
        self._client = None

    @property
    def available(self) -> bool:
        return bool(self.api_key)

    def _load(self):
        if self._client is None:
            if not self.api_key:
                raise RuntimeError(
                    "no API key: set CCR_LLM_API_KEY (OpenAI-compatible, e.g. DeepSeek)"
                )
            from openai import OpenAI

            self._client = OpenAI(api_key=self.api_key, base_url=self.base_url)
        return self._client

    def answer(
        self, question: str, chunks: Sequence[Chunk], top_score: float,
        refusal_threshold: float,
    ) -> Answer:
        provided = sorted({n for c in chunks for n in c.article_numbers})
        if should_refuse(top_score, refusal_threshold):
            return Answer(
                question=question,
                refused=True,
                refusal_reason=f"top score {top_score:.3f} < threshold {refusal_threshold:.3f}",
                provided_articles=provided,
            )
        resp = self._load().chat.completions.create(
            model=self.model,
            messages=build_messages(question, chunks),
            temperature=0.0,
        )
        text = resp.choices[0].message.content or ""
        cited = extract_citations(text)
        return Answer(
            question=question,
            refused=False,
            text=text,
            cited_articles=cited,
            provided_articles=provided,
            unfaithful_citations=unfaithful_citations(cited, provided),
            model=self.model,
        )


__all__ = [
    "Answer",
    "Generator",
    "SYSTEM_PROMPT",
    "build_messages",
    "extract_citations",
    "should_refuse",
    "unfaithful_citations",
]
