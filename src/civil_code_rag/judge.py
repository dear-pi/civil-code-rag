"""LLM-as-judge scoring of generated answers (requires an API key).

Two independent 1–5 scores per question:

- faithfulness: is the answer fully supported by the provided articles?
- relevance: does the answer address the question?

The judge sees only the question, the provided article texts and the answer
— never the golden answer — so the score reflects groundedness, not
answer-key leakage.
"""

from __future__ import annotations

import json
import os
import re
from collections.abc import Sequence

from .chunking import Chunk
from .generation import Generator

JUDGE_PROMPT = """你是评测员。根据【提供的条文】评判【回答】的两个维度，各打 1–5 分：

1. faithfulness（忠实度）：回答的每个论断是否都被提供的条文支持，有无编造。
2. relevance（相关性）：回答是否切题地回答了问题。

只输出 JSON，格式：{{"faithfulness": <1-5>, "relevance": <1-5>, "reason": "<不超过50字>"}}

【提供的条文】
{context}

【问题】{question}

【回答】{answer}"""


def judge_answer(
    generator: Generator, question: str, answer: str, chunks: Sequence[Chunk]
) -> dict:
    context = "\n\n".join(c.text for c in chunks)
    prompt = JUDGE_PROMPT.format(context=context, question=question, answer=answer)
    resp = generator._load().chat.completions.create(  # noqa: SLF001 — same client, judge role
        model=generator.model,
        messages=[{"role": "user", "content": prompt}],
        temperature=0.0,
    )
    raw = resp.choices[0].message.content or ""
    m = re.search(r"\{.*\}", raw, re.S)
    if not m:
        return {"faithfulness": None, "relevance": None, "reason": f"unparseable: {raw[:60]}"}
    try:
        data = json.loads(m.group(0))
        return {
            "faithfulness": int(data["faithfulness"]),
            "relevance": int(data["relevance"]),
            "reason": str(data.get("reason", ""))[:80],
        }
    except (KeyError, ValueError, json.JSONDecodeError) as e:
        return {"faithfulness": None, "relevance": None, "reason": f"parse error: {e}"}


def llm_eval_ready() -> bool:
    return bool(os.environ.get("CCR_LLM_API_KEY"))
