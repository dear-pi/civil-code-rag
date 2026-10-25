"""Golden QA set loading and validation against the parsed corpus."""

from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path

from .corpus import Article


@dataclass(frozen=True)
class QAItem:
    id: str
    topic: str
    scope: str  # "in" | "out"
    question: str
    golden_articles: frozenset[int]
    answer_keywords: tuple[str, ...]


def load_qa(path: str | Path) -> list[QAItem]:
    data = json.loads(Path(path).read_text(encoding="utf-8"))
    items = []
    for q in data["questions"]:
        items.append(
            QAItem(
                id=q["id"],
                topic=q.get("topic", ""),
                scope=q["scope"],
                question=q["question"],
                golden_articles=frozenset(q.get("golden_articles", [])),
                answer_keywords=tuple(q.get("answer_keywords", [])),
            )
        )
    return items


def validate_qa(items: list[QAItem], corpus: list[Article]) -> list[str]:
    """Return a list of human-readable validation errors (empty = valid).

    Checks: unique ids, in-scope items have golden articles that exist in the
    corpus, and every answer keyword literally appears in the concatenation
    of the golden articles' texts. Out-of-scope items must have no golden.
    """
    errors: list[str] = []
    seen_ids: set[str] = set()
    by_no = {a.number: a for a in corpus}
    for it in items:
        if it.id in seen_ids:
            errors.append(f"{it.id}: duplicate id")
        seen_ids.add(it.id)
        if it.scope == "out":
            if it.golden_articles:
                errors.append(f"{it.id}: out-of-scope item must have no golden articles")
            continue
        if not it.golden_articles:
            errors.append(f"{it.id}: in-scope item without golden articles")
            continue
        missing = [n for n in it.golden_articles if n not in by_no]
        if missing:
            errors.append(f"{it.id}: golden article(s) not in corpus: {missing}")
            continue
        text = "".join(by_no[n].text for n in sorted(it.golden_articles))
        for kw in it.answer_keywords:
            if kw not in text:
                errors.append(f"{it.id}: keyword {kw!r} not found in golden article text")
    return errors
