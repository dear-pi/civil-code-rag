"""Parse the Civil Code full text into a structured list of articles.

Input: text extracted from the official print-PDF of the NPC web version
(one text line per PDF line, hard-wrapped mid-sentence, two header lines per
page). The parser is line-based because article labels are only trustworthy
at the start of a line: cross-references such as ``适用本法第一百八十八条``
contain the same ``第X条`` pattern but never open a line.

Hierarchy: 编 (part) > 分编 (sub-part, only in the property and contract
books) > 章 (chapter) > 节 (section) > 条 (article). Heading state is tracked
before the first article too, because the table of contents re-states the
hierarchy before ``第一条`` in the body.

Validation hook: ``assert_sequence`` — a correct parse must yield exactly
articles 1..N with no gaps and no duplicates.
"""

from __future__ import annotations

import json
import re
from dataclasses import asdict, dataclass
from pathlib import Path

from .cnnum import NUMERAL_RE, cn2int

CN = NUMERAL_RE  # [一二三四五六七八九十百千零]+

_SUB_PART_RE = re.compile(r"^(第[一二三四五六]分编)\s*(.*)$")
_PART_RE = re.compile(r"^(第[一二三四五六七]编)\s*(.*)$")
_APPENDIX_RE = re.compile(r"^附\s*则\s*$")
_CHAPTER_RE = re.compile(rf"^(第{CN}章)\s*(.*)$")
_SECTION_RE = re.compile(rf"^(第{CN}节)\s*(.*)$")
_ARTICLE_RE = re.compile(rf"^(第{CN}条)\s+(.*)$")
_ARTICLE_TAIL_RE = re.compile(rf"^(第{CN}条)\s*$")

# lines whose space-stripped form matches these are page furniture, not law text
_NOISE_EXACT = {
    "关于我们|网站地图",
    "版权所有：《中国人大》杂志社",
    "<<返回首页",
    "目录",
    "中华人民共和国民法典",
}
_NOISE_PREFIX = ("8/23/2020", "www.npc.gov.cn", "京ICP备", "FAOLEX", "2020年06月02日")
_NOISE_CONTAINS = ("_中国人大网", "全国人大全媒体", "返回首页")


def _is_noise(line: str) -> bool:
    stripped = line.strip()
    compact = re.sub(r"\s+", "", stripped)
    if not compact:
        return True
    if compact in _NOISE_EXACT:
        return True
    if stripped.startswith(_NOISE_PREFIX):
        return True
    return any(pat in compact for pat in _NOISE_CONTAINS)


@dataclass
class Article:
    """One legal article (条文) with its position in the code hierarchy."""

    number: int  # 1077
    label: str  # 第一千零七十七条
    part: str  # 第五编 婚姻家庭
    sub_part: str | None  # 第一分编 通则 (only in books 2/3)
    chapter: str | None  # 第一章 一般规定
    section: str | None  # 第一节 ...
    text: str  # article body, label stripped

    @property
    def heading(self) -> str:
        bits = [self.part, self.sub_part, self.chapter, self.section]
        return " · ".join(b for b in bits if b)


def _squeeze(s: str) -> str:
    """Remove all whitespace — the PDF inserts cosmetic gaps inside headings."""
    return re.sub(r"\s+", "", s)


def parse(text: str) -> list[Article]:
    """Parse raw extracted text into a list of articles.

    Lines before the first article belong to the title page and table of
    contents; their text is dropped but hierarchy headings are still tracked,
    because the body re-states them right before ``第一条``.
    """
    articles: list[Article] = []
    part = sub_part = chapter = section = ""
    label: str | None = None
    body: list[str] = []
    started = False

    def flush() -> None:
        nonlocal label, body
        if label is None:
            return
        articles.append(
            Article(
                number=cn2int(label[1:-1]),
                label=label,
                part=part,
                sub_part=sub_part or None,
                chapter=chapter or None,
                section=section or None,
                text=re.sub(r"\s+", "", "".join(body)),
            )
        )
        label, body = None, []

    for raw_line in text.splitlines():
        if _is_noise(raw_line):
            continue
        line = raw_line.strip()
        if not line:
            continue

        m = _ARTICLE_RE.match(line) or _ARTICLE_TAIL_RE.match(line)
        if m:
            flush()
            started = True
            label = m.group(1)
            if m.group(2):
                body.append(m.group(2))
            continue

        if _APPENDIX_RE.match(line):
            flush()
            part, sub_part, chapter, section = "附则", "", "", ""
            continue
        sp = _SUB_PART_RE.match(line)
        if sp and len(line) < 40:
            flush()
            sub_part = _squeeze(sp.group(1)) + " " + _squeeze(sp.group(2))
            chapter = section = ""
            continue
        p = _PART_RE.match(line)
        if p and len(line) < 40:
            flush()
            part = _squeeze(p.group(1)) + " " + _squeeze(p.group(2))
            sub_part = chapter = section = ""
            continue
        c = _CHAPTER_RE.match(line)
        if c and len(line) < 40:
            flush()
            chapter = _squeeze(c.group(1)) + " " + _squeeze(c.group(2))
            section = ""
            continue
        s = _SECTION_RE.match(line)
        if s and len(line) < 40:
            flush()
            section = _squeeze(s.group(1)) + " " + _squeeze(s.group(2))
            continue

        if started:
            body.append(line)

    flush()
    return articles


def assert_sequence(articles: list[Article]) -> None:
    """A correct parse is exactly 1..N with no gaps or duplicates."""
    numbers = [a.number for a in articles]
    expected = list(range(1, len(numbers) + 1))
    first_bad = next(
        (i for i, (a, b) in enumerate(zip(numbers, expected, strict=True)) if a != b), "?"
    )
    assert numbers == expected, (
        f"article sequence broken: got {len(numbers)} articles, "
        f"first mismatch at index {first_bad}"
    )


def load_corpus(path: str | Path) -> list[Article]:
    data = json.loads(Path(path).read_text(encoding="utf-8"))
    return [Article(**item) for item in data]


def save_corpus(articles: list[Article], path: str | Path) -> None:
    Path(path).write_text(
        json.dumps([asdict(a) for a in articles], ensure_ascii=False, indent=1),
        encoding="utf-8",
    )
