"""Chinese-numeral parsing for Civil Code article/part/chapter labels.

The Civil Code numbers every legal article in Chinese numerals, e.g.
``第一千零七十七条`` -> 1077. Compound forms (``十一`` = 11,
``一百零一`` = 101, ``二百六十`` = 260) make naive digit-accumulation
wrong, so units are applied to their preceding digit and summed.
"""

CN_DIGIT = {
    "零": 0, "一": 1, "二": 2, "三": 3, "四": 4,
    "五": 5, "六": 6, "七": 7, "八": 8, "九": 9,
}
CN_UNIT = {"十": 10, "百": 100, "千": 1000}

NUMERAL_RE = "[一二三四五六七八九十百千零]+"
ARTICLE_RE = f"第{NUMERAL_RE}条"


def cn2int(text: str) -> int:
    """Parse a Chinese numeral between 1 and 9999 into an int.

    Raises ``ValueError`` on empty or non-numeral input.
    """
    if not text or any(ch not in CN_DIGIT and ch not in CN_UNIT for ch in text):
        raise ValueError(f"not a Chinese numeral: {text!r}")
    section = 0
    digit: int | None = None
    for ch in text:
        if ch in CN_DIGIT:
            digit = CN_DIGIT[ch]  # 零 sets 0; a following unit is not expected
        else:
            unit = CN_UNIT[ch]
            if digit is None:
                digit = 1  # bare 十 as in 十六 -> 16
            section += digit * unit
            digit = None
    return section + (digit or 0)
