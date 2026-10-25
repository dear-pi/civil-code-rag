import pytest

from civil_code_rag.cnnum import cn2int


@pytest.mark.parametrize(
    ("text", "expected"),
    [
        ("一", 1),
        ("八", 8),
        ("十", 10),
        ("十一", 11),
        ("十六", 16),
        ("二十", 20),
        ("三十四", 34),
        ("一百", 100),
        ("一百零一", 101),
        ("一百八十八", 188),
        ("二百六十", 260),
        ("五百九十七", 597),
        ("一千零七十七", 1077),
        ("一千二百六十", 1260),
    ],
)
def test_cn2int(text, expected):
    assert cn2int(text) == expected


@pytest.mark.parametrize("bad", ["", "条", "第", "X", "1百"])
def test_cn2int_rejects(bad):
    with pytest.raises(ValueError):
        cn2int(bad)
