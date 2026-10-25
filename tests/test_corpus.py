from civil_code_rag.corpus import assert_sequence, parse

FIXTURE = """
8/23/2020 中华人民共和国民法典 _ 中国人大网
www.npc.gov.cn/whatever 1/2
目    录
第一编  总    则
        第一章  基本规定
第一编  总    则
        第一章  基本规定
第一条  为了保护民事主体的合法权益，调整民事关系，维护社会和
经济秩序，适应中国特色社会主义发展要求，弘扬社会主义核心价值观。
第二条  民法调整平等主体的自然人、法人和非法人组织之间的人身
关系和财产关系。
第二编  物  权
        第一分编  通  则
            第一章  一般规定
第二百零五条  本编调整因物的归属和利用产生的民事关系。
附  则
第一千二百六十条  本法自2021年1月1日起施行。
版权所有：《中国人大》杂志社
"""


def test_parse_fixture_structure():
    articles = parse(FIXTURE)
    assert [a.number for a in articles] == [1, 2, 205, 1260]

    a1 = articles[0]
    assert a1.label == "第一条"
    assert a1.part == "第一编 总则"
    assert a1.chapter == "第一章 基本规定"
    assert a1.sub_part is None
    assert a1.text.startswith("为了保护民事主体的合法权益")
    assert "经济秩序" in a1.text  # hard line-wrap was re-joined

    a205 = articles[2]
    assert a205.part == "第二编 物权"
    assert a205.sub_part == "第一分编 通则"
    assert a205.chapter == "第一章 一般规定"

    a1260 = articles[3]
    assert a1260.part == "附则"
    assert "2021年1月1日" in a1260.text
    assert "版权所有" not in a1260.text  # footer noise filtered


def test_assert_sequence_detects_gaps_and_dupes():
    a1 = parse("第一条  内容一。\n")[0]
    a2 = parse("第二条  内容二。\n")[0]
    a3 = parse("第三条  内容三。\n")[0]
    assert_sequence([a1, a2, a3])  # consecutive passes
    import pytest

    with pytest.raises(AssertionError):
        assert_sequence([a1, a3])  # gap
    with pytest.raises(AssertionError):
        assert_sequence([a1, a1])  # duplicate


def test_cross_reference_is_not_a_new_article():
    text = "第一条  依照本法第一百八十八条的规定，向人民法院请求保护民事权利。\n"
    articles = parse(text)
    assert len(articles) == 1
    assert articles[0].number == 1
    assert "第一百八十八条" in articles[0].text
