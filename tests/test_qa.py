"""Golden-set integrity: the committed QA set must validate against the
committed corpus. If someone edits either file inconsistently, this fails."""

from pathlib import Path

from civil_code_rag.corpus import load_corpus
from civil_code_rag.qa import load_qa, validate_qa

ROOT = Path(__file__).resolve().parent.parent


def test_committed_qa_validates_against_committed_corpus():
    corpus = load_corpus(ROOT / "data" / "corpus" / "civil_code_2020.json")
    assert len(corpus) == 1260
    items = load_qa(ROOT / "data" / "golden" / "qa_set.json")
    assert len(items) >= 50
    assert sum(1 for i in items if i.scope == "out") >= 5
    errors = validate_qa(items, corpus)
    assert errors == [], "\n".join(errors)


def test_qa_topics_cover_all_seven_books():
    corpus = load_corpus(ROOT / "data" / "corpus" / "civil_code_2020.json")
    items = load_qa(ROOT / "data" / "golden" / "qa_set.json")
    by_no = {a.number: a for a in corpus}
    parts = {by_no[n].part for i in items if i.scope == "in" for n in i.golden_articles}
    for book in ("总则", "物权", "合同", "人格权", "婚姻家庭", "继承", "侵权责任"):
        assert any(book in p for p in parts), f"no golden question covers {book}"
