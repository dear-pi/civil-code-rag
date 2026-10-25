import pytest

from civil_code_rag.metrics import evaluate_runs, rank_of_first_hit


def test_rank_of_first_hit():
    ranked = [frozenset({10}), frozenset({20, 30}), frozenset({99})]
    assert rank_of_first_hit(ranked, {20}) == 2
    assert rank_of_first_hit(ranked, {99}) == 3
    assert rank_of_first_hit(ranked, {777}) is None


def test_evaluate_runs_hand_computed():
    runs = [
        ([frozenset({1}), frozenset({2}), frozenset({3})], {1}),  # hit rank 1
        ([frozenset({9}), frozenset({1}), frozenset({2})], {1}),  # hit rank 2
        ([frozenset({9}), frozenset({8})], {1}),  # miss
    ]
    m = evaluate_runs(runs, k=2)
    assert m["n"] == 3
    assert m["hit@k"] == pytest.approx(2 / 3)
    assert m["mrr@k"] == pytest.approx((1 + 0.5) / 3)
    assert m["recall@k"] == pytest.approx(2 / 3)


def test_evaluate_runs_multi_golden_recall():
    runs = [
        ([frozenset({1, 2}), frozenset({3})], {1, 2, 3, 4}),  # covers 3 of 4 golden
    ]
    m = evaluate_runs(runs, k=2)
    assert m["hit@k"] == 1.0
    assert m["recall@k"] == pytest.approx(3 / 4)


def test_empty_runs():
    assert evaluate_runs([], k=5)["n"] == 0
