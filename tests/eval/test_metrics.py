"""Tests for retrieval metrics. These should pass on day one and stay passing."""

from __future__ import annotations

import pytest

from finrag_eval.eval.metrics import (
    evidence_hit_rate,
    mean_reciprocal_rank,
    ndcg_at_k,
    precision_at_k,
    recall_at_k,
)


class TestRecallAtK:
    def test_perfect_recall(self) -> None:
        assert recall_at_k(["a", "b", "c"], {"a", "b", "c"}, k=3) == 1.0

    def test_zero_recall(self) -> None:
        assert recall_at_k(["x", "y", "z"], {"a", "b", "c"}, k=3) == 0.0

    def test_partial_recall(self) -> None:
        assert recall_at_k(["a", "x", "y"], {"a", "b", "c"}, k=3) == pytest.approx(1 / 3)

    def test_empty_gold_returns_zero(self) -> None:
        assert recall_at_k(["a"], set(), k=1) == 0.0

    def test_k_truncates_retrieved(self) -> None:
        # Only "a" is in top-2, "b" is at rank 3
        assert recall_at_k(["a", "x", "b"], {"a", "b"}, k=2) == 0.5


class TestPrecisionAtK:
    def test_perfect_precision(self) -> None:
        assert precision_at_k(["a", "b"], {"a", "b", "c"}, k=2) == 1.0

    def test_zero_precision(self) -> None:
        assert precision_at_k(["x", "y"], {"a", "b"}, k=2) == 0.0

    def test_partial_precision(self) -> None:
        assert precision_at_k(["a", "x"], {"a", "b"}, k=2) == 0.5

    def test_k_zero_returns_zero(self) -> None:
        assert precision_at_k(["a"], {"a"}, k=0) == 0.0


class TestMRR:
    def test_first_rank_gives_one(self) -> None:
        assert mean_reciprocal_rank(["a", "x"], {"a"}) == 1.0

    def test_second_rank_gives_half(self) -> None:
        assert mean_reciprocal_rank(["x", "a"], {"a"}) == 0.5

    def test_no_gold_gives_zero(self) -> None:
        assert mean_reciprocal_rank(["x", "y"], {"a"}) == 0.0

    def test_fourth_rank(self) -> None:
        assert mean_reciprocal_rank(["w", "x", "y", "a"], {"a"}) == pytest.approx(0.25)


class TestNDCGAtK:
    def test_perfect_order(self) -> None:
        assert ndcg_at_k(["a", "b"], {"a", "b"}, k=2) == pytest.approx(1.0)

    def test_no_relevant_in_top_k(self) -> None:
        assert ndcg_at_k(["x", "y"], {"a"}, k=2) == 0.0

    def test_inverted_order_is_lower_than_perfect(self) -> None:
        gold = {"a", "b"}
        perfect = ndcg_at_k(["a", "b", "x"], gold, k=3)
        suboptimal = ndcg_at_k(["x", "a", "b"], gold, k=3)
        assert suboptimal < perfect

    def test_single_hit_at_rank_1(self) -> None:
        # DCG = 1/log2(2) = 1, IDCG = 1/log2(2) = 1, nDCG = 1
        assert ndcg_at_k(["a"], {"a"}, k=1) == pytest.approx(1.0)

    def test_returns_zero_when_no_gold(self) -> None:
        assert ndcg_at_k(["a", "b"], set(), k=2) == 0.0


class TestEvidenceHitRate:
    def test_hit_returns_one(self) -> None:
        assert evidence_hit_rate(["a", "x"], {"a"}, k=2) == 1.0

    def test_miss_returns_zero(self) -> None:
        assert evidence_hit_rate(["x", "y"], {"a"}, k=2) == 0.0

    def test_gold_outside_k_is_miss(self) -> None:
        assert evidence_hit_rate(["x", "y", "a"], {"a"}, k=2) == 0.0
