"""Tests for AnswerJudge.calibrate_against_humans (inter-rater kappa)."""

from __future__ import annotations

import math

import pytest

from finrag_eval.eval.judge import AnswerJudge, JudgeScore


def _score(correctness: float) -> JudgeScore:
    return JudgeScore(
        correctness=correctness,
        completeness=0.0,
        faithfulness=0.0,
        citation_support=0.0,
        abstention_correct=False,
        reasoning="",
    )


def test_perfect_agreement() -> None:
    human = [_score(0.0), _score(0.5), _score(1.0), _score(0.1), _score(0.9)]
    judge = [_score(0.2), _score(0.5), _score(0.8), _score(0.0), _score(1.0)]
    out = AnswerJudge().calibrate_against_humans(human, judge, dimensions=["correctness"])
    assert out["correctness_agreement"] == 1.0
    assert out["correctness_kappa"] == 1.0
    assert out["correctness_kappa_weighted"] == 1.0
    assert out["correctness_n"] == 5.0


def test_adjacent_disagreement_weighted_beats_unweighted() -> None:
    # All disagreements are one bin apart → weighted kappa > unweighted.
    human = [_score(0.5), _score(0.5), _score(0.5), _score(1.0), _score(0.0)]
    judge = [_score(1.0), _score(1.0), _score(0.5), _score(1.0), _score(0.0)]
    out = AnswerJudge().calibrate_against_humans(human, judge, dimensions=["correctness"])
    assert out["correctness_kappa_weighted"] > out["correctness_kappa"]


def test_no_variance_yields_nan() -> None:
    human = [_score(1.0), _score(0.9), _score(0.8)]
    judge = [_score(0.9), _score(1.0), _score(0.7)]
    out = AnswerJudge().calibrate_against_humans(human, judge, dimensions=["correctness"])
    assert out["correctness_agreement"] == 1.0
    assert math.isnan(out["correctness_kappa"])


def test_abstention_is_binary() -> None:
    def s(ab: bool) -> JudgeScore:
        return JudgeScore(
            correctness=0.0,
            completeness=0.0,
            faithfulness=0.0,
            citation_support=0.0,
            abstention_correct=ab,
            reasoning="",
        )

    human = [s(True), s(False), s(True), s(False)]
    judge = [s(True), s(False), s(True), s(False)]
    out = AnswerJudge().calibrate_against_humans(
        human, judge, dimensions=["abstention_correct"]
    )
    assert out["abstention_correct_agreement"] == 1.0
    assert out["abstention_correct_kappa"] == 1.0


def test_mismatched_lengths_raise() -> None:
    with pytest.raises(ValueError):
        AnswerJudge().calibrate_against_humans([_score(1.0)], [], dimensions=["correctness"])


def test_unknown_dimension_raises() -> None:
    with pytest.raises(ValueError):
        AnswerJudge().calibrate_against_humans(
            [_score(1.0)], [_score(1.0)], dimensions=["nonsense"]
        )
