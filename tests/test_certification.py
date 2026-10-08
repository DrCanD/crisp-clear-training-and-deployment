"""Compare exact acceptance to an independent binomial implementation."""
from fractions import Fraction
import math

import numpy as np
import pytest
from scipy.stats import binomtest

from reproduce.certification import (
    counts_top2, exact_threshold_table, fixed_predict, sequential_predict,
    threshold_table, two_sided_pvalue,
)


@pytest.mark.parametrize("denominator", [1000, 5000])
def test_thresholds_match_independent_two_sided_test(denominator):
    table = exact_threshold_table(denominator, 256)
    for total in range(1, 257):
        for top in range((total + 1) // 2, total + 1):
            expected = binomtest(top, total, 0.5, alternative="two-sided").pvalue <= 1 / denominator
            assert (top >= table[total]) == expected


def test_formal_integer_boundaries():
    fixed = exact_threshold_table(1000, 16)
    sequential = exact_threshold_table(5000, 16)
    assert 10 < fixed[10] and 11 >= fixed[11]
    assert 13 < sequential[13] and 14 >= sequential[14]
    assert 16 >= sequential[16] and 15 < sequential[16]
    assert 15 >= fixed[16]


def test_fractional_level_boundary_has_no_roundoff():
    assert two_sided_pvalue(15, 1) == Fraction(17, 32768)
    table = threshold_table(Fraction(17, 32768), 16)
    assert table[16] == 15
    assert threshold_table(Fraction(17, 32768) - Fraction(1, 10**12), 16)[16] == 16


def test_multiclass_ranking_and_ties():
    votes = np.array([[2] * 12 + [0] * 2 + [1] * 2,
                      [2] * 8 + [1] * 8])
    classes, top, second, counts = counts_top2(votes, 3)
    np.testing.assert_array_equal(classes, [2, 1])
    np.testing.assert_array_equal(top, [12, 8])
    np.testing.assert_array_equal(second, [2, 8])
    np.testing.assert_array_equal(counts.sum(axis=1), [16, 16])
    assert fixed_predict(votes, 3)[1] == -1


def test_sequential_first_acceptance_and_budget():
    votes = np.zeros((3, 256), dtype=np.int64)
    votes[1, 15] = 1  # 15-1 passes fixed alpha but not alpha/5 at first look.
    votes[2] = np.arange(256) % 2
    decisions, draws = sequential_predict(votes, 2)
    np.testing.assert_array_equal(decisions, [0, 0, -1])
    np.testing.assert_array_equal(draws, [16, 32, 256])
    assert fixed_predict(votes[1:2, :16], 2)[0] == 0


def test_small_multinomial_error_probability():
    # Exhaust all count outcomes under a nonuniform three-class distribution.
    # The wrong accepted class event is unconditional, including abstentions.
    draws, probability = 16, (0.55, 0.30, 0.15)
    wrong_probability = 0.0
    for a in range(draws + 1):
        for b in range(draws - a + 1):
            counts = (a, b, draws - a - b)
            votes = np.repeat(np.arange(3), counts)[None]
            decision = fixed_predict(votes, 3, alpha=0.001)[0]
            if decision > 0:
                multiplicity = math.factorial(draws) / math.prod(math.factorial(c) for c in counts)
                wrong_probability += multiplicity * math.prod(p**c for p, c in zip(probability, counts))
    assert wrong_probability < 0.001


@pytest.mark.parametrize("looks", [(), (16, 16), (32, 16), (0, 16), (300,)])
def test_invalid_looks_fail(looks):
    with pytest.raises(ValueError):
        sequential_predict(np.zeros((1, 256), dtype=int), 2, looks=looks)


def test_invalid_votes_fail():
    with pytest.raises(TypeError):
        fixed_predict(np.zeros((1, 16), dtype=float), 2)
    with pytest.raises(ValueError):
        fixed_predict(np.array([[2]]), 2)
