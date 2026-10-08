"""Two-sided PREDICT decisions for a fixed input under independent draws.

The accepted label estimates the most probable sampled class. The certificate
controls the unconditional probability of accepting a different class; it is
neither a true-label guarantee nor an adversarial-robustness certificate.
Sequential looks reuse cumulative draws and divide the total error budget.
Integer thresholds retain the acceptance arithmetic used by the FPGA export.
"""
from fractions import Fraction
import math

import numpy as np

DEFAULT_LOOKS = (16, 32, 64, 128, 256)


def _level(alpha):
    level = Fraction(str(alpha))
    if not 0 < level < 1:
        raise ValueError("alpha must lie strictly between zero and one")
    return level


def threshold_table(alpha, n_max):
    """Minimum top count for each top-two total at a two-sided level.

    Values above n_max indicate that acceptance is impossible. All comparisons
    use exact integers, including at boundaries of the requested error level.
    """
    level = _level(alpha)
    if not isinstance(n_max, (int, np.integer)) or n_max < 1:
        raise ValueError("n_max must be a positive integer")
    thresholds = np.full(n_max + 1, n_max + 1, dtype=np.int64)
    numerator, denominator = level.numerator, level.denominator
    for n in range(1, n_max + 1):
        tail = 0
        for top in range(n, -1, -1):
            tail += math.comb(n, top)
            if 2 * denominator * tail > numerator * (1 << n):
                break
            if 2 * top > n:
                thresholds[n] = top
    return thresholds


def exact_threshold_table(denom, n_max):
    """Original integer-table interface for alpha = 1 / denom."""
    if not isinstance(denom, (int, np.integer)) or denom < 2:
        raise ValueError("denom must be an integer of at least two")
    return threshold_table(Fraction(1, int(denom)), n_max)


def _votes(votes, n_classes):
    votes = np.asarray(votes)
    if not isinstance(n_classes, (int, np.integer)) or n_classes < 2:
        raise ValueError("n_classes must be an integer of at least two")
    if votes.ndim != 2 or votes.shape[1] < 1:
        raise ValueError("votes must have shape (inputs, draws), with at least one draw")
    if not np.issubdtype(votes.dtype, np.integer):
        raise TypeError("votes must contain integer class indices")
    if votes.size and (votes.min() < 0 or votes.max() >= n_classes):
        raise ValueError("vote indices must lie between zero and n_classes - 1")
    return votes


def counts_top2(votes_prefix, n_classes):
    """Count classes; ties are ordered by increasing class index."""
    votes_prefix = _votes(votes_prefix, n_classes)
    n, draws = votes_prefix.shape
    counts = np.zeros((n, n_classes), dtype=np.int64)
    np.add.at(counts, (np.repeat(np.arange(n), draws), votes_prefix.ravel()), 1)
    order = np.argsort(-counts, axis=1, kind="stable")
    top_class = order[:, 0]
    top = counts[np.arange(n), top_class]
    second = counts[np.arange(n), order[:, 1]]
    return top_class, top, second, counts


def predict_table(votes_prefix, n_classes, thresholds):
    """Return accepted classes or -1 (abstention) using a supplied table."""
    top_class, top, second, _ = counts_top2(votes_prefix, n_classes)
    thresholds = np.asarray(thresholds)
    if thresholds.ndim != 1 or len(thresholds) <= np.asarray(votes_prefix).shape[1]:
        raise ValueError("threshold table must cover every possible top-two total")
    if not np.issubdtype(thresholds.dtype, np.integer):
        raise TypeError("thresholds must be integers")
    accepted = (top > second) & (top >= thresholds[top + second])
    return np.where(accepted, top_class, -1)


def fixed_predict(votes, n_classes, alpha=0.001):
    """Apply the two-sided test once to all supplied draws (normally 256)."""
    votes = _votes(votes, n_classes)
    return predict_table(votes, n_classes, threshold_table(alpha, votes.shape[1]))


def sequential(votes, n_classes, looks, thresholds):
    """Original interface: first accepted cumulative look and its draw count."""
    votes = _votes(votes, n_classes)
    looks = tuple(looks)
    if (not looks or any(not isinstance(x, (int, np.integer)) or x < 1 for x in looks)
            or any(a >= b for a, b in zip(looks, looks[1:]))
            or looks[-1] > votes.shape[1]):
        raise ValueError("looks must be strictly increasing positive draw counts within votes")
    n = votes.shape[0]
    decisions = np.full(n, -1, dtype=np.int64)
    used = np.full(n, looks[-1], dtype=np.int64)
    active = np.ones(n, dtype=bool)
    for draws in looks:
        result = predict_table(votes[:, :draws], n_classes, thresholds)
        accepted = active & (result >= 0)
        decisions[accepted] = result[accepted]
        used[accepted] = draws
        active &= ~accepted
    return decisions, used


def sequential_predict(votes, n_classes, looks=DEFAULT_LOOKS, alpha=0.001):
    """Allocate alpha equally across looks and stop at the first acceptance."""
    looks = tuple(looks)
    if not looks:
        raise ValueError("at least one look is required")
    per_look = _level(alpha) / len(looks)
    votes = _votes(votes, n_classes)
    return sequential(votes, n_classes, looks, threshold_table(per_look, votes.shape[1]))


def two_sided_pvalue(top, second):
    """Exact rational binomial p-value for already ranked top-two counts."""
    if (not isinstance(top, (int, np.integer))
            or not isinstance(second, (int, np.integer)) or top < second or second < 0):
        raise ValueError("counts must be integers with top >= second >= 0")
    total = int(top + second)
    tail = sum(math.comb(total, k) for k in range(int(top), total + 1))
    return min(Fraction(1), Fraction(2 * tail, 1 << total))
