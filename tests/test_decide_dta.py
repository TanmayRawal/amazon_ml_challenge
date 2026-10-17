"""The O(n^2) expected-F0.5 must equal brute-force enumeration under the metric's rules."""
import itertools
import math

import numpy as np
import polars as pl
import pytest

from decide import dta_select, expected_f, single_owner
from score import f05


def brute(p, lam, k, m_cap=40):
    """E[F0.5 of predicting the first k items]; M extra true matches ~ Poisson(lam)."""
    total = 0.0
    for labels in itertools.product([0, 1], repeat=len(p)):
        prob = math.prod(pi if y else 1 - pi for pi, y in zip(p, labels))
        for m in range(m_cap + 1 if lam > 0 else 1):
            pm = math.exp(-lam) * lam ** m / math.factorial(m) if lam > 0 else 1.0
            true = {i for i, y in enumerate(labels) if y} | {f"miss{j}" for j in range(m)}
            total += prob * pm * f05(set(range(k)), true)
    return total


@pytest.mark.parametrize("seed,lam", [(0, 0.0), (1, 0.0), (2, 0.7), (3, 2.5)])
def test_expected_f_matches_enumeration(seed, lam):
    """expected_f equals brute-force expected F0.5 for every prefix length k, with and without unseen matches."""
    rng = np.random.default_rng(seed)
    p = np.sort(rng.random(6))[::-1]
    got = expected_f(p[None, :], np.array([lam]), max_k=6)[0]
    for k in range(7):
        assert got[k] == pytest.approx(brute(list(p), lam, k), abs=1e-9)


def test_single_candidate_abstains_below_half():
    """A lone candidate is predicted only above p = 0.5, and a chance of unseen matches lowers that bar."""
    # One candidate, nothing else possible: predicting it scores p, abstaining scores 1 - p.
    ef = expected_f(np.array([[0.49], [0.51]]), np.zeros(2), max_k=1)
    assert np.argmax(ef[0]) == 0 and np.argmax(ef[1]) == 1
    # A chance of unseen true matches (lam > 0) makes abstaining less attractive.
    ef = expected_f(np.array([[0.45]]), np.array([0.5]), max_k=1)
    assert np.argmax(ef[0]) == 1


def test_dta_select_prefix_and_empty():
    """dta_select predicts the best-scoring prefix of candidates and leaves a weak-only entity empty."""
    scores = pl.DataFrame({"s1": [0, 0, 0, 1], "t": [1, 2, 3, 4], "p": [0.95, 0.9, 0.05, 0.2]},
                          schema={"s1": pl.UInt32, "t": pl.UInt32, "p": pl.Float64})
    out = dta_select(scores, missed=0.0)
    assert sorted(out.select("s1", "t").rows()) == [(0, 1), (0, 2)]


def test_single_owner_normalization():
    """single_owner splits a shared target's probability across its S1 owners and leaves unshared targets unchanged."""
    scores = pl.DataFrame({"s1": [0, 1, 2], "t": [7, 7, 8], "p": [0.9, 0.9, 0.6]})
    out = single_owner(scores).sort("s1")["p"].to_list()
    assert out[0] == pytest.approx(9 / 19) and out[1] == pytest.approx(9 / 19)
    assert out[2] == pytest.approx(0.6)


def test_prior_shift_scales_odds():
    """prior_shift multiplies each probability's odds by the given factor (1.0 is a no-op)."""
    from decide import prior_shift
    s = pl.DataFrame({"s1": [0, 0], "t": [1, 2], "p": [0.5, 0.8]})
    assert prior_shift(s, 1.0)["p"].to_list() == [0.5, 0.8]
    out = prior_shift(s, 0.5)["p"].to_list()
    assert out[0] == pytest.approx(1 / 3)            # odds 1 -> 0.5
    assert out[1] == pytest.approx(2 / 3)            # odds 4 -> 2
