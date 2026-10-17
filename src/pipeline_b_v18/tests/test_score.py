"""F0.5 metric: README example, edge cases, vectorised per-entity scoring, and candidate recall/oracle metrics."""
import random

import polars as pl
import pytest

from score import candidate_metrics, f05, per_entity, summarize


def test_readme_example():
    """f05 reproduces the worked example from the challenge README (~0.714)."""
    pred = {"S2-00047", "S2-00193", "S3-00812"}
    true = {"S2-00047", "S3-00812"}
    assert f05(pred, true) == pytest.approx(0.714, abs=5e-4)


@pytest.mark.parametrize(
    "pred,true,expected",
    [
        (set(), set(), 1.0),          # correct singleton
        ({"a"}, set(), 0.0),          # false merge on a singleton
        (set(), {"a"}, 0.0),          # missed entity
        ({"b"}, {"a"}, 0.0),          # wrong match
        ({"a"}, {"a"}, 1.0),
        ({"a", "b"}, {"a"}, 1.25 * 0.5 * 1.0 / (0.25 * 0.5 + 1.0)),
    ],
)
def test_edge_cases(pred, true, expected):
    """f05 handles empty sets, false merges, misses, wrong matches and partial precision as the metric defines."""
    assert f05(pred, true) == pytest.approx(expected)


def _random_case(seed: int):
    """Generate a seeded random ground truth and noisy prediction list of (s1, t) pairs for 300 entities."""
    rng = random.Random(seed)
    n_s1, n_t = 300, 400
    truth, pred = [], []
    for s in range(n_s1):
        k = rng.choice([0, 0, 1, 2, 3, 5])
        true_t = rng.sample(range(n_t), k)
        truth += [(s, t) for t in true_t]
        keep = [t for t in true_t if rng.random() < 0.7]
        noise = rng.sample(range(n_t), rng.choice([0, 0, 1, 2]))
        if rng.random() < 0.2:
            keep, noise = [], []
        pred += [(s, t) for t in set(keep) | set(noise)]
    return n_s1, truth, pred


def _frame(rows):
    """Build a typed (s1, t) pair frame from a list of row tuples."""
    return pl.DataFrame(rows, schema={"s1": pl.UInt32, "t": pl.UInt32}, orient="row")


@pytest.mark.parametrize("seed", [0, 1, 2])
def test_vectorized_matches_reference(seed):
    """Vectorised per_entity/summarize scores equal a per-entity loop over the reference f05 on random data."""
    n_s1, truth, pred = _random_case(seed)
    ids = pl.DataFrame({"s1": pl.Series(range(n_s1), dtype=pl.UInt32)})
    ent = per_entity(ids, _frame(pred), _frame(truth))
    ref = []
    for s in range(n_s1):
        ref.append(f05({t for a, t in pred if a == s}, {t for a, t in truth if a == s}))
    assert ent.sort("s1")["f"].to_list() == pytest.approx(ref)
    assert summarize(ent)["macro_f05"] == pytest.approx(sum(ref) / n_s1)


def test_missing_prediction_rows_count_as_empty():
    """Entities with no prediction rows are scored as empty predictions (0 if they have matches, 1 if singleton)."""
    ids = pl.DataFrame({"s1": pl.Series([0, 1], dtype=pl.UInt32)})
    truth = _frame([(0, 5)])
    ent = per_entity(ids, _frame([]), truth)
    # entity 0 has a true match but no prediction -> 0; entity 1 is a singleton -> 1
    assert ent.sort("s1")["f"].to_list() == [0.0, 1.0]


def test_candidate_oracle_and_recall():
    """candidate_metrics reports pair recall, oracle macro F0.5 and the within-country reduction ratio of a candidate set."""
    ids = pl.DataFrame({"s1": pl.Series([0, 1, 2], dtype=pl.UInt32), "country": ["A", "A", "B"]})
    truth = _frame([(0, 1), (0, 2), (1, 3)])
    cand = _frame([(0, 1), (0, 9), (1, 3), (2, 4)])
    m = candidate_metrics(ids, cand, truth, pool_sizes={"A": 10, "B": 10})
    assert m["pair_recall"] == pytest.approx(2 / 3)
    # oracle picks the covered true pairs; singleton 2 predicts nothing
    expected = (f05({1}, {1, 2}) + 1.0 + 1.0) / 3
    assert m["oracle_macro_f05"] == pytest.approx(expected)
    assert m["reduction_ratio_within_country"] == pytest.approx(1 - 4 / 30)
