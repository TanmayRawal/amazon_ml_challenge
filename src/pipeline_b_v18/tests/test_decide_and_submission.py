"""Match decision rules (threshold, relative cut, exclusivity, 11-match cap) and submission TSV write/read-back."""
import polars as pl
import pytest

from data_io import MATCH_HEADER, write_id_lists
from make_submission import _read_back
from model_a import decide


def _scores(rows):
    """Build a typed (s1, t, p) score frame from a list of row tuples."""
    return pl.DataFrame(rows, schema={"s1": pl.UInt32, "t": pl.UInt32, "p": pl.Float32}, orient="row")


def test_threshold_and_cap():
    """decide keeps only pairs whose probability clears tau."""
    scores = _scores([(0, 1, 0.9), (0, 2, 0.4), (1, 3, 0.2)])
    out = decide(scores, tau=0.5, rho=0.0, excl=False)
    assert out.sort("t").rows() == [(0, 1)]


def test_target_exclusivity_keeps_best_owner():
    """With excl=True a target shared by two S1 entities goes only to its highest-scoring owner."""
    scores = _scores([(0, 7, 0.9), (1, 7, 0.6), (1, 8, 0.7)])
    out = decide(scores, tau=0.5, rho=0.0, excl=True)
    assert sorted(out.rows()) == [(0, 7), (1, 8)]


def test_relative_cut():
    """rho drops candidates scoring below rho times the entity's best score."""
    scores = _scores([(0, 1, 0.95), (0, 2, 0.55)])
    assert sorted(decide(scores, tau=0.5, rho=0.8, excl=False).rows()) == [(0, 1)]
    assert sorted(decide(scores, tau=0.5, rho=0.5, excl=False).rows()) == [(0, 1), (0, 2)]


def test_at_most_eleven_matches():
    """decide caps each S1 entity at 11 predicted matches even when more clear the threshold."""
    scores = _scores([(0, t, 0.9) for t in range(20)])
    assert decide(scores, tau=0.5, rho=0.0, excl=False).height == 11


def test_write_and_read_back(tmp_path):
    """write_id_lists writes one sorted, comma-joined row per S1 in order (blank if none) and _read_back parses it back."""
    order = pl.Series(["S1-1", "S1-2", "S1-3"])
    pairs = pl.DataFrame({"s1_id": ["S1-1", "S1-1", "S1-3"], "t_id": ["S3-9", "S2-5", "S2-5"]})
    path = tmp_path / "m.tsv"
    write_id_lists(path, MATCH_HEADER, order, pairs)
    assert path.read_text(encoding="utf-8") == (
        "source1_entity_id\tmatched_entity_ids\nS1-1\tS2-5,S3-9\nS1-2\t\nS1-3\tS2-5\n"
    )
    got = _read_back(path, MATCH_HEADER, order.to_list(), {"S2-5", "S3-9"})
    assert got == {"S1-1": {"S2-5", "S3-9"}, "S1-2": set(), "S1-3": {"S2-5"}}


def test_read_back_rejects_unknown_ids(tmp_path):
    """_read_back raises when the file lists a matched ID that is not in the known target set."""
    path = tmp_path / "m.tsv"
    path.write_text("source1_entity_id\tmatched_entity_ids\nS1-1\tS2-404\n", encoding="utf-8")
    with pytest.raises(AssertionError):
        _read_back(path, MATCH_HEADER, ["S1-1"], {"S2-5"})
