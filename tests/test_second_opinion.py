"""V21 second opinion: the apply_policy hook corrects only the listed pairs, in logit space, after ownership."""
import math

import numpy as np
import polars as pl
import pytest

from matcher_b import apply_policy
from xlink import SECOND, second_adjust


class _Stub:
    """Fake second-opinion model that checks its input shape and always returns a raw logit of 2.0."""

    def predict(self, X, raw_score=False):
        """Assert raw-score mode and SECOND-shaped features (NaN first column), then return 2.0 per row."""
        assert raw_score and X.shape[1] == len(SECOND) and np.isnan(X[:, 0]).all()
        return np.full(len(X), 2.0)


def test_second_adjust_shifts_logit_of_listed_pairs_only():
    """second_adjust adds the model's logit only to listed pairs, and apply_policy uses the hook to change decisions."""
    s = pl.DataFrame({"s1": [0, 0], "t": [5, 6], "p": [0.4, 0.3]})
    feat = pl.DataFrame({"s1": [0], "t": [5]}).with_columns(
        [pl.lit(None, pl.Float32).alias(c) for c in SECOND if c != "l_logit"])
    out = second_adjust(feat, _Stub())(s).sort("t")["p"].to_list()
    assert out[0] == pytest.approx(1 / (1 + math.exp(-(math.log(0.4 / 0.6) + 2.0))), abs=1e-6)
    assert out[1] == 0.3
    pol = {"name": "dta_own", "missed": 0.0}
    assert apply_policy(s, pol, lambda d: d).equals(apply_policy(s, pol))
    assert apply_policy(s, pol).height == 0 and apply_policy(s, pol, second_adjust(feat, _Stub()))["t"].to_list() == [5]
