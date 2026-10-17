"""Decision layer: single-owner normalization and exact expected-F0.5 set selection.

Expected F0.5 (decision-theoretic approach, Ye et al., ICML 2012): with candidate
probabilities sorted descending and labels treated as independent, the best
prediction is always a top-k prefix. For each k we compute

    E[F(top-k)] = sum_{k1,k2} P(S_1..k = k1) P(S_k+1..T + M = k2) * 1.25 k1 / (k + 0.25 (k1 + k2))

using Poisson-binomial distributions, where M ~ Poisson(lam) stands for true matches
outside the top T (lower-ranked candidates plus blocking misses). The empty prediction
scores 1 only when there is no true match at all, so E[F(empty)] = P(S_1..T = 0) P(M = 0),
matching the challenge metric. At most 11 matches are ever predicted (ground-truth max).
"""
from __future__ import annotations

import numpy as np
import polars as pl

BETA2 = 0.25
MAX_MATCHES = 11


def expected_f(p: np.ndarray, lam: np.ndarray, max_k: int = MAX_MATCHES, m_max: int = 25, g=None) -> np.ndarray:
    """p: (N, T) probabilities sorted descending, zero-padded. lam: (N,) Poisson mean of
    true matches outside the top T. Returns (N, max_k + 1) expected F0.5 for k = 0..max_k.
    g: optional size tilt over the S1's total number of true matches s (g[s], 1 beyond its length): the size
    distribution becomes P(s) g(s) up to a constant, which leaves the argmax over k unchanged, so values are
    returned unnormalised. True copies come in groups, so independent labels overstate s = 1 (g = [1, 0.8])."""
    n, t = p.shape
    max_k = min(max_k, t)
    pre = np.zeros((n, t + 1, t + 1))
    pre[:, 0, 0] = 1.0
    for k in range(1, t + 1):
        q = p[:, k - 1:k]
        pre[:, k, :] = pre[:, k - 1, :] * (1.0 - q)
        pre[:, k, 1:] += pre[:, k - 1, :-1] * q
    suf = np.zeros((n, t + 1, t + 1))
    suf[:, t, 0] = 1.0
    for k in range(t - 1, -1, -1):
        q = p[:, k:k + 1]
        suf[:, k, :] = suf[:, k + 1, :] * (1.0 - q)
        suf[:, k, 1:] += suf[:, k + 1, :-1] * q
    j = np.arange(m_max + 1)
    fact = np.cumprod(np.r_[1.0, np.arange(1, m_max + 1, dtype=float)])
    pois = np.exp(-lam)[:, None] * lam[:, None] ** j[None, :] / fact[None, :]

    width = t + m_max + 1
    G = np.ones(2 * width)
    if g is not None:
        G[:len(g)] = g
    out = np.zeros((n, max_k + 1))
    out[:, 0] = pre[:, t, 0] * pois[:, 0] * G[0]
    for k in range(1, max_k + 1):
        rest = np.zeros((n, width))
        tail = suf[:, k, : t - k + 1]
        for a in range(t - k + 1):
            rest[:, a:a + m_max + 1] += tail[:, a:a + 1] * pois
        k1 = np.arange(k + 1)[:, None]
        k2 = np.arange(width)[None, :]
        w = (1.0 + BETA2) * k1 / (k + BETA2 * (k1 + k2)) * G[k1 + k2]
        out[:, k] = ((pre[:, k, : k + 1] @ w) * rest).sum(axis=1)
    return out


def prior_shift(scores: pl.DataFrame, r: float, col: str = "p") -> pl.DataFrame:
    """Label-shift adjustment of calibrated probabilities: odds times r.

    r < 1 expects fewer true pairs than the calibration data had. Test is inferred to hold
    ~41-48% ownerless records against ~26% in train; r is chosen on distractor injection.
    """
    if r == 1.0:
        return scores
    odds = pl.col(col) / (1.0 - pl.col(col)).clip(1e-12, None)
    return scores.with_columns(((odds * r) / (1.0 + odds * r)).alias(col))


def single_owner(scores: pl.DataFrame, col: str = "p", power: float = 1.0) -> pl.DataFrame:
    """Renormalize each target's probabilities across the S1s that list it.

    Every ground-truth target has at most one owner. Treating each S1's probability as
    independent evidence and conditioning on "at most one owner" gives
    p'_s = odds_s / (1 + sum over competing S1 of odds). A target listed by one S1 keeps p. power > 1 sharpens the odds first (the stack sees competitor features,
    so its mid-range p are under-confident for the set decision).
    """
    odds = (pl.col(col).clip(0.0, 1.0 - 1e-6) / (1.0 - pl.col(col).clip(0.0, 1.0 - 1e-6))) ** power
    return scores.with_columns((odds / (1.0 + odds.sum().over("t"))).alias(col))


def hard_owner(scores: pl.DataFrame, col: str = "p") -> pl.DataFrame:
    """Zero every claim on a target except the strongest one."""
    return scores.with_columns(
        pl.when(pl.col(col) == pl.col(col).max().over("t")).then(pl.col(col)).otherwise(0.0).alias(col)
    )


def dta_select(scores: pl.DataFrame, missed: float, top: int = 12, chunk: int = 200_000, g=None) -> pl.DataFrame:
    """Choose each S1's predicted set by exact expected F0.5.

    scores: s1, t, p (calibrated). missed: expected number of true matches per S1 that
    are not among its candidates at all (blocking misses). Returns predicted (s1, t) and
    the chosen k per S1 in column k.
    """
    ranked = (
        scores.sort(["s1", "p", "t"], descending=[False, True, False])
        .with_columns(r=pl.int_range(pl.len()).over("s1"))
    )
    lam = ranked.filter(pl.col("r") >= top).group_by("s1").agg(rest=pl.col("p").sum())
    wide = (
        ranked.filter(pl.col("r") < top)
        .group_by("s1")
        .agg(pl.col("p"))
        .join(lam, on="s1", how="left")
        .with_columns(pl.col("rest").fill_null(0.0) + missed)
    )
    chosen = []
    for lo in range(0, wide.height, chunk):
        part = wide.slice(lo, chunk)
        mat = np.zeros((part.height, top))
        lists = part["p"].to_list()
        for i, row in enumerate(lists):
            mat[i, :len(row)] = row
        ef = expected_f(mat, part["rest"].to_numpy().astype(float), max_k=min(MAX_MATCHES, top), g=g)
        chosen.append(pl.DataFrame({"s1": part["s1"], "k": np.argmax(ef, axis=1).astype(np.int64)}))
    k = pl.concat(chosen)
    return ranked.join(k, on="s1").filter(pl.col("r") < pl.col("k")).select("s1", "t", "k")


def threshold_select(scores: pl.DataFrame, tau: float, rho: float = 0.0) -> pl.DataFrame:
    """EUM challenger: all candidates with p >= tau (and >= rho x the S1's best), capped at 11."""
    d = scores.filter(pl.col("p") >= tau)
    if rho > 0:
        d = d.filter(pl.col("p") >= rho * pl.col("p").max().over("s1"))
    return (
        d.sort(["s1", "p", "t"], descending=[False, True, False])
        .group_by("s1", maintain_order=True)
        .head(MAX_MATCHES)
        .select("s1", "t")
    )
