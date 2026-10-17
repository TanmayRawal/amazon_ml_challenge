"""Second opinion for the v21e ensemble (US/India): a small LightGBM that corrects the probabilities the set decision
sees (after single owner), boosted from their logit, with the list context on the ENSEMBLE probabilities, both models'
logits, the v20 transformer scores and v20 stack number / street / sibling signals. Self-contained (polars, numpy,
lightgbm + files of a v20/V21 work dir), so train (ens_exp.py) and test build the features with the same code.

Test use inside v21e_bundle/scripts/ensemble_v21e.py, replacing step 4 (decoy shift -> bw boost -> apply_policy):
    sys.path.insert(0, r"<...>/experiments/decision2"); import ens_second as es
    pool = xlink.bw_boost(cfg, "test", pool, 3.0, log)                       # NO decoy shift here
    feat = es.features(args.work, "test", pool, p20, p18)                    # p20 / p18: the frames of steps 1 and 2
    pred_usin = es.decide(pool, feat, es.load(), shift=lambda s: xlink.decoy_shift(cfg, "test", s, 0.5, log))
(the decoy shift is applied after the correction: the model never saw it in training.)
"""
from __future__ import annotations

import glob
import os

import numpy as np
import polars as pl

HERE = os.path.dirname(os.path.abspath(__file__))
MODEL = os.path.join(HERE, "ens_second.txt")
STACK = ["num_conflict", "num_match", "num_near", "num_delta", "num_off", "num_s_only", "num_t_only", "sfx_conflict",
         "st_tset", "addr_wjac", "num_view", "num_new_min", "tok_view", "tok_new_min", "sa_same", "sa_other",
         "addr_eq_same", "sup_num_share", "s_num_share", "t_num_mode", "nm_edit", "name_tset", "leg_swap"]
CTX = ["l_rank", "l_sum_o", "l_nconf_o", "l_nconf_same", "l_nconf_oth", "l_t_n", "l_t_odds_o", "l_best_o"]
PAIR = ["lp20", "lp18", "src_both"]
FEATS = ["in_band", "bw"] + STACK + ["xenc1", "xenc2", "xenc3"] + CTX + PAIR + ["l_logit"]
MIN_P = 0.01
MISSED = 0.031


def _lg(c: str) -> pl.Expr:
    """Polars expression for the logit of probability column `c`, clipped to [1e-6, 1 - 1e-6]."""
    return (pl.col(c).clip(1e-6, 1 - 1e-6) / (1 - pl.col(c).clip(1e-6, 1 - 1e-6))).log()


def context(pool: pl.DataFrame, tg: pl.DataFrame) -> pl.DataFrame:
    """s1, t + CTX for the pairs with p > MIN_P: the S1's other candidates and the target's rivals on pool p."""
    conf = (pl.col("p") > 0.5).cast(pl.Float32)
    odds = pl.col("p").clip(0, 1 - 1e-6) / (1 - pl.col("p").clip(0, 1 - 1e-6))
    y = (pool.filter(pl.col("p") > MIN_P).select("s1", "t", "p").join(tg.select("t", "src"), on="t", how="left")
         .with_columns(l_rank=pl.col("p").rank("ordinal", descending=True).over("s1").cast(pl.Float32),
                       l_sum_o=(pl.col("p").sum().over("s1") - pl.col("p")).cast(pl.Float32),
                       l_nconf_o=conf.sum().over("s1") - conf,
                       l_nconf_same=conf.sum().over("s1", "src") - conf,
                       l_nconf_oth=conf.sum().over("s1") - conf.sum().over("s1", "src"),
                       l_t_n=pl.len().over("t").cast(pl.Float32),
                       l_t_odds_o=(odds.sum().over("t") - odds).cast(pl.Float32)))
    top2 = pl.col("p").sort(descending=True).slice(1, 1).first().over("s1")
    return y.with_columns(l_best_o=pl.when(pl.col("l_rank") == 1).then(top2).otherwise(pl.col("p").max().over("s1"))
                          .fill_null(0.0).cast(pl.Float32)).drop("p", "src")


def features(work: str, split: str, pool: pl.DataFrame, p20: pl.DataFrame, p18: pl.DataFrame) -> pl.DataFrame:
    """pool: s1, t, p = the ensemble probability the decision gets (after bw_boost); p20: s1, t, p20 (v20/V21 linked
    pool); p18: s1, t, p18 (v18 stage 2). Returns s1, t + FEATS (without l_logit) for the pairs with p > MIN_P."""
    d = os.path.join(work, split)
    tg = pl.read_parquet(os.path.join(d, "tgt.parquet"), columns=["t", "src", "addr_blank", "is_domain"])
    y = context(pool, tg)
    keys = y.select("s1", "t")
    band = pl.read_parquet(os.path.join(d, "xenc_band.parquet"), columns=["s1", "t"]).join(keys, on=["s1", "t"], how="semi")
    for k, name in ((1, "xenc_scores.parquet"), (2, "xenc_scores2.parquet"), (3, "xenc_scores3.parquet")):
        path = os.path.join(d, name)
        xs = (pl.read_parquet(path, columns=["s1", "t", "xenc"]).join(band, on=["s1", "t"], how="semi")
              .select("s1", "t", pl.col("xenc").cast(pl.Float32).alias(f"xenc{k}")) if os.path.exists(path)
              else band.with_columns(pl.lit(None, pl.Float32).alias(f"xenc{k}")))
        y = y.join(xs, on=["s1", "t"], how="left")
    fd = pl.concat([pl.scan_parquet(p).select(["s1", "t"] + STACK).join(keys.lazy(), on=["s1", "t"], how="semi").collect()
                    .with_columns([pl.col(c).cast(pl.Float32) for c in STACK])
                    for p in sorted(glob.glob(os.path.join(d, "feat_d", "part_*.parquet")))])
    y = (y.join(fd, on=["s1", "t"], how="left")
         .join(tg.select("t", bw=(pl.col("addr_blank") | pl.col("is_domain")).cast(pl.Float32)), on="t", how="left")
         .join(p20.select("s1", "t", "p20"), on=["s1", "t"], how="left").join(p18.select("s1", "t", "p18"), on=["s1", "t"], how="left")
         .with_columns(in_band=pl.col("xenc1").is_not_null().cast(pl.Float32),
                       lp20=_lg("p20").cast(pl.Float32), lp18=_lg("p18").cast(pl.Float32),
                       src_both=(pl.col("p20").is_not_null().cast(pl.Int8) + 2 * pl.col("p18").is_not_null().cast(pl.Int8))
                       .cast(pl.Float32))
         .drop("p20", "p18"))
    return y.select(["s1", "t"] + [c for c in FEATS if c != "l_logit"])


def matrix(y: pl.DataFrame, feats: list = FEATS) -> tuple[np.ndarray, np.ndarray]:
    """y: s1, t, p (owned) + features -> (X, offset = logit p)."""
    y = y.with_columns(l_logit=_lg("p").cast(pl.Float32))
    return y.select(feats).to_numpy().astype(np.float32), y["l_logit"].to_numpy().astype(np.float64)


def adjust(feat: pl.DataFrame, booster, feats: list = FEATS):
    """owned p -> sigmoid(logit p + booster raw score) for the pairs in feat (others unchanged)."""
    def f(s: pl.DataFrame) -> pl.DataFrame:
        """Return `s` (s1, t, p) with p replaced by the corrected probability for the pairs present in `feat`."""
        y = s.join(feat, on=["s1", "t"], how="inner")
        if y.height == 0:
            return s
        X, off = matrix(y, feats)
        pc = 1.0 / (1.0 + np.exp(-(off + booster.predict(X, raw_score=True))))
        return (s.join(y.select("s1", "t", pc=pl.Series(pc)), on=["s1", "t"], how="left")
                .with_columns(p=pl.coalesce("pc", "p")).drop("pc"))
    return f


def owned(pool: pl.DataFrame) -> pl.DataFrame:
    """decide.single_owner (v20 policy dta_own, power 1)."""
    odds = pl.col("p").clip(0.0, 1.0 - 1e-6) / (1.0 - pl.col("p").clip(0.0, 1.0 - 1e-6))
    return pool.filter(pl.col("p") > 1e-4).select("s1", "t", "p").with_columns(p=odds / (1.0 + odds.sum().over("t")))


def decide(pool: pl.DataFrame, feat: pl.DataFrame | None, booster, shift=None, missed: float = MISSED, feats=FEATS,
           adj=None) -> pl.DataFrame:
    """v20's dta_own decision with the second opinion inside: single owner -> correction -> shift (optional
    s -> s, e.g. the decoy shift) -> exact expected-F0.5 sets (decide.dta_select, imported from the caller's path)."""
    from decide import dta_select
    s = owned(pool)
    if adj is not None:
        s = adj(s)
    elif booster is not None:
        s = adjust(feat, booster, feats)(s)
    if shift is not None:
        s = shift(s)
    return dta_select(s.filter(pl.col("p") > 1e-4), missed).select("s1", "t")


def load(path: str = MODEL):
    """Load the trained second-opinion LightGBM booster from `path` (default: ens_second.txt next to this file)."""
    import lightgbm as lgb
    return lgb.Booster(model_file=path)


if __name__ == "__main__":   # self-check: the hook shifts the logit of listed pairs only; decide == v20 dta_own without it
    import sys
    sys.path.insert(0, "C:/Users/Lenovo/Desktop/Ml challange/code_V21/business_entity_resolution/src")
    from matcher_b import apply_policy

    class _Stub:
        """Fake booster for the self-check: adds a constant +2.0 to the logit of every listed pair."""

        def predict(self, X, raw_score=False):
            """Return a raw score of 2.0 per row (asserts raw_score=True and the full feature width)."""
            assert raw_score and X.shape[1] == len(FEATS)
            return np.full(len(X), 2.0)

    s = pl.DataFrame({"s1": [0, 0, 1], "t": [5, 6, 5], "p": [0.45, 0.3, 0.2]})
    feat = pl.DataFrame({"s1": [0], "t": [5]}).with_columns([pl.lit(None, pl.Float32).alias(c) for c in FEATS if c != "l_logit"])
    out = adjust(feat, _Stub())(owned(s)).sort("s1", "t")["p"].to_list()
    p0 = owned(s).sort("s1", "t")["p"].to_list()
    assert abs(out[0] - 1 / (1 + np.exp(-(np.log(p0[0] / (1 - p0[0])) + 2.0)))) < 1e-9 and out[1:] == p0[1:]
    assert decide(s, None, None).sort("s1", "t").equals(apply_policy(s, {"name": "dta_own", "missed": MISSED}).sort("s1", "t"))
    assert decide(s, feat, _Stub()).sort("s1", "t")["t"].to_list() == [5]
    print("ens_second self-check ok")
