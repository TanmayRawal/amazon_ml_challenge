"""Entity-disjoint folds over Source-1 entities.

Each Source-1 entity and all of its matched targets form one group (targets have a
single owner in the ground truth, so groups never overlap). Folds are stratified by
country x match-count bucket x "has an Indic-script target", using a seeded hash so
the assignment is reproducible and independent of row order.
"""
from __future__ import annotations

import polars as pl


def make_folds(s1: pl.DataFrame, gt: pl.DataFrame, tgt: pl.DataFrame, n_folds: int, seed: int) -> pl.DataFrame:
    """s1: s1, entity_id, country. gt: s1, t. tgt: t, script. Returns s1, fold, stratum."""
    per_s1 = (
        gt.join(tgt.select("t", "script"), on="t", how="left")
        .group_by("s1")
        .agg(n_true=pl.len(), indic=pl.col("script").is_in(["indic", "mixed"]).any())
    )
    df = (
        s1.select("s1", "entity_id", "country")
        .join(per_s1, on="s1", how="left")
        .with_columns(pl.col("n_true").fill_null(0), pl.col("indic").fill_null(False))
    )
    bucket = (
        pl.when(pl.col("n_true") == 0).then(pl.lit("0"))
        .when(pl.col("n_true") == 1).then(pl.lit("1"))
        .when(pl.col("n_true") <= 3).then(pl.lit("2-3"))
        .when(pl.col("n_true") <= 5).then(pl.lit("4-5"))
        .otherwise(pl.lit("6+"))
    )
    df = df.with_columns(
        stratum=pl.concat_str([pl.col("country"), bucket, pl.col("indic").cast(pl.String)], separator="|"),
        _h=pl.col("entity_id").hash(seed=seed),
    )
    df = df.with_columns(
        fold=((pl.col("_h").rank("ordinal").over("stratum") - 1) % n_folds).cast(pl.UInt8)
    )
    return df.select("s1", "fold", "stratum", "n_true").sort("s1")
