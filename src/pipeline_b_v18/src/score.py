"""Exact challenge metric (macro F0.5 over Source-1 entities) and blocking metrics.

Semantics, per Source-1 entity with true set T and predicted set P:
  |T| = 0: score 1 if |P| = 0 else 0
  |T| > 0: score 0 if |P ∩ T| = 0 (this covers P = ∅), else
           1.25·prec·rec / (0.25·prec + rec)
The macro score averages over every entity in the evaluation set; an entity with
no prediction rows counts as an empty prediction.
"""
from __future__ import annotations

import math

import polars as pl


def f05(pred: set, true: set) -> float:
    """Reference implementation for one entity (used by tests)."""
    if not true:
        return 1.0 if not pred else 0.0
    tp = len(pred & true)
    if tp == 0:
        return 0.0
    p, r = tp / len(pred), tp / len(true)
    return 1.25 * p * r / (0.25 * p + r)


def per_entity(eval_s1: pl.DataFrame, pred: pl.DataFrame, truth: pl.DataFrame) -> pl.DataFrame:
    """Per-entity counts and F0.5.

    eval_s1: column s1 (the entities to score). pred/truth: columns s1, t.
    Returns s1, n_true, n_pred, tp, f.
    """
    pred = pred.select("s1", "t").unique()
    n_true = truth.group_by("s1").agg(n_true=pl.len())
    n_pred = pred.group_by("s1").agg(n_pred=pl.len())
    tp = pred.join(truth.select("s1", "t"), on=["s1", "t"], how="inner").group_by("s1").agg(tp=pl.len())
    df = (
        eval_s1.select("s1")
        .join(n_true, on="s1", how="left")
        .join(n_pred, on="s1", how="left")
        .join(tp, on="s1", how="left")
        .with_columns(pl.col("n_true", "n_pred", "tp").fill_null(0).cast(pl.Int64))
    )
    prec = pl.col("tp") / pl.col("n_pred")
    rec = pl.col("tp") / pl.col("n_true")
    f = (
        pl.when(pl.col("n_true") == 0)
        .then((pl.col("n_pred") == 0).cast(pl.Float64))
        .when(pl.col("tp") == 0)
        .then(0.0)
        .otherwise(1.25 * prec * rec / (0.25 * prec + rec))
    )
    return df.with_columns(f=f)


def ci95(values: pl.Series) -> tuple[float, float]:
    """Mean and half-width of a normal-approximation 95% interval (n is in the millions)."""
    n = values.len()
    mean = float(values.mean())
    sd = float(values.std()) if n > 1 else 0.0
    return mean, 1.96 * sd / math.sqrt(max(n, 1))


def summarize(ent: pl.DataFrame) -> dict:
    """Headline metrics from per_entity() output."""
    mean, half = ci95(ent["f"])
    single = ent.filter(pl.col("n_true") == 0)
    multi = ent.filter(pl.col("n_true") >= 2)
    fp = ent["n_pred"] - ent["tp"]
    return {
        "n_s1": ent.height,
        "macro_f05": mean,
        "ci95": half,
        "singleton_acc": float((single["n_pred"] == 0).mean()) if single.height else float("nan"),
        "fp_pairs": int(fp.sum()),
        "s1_with_fp": float((fp > 0).mean()),
        "multi_recall": float(multi["tp"].sum() / max(multi["n_true"].sum(), 1)),
        "pair_precision": float(ent["tp"].sum() / max(ent["n_pred"].sum(), 1)),
        "pair_recall": float(ent["tp"].sum() / max(ent["n_true"].sum(), 1)),
        "mean_pred_len": float(ent["n_pred"].mean()),
        "mean_true_len": float(ent["n_true"].mean()),
    }


def candidate_metrics(
    eval_s1: pl.DataFrame, cand: pl.DataFrame, truth: pl.DataFrame, pool_sizes: dict | None = None
) -> dict:
    """Recall ceiling, oracle macro F0.5 and reduction ratio of a candidate set.

    eval_s1 needs columns s1 and country; pool_sizes maps country -> number of targets.
    """
    ids = eval_s1.select("s1")
    truth_e = truth.join(ids, on="s1", how="semi")
    cand_e = cand.select("s1", "t").join(ids, on="s1", how="semi")
    hit = cand_e.join(truth_e, on=["s1", "t"], how="inner")
    oracle = per_entity(ids, hit, truth_e)
    covered = oracle.filter(pl.col("n_true") > 0)
    per_s1 = cand_e.group_by("s1").agg(n=pl.len())
    counts = ids.join(per_s1, on="s1", how="left").with_columns(pl.col("n").fill_null(0))["n"]
    out = {
        "pairs_true": truth_e.height,
        "pair_recall": hit.height / max(truth_e.height, 1),
        "s1_full_cover": float((covered["tp"] == covered["n_true"]).mean()) if covered.height else float("nan"),
        "oracle_macro_f05": float(oracle["f"].mean()),
        "cand_pairs": cand_e.height,
        "cand_per_s1_mean": float(counts.mean()),
        "cand_per_s1_p90": float(counts.quantile(0.9)),
        "cand_per_s1_max": int(counts.max()),
    }
    if pool_sizes is not None:
        by_c = eval_s1.group_by("country").agg(n=pl.len())
        space = sum(row["n"] * pool_sizes.get(row["country"], 0) for row in by_c.iter_rows(named=True))
        out["reduction_ratio_within_country"] = 1.0 - cand_e.height / max(space, 1)
    return out
