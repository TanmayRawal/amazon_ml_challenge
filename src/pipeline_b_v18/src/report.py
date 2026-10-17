"""Blocking and matching reports, overall and by stratum (train universe)."""
from __future__ import annotations

import json

import polars as pl

from blocking import PASS_COLS, load_candidates
from prepare import load
from score import candidate_metrics, per_entity, summarize


def pool_sizes(cfg, split: str) -> dict:
    """Number of target records per country in a split."""
    return dict(load(cfg, split, "tgt", ["country"]).group_by("country").len().iter_rows())


def _fmt(d: dict) -> str:
    """Format a metrics dict as "key=value" pairs for logging."""
    return "  ".join(f"{k}={v:.4f}" if isinstance(v, float) else f"{k}={v:,}" for k, v in d.items())


def candidate_report(cfg, log=print, name: str = "cand_a") -> dict:
    """Blocking report on train: candidate metrics overall and per country, pair recall by stratum
    and the share of true pairs found by each pass; saved as train/report_<name>.json."""
    s1 = load(cfg, "train", "s1", ["s1", "country"])
    gt = load(cfg, "train", "gt")
    cand = load_candidates(cfg, "train", ["s1", "t"] + PASS_COLS)
    pool = pool_sizes(cfg, "train")
    res = {"overall": candidate_metrics(s1, cand, gt, pool)}
    for (country,), grp in s1.group_by(["country"], maintain_order=True):
        res[country] = candidate_metrics(grp, cand, gt, pool)
    for k, v in res.items():
        log(f"[blocking] {k}: {_fmt(v)}")

    tg = load(cfg, "train", "tgt", ["t", "src", "script", "addr_blank"])
    hits = (
        gt.join(cand, on=["s1", "t"], how="left")
        .with_columns(hit=pl.col("f_exact").is_not_null())
        .with_columns(pl.col(PASS_COLS).fill_null(0))
        .join(tg, on="t")
        .join(s1, on="s1")
    )
    strata = {}
    for col in ["src", "script", "addr_blank"]:
        tab = hits.group_by("country", col).agg(recall=pl.col("hit").mean(), n=pl.len()).sort("country", col)
        strata[col] = tab.to_dicts()
        log(f"[blocking] pair recall by {col}:\n{tab}")
    alone = hits.select(
        exact=(pl.col("f_exact") > 0).mean(), rare=(pl.col("rare_n") > 0).mean(),
        addr=(pl.col("addr_n") > 0).mean(), concat=(pl.col("f_concat") > 0).mean(),
    ).to_dicts()[0]
    log(f"[blocking] share of true pairs found by each pass (after the cut): {_fmt(alone)}")
    res["strata"] = strata
    res["pass_share"] = alone
    (cfg.split_dir("train") / f"report_{name}.json").write_text(json.dumps(res, indent=2, default=str))
    return res


def strata_report(cfg, pred: pl.DataFrame, fold: int, log=print) -> dict:
    """Error analysis on one fold: macro F0.5 and error counts by entity stratum."""
    from features import name_frequencies

    gt = load(cfg, "train", "gt")
    tg = load(cfg, "train", "tgt", ["t", "country", "core_key", "script", "addr_blank"])
    s1 = name_frequencies(load(cfg, "train", "s1", ["s1", "country", "core_key"]), tg)
    folds = load(cfg, "train", "folds", ["s1", "fold", "n_true"])
    ids = s1.join(folds, on="s1").filter(pl.col("fold") == fold)
    tflags = (
        gt.join(tg.select("t", "script", "addr_blank"), on="t")
        .group_by("s1")
        .agg(has_indic=pl.col("script").is_in(["indic", "mixed"]).any(), has_blank=pl.col("addr_blank").any())
    )
    ent = (
        per_entity(ids.select("s1"), pred, gt)
        .join(ids, on="s1")
        .join(tflags, on="s1", how="left")
        .with_columns(pl.col("has_indic", "has_blank").fill_null(False))
        .with_columns(
            truth=pl.when(pl.col("n_true") == 0).then(pl.lit("0"))
            .when(pl.col("n_true") == 1).then(pl.lit("1"))
            .when(pl.col("n_true") <= 3).then(pl.lit("2-3"))
            .when(pl.col("n_true") <= 5).then(pl.lit("4-5")).otherwise(pl.lit("6+")),
            name_repeated=pl.col("name_rep") > 1,
            singleton_trap=(pl.col("n_true") == 0) & (pl.col("name_tdf") > 0),
        )
    )
    out = {}
    for cols in (["country", "truth"], ["country", "name_repeated"], ["country", "singleton_trap"],
                 ["country", "has_indic"], ["country", "has_blank"]):
        tab = (
            ent.group_by(cols)
            .agg(n=pl.len(), macro_f05=pl.col("f").mean(),
                 lost=(1 - pl.col("f")).sum(), fp=(pl.col("n_pred") - pl.col("tp")).sum(),
                 fn=(pl.col("n_true") - pl.col("tp")).sum())
            .with_columns(share_of_loss=pl.col("lost") / (1 - ent["f"]).sum())
            .sort(cols)
        )
        out["/".join(cols)] = tab.to_dicts()
        log(f"[strata fold {fold}] by {cols}:\n{tab}")

    # Where do false-positive pairs point: an unmatched distractor or another entity's target?
    owners = gt.select("t", owner="s1")
    fps = pred.join(ids.select("s1"), on="s1", how="semi").join(gt, on=["s1", "t"], how="anti")
    kinds = fps.join(owners, on="t", how="left").select(
        to_distractor=pl.col("owner").is_null().mean(), n=pl.len()).to_dicts()[0]
    log(f"[strata fold {fold}] false-positive pairs: {kinds['n']:,}; pointing at unmatched distractors: "
        f"{kinds['to_distractor']:.1%}, at another entity's target: {1 - kinds['to_distractor']:.1%}")
    out["fp_kinds"] = kinds
    return out


def matching_report(eval_s1: pl.DataFrame, pred: pl.DataFrame, gt: pl.DataFrame, label: str, log=print) -> dict:
    """Headline metrics plus a per-country breakdown; eval_s1 needs s1 and country."""
    ent = per_entity(eval_s1, pred, gt).join(eval_s1.select("s1", "country"), on="s1")
    out = {"overall": summarize(ent)}
    log(f"[{label}] overall: {_fmt(out['overall'])}")
    for (country,), grp in ent.group_by(["country"], maintain_order=True):
        out[country] = summarize(grp)
        log(f"[{label}] {country}: {_fmt(out[country])}")
    return out
