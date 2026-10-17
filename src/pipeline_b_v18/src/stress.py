"""Stress test: distractor injection (plan §6.4).

The test pool is inferred to hold ~41-48% ownerless targets versus ~26% in train. Here
a share of train S1 entities is hidden: their targets stay in every other entity's
candidate list but lose their owner, exactly like test distractors. Everything that
depends on which S1s exist is recomputed (target competition features and single-owner
normalization); candidate lists themselves do not change, because blocking and ranking
are computed per S1. The trained matchers are reused unchanged.

Scores are reported on one fixed sample of sealed-fold entities that stay visible at
every hiding level, so the numbers are directly comparable. Their pairs are rescored
exactly; competing claims on the same targets by other visible entities keep their
stored scores (recomputing those too would touch ~45M rows), and claims by hidden
entities are dropped, which is the effect being measured.
"""
from __future__ import annotations

import json
import time

import lightgbm as lgb
import numpy as np
import polars as pl

from config import Config
from decide import dta_select, hard_owner, prior_shift, single_owner, threshold_select
from features import FEATURES_C, feature_parts
from matcher_b import GROUPS, _boosters, _dir
from prepare import load
from score import per_entity, summarize


def _decide(pool: pl.DataFrame, policy: dict, ids: pl.DataFrame) -> pl.DataFrame:
    """matcher_b.apply_policy, with owner adjustment over all claims but set selection
    only for the evaluated entities (identical result for them, much less work)."""
    s = pool.select("s1", "t", "p")
    if policy["name"] in ("eum", "dta_hard"):
        s = hard_owner(s)
    elif policy["name"] == "dta_own":
        s = single_owner(s)
    s = s.join(ids.select("s1"), on="s1", how="semi")
    if policy["name"] == "eum":
        return threshold_select(s, policy["tau"], policy["rho"])
    return dta_select(s.filter(pl.col("p") > 1e-4), policy["missed"]).select("s1", "t")


def injection(cfg: Config, fracs=(0.0, 0.2, 0.3), n_eval: int = 100_000, log=print) -> dict:
    """Stage-1 distractor injection: hide each share `fracs` of train S1s and re-score matcher B.

    Evaluates up to `n_eval` sealed-fold entities that stay visible at every level; returns and
    saves (report_stress_injection.json) macro F0.5 and related metrics per hidden share.
    """
    t0 = time.time()
    meta = json.loads((_dir(cfg) / "meta.json").read_text())
    xs, ys = np.array(meta["isotonic"]["x"]), np.array(meta["isotonic"]["y"])
    boosters = _boosters(cfg)
    gt = load(cfg, "train", "gt")
    s1 = load(cfg, "train", "s1", ["s1", "country"]).join(load(cfg, "train", "folds", ["s1", "fold"]), on="s1")
    cand = pl.concat([
        pl.read_parquet(p, columns=["s1", "t", "cheap", "rrank"]).filter(pl.col("rrank") <= cfg.rk_cand_k)
        .select("s1", "t", "cheap")
        for p in sorted((cfg.split_dir("train") / "cand_b").glob("part_*.parquet"))
    ])

    hide_key = pl.col("s1").hash(seed=cfg.seed + 7) % 1_000_000
    hidden_max = s1.filter(hide_key < int(max(fracs) * 1_000_000)).select("s1")
    pool = s1.filter(pl.col("fold") == cfg.sealed_fold).join(hidden_max, on="s1", how="anti")
    rate = min(1.0, n_eval / max(pool.height, 1))
    eval_ids = pool.filter((pl.col("s1").hash(seed=cfg.seed + 8) % 1_000_000) < int(rate * 1_000_000))
    targets = cand.join(eval_ids.select("s1"), on="s1", how="semi").select("t").unique()
    feats = pl.concat([
        pl.read_parquet(p).join(eval_ids.select("s1"), on="s1", how="semi") for p in feature_parts(cfg, "train", "c")
    ])
    stored = pl.read_parquet(cfg.split_dir("train") / "scores_b.parquet", columns=["s1", "t", "p"])
    others = stored.join(targets, on="t", how="semi").join(eval_ids.select("s1"), on="s1", how="anti")
    log(f"[stress] eval entities {eval_ids.height:,}; their pairs {feats.height:,}; competing claims "
        f"{others.height:,} ({time.time() - t0:.0f}s)")

    results = {}
    for frac in fracs:
        hidden = s1.filter(hide_key < int(frac * 1_000_000)).select("s1")
        comp = (
            cand.join(hidden, on="s1", how="anti").join(targets, on="t", how="semi")
            .group_by("t").agg(n_vis=pl.len(), mx_vis=pl.col("cheap").max())
        )
        f = feats.join(comp, on="t", how="left").with_columns(
            t_n_s1=pl.col("n_vis").cast(pl.Float32).log1p(),
            t_best_gap=(pl.col("rk_p") - pl.col("mx_vis")).cast(pl.Float32),
            t_is_best=(pl.col("rk_p") >= pl.col("mx_vis")).cast(pl.Float32),
        )
        x = f.select(FEATURES_C).to_numpy()
        ca = np.interp(boosters["A"].predict(x, num_threads=0), xs, ys)
        cb = np.interp(boosters["B"].predict(x, num_threads=0), xs, ys)
        fold = f["fold"].to_numpy()
        p = np.where(np.isin(fold, GROUPS["B"]), ca, np.where(np.isin(fold, GROUPS["A"]), cb, 0.5 * (ca + cb)))
        own = f.select("s1", "t").with_columns(p=pl.Series(p))
        if frac == 0.0:
            drift = own.join(stored, on=["s1", "t"], suffix="_stored").select(
                (pl.col("p") - pl.col("p_stored")).abs().max()).item()
            log(f"[stress] hide 0% reproduces the stored scores: max |p diff| = {drift:.2e}")
        scores = pl.concat([own, others.join(hidden, on="s1", how="anti")])
        pred = _decide(scores.filter(pl.col("p") > 1e-4), meta["policy"], eval_ids)
        ent = per_entity(eval_ids.select("s1"), pred, gt)
        summ = summarize(ent)
        share = 0.26 + frac * 0.74
        results[str(frac)] = {"approx_distractor_share": share, **summ}
        log(f"[stress] hide {frac:.0%} (distractors ~{share:.0%}): macro F0.5={summ['macro_f05']:.4f} "
            f"singleton_acc={summ['singleton_acc']:.4f} s1_with_fp={summ['s1_with_fp']:.4f} "
            f"precision={summ['pair_precision']:.4f} recall={summ['pair_recall']:.4f} "
            f"mean_pred_len={summ['mean_pred_len']:.3f} ({time.time() - t0:.0f}s)")
    (cfg.split_dir("train") / "report_stress_injection.json").write_text(json.dumps(results, indent=2))
    return results


def injection_stack(cfg: Config, fracs=(0.0, 0.2, 0.3, 0.4), n_eval: int = 100_000, log=print,
                    shifts=(1.0, 0.8, 0.6, 0.45, 0.3), pool_folds: str = "sealed") -> dict:
    """Distractor injection through both stages (v4+).

    Stage 1 is rescored for the evaluated entities as in injection(). Stage 2 is then
    rebuilt for them from scratch: its target context uses stage-1 claims by visible
    entities only, and its list and consensus features use the rescored stage-1 values.
    Final claims by other visible entities keep their stored stage-2 scores. Each level is
    also decided after a prior shift r of the final odds (decide.prior_shift).

    pool_folds="dev" evaluates dev-fold entities instead (out-of-fold scores), which is
    what the shift r is chosen on; the sealed fold is only reported.
    """
    import stack

    t0 = time.time()
    m1 = json.loads((_dir(cfg) / "meta.json").read_text())
    m2 = json.loads((stack._dir(cfg) / "meta.json").read_text())
    b1 = _boosters(cfg)
    b2 = {n: lgb.Booster(model_file=str(stack._dir(cfg) / f"stack_{n}.txt")) for n in GROUPS}
    gt = load(cfg, "train", "gt")
    s1 = load(cfg, "train", "s1", ["s1", "country"]).join(load(cfg, "train", "folds", ["s1", "fold"]), on="s1")
    cand = pl.concat([
        pl.read_parquet(p, columns=["s1", "t", "cheap", "rrank"]).filter(pl.col("rrank") <= cfg.rk_cand_k)
        .select("s1", "t", "cheap")
        for p in sorted((cfg.split_dir("train") / "cand_b").glob("part_*.parquet"))
    ])
    tg = (
        load(cfg, "train", "tgt", ["t", "country", "src", "core", "core_key", "addr_toks", "legal", "addr_nums"]).sort("t")
        .with_columns(dupkey=pl.concat_str(["core_key", "addr_toks"], separator="|"))
    )
    s1t = load(cfg, "train", "s1", ["s1", "country", "core", "legal", "addr_nums"]).sort("s1")
    tdf = stack.token_df(tg)

    hide_key = pl.col("s1").hash(seed=cfg.seed + 7) % 1_000_000
    hidden_max = s1.filter(hide_key < int(max(fracs) * 1_000_000)).select("s1")
    in_pool = (pl.col("fold") == cfg.sealed_fold) if pool_folds == "sealed" else (pl.col("fold") != cfg.sealed_fold)
    pool = s1.filter(in_pool).join(hidden_max, on="s1", how="anti")
    rate = min(1.0, n_eval / max(pool.height, 1))
    eval_ids = pool.filter((pl.col("s1").hash(seed=cfg.seed + 8) % 1_000_000) < int(rate * 1_000_000))
    targets = cand.join(eval_ids.select("s1"), on="s1", how="semi").select("t").unique()
    feats = pl.concat([
        pl.read_parquet(p).join(eval_ids.select("s1"), on="s1", how="semi") for p in feature_parts(cfg, "train", "c")
    ])
    stored1 = pl.read_parquet(cfg.split_dir("train") / "scores_b.parquet", columns=["s1", "t", "p"])
    stored2 = pl.read_parquet(cfg.split_dir("train") / "scores_c.parquet", columns=["s1", "t", "p"])
    others1 = stored1.join(targets, on="t", how="semi").join(eval_ids.select("s1"), on="s1", how="anti")
    others2 = stored2.join(targets, on="t", how="semi").join(eval_ids.select("s1"), on="s1", how="anti")
    own2_stored = stored2.join(eval_ids.select("s1"), on="s1", how="semi")
    del stored1, stored2
    log(f"[stress2] eval entities {eval_ids.height:,}; their pairs {feats.height:,}; competing claims "
        f"{others1.height:,} ({time.time() - t0:.0f}s)")

    def both(boosters, x, xs, ys, fold):
        """Calibrated cross-fitted p: the model that did not train on the row's fold, mean of both otherwise."""
        ca = np.interp(boosters["A"].predict(x, num_threads=0), xs, ys)
        cb = np.interp(boosters["B"].predict(x, num_threads=0), xs, ys)
        return np.where(np.isin(fold, GROUPS["B"]), ca, np.where(np.isin(fold, GROUPS["A"]), cb, 0.5 * (ca + cb)))

    results = {}
    for frac in fracs:
        hidden = s1.filter(hide_key < int(frac * 1_000_000)).select("s1")
        comp = (
            cand.join(hidden, on="s1", how="anti").join(targets, on="t", how="semi")
            .group_by("t").agg(n_vis=pl.len(), mx_vis=pl.col("cheap").max())
        )
        f = feats.join(comp, on="t", how="left").with_columns(
            t_n_s1=pl.col("n_vis").cast(pl.Float32).log1p(),
            t_best_gap=(pl.col("rk_p") - pl.col("mx_vis")).cast(pl.Float32),
            t_is_best=(pl.col("rk_p") >= pl.col("mx_vis")).cast(pl.Float32),
        )
        p1 = both(b1, f.select(FEATURES_C).to_numpy(), np.array(m1["isotonic"]["x"]), np.array(m1["isotonic"]["y"]),
                  f["fold"].to_numpy())
        claims1 = pl.concat([f.select("s1", "t").with_columns(p=pl.Series(p1)),
                             others1.join(hidden, on="s1", how="anti")])
        tctx = stack._target_context(claims1, tg.height)
        df = f.select(["s1", "t", "label", "fold"] + stack.RAW_KEEP).with_columns(p=pl.Series(p1))
        x2 = stack._stack_part(df, tctx, tg, s1t, tdf)
        p2 = both(b2, x2.select(stack.FEATURES_D).to_numpy(), np.array(m2["isotonic"]["x"]),
                  np.array(m2["isotonic"]["y"]), x2["fold"].to_numpy())
        own = x2.select("s1", "t").with_columns(p=pl.Series(p2))
        if frac == 0.0:
            drift = own.join(own2_stored, on=["s1", "t"], suffix="_stored").select(
                (pl.col("p") - pl.col("p_stored")).abs().max()).item()
            log(f"[stress2] hide 0% reproduces the stored stage-2 scores: max |p diff| = {drift:.2e}")
        scores = pl.concat([own, others2.join(hidden, on="s1", how="anti")])
        share = 0.26 + frac * 0.74
        for r in shifts:
            pred = _decide(prior_shift(scores, r).filter(pl.col("p") > 1e-4), m2["policy"], eval_ids)
            summ = summarize(per_entity(eval_ids.select("s1"), pred, gt))
            results[f"{frac}|{r}"] = {"hide": frac, "shift": r, "approx_distractor_share": share, **summ}
            log(f"[stress2] hide {frac:.0%} (distractors ~{share:.0%}) shift {r:.2f}: macro F0.5={summ['macro_f05']:.4f} "
                f"singleton_acc={summ['singleton_acc']:.4f} s1_with_fp={summ['s1_with_fp']:.4f} "
                f"precision={summ['pair_precision']:.4f} recall={summ['pair_recall']:.4f} "
                f"mean_pred_len={summ['mean_pred_len']:.3f} ({time.time() - t0:.0f}s)")
    (cfg.split_dir("train") / f"report_stress_injection_stack_{pool_folds}.json").write_text(json.dumps(results, indent=2))
    return results
