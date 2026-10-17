"""v21e ensemble (v20 linked p + v18 stage-2 p, logit avg 0.5) on the 1.43M shared train S1s + the second opinion on
top of it. Stages: reproduce ens_eval.py; test-like baseline (orphan blank/website targets dropped, bw x3); second
opinion cross-fitted on dev folds (fold 4 by the dev model), with / without the two models' own logits; decoy shift."""
import json
import sys
import time

import lightgbm as lgb
import numpy as np
import polars as pl

R = "C:/Users/Lenovo/Desktop/Ml challange/"
sys.path.insert(0, R + "experiments/decision2/v21/business_entity_resolution/src"); sys.path.insert(0, R + "experiments/decision2")
from config import from_args  # noqa: E402
from matcher_b import apply_policy  # noqa: E402
from score import per_entity  # noqa: E402
from xlink import decoy_shift  # noqa: E402
import ens_second as es  # noqa: E402

OUT = R + "experiments/ens_v21/"
B = R + "v21e_bundle/v18_model/"
W = R + "work_v21"
POL = {"name": "dta_own", "missed": 0.031}
PARAMS = {"objective": "binary", "learning_rate": 0.05, "num_leaves": 31, "min_data_in_leaf": 2000,
          "feature_fraction": 0.8, "bagging_fraction": 0.5, "bagging_freq": 1, "lambda_l2": 10.0,
          "verbosity": -1, "seed": 11, "deterministic": True, "force_row_wise": True, "num_threads": 12}
ROUNDS = 400
t0 = time.time()
log = lambda *a: print(*a, f"({time.time() - t0:.0f}s)", flush=True)  # noqa: E731

m = pl.read_parquet(B + "pipeline_a_train_s1_map.parquet").drop_nulls("s1_ours")
pool20 = (pl.read_parquet(R + "experiments/v21_fault/pool.parquet", columns=["s1", "t", "p", "fold", "country"])
          .join(m.select(s1="s1_theirs"), on="s1", how="semi"))
fold = pool20.select("s1", "fold", "country").unique("s1")
p18 = (pl.scan_parquet(B + "train_scores_c_v18_oof.parquet").select("s1", "t", p18="p").filter(pl.col("p18") > 1e-4).collect()
       .join(m.select(s1="s1_ours", s1n="s1_theirs"), on="s1").select(s1="s1n", t="t", p18="p18"))
p20 = pool20.select("s1", "t", p20="p")
j = p20.join(p18, on=["s1", "t"], how="full", coalesce=True)
z = 0.5 * es._lg("p20") + 0.5 * es._lg("p18")
j = j.with_columns(p=pl.when(pl.col("p20").is_null()).then(pl.col("p18")).when(pl.col("p18").is_null()).then(pl.col("p20"))
                   .otherwise(1 / (1 + (-z).exp())))
log(f"shared S1 {m.height:,}; pairs {j.height:,}: both {j.filter(pl.col('p20').is_not_null() & pl.col('p18').is_not_null()).height:,}, "
    f"only v20 {j.filter(pl.col('p18').is_null()).height:,}, only v18 {j.filter(pl.col('p20').is_null()).height:,}")
gt = pl.read_parquet(W + "/train/gt.parquet", columns=["s1", "t"]).join(m.select(s1="s1_theirs"), on="s1", how="semi")
ids = m.select(s1="s1_theirs").join(fold, on="s1", how="left")


def score(pred):
    """Per-S1 F0.5 of the predicted pairs `pred` against train truth, joined with each S1's fold and country."""
    return per_entity(ids.select("s1"), pred, gt).select("s1", "f").join(ids, on="s1")


def summ(f, tag):
    """Log the mean per-S1 F0.5 of the score frame `f` on fold 4 and on the dev folds, per country, labelled `tag`."""
    g = (f.filter(pl.col("fold").is_not_null()).with_columns(sp=pl.when(pl.col("fold") == 4).then(pl.lit("f4")).otherwise(pl.lit("dev")))
         .group_by("sp", "country").agg(pl.col("f").mean()).sort("sp", "country"))
    tot = f.filter(pl.col("fold") == 4)["f"].mean()
    log(f"{tag:34s} f4 {tot:.5f} " + "  ".join(f"{a}_{b} {v:.5f}" for a, b, v in g.iter_rows()))


def delta(f, base, tag):
    """Log the mean per-S1 F0.5 change of `f` vs `base` (with standard error and number of changed S1s) per split/country."""
    d = (f.join(base.select("s1", f0="f"), on="s1").filter(pl.col("fold").is_not_null()).with_columns(d=pl.col("f") - pl.col("f0"),
         sp=pl.when(pl.col("fold") == 4).then(pl.lit("f4")).otherwise(pl.lit("dev")))
         .group_by("sp", "country").agg(d=pl.col("d").mean(), se=pl.col("d").std() / pl.len().sqrt(), ch=(pl.col("d") != 0).sum()).sort("sp", "country"))
    log(f"{tag:34s} " + "  ".join(f"{a}_{b} {v * 1e5:+.1f}e-5 (se {s * 1e5:.1f}, n{c})" for a, b, v, s, c in d.iter_rows()))


raw = j.select("s1", "t", "p").filter(pl.col("p") > 1e-4)
summ(score(apply_policy(raw, POL)), "ensemble raw (ens_eval)")
tg = pl.read_parquet(W + "/train/tgt.parquet", columns=["t", "addr_blank", "is_domain"]).select("t", bw=pl.col("addr_blank") | pl.col("is_domain"))
orphan = tg.filter("bw").select("t").join(pl.read_parquet(W + "/train/gt.parquet", columns=["t"]).unique(), on="t", how="anti")
odds = pl.col("p") / (1 - pl.col("p")).clip(1e-12, None)
tl = (raw.join(orphan, on="t", how="anti").join(tg, on="t", how="left")
      .with_columns(p=pl.when(pl.col("bw").fill_null(False)).then(odds * 3 / (1 + odds * 3)).otherwise(pl.col("p"))).drop("bw"))
base = score(es.decide(tl, None, None))
summ(base, "ensemble test-like (bw x3)")
chk = score(apply_policy(tl, POL))
assert chk["f"].equals(base["f"]), "es.decide != apply_policy"
cfg = from_args(None, W)
quiet = lambda *a: None  # noqa: E731
base_sh = score(apply_policy(decoy_shift(cfg, "train", raw.join(orphan, on="t", how="anti"), 0.5, quiet)
                             .join(tg, on="t", how="left").with_columns(p=pl.when(pl.col("bw").fill_null(False)).then(odds * 3 / (1 + odds * 3)).otherwise(pl.col("p"))).drop("bw"), POL))
summ(base_sh, "ensemble test-like + shift 0.5 (v21e)")

feat = es.features(W, "train", tl, p20, p18)
feat.write_parquet(OUT + "ens_feat.parquet")
log(f"features {feat.height:,} x {feat.width}")
lab = pl.read_parquet(W + "/train/gt.parquet", columns=["s1", "t"]).with_columns(y=pl.lit(1.0))
y = (es.owned(tl).join(feat, on=["s1", "t"]).join(fold.select("s1", "fold"), on="s1")
     .join(lab, on=["s1", "t"], how="left").with_columns(pl.col("y").fill_null(0.0)))
fo = y["fold"].to_numpy()
yy = y["y"].to_numpy()
res = {}
for name, feats in (("with_p18_p20", es.FEATS), ("no_pair", [f for f in es.FEATS if f not in es.PAIR])):
    X, off = es.matrix(y, feats)
    out = np.zeros(len(yy))
    for f in (4, 0, 1, 2, 3):
        tr = (fo != 4) & (fo != f)
        te = fo == f
        mdl = lgb.train(PARAMS, lgb.Dataset(X[tr], yy[tr], init_score=off[tr], feature_name=feats), num_boost_round=ROUNDS)
        out[te] = 1 / (1 + np.exp(-(off[te] + mdl.predict(X[te], raw_score=True))))
        if f == 4:
            mdl.save_model(OUT + f"ens_second_{name}.txt")
            imp = sorted(zip(feats, mdl.feature_importance("gain")), key=lambda kv: -kv[1])
            log(f"{name} fold-4 model gain: " + ", ".join(f"{k} {v:.0f}" for k, v in imp[:14]))
    pc = y.select("s1", "t", pc=pl.Series(out))
    pc.write_parquet(OUT + f"ens_pc_{name}.parquet")
    adj = lambda s, pc=pc: s.join(pc, on=["s1", "t"], how="left").with_columns(p=pl.coalesce("pc", "p")).drop("pc")  # noqa: E731
    f_ = score(es.decide(tl, None, None, adj=adj))
    delta(f_, base, f"second[{name}] vs test-like")
    tl_nobw = tl  # shift after the correction (production order)
    sh = lambda s: decoy_shift(cfg, "train", s, 0.5, quiet)  # noqa: E731
    f_sh = score(es.decide(tl_nobw, None, None, adj=adj, shift=sh))
    delta(f_sh, base_sh, f"second[{name}]+shift vs v21e(+shift)")
log("done")
