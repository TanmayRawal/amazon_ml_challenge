"""Reproduce the v21e3b US/India decision from pipeline A's precomputed test features:
ensemble p (lp20/lp18) -> bw x3 -> single owner -> second-opinion correction -> [optional CE blend] -> decoy shift x0.5 -> dta_select."""
import sys, time
import numpy as np, polars as pl, lightgbm as lgb
sys.path.insert(0, r"work/w20/code_V21/business_entity_resolution/src")
from decide import dta_select
W = r"work"
STACK = ["num_conflict", "num_match", "num_near", "num_delta", "num_off", "num_s_only", "num_t_only", "sfx_conflict",
         "st_tset", "addr_wjac", "num_view", "num_new_min", "tok_view", "tok_new_min", "sa_same", "sa_other",
         "addr_eq_same", "sup_num_share", "s_num_share", "t_num_mode", "nm_edit", "name_tset", "leg_swap"]
CTX = ["l_rank", "l_sum_o", "l_nconf_o", "l_nconf_same", "l_nconf_oth", "l_t_n", "l_t_odds_o", "l_best_o"]
FEATS = ["in_band", "bw"] + STACK + ["xenc1", "xenc2", "xenc3"] + CTX + ["lp20", "lp18", "src_both"] + ["l_logit"]
lg = lambda c: (pl.col(c).clip(1e-6, 1 - 1e-6) / (1 - pl.col(c).clip(1e-6, 1 - 1e-6))).log()
sig = lambda e: 1 / (1 + (-e).exp())

def corrected():
    """Second-opinion corrected US/India test probabilities.

    From the precomputed v21e3 test features: ensemble logit (0.5 lp20 + 0.5 lp18, or the one available), odds x3 for
    blank/website targets, single-owner renormalisation, then the LightGBM correction (ens_second_with_p18_p20.txt)
    added to the logit. Returns s1, t, in_band, p.
    """
    t0 = time.time()
    f = pl.read_parquet(rf"{W}\w21\test_v21\test_feat_v21e3.parquet")
    z = pl.when(pl.col("src_both") == 3).then(0.5 * pl.col("lp20") + 0.5 * pl.col("lp18")).when(pl.col("src_both") == 1).then(pl.col("lp20")).otherwise(pl.col("lp18"))
    f = f.with_columns(p=sig(z))
    odds = pl.col("p").clip(0, 1 - 1e-6) / (1 - pl.col("p").clip(0, 1 - 1e-6))
    f = f.with_columns(p=pl.when(pl.col("bw") > 0.5).then(odds * 3 / (1 + odds * 3)).otherwise(pl.col("p")))      # bw x3
    f = f.filter(pl.col("p") > 1e-4)
    odds = pl.col("p").clip(0, 1 - 1e-6) / (1 - pl.col("p").clip(0, 1 - 1e-6))
    f = f.with_columns(p=odds / (1 + odds.sum().over("t")))                                                         # single owner
    f = f.with_columns(l_logit=lg("p").cast(pl.Float32))
    bst = lgb.Booster(model_file=r"src/ensemble/ens_second_with_p18_p20.txt")
    X = f.select(FEATS).to_numpy().astype(np.float32)
    pc = 1.0 / (1.0 + np.exp(-(f["l_logit"].to_numpy().astype(np.float64) + bst.predict(X, raw_score=True))))
    out = f.select("s1", "t", "in_band").with_columns(p=pl.Series(pc))
    print(f"corrected p for {out.height:,} pairs ({time.time() - t0:.0f}s)", flush=True)
    return out

def decoy_flags():
    """Load the address numbers and tokens of US/India test S1s and of all test targets (for the decoy shift)."""
    s1 = pl.read_parquet(rf"{W}\w20\test\s1.parquet", columns=["s1", "country", "addr_nums", "addr_toks"]).filter(pl.col("country").is_in(["US", "India"]))
    tg = pl.read_parquet(rf"{W}\w20\test\tgt.parquet", columns=["t", "addr_nums", "addr_toks"])
    return s1, tg

def shift_decide(s, r=0.5):
    """Decoy shift then set decision.

    Pairs (p > 0.01) whose first house numbers differ (not a reordering) while the street tokens have Jaccard >= 0.5
    get their odds multiplied by `r`; then dta_select (missed = 0.031) picks each S1's set. Returns s1, t.
    """
    s1, tg = decoy_flags()
    first = lambda c: pl.col(c).str.split(" ").list.first()
    toks = lambda c: pl.col(c).str.split(" ").list.eval(pl.element().filter(pl.element().str.len_chars() > 1))
    y = (s.filter(pl.col("p") > 0.01).select("s1", "t").join(s1, on="s1").join(tg, on="t", suffix="_t")
         .with_columns(a=toks("addr_toks"), b=toks("addr_toks_t"), n1=first("addr_nums"), n2=first("addr_nums_t"),
                       m1=pl.col("addr_nums").str.split(" "), m2=pl.col("addr_nums_t").str.split(" ")))
    jac = pl.col("a").list.set_intersection("b").list.len() / pl.col("a").list.set_union("b").list.len().clip(1, None)
    reorder = pl.col("m2").list.contains(pl.col("n1")) | pl.col("m1").list.contains(pl.col("n2"))
    flag = y.filter((pl.col("n1").fill_null("") != "") & (pl.col("n2").fill_null("") != "") & (pl.col("n1") != pl.col("n2")) & ~reorder & (jac >= 0.5)).select("s1", "t", shift=pl.lit(True))
    odds = pl.col("p") / (1.0 - pl.col("p")).clip(1e-12, None)
    s = s.join(flag, on=["s1", "t"], how="left").with_columns(p=pl.when(pl.col("shift")).then(odds * r / (1.0 + odds * r)).otherwise(pl.col("p"))).drop("shift")
    return dta_select(s.select("s1", "t", "p").filter(pl.col("p") > 1e-4), 0.031).select("s1", "t")

if __name__ == "__main__":
    c = corrected(); c.write_parquet(rf"{W}\ce\test_corrected.parquet")
    pred = shift_decide(c.select("s1", "t", "p"))
    pred.write_parquet(rf"{W}\ce\usin_repro.parquet")
    s1i = pl.read_parquet(rf"{W}\test\s1.parquet", columns=["s1", "entity_id", "country"])
    tgi = pl.read_parquet(rf"{W}\test\tgt.parquet", columns=["t", "entity_id"])
    ref = (pl.read_csv(rf"{W}\w20\submissions\v21e3b\submission_v21e3b\matching_results.tsv", separator="\t", quote_char=None, schema_overrides={"matched_entity_ids": pl.String})
           .with_columns(m=pl.col("matched_entity_ids").fill_null("").str.split(",")).explode("m").filter(pl.col("m") != "")
           .join(s1i.select("s1", source1_entity_id="entity_id", country="country"), on="source1_entity_id").filter(pl.col("country") != "France")
           .join(tgi.select("t", m="entity_id"), on="m").select("s1", "t"))
    print(f"repro {pred.height:,} vs v21e3b US/India {ref.height:,}: only repro {pred.join(ref, on=['s1','t'], how='anti').height:,}, only v21e3b {ref.join(pred, on=['s1','t'], how='anti').height:,}")
