"""Singleton guard (v24), application to US/India test.

Recomputes the US/India decision from the second-opinion corrected test probabilities (ce/test_corrected.parquet)
with the same decoy shift as usin_decide.shift_decide, checks it against the reproduced rows (ce/usin_repro.parquet),
counts the S1s the guard would empty at thresholds 0.5/0.6/0.7, and writes the reproduced pairs minus the S1s whose
only predicted pair has corrected p < 0.6 (usin_pred_CANDIDATE_guard06.parquet; -267 pairs in v24).
"""
import sys, polars as pl
sys.path.insert(0, r"work/w20")
sys.path.insert(0, r"work/w20/code_V21/business_entity_resolution/src")
import usin_decide as U
from decide import dta_select
W = r"work"
c = pl.read_parquet(rf"{W}\ce\test_corrected.parquet").select("s1", "t", "p")
# same shift as U.shift_decide, but keep shifted p
s1, tg = U.decoy_flags()
first = lambda k: pl.col(k).str.split(" ").list.first()
toks = lambda k: pl.col(k).str.split(" ").list.eval(pl.element().filter(pl.element().str.len_chars() > 1))
y = (c.filter(pl.col("p") > 0.01).select("s1", "t").join(s1, on="s1").join(tg, on="t", suffix="_t")
     .with_columns(a=toks("addr_toks"), b=toks("addr_toks_t"), n1=first("addr_nums"), n2=first("addr_nums_t"), m1=pl.col("addr_nums").str.split(" "), m2=pl.col("addr_nums_t").str.split(" ")))
jac = pl.col("a").list.set_intersection("b").list.len() / pl.col("a").list.set_union("b").list.len().clip(1, None)
reorder = pl.col("m2").list.contains(pl.col("n1")) | pl.col("m1").list.contains(pl.col("n2"))
flag = y.filter((pl.col("n1").fill_null("") != "") & (pl.col("n2").fill_null("") != "") & (pl.col("n1") != pl.col("n2")) & ~reorder & (jac >= 0.5)).select("s1", "t", shift=pl.lit(True))
odds = pl.col("p") / (1.0 - pl.col("p")).clip(1e-12, None)
s = c.join(flag, on=["s1", "t"], how="left").with_columns(p=pl.when(pl.col("shift")).then(odds * 0.5 / (1.0 + odds * 0.5)).otherwise(pl.col("p"))).drop("shift")
pred = dta_select(s.filter(pl.col("p") > 1e-4), 0.031).select("s1", "t")
ref = pl.read_parquet(rf"{W}\ce\usin_repro.parquet").select("s1", "t")
print("recomputed vs usin_repro: only new", pred.join(ref, on=["s1", "t"], how="anti").height, "only repro", ref.join(pred, on=["s1", "t"], how="anti").height, flush=True)
x = ref.join(s, on=["s1", "t"])
cnt = x.join(s1.select("s1", "country"), on="s1").group_by("s1", "country").agg(n=pl.len(), pm=pl.col("p").max())
print("test US/India S1s in s1 table:", s1.group_by("country").len().to_dicts())
for th in (0.5, 0.6, 0.7):
    print("guard", th, cnt.filter((pl.col("n") == 1) & (pl.col("pm") < th)).group_by("country").len().to_dicts())
bad = cnt.filter((pl.col("n") == 1) & (pl.col("pm") < 0.6)).select("s1")
out = ref.join(bad, on="s1", how="anti")
out.write_parquet("usin_pred_CANDIDATE_guard06.parquet")
print("candidate pairs", out.height, "removed", ref.height - out.height, "added 0")
