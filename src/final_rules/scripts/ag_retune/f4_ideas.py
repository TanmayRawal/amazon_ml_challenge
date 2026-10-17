"""Fold-4 proxy: US/India decision-layer ideas vs baseline (single owner -> dta_select 0.031)."""
import sys, time
import numpy as np, polars as pl
sys.path.insert(0, r"work/w20/code_V21/business_entity_resolution/src")
from decide import dta_select
from score import per_entity
W = r"work"
t0 = time.time()
pool = pl.read_parquet(rf"{W}\ce\f4_pool.parquet").filter(pl.col("p") > 1e-4)
gt = pool.filter(pl.col("label") == 1).select("s1", "t")
ids = pool.select("s1", "country").unique("s1")
src = pl.read_parquet(rf"{W}\train\tgt.parquet", columns=["t", "src"])

def own(k=1.0):
    """Single-owner renormalisation of the fold-4 pool: p -> odds**k / (1 + sum of odds**k over the target's S1s)."""
    odds = (pl.col("p").clip(0, 1 - 1e-6) / (1 - pl.col("p").clip(0, 1 - 1e-6))) ** k
    return pool.select("s1", "t", "p").with_columns(p=odds / (1 + odds.sum().over("t")))

def score(pred):
    """Per-S1 F0.5 (f) and number of predicted pairs (n_pred) of `pred` against fold-4 labels, with country."""
    return per_entity(ids.select("s1"), pred.select("s1", "t"), gt).join(ids, on="s1").select("s1", "country", "f", "n_pred")

def cap_post(pred, s):
    """Cap applied after the decision: keep at most 5 S2 / 6 S3 predicted targets per S1 (highest p from `s` first)."""
    x = pred.select("s1", "t").join(s, on=["s1", "t"]).join(src, on="t")
    x = x.sort(["s1", "src", "p"], descending=[False, False, True]).with_columns(r=pl.int_range(pl.len()).over("s1", "src"))
    return x.filter(pl.col("r") < pl.when(pl.col("src") == 2).then(5).otherwise(6)).select("s1", "t")

def cap_pre(s):
    """Cap applied before the decision: keep only each S1's top 5 S2 / top 6 S3 candidates by p. Returns s1, t, p."""
    x = s.join(src, on="t").sort(["s1", "src", "p"], descending=[False, False, True]).with_columns(r=pl.int_range(pl.len()).over("s1", "src"))
    return x.filter(pl.col("r") < pl.when(pl.col("src") == 2).then(5).otherwise(6)).select("s1", "t", "p")

def single_guard(pred, s, thr):
    """Singleton guard: drop the prediction of S1s whose only predicted pair has p (from `s`) below `thr`."""
    x = pred.select("s1", "t").join(s, on=["s1", "t"])
    bad = x.group_by("s1").agg(n=pl.len(), pm=pl.col("p").max()).filter((pl.col("n") == 1) & (pl.col("pm") < thr)).select("s1")
    return pred.join(bad, on="s1", how="anti").select("s1", "t")

s1_ = own(1.0)
base_pred = dta_select(s1_.filter(pl.col("p") > 1e-4), 0.031).select("s1", "t")
base = score(base_pred).rename({"f": "f0", "n_pred": "n0"})
print(f"base ready {time.time()-t0:.0f}s", {c: round(base.filter(pl.col('country') == c)['f0'].mean(), 6) for c in ("US", "India")}, flush=True)
rows = []

def report(name, pred):
    """Print and record the per-country F0.5 change of `pred` vs the baseline (mean, s.e., z in 1e-5 units, changed
    S1s); an idea passes when both US and India have z > 2."""
    f = score(pred).join(base.select("s1", "f0", "n0"), on="s1").with_columns(d=pl.col("f") - pl.col("f0"))
    r = {"idea": name}
    for c in ("US", "India"):
        x = f.filter(pl.col("country") == c)
        m, se = x["d"].mean() * 1e5, x["d"].std() / np.sqrt(x.height) * 1e5
        r[f"{c}_d_e5"] = round(m, 2); r[f"{c}_se_e5"] = round(se, 2); r[f"{c}_z"] = round(m / se, 2) if se > 0 else 0.0
        r[f"{c}_nchg"] = int((x["d"] != 0).sum())
    r["pass"] = r["US_z"] > 2 and r["India_z"] > 2
    rows.append(r); print(r, f"{time.time()-t0:.0f}s", flush=True)

# (1) missed parameter, country-specific: evaluate each value; per-country deltas are independent (dta is per-S1)
for m in (0.02, 0.045):
    report(f"missed={m}", dta_select(s1_.filter(pl.col("p") > 1e-4), m))
# (2) power on single-owner renorm
for k in (0.8, 1.2):
    s = own(k)
    report(f"so_power={k}", dta_select(s.filter(pl.col("p") > 1e-4), 0.031))
# (3) per-S1 generator caps
report("cap_post(5 S2/6 S3)", cap_post(base_pred, s1_))
report("cap_pre(5 S2/6 S3)", dta_select(cap_pre(s1_).filter(pl.col("p") > 1e-4), 0.031))
# (4) singleton guard
for thr in (0.5, 0.6, 0.7):
    report(f"single_guard<{thr}", single_guard(base_pred, s1_, thr))
pl.DataFrame(rows).write_csv(r"work/scratch/ag_retune/f4_results.csv")
# diagnostics: how often base prediction exceeds caps / is a low-p singleton
x = base_pred.join(s1_, on=["s1", "t"]).join(src, on="t")
print("S1 x src over cap:", x.group_by("s1", "src").len().filter(pl.col("len") > pl.when(pl.col("src") == 2).then(5).otherwise(6)).height)
