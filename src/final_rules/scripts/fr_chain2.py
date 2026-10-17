"""Run the approximate France chain R(p) for V20 p and ensemble p (w = weight of the V20 logit), then derive the delta."""
import sys, os, time
import polars as pl
sys.path.insert(0, r"work/w20/code_V21/business_entity_resolution/src")
from matcher_b import apply_policy
W = r"work"
OUT = rf"{W}\w20\frens"
POLICY = {"name": "dta_own", "missed": 0.031}
cls = pl.read_parquet(rf"{OUT}\classes.parquet").select("s1", "t", "vocab", "kill", "sufB", "floor")
j = pl.read_parquet(rf"{W}\w20\fr_ens_pool.parquet")
extra = sys.argv[1] if len(sys.argv) > 1 else None          # optional third score file (s1,t,p) e.g. transformer
if extra:
    j = j.join(pl.read_parquet(extra).select("s1", "t", p3="p"), on=["s1", "t"], how="left")
lg = lambda c: (pl.col(c).clip(1e-6, 1 - 1e-6) / (1 - pl.col(c).clip(1e-6, 1 - 1e-6))).log()
def chain(p_expr, tag):
    """Run the approximate France rule chain on the probability expression `p_expr` and save it as R_<tag>.parquet.

    Vocabulary swaps and audit kills get p = 0, suffix copies (sufB) get odds x5, list-word floors get p >= 0.9, then
    the v20 dta_own decision is applied. Returns the predicted pairs (s1, t).
    """
    t0 = time.time()
    sc = j.select("s1", "t", p=p_expr).filter(pl.col("p").is_not_null() & (pl.col("p") > 1e-4))
    sc = sc.join(cls, on=["s1", "t"], how="left").with_columns([pl.col(c).fill_null(False) for c in ["vocab", "kill", "sufB", "floor"]])
    odds = pl.col("p") / (1 - pl.col("p")).clip(1e-12, None)
    sc = sc.with_columns(p=pl.when(pl.col("vocab") | pl.col("kill")).then(0.0)
                           .when(pl.col("sufB")).then(odds * 5 / (1 + odds * 5)).otherwise(pl.col("p")))
    sc = sc.with_columns(p=pl.when(pl.col("floor") & ~pl.col("kill") & ~pl.col("vocab")).then(pl.max_horizontal("p", pl.lit(0.9))).otherwise(pl.col("p")))
    pred = apply_policy(sc.filter(pl.col("p") > 1e-4).select("s1", "t", "p"), POLICY)
    pred.write_parquet(rf"{OUT}\R_{tag}.parquet")
    print(f"R[{tag}]: {pred.height} pairs ({time.time() - t0:.0f}s)", flush=True)
    return pred
A = chain(pl.col("p20"), "v20")
for w in (0.5, 0.7):
    z = w * lg("p20") + (1 - w) * lg("p18")
    chain(pl.when(pl.col("p18").is_null()).then(pl.col("p20")).when(pl.col("p20").is_null()).then(pl.col("p18")).otherwise(1 / (1 + (-z).exp())), f"w{w}")
    # SAFE variant: V20 pair universe only (v18-only pairs are ignored)
    chain(pl.when(pl.col("p20").is_null()).then(None).when(pl.col("p18").is_null()).then(pl.col("p20")).otherwise(1 / (1 + (-z).exp())), f"w{w}_v20univ")
