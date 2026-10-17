"""Build v22c pairs (+ model p via logit average of the pipeline A and pipeline B stacks)."""
import polars as pl
W = r"work"
def rd(f):
    """Read the (s1, t) columns of parquet file `f` as UInt32."""
    return pl.read_parquet(f).select(pl.col("s1").cast(pl.UInt32), pl.col("t").cast(pl.UInt32))
fp = rd(rf"{W}\w20\final_pairs.parquet")
adds = [rd(rf"{W}\w20\{f}.parquet") for f in ("acro_add", "fr_tie_add", "fr_tie2_add")]
rem = rd(rf"{W}\w20\fr_fp_remove.parquet")
v = pl.concat([fp] + adds).unique().join(rem, on=["s1", "t"], how="anti")
print("final_pairs", fp.height, "adds", [a.height for a in adds], "new from adds", pl.concat(adds).unique().join(fp, on=["s1","t"], how="anti").height, "rem", rem.height, "rem present", rem.join(pl.concat([fp]+adds), on=["s1","t"], how="semi").height, "v22c", v.height)
def lg(c):
    """Polars expression for the logit of probability column `c`, clipped to [1e-6, 1 - 1e-6]."""
    return (pl.col(c).clip(1e-6, 1 - 1e-6) / (1 - pl.col(c).clip(1e-6, 1 - 1e-6))).log()
def attach(x):
    """Attach model p to pairs `x`: logit average of the pipeline A V20 stack (p_them) and the pipeline B v18 stack (p_ours),
    or whichever one exists; p is null when neither scored the pair."""
    k = x.select("s1", "t").lazy()
    a = pl.scan_parquet(rf"{W}\w20\test\scores_c.parquet").select("s1", "t", p_them="p").join(k, on=["s1", "t"], how="semi").collect()
    b = pl.scan_parquet(rf"{W}\test\scores_c.parquet").select("s1", "t", p_ours="p").join(k, on=["s1", "t"], how="semi").collect()
    x = x.join(a, on=["s1", "t"], how="left").join(b, on=["s1", "t"], how="left")
    z = pl.when(pl.col("p_them").is_not_null() & pl.col("p_ours").is_not_null()).then((lg("p_them") + lg("p_ours")) / 2) \
          .when(pl.col("p_them").is_not_null()).then(lg("p_them")).when(pl.col("p_ours").is_not_null()).then(lg("p_ours"))
    return x.with_columns(p=1 / (1 + (-z).exp()))
if __name__ == "__main__":
    v = attach(v)
    print(v.select(pl.len(), pl.col("p").is_null().sum().alias("p_null"), pl.col("p_them").is_null().sum().alias("them_null"), pl.col("p_ours").is_null().sum().alias("ours_null")))
    v.write_parquet("v22c.parquet")
