"""French decoys (v22), examples: accepted exact-name copies at a non-decoy positive house-number offset (6-20) with
the same legal form, for S1s that also have a same-number copy. Prints each S1 with its top-10 candidates (pool_d2,
w20/fr_pool_classes.parquet) for manual inspection.
"""
import polars as pl
pl.Config.set_tbl_rows(80); pl.Config.set_tbl_width_chars(250); pl.Config.set_fmt_str_lengths(55)
W = r"work"
d = pl.read_parquet("pool_d2.parquet")
x = pl.read_parquet("fr_acc.parquet")
cp = pl.read_parquet(rf"{W}\w20\fr_pool_classes.parquet")
tg = pl.read_parquet(rf"{W}\test\tgt.parquet", columns=["t","business_name","business_address","src"])
s1 = pl.read_parquet(rf"{W}\test\s1.parquet", columns=["s1","business_name","business_address"])
pick = d.filter(pl.col("exact") & (pl.col("g")=="pos") & pl.col("acc") & pl.col("has_eq") & (pl.col("leg")=="leg_eq") & pl.col("d").is_between(6,20)).sample(6, seed=3)
for r in pick.iter_rows(named=True):
    print("S1:", s1.filter(pl.col("s1")==r["s1"]).select("business_name","business_address").row(0), "d=", r["d"])
    print(cp.filter(pl.col("s1")==r["s1"]).join(tg, on="t").select("t","src","business_name","business_address","name","num","leg","p","acc").sort("p", descending=True).head(10))
