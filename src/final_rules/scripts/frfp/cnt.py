"""French decoys (v22), exploration: copies per S1 by source.

Compares the distribution of S2 / S3 copies per S1 on US/India train truth (gt.parquet) with the accepted French test
pairs (fr_acc.parquet), to see whether France is over-predicting copies per S1. Prints only.
"""
import polars as pl
pl.Config.set_tbl_rows(80); pl.Config.set_tbl_width_chars(250)
W = r"work"
gt = pl.read_parquet(rf"{W}\train\gt.parquet"); print(gt.schema)
ttg = pl.read_parquet(rf"{W}\train\tgt.parquet", columns=["t","src"])
ts1 = pl.read_parquet(rf"{W}\train\s1.parquet", columns=["s1","country"])
g = gt.join(ttg, on="t").group_by("s1").agg(n2=(pl.col("src")==2).sum(), n3=(pl.col("src")==3).sum())
g = ts1.join(g, on="s1", how="left").fill_null(0)
print(g.group_by("country","n2","n3").len().sort("country","len", descending=[False,True]).group_by("country").head(15))
print(g.group_by("country").agg(pl.col("n2").mean(), pl.col("n3").mean(), (pl.col("n2")>=3).mean().alias("n2ge3"), (pl.col("n3")>=3).mean().alias("n3ge3")))
x = pl.read_parquet("fr_acc.parquet")
s1 = pl.read_parquet(rf"{W}\test\s1.parquet", columns=["s1","country"]).filter(pl.col("country")=="France")
h = x.group_by("s1").agg(n2=(pl.col("src")==2).sum(), n3=(pl.col("src")==3).sum())
h = s1.join(h, on="s1", how="left").fill_null(0)
print(h.group_by("n2","n3").len().sort("len", descending=True).head(25))
print(h.select(pl.col("n2").mean(), pl.col("n3").mean(), (pl.col("n2")>=3).mean().alias("n2ge3"), (pl.col("n3")>=3).mean().alias("n3ge3")))
print("n2 dist", h.group_by("n2").len().sort("n2")); print("n3 dist", h.group_by("n3").len().sort("n3"))
print("train n2 dist", g.group_by("country","n2").len().sort("country","n2")); print("train n3", g.group_by("country","n3").len().sort("country","n3"))
