"""French decoys (v22), exploration: legal-form drop/add rates.

For accepted exact-name French pairs (fr_acc.parquet), prints how often the legal form is dropped or added, by
whether the S1 has a legal form, source and number class; and the predicted copies per French S1 with / without a
legal form next to the US/India train analog (true copies per S1). Prints only.
"""
import polars as pl
pl.Config.set_tbl_rows(60); pl.Config.set_tbl_width_chars(250)
W = r"work"
x = pl.read_parquet("fr_acc.parquet").with_columns(s1leg=pl.col("legal").fill_null("")!="")
s1 = pl.read_parquet(rf"{W}\test\s1.parquet", columns=["s1","country","legal"]).filter(pl.col("country")=="France").with_columns(s1leg=pl.col("legal").fill_null("")!="")
print("drop/add rate by src & num:")
print(x.filter(pl.col("name")=="exact").group_by("s1leg","src","num").agg(n=pl.len(), chg=(pl.col("leg").is_in(["leg_drop","leg_add"])).mean()).sort("s1leg","src","num"))
h = x.group_by("s1").agg(n=pl.len(), n2=(pl.col("src")==2).sum(), n3=(pl.col("src")==3).sum())
h = s1.join(h, on="s1", how="left").fill_null(0)
print(h.group_by("s1leg").agg(pl.len(), pl.col("n").mean(), pl.col("n2").mean(), pl.col("n3").mean(), (pl.col("n")==0).mean().alias("zero")))
# train analog: US/India legal present?
ts1 = pl.read_parquet(rf"{W}\train\s1.parquet", columns=["s1","country","legal"]).with_columns(s1leg=pl.col("legal").fill_null("")!="")
gt = pl.read_parquet(rf"{W}\train\gt.parquet")
g = gt.group_by("s1").agg(n=pl.len())
print(ts1.join(g, on="s1", how="left").fill_null(0).group_by("country","s1leg").agg(pl.len(), pl.col("n").mean()))
