"""French decoys (v22), exploration: S1-S1 pairs with the same name and street.

Self-joins French test S1s on core name and street tokens and prints the house-number difference distribution and
legal-form relation, to check whether the decoy offsets also occur between genuine distinct S1s. Prints only.
"""
import polars as pl
pl.Config.set_tbl_rows(100); pl.Config.set_tbl_width_chars(250); pl.Config.set_fmt_str_lengths(50)
W = r"work"
s1 = pl.read_parquet(rf"{W}\test\s1.parquet", columns=["s1","country","business_name","business_address","core","core_key","legal","addr_nums","addr_toks"]).filter(pl.col("country")=="France")
s1 = s1.with_columns(n0=pl.col("addr_nums").fill_null("").str.split(" ").list.first().cast(pl.Int64, strict=False),
                     leg=pl.col("legal").fill_null(""))
print(s1.select(pl.col("addr_toks").head(5)))
# same street tokens, exact name
a = s1.join(s1, on=["core_key","addr_toks"], suffix="_b").filter(pl.col("s1") < pl.col("s1_b"))
a = a.with_columns(d=(pl.col("n0_b")-pl.col("n0")).abs(),
    lrel=pl.when(pl.col("leg")==pl.col("leg_b")).then(pl.lit("eq")).when((pl.col("leg")=="")|(pl.col("leg_b")=="")).then(pl.lit("adddrop")).otherwise(pl.lit("swap")))
print("S1-S1 same name+street:", a.height)
print(a.group_by("d").agg(n=pl.len(), eq=(pl.col("lrel")=="eq").mean(), ad=(pl.col("lrel")=="adddrop").mean(), sw=(pl.col("lrel")=="swap").mean()).sort("d").head(30))
print(a.filter(pl.col("d")==0).select("business_name","business_address","business_name_b","business_address_b").head(20))
