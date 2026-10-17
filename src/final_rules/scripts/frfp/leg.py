"""French decoys (v22), exploration: legal-form transition mix.

Prints the mix of legal-form relations (equal / added / dropped / swapped) by whether the S1 has a legal form, for
accepted French pairs in several classes (blank address, same number, no number) and for the same-number exact pool.
Prints only.
"""
import polars as pl
pl.Config.set_tbl_rows(100); pl.Config.set_tbl_width_chars(250); pl.Config.set_fmt_str_lengths(50)
W = r"work"
x = pl.read_parquet("fr_acc.parquet").with_columns(s1leg=pl.col("legal").fill_null("")!="")
cp = pl.read_parquet(rf"{W}\w20\fr_pool_classes.parquet")
s1 = pl.read_parquet(rf"{W}\test\s1.parquet", columns=["s1","legal"])
cp = cp.join(s1, on="s1").with_columns(s1leg=pl.col("legal").fill_null("")!="")
def mix(df, label):
    """Print, under `label`, the share of each legal-form relation within S1s with and without a legal form."""
    r = df.group_by("s1leg","leg").len().with_columns(f=pl.col("len")/pl.col("len").sum().over("s1leg")).sort("s1leg","leg")
    print(label); print(r)
mix(x.filter(pl.col("num")=="blank"), "ACC blank-address, all names")
mix(x.filter((pl.col("num")=="blank") & (pl.col("name")=="exact")), "ACC blank exact")
mix(x.filter((pl.col("num")=="num_eq") & (pl.col("name")=="exact") & (pl.col("st")=="st")), "ACC num_eq exact st")
mix(x.filter((pl.col("num")=="num_eq") & (pl.col("name")!="exact") & (pl.col("st")=="st")), "ACC num_eq nonexact st")
mix(x.filter((pl.col("num")=="nonum") & (pl.col("name")=="exact")), "ACC nonum exact")
mix(cp.filter((pl.col("num")=="num_eq") & (pl.col("name")=="exact") & (pl.col("st")=="st")), "POOL num_eq exact st")
print(cp.filter((pl.col("num")=="num_eq") & (pl.col("name")=="exact") & (pl.col("st")=="st")).group_by("s1leg","leg").agg(n=pl.len(), acc=pl.col("acc").mean(), p=pl.col("p").mean()).sort("s1leg","leg"))
