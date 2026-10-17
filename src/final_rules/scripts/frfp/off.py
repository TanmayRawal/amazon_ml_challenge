"""French decoys (v22), step 2: house-number offsets of French pool pairs.

Adds d = first target house number minus first S1 house number to the French candidate pool with pair classes
(w20/fr_pool_classes.parquet) and writes pool_d.parquet. Prints, for same-street pairs with a substituted/truncated
number, pool size, accepted count, mean p and legal-form agreement per offset |d| <= 40 (exact vs non-exact names),
which exposes the decoy offsets.
"""
import polars as pl
pl.Config.set_tbl_rows(100); pl.Config.set_tbl_width_chars(250)
W = r"work"
cp = pl.read_parquet(rf"{W}\w20\fr_pool_classes.parquet")
print(cp.height, cp.schema)
s1 = pl.read_parquet(rf"{W}\test\s1.parquet", columns=["s1", "addr_nums"])
tg = pl.read_parquet(rf"{W}\test\tgt.parquet", columns=["t", "addr_nums"])
x = cp.join(s1, on="s1").join(tg, on="t", suffix="_t")
f1 = lambda c: pl.col(c).fill_null("").str.split(" ").list.first().cast(pl.Int64, strict=False)
x = x.with_columns(d=f1("addr_nums_t") - f1("addr_nums"), exact=pl.col("name")=="exact")
x.write_parquet("pool_d.parquet")
r = (x.filter(pl.col("num").is_in(["num_sub","num_trunc"]) & (pl.col("st")=="st"))
      .group_by("exact","d").agg(n=pl.len(), nacc=pl.col("acc").sum(), p=pl.col("p").mean(), leg_eq=(pl.col("leg")=="leg_eq").mean())
      .filter(pl.col("d").abs()<=40).sort("exact","d"))
print(r.filter(pl.col("exact")))
print(r.filter(~pl.col("exact")))
