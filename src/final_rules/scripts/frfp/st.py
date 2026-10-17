"""French decoys (v22), exploration: accepted typo'd (swap_typo) and added-word names per house-number offset in
[-25, 25] (pool_d2.parquet). Prints only; motivates rule C (typos at +1 / +2)."""
import polars as pl
d = pl.read_parquet("pool_d2.parquet")
a = d.filter(pl.col("acc") & pl.col("d").is_between(-25,25))
r = a.group_by("d").agg(st=(pl.col("name")=="swap_typo").sum(), ad=(pl.col("name")=="added").sum()).sort("d")
for row in r.iter_rows(): print(row, end="  ")
