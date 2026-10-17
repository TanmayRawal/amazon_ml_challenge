"""French decoys (v22 fr_fp_remove), last step: keep only removals from S1s with 2+ predicted pairs (k >= 2), so no
S1 is emptied. Overwrites remove.parquet and remove_detail.parquet (-89 pairs in v22c)."""
import polars as pl
R = pl.read_parquet("remove_detail.parquet")
print("k==1:", R.filter(pl.col("k")==1).group_by("rule").len())
R = R.filter(pl.col("k")>=2)
print(R.group_by("rule").agg(n=pl.len(), k=pl.col("k").mean()))
R.select("s1","t").write_parquet("remove.parquet"); R.write_parquet("remove_detail.parquet")
print("final", R.height)
