"""French decoys (v22), sanity check: entity IDs of test S1s and targets are unique and never shared. Prints only."""
import polars as pl
W = r"work"
s1 = pl.read_parquet(rf"{W}\test\s1.parquet", columns=["s1","entity_id","country"])
tg = pl.read_parquet(rf"{W}\test\tgt.parquet", columns=["t","entity_id"])
print(s1.head(3), tg.head(3))
print(s1["entity_id"].n_unique(), s1.height, tg["entity_id"].n_unique(), tg.height)
print(s1.join(tg, on="entity_id").height)
