"""French decoys (v22), step 3: offset groups and same-number copies.

For French pool pairs at a substituted/truncated house number on the same street (pool_d.parquet), flags whether the
S1 also has an accepted same-number copy (has_eq) and groups the offset d into decoy offsets (1,2,3,4,5,7,9,11,13,21),
other positive, negative. Writes pool_d2.parquet (used by ex.py, st.py, final.py) and prints the breakdown.
"""
import polars as pl
pl.Config.set_tbl_rows(80); pl.Config.set_tbl_width_chars(250)
W = r"work"
fc = pl.read_parquet(rf"{W}\w20\final_classes.parquet").filter(pl.col("country")=="France")
eqs = fc.filter(pl.col("num")=="num_eq").group_by("s1").agg(neq=pl.len())
d = pl.read_parquet("pool_d.parquet").filter(pl.col("num").is_in(["num_sub","num_trunc"]) & (pl.col("st")=="st"))
d = d.with_columns(cs=pl.len().over("s1","d")).join(eqs, on="s1", how="left").with_columns(has_eq=pl.col("neq").fill_null(0)>0)
DEC=[1,2,3,4,5,7,9,11,13,21]
d = d.with_columns(g=pl.when(pl.col("d").is_in(DEC)).then(pl.lit("dec")).when(pl.col("d")<0).then(pl.lit("neg")).when(pl.col("d")>0).then(pl.lit("pos")).otherwise(pl.lit("na")))
d.write_parquet("pool_d2.parquet")
print(d.group_by("exact","g","leg").agg(n=pl.len(), has_eq=pl.col("has_eq").mean(), acc=pl.col("acc").sum(), acc_has_eq=pl.col("has_eq").filter(pl.col("acc")).mean()).sort("exact","g","leg"))
print(d.filter(~pl.col("exact")).group_by("g","name").agg(n=pl.len(), has_eq=pl.col("has_eq").mean(), acc=pl.col("acc").sum(), acc_has_eq=pl.col("has_eq").filter(pl.col("acc")).mean()).sort("g","name"))
