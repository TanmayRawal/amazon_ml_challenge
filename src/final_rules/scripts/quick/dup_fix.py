"""v23 duplicate fix (double-claimed targets) and French tie band check.

For targets claimed by 2+ S1s in v22c.parquet, keeps the highest-p claimant and writes the others to
dup_remove.parquet (exact-probability ties and all-null groups are skipped). Then prints the agreement of the stop-word
tie posterior with model-claimed ties in the bands [0.85, 0.9) and [0.9, 1], and summarises fr_tie3_add.parquet
(French ties at posterior 0.85-0.9, p >= 0.2).
"""
import polars as pl
v = pl.read_parquet("v22c.parquet")
d = v.filter(pl.len().over("t") > 1).filter(~pl.col("p").is_null().all().over("t"))
d = d.with_columns(pmax=pl.col("p").max().over("t"), nmax=(pl.col("p") == pl.col("p").max().over("t")).sum().over("t"))
ties = d.filter(pl.col("nmax") > 1)["t"].n_unique()
rm = d.filter((pl.col("nmax") == 1) & (pl.col("p") < pl.col("pmax")) | ((pl.col("nmax") == 1) & pl.col("p").is_null()))
print("dup targets", d["t"].n_unique(), "exact-tie targets skipped", ties, "removals", rm.height)
rm.select("s1", "t").write_parquet("dup_remove.parquet")
ps = pl.read_parquet(r"work/scratch/fr_tie2/post_sw.parquet")
for lo, hi in ((0.85, 0.9), (0.9, 1.01)):
    c = ps.filter(pl.col("claimed") & ~pl.col("prev") & (pl.col("post") >= lo) & (pl.col("post") < hi))
    c2 = c.filter(pl.col("p") >= 0.2)
    print(f"band [{lo},{hi}) model-claimed non-prev ties: n={c.height} agree={c['agree'].mean():.3f} | p>=0.2: n={c2.height} agree={c2['agree'].mean():.3f}")
a = pl.read_parquet("fr_tie3_add.parquet"); print("tie3 adds", a.height, "mean post", round(a["post"].mean(), 4), "p median", round(a["p"].median(), 3))
