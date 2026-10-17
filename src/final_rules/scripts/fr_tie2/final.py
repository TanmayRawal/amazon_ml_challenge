"""French ties by stop words (v22), final step: select the fr_tie2_add pairs.

Compares the stop-word tie posterior (post_sw.parquet from nb2.py) with the baseline formatting posterior
(w20/fr_tie_nb.parquet) and the earlier tie additions (w20/fr_tie_add.parquet): agreement with model-claimed ties,
and whether earlier additions flip. Writes add.parquet: unclaimed tie targets not added before, with posterior >= 0.9
and model p >= 0.2 (+582 pairs in v22c, fr_tie2_add).
"""
import polars as pl
W = r"work"
r1 = pl.read_parquet("post_sw.parquet")
b = pl.read_parquet(rf"{W}\w20\fr_tie_nb.parquet").select("t", pl.col("s1").alias("s1_base"), pl.col("post").alias("post_base"))
prev = pl.read_parquet(rf"{W}\w20\fr_tie_add.parquet").select("t", pl.col("s1").alias("s1_prev"))
r = r1.join(b, on="t").join(prev, on="t", how="left")
# honest agreement: claimed ties excluding prev adds
c = r.filter(pl.col("claimed") & ~pl.col("prev"))
for th in (0.75, 0.9):
    q = c.filter(pl.col("post")>=th); print(f"model-claimed (excl prev) post>={th}: n={q.height} agree={q['agree'].mean():.4f}")
    d = q.filter(pl.col("post_base")<0.9); print(f"   of which baseline post<0.9 (stopword decisive): n={d.height} agree={d['agree'].mean():.4f}")
    d = q.filter(pl.col("s1")!=pl.col("s1_base")); print(f"   argmax changed vs baseline: n={d.height} agree={d['agree'].mean() if d.height else float('nan'):.4f}")
# prev adds flipped by new model?
pv = r.filter(pl.col("s1_prev").is_not_null())
print("prev adds:", pv.height, "new argmax differs:", (pv["s1"]!=pv["s1_prev"]).sum(), "new post<0.9:", (pv["post"]<0.9).sum())
add = r.filter(~pl.col("claimed") & ~pl.col("prev") & (pl.col("post")>=0.9) & (pl.col("p")>=0.2))
print("adds:", add.height, "k dist:", add.join(pl.read_parquet(rf"{W}\w20\fr_tie_nb.parquet").select("t","k"), on="t")["k"].describe())
print("post quantiles", add["post"].quantile(0.1), add["post"].median(), "p median", add["p"].median())
assert add["t"].n_unique()==add.height
add.select(pl.col("s1").cast(pl.UInt32), pl.col("t").cast(pl.UInt32), "post", "p").write_parquet("add.parquet")
nop = r.filter(~pl.col("claimed") & ~pl.col("prev") & (pl.col("post")>=0.9))
print("without p filter:", nop.height, "p<0.2:", (nop["p"]<0.2).sum(), "p null:", nop["p"].is_null().sum())
# examples
s1 = pl.read_parquet(rf"{W}\test\s1.parquet", columns=["s1","business_name"]); tg = pl.read_parquet(rf"{W}\test\tgt.parquet", columns=["t","business_name"])
pl.Config.set_fmt_str_lengths(50); pl.Config.set_tbl_width_chars(200)
print(add.sample(8, seed=2).join(tg, on="t").join(s1, on="s1", suffix="_s1").select("business_name","business_name_s1","post","p"))
