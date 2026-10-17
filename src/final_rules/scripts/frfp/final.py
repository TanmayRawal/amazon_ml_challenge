"""French decoys (v22 fr_fp_remove): select the accepted French pairs to remove.

Rules on accepted pairs (pool_d2.parquet; d = target house number minus S1 house number):
- A: exact name at offset +13 or +21 with the legal form added (same signature as the decoy offsets +1...+11);
- B: added word at offset +13 or +21;
- C: typo'd name (swap_typo) at offset +1 or +2.
Keeps only pairs present in w20/final_pairs.parquet and records k, the number of predicted pairs of each affected S1.
Writes remove.parquet (s1, t) and remove_detail.parquet; final2.py then filters them.
"""
import polars as pl
W = r"work"
d = pl.read_parquet("pool_d2.parquet").filter(pl.col("acc"))
A = d.filter((pl.col("name")=="exact") & pl.col("d").is_in([13,21]) & (pl.col("leg")=="leg_add")).with_columns(rule=pl.lit("A_exact_off13_21_legadd"))
B = d.filter((pl.col("name")=="added") & pl.col("d").is_in([13,21])).with_columns(rule=pl.lit("B_added_off13_21"))
C = d.filter((pl.col("name")=="swap_typo") & pl.col("d").is_in([1,2])).with_columns(rule=pl.lit("C_swaptypo_off1_2"))
R = pl.concat([A,B,C]).select("s1","t","rule","d","leg","p")
fp = pl.read_parquet(rf"{W}\w20\final_pairs.parquet")
R = R.join(fp, on=["s1","t"], how="semi")
print(R.group_by("rule").len(), R.height, R.select(pl.struct("s1","t").n_unique()))
n = fp.join(R.select("s1").unique(), on="s1", how="semi").group_by("s1").agg(k=pl.len())
R = R.join(n, on="s1")
print("list size k of affected S1s:", R["k"].describe())
R.select("s1","t").write_parquet("remove.parquet"); R.write_parquet("remove_detail.parquet")
