"""French decoys (v22), exploration: accepted pairs by offset group and legal-form relation.

Groups same-street, number-substituted pool pairs (pool_d.parquet) into decoy offsets, non-decoy positive (6-20),
negative and other, and prints the legal-form mix and mean p of accepted exact and non-exact pairs, with a focus on
offsets +13 / +21 (candidates for rule A in final.py). Prints only.
"""
import polars as pl
pl.Config.set_tbl_rows(100); pl.Config.set_tbl_width_chars(250); pl.Config.set_fmt_str_lengths(50)
x = pl.read_parquet("pool_d.parquet").filter(pl.col("num").is_in(["num_sub","num_trunc"]) & (pl.col("st")=="st"))
DEC = [1,2,3,4,5,7,9,11,13,21]
x = x.with_columns(grp=pl.when(pl.col("d").is_in(DEC)).then(pl.lit("decoy")).when(pl.col("d").is_between(6,20)).then(pl.lit("pos_nondec")).when(pl.col("d").is_between(-20,-1)).then(pl.lit("neg")).otherwise(pl.lit("other")))
a = x.filter(pl.col("acc"))
print("EXACT accepted by grp/d leg:")
print(a.filter(pl.col("exact")).group_by("grp").agg(n=pl.len(), leg_eq=(pl.col("leg")=="leg_eq").mean(), drop=(pl.col("leg")=="leg_drop").mean(), add=(pl.col("leg")=="leg_add").mean(), swap=(pl.col("leg")=="leg_swap").mean(), p=pl.col("p").mean()))
print(a.filter(pl.col("exact") & pl.col("d").is_in([13,21])).group_by("d","leg").agg(n=pl.len(), p=pl.col("p").mean()).sort("d","leg"))
print("pool exact d13/21 leg:", x.filter(pl.col("exact") & pl.col("d").is_in([13,21])).group_by("d","leg").agg(n=pl.len(), acc=pl.col("acc").sum(), p=pl.col("p").mean()).sort("d","leg"))
print("pool exact pos_nondec leg:", x.filter(pl.col("exact") & (pl.col("grp")=="pos_nondec")).group_by("leg").agg(n=pl.len(), acc=pl.col("acc").sum(), p=pl.col("p").mean()).sort("leg"))
print("pool exact decoy d<=11 leg:", x.filter(pl.col("exact") & pl.col("d").is_in(DEC[:8])).group_by("leg").agg(n=pl.len(), p=pl.col("p").mean(), pgt5=(pl.col("p")>0.5).sum()).sort("leg"))
print("NONEXACT accepted:")
print(a.filter(~pl.col("exact")).group_by("grp","name").agg(n=pl.len(), leg_eq=(pl.col("leg")=="leg_eq").mean(), p=pl.col("p").mean()).sort("grp","name"))
print(x.filter(~pl.col("exact")).group_by("grp","name").agg(n=pl.len(), acc=pl.col("acc").sum(), leg_eq=(pl.col("leg")=="leg_eq").mean()).sort("grp","name"))
