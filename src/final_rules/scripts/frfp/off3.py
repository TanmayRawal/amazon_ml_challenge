"""French decoys (v22), exploration: accepted non-exact names at decoy offsets.

Prints, per decoy offset, the accepted non-exact pairs by name class with examples (added word, typo, swapped word),
and scans all offsets for spikes of exact names with the legal form added. Supports rules B and C in final.py.
Prints only.
"""
import polars as pl
pl.Config.set_tbl_rows(100); pl.Config.set_tbl_width_chars(250); pl.Config.set_fmt_str_lengths(45)
W = r"work"
x = pl.read_parquet("pool_d.parquet").filter(pl.col("num").is_in(["num_sub","num_trunc"]) & (pl.col("st")=="st"))
DEC = [1,2,3,4,5,7,9,11,13,21]
s1 = pl.read_parquet(rf"{W}\test\s1.parquet", columns=["s1","business_name","business_address"])
tg = pl.read_parquet(rf"{W}\test\tgt.parquet", columns=["t","business_name","business_address","src"])
a = x.filter(pl.col("acc") & ~pl.col("exact") & pl.col("d").is_in(DEC)).join(s1, on="s1").join(tg, on="t", suffix="_t")
print(a.group_by("d","name").len().pivot(on="name", index="d", values="len").sort("d"))
for nm in ["added","swap_typo","swap_word"]:
    print(nm); print(a.filter(pl.col("name")==nm).select("business_name","business_address","business_name_t","business_address_t","d","leg","p").head(14))
# exact: scan all d for leg_add spikes
e = x.filter(pl.col("exact")).group_by("d").agg(n=pl.len(), add=(pl.col("leg")=="leg_add").sum(), acc=pl.col("acc").sum(), acc_add=(pl.col("acc") & (pl.col("leg")=="leg_add")).sum()).sort("n", descending=True)
print(e.filter(pl.col("add")>=15).sort("d"))
