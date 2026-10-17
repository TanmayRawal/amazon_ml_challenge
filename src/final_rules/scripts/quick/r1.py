"""v23 recovered US/India pairs (R1): de-duplicate against v22c and the domain rule.

R1 pairs (never retrieved, name >= 90, same numbers and street, unique S1; usinfn/add.parquet) are checked against the
pairs and targets already in v22c.parquet and against the US/India domain-rule candidates. Writes r1_add.parquet (R1
minus targets in v22c or the domain rule) and r1_add_nodom.parquet (minus v22c only), and prints examples.
"""
import polars as pl
W = r"work"
SP = r"work/scratch"
pl.Config.set_tbl_width_chars(250); pl.Config.set_fmt_str_lengths(45); pl.Config.set_tbl_rows(60)
r = pl.read_parquet(rf"{SP}\usinfn\add.parquet").select(pl.col("s1").cast(pl.UInt32), pl.col("t").cast(pl.UInt32))
v = pl.read_parquet("v22c.parquet", columns=["s1", "t"])
d = pl.read_parquet("dom_usin_add_candidate.parquet")
print("r1 rows", r.height, "unique t", r["t"].n_unique(), "pairs already in v22c", r.join(v, on=["s1", "t"], how="semi").height, "targets in v22c", r.join(v, on="t", how="semi").height)
print("r1 targets in dom cand", r.join(d, on="t", how="semi").height, " same pair", r.join(d, on=["s1", "t"], how="semi").height)
out = r.join(v, on="t", how="anti").join(d, on="t", how="anti")
out2 = r.join(v, on="t", how="anti")
print("r1_add (minus v22c & dom):", out.height, " (minus v22c only):", out2.height)
out.write_parquet("r1_add.parquet"); out2.write_parquet("r1_add_nodom.parquet")
s1 = pl.read_parquet(rf"{W}\test\s1.parquet", columns=["s1", "business_name", "business_address"])
tg = pl.read_parquet(rf"{W}\test\tgt.parquet", columns=["t", "business_name", "business_address", "country"])
print(out2.join(tg, on="t").join(s1, on="s1", suffix="_s1").select("country", "business_name", "business_name_s1", "business_address", "business_address_s1").head(12))
