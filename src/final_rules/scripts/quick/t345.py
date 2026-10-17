"""v23 small fixes 3-5 on top of v22c (v22c.parquet from base.py).

3. Duplicates: for targets claimed by 2+ S1s, keep the highest-p claimant; others -> dup_remove.parquet.
4. Caps: after the duplicate fix, S1 x source groups above the generator cap (5 S2 / 6 S3 copies, checked on train)
   lose their lowest-p extras -> cap_remove.parquet.
5. French ties at posterior 0.85-0.9 with model p >= 0.2, target not yet in v22c -> fr_tie3_add.parquet.
Also prints which input list the removed duplicates come from, and example rows.
"""
import polars as pl
W = r"work"
SP = r"work/scratch"
pl.Config.set_tbl_cols(30); pl.Config.set_tbl_width_chars(250); pl.Config.set_fmt_str_lengths(40); pl.Config.set_tbl_rows(30)
v = pl.read_parquet("v22c.parquet")
tg = pl.read_parquet(rf"{W}\test\tgt.parquet", columns=["t", "src", "country", "business_name"])
s1n = pl.read_parquet(rf"{W}\test\s1.parquet", columns=["s1", "business_name"])
v = v.join(tg.select("t", "src", "country"), on="t", how="left")
# ---- 3. duplicates
d = v.filter(pl.len().over("t") > 1)
print("dup targets", d["t"].n_unique(), "dup rows", d.height, "claimants per target", d.group_by("t").len()["len"].value_counts().sort("len"))
d = d.with_columns(allnull=pl.col("p").is_null().all().over("t"), anynull=pl.col("p").is_null().any().over("t"))
dd = d.filter(~pl.col("allnull")).with_columns(r=pl.col("p").fill_null(-1.0).rank("ordinal", descending=True).over("t"))
dup_rm = dd.filter(pl.col("r") > 1)
print("dup skipped (all p null) targets:", d.filter(pl.col("allnull"))["t"].n_unique(), " targets with some-null p:", dd.filter(pl.col("anynull"))["t"].n_unique(), " removals:", dup_rm.height, " removals with null p:", dup_rm["p"].is_null().sum())
print(dd.with_columns(pk=pl.col("p").max().over("t")).filter(pl.col("r") > 1).select(pl.col("p").describe()) if False else dup_rm.select(pl.col("p").quantile(q).alias(f"q{q}") for q in (0.1, 0.5, 0.9)))
print("kept-p for removed dups:", dd.filter(pl.col("r") == 1).join(dup_rm.select("t").unique(), on="t", how="semi").select(pl.col("p").quantile(q).alias(f"q{q}") for q in (0.1, 0.5, 0.9)))
print("dup by country", dup_rm.group_by("country").len().sort("len", descending=True).head(10))
# which input list do the removed pairs come from?
def rd(f):
    """Read the (s1, t) columns of parquet file `f` as UInt32."""
    return pl.read_parquet(f).select(pl.col("s1").cast(pl.UInt32), pl.col("t").cast(pl.UInt32))
srcs = {"final": rd(rf"{W}\w20\final_pairs.parquet"), "acro": rd(rf"{W}\w20\acro_add.parquet"), "tie": rd(rf"{W}\w20\fr_tie_add.parquet"), "tie2": rd(rf"{W}\w20\fr_tie2_add.parquet")}
for k, s in srcs.items():
    print("  dup removals from", k, dup_rm.join(s, on=["s1", "t"], how="semi").height, "| kept from", k, dd.filter(pl.col("r") == 1).join(s, on=["s1", "t"], how="semi").join(dup_rm.select("t").unique(), on="t", how="semi").height)
print(dd.join(dup_rm.select("t").unique(), on="t", how="semi").join(tg.select("t", tn="business_name"), on="t").join(s1n, on="s1").sort("t", "r").select("t", "r", "p", "tn", "business_name", "country").head(16))
dup_rm.select("s1", "t").write_parquet("dup_remove.parquet")
# ---- 4. caps (computed after dup removal so the two lists compose)
gt = pl.read_parquet(rf"{W}\train\gt.parquet").join(pl.read_parquet(rf"{W}\train\tgt.parquet", columns=["t", "src"]), on="t")
print("train max copies per (s1,src):", gt.group_by("s1", "src").len().group_by("src").agg(pl.col("len").max(), (pl.col("len") > 4).sum().alias("n>4")))
for label, base in (("raw v22c", v), ("v22c - dup_remove", v.join(dup_rm.select("s1", "t"), on=["s1", "t"], how="anti"))):
    b = base.with_columns(cap=pl.when(pl.col("src") == 2).then(5).otherwise(6), n=pl.len().over("s1", "src"))
    over = b.filter(pl.col("n") > pl.col("cap"))
    over = over.with_columns(r=pl.col("p").fill_null(-1.0).rank("ordinal", descending=True).over("s1", "src"))
    rm = over.filter(pl.col("r") > pl.col("cap"))
    print(label, "groups over cap:", over.select("s1", "src").unique().height, "removals:", rm.height, "null-p removals:", rm["p"].is_null().sum(), "by src", rm.group_by("src").len().to_dicts())
    print("  removed p quantiles", rm.select(pl.col("p").quantile(q).alias(f"q{q}") for q in (0.1, 0.5, 0.9)).to_dicts())
cap_rm = rm
print(over.join(tg.select("t", tn="business_name"), on="t").join(s1n, on="s1").sort("s1", "src", "r").select("s1", "src", "r", "p", "tn", "business_name", "country").head(14))
cap_rm.select("s1", "t").write_parquet("cap_remove.parquet")
# ---- 5. fr tie3
ps = pl.read_parquet(rf"{SP}\fr_tie2\post_sw.parquet")
print("post_sw rows", ps.height, "unique t", ps["t"].n_unique(), "claimed", ps["claimed"].sum(), "prev", ps["prev"].sum())
a = ps.filter((pl.col("post") >= 0.85) & (pl.col("post") < 0.9) & (pl.col("p") >= 0.2)).with_columns(pl.col("s1").cast(pl.UInt32), pl.col("t").cast(pl.UInt32))
print("band 0.85-0.9 & p>=0.2:", a.height, " claimed:", a["claimed"].sum(), " prev:", a["prev"].sum())
vt = v.select("t").unique()
a2 = a.join(vt, on="t", how="anti")
print("not in v22c:", a2.height, " of which claimed/prev flags:", a2["claimed"].sum(), a2["prev"].sum())
# sanity: compare post_sw p with our p for the same pair
chk = base.select("s1", "t").head(0)
print(a2.join(tg.select("t", "country", tn="business_name"), on="t").join(s1n, on="s1").select("post", "p", "tn", "business_name", "country").sample(min(10, a2.height), seed=1))
print("countries", a2.join(tg.select("t", "country"), on="t")["country"].value_counts())
a2.select("s1", "t", "post", "p").write_parquet("fr_tie3_add.parquet")
# already-added fr_tie2 (>=0.9) are in v22c? check
print("fr_tie2_add in v22c:", srcs["tie2"].join(v.select("s1", "t"), on=["s1", "t"], how="semi").height, "/", srcs["tie2"].height)
