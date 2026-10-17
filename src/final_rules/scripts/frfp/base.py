"""French decoys (v22 fr_fp_remove), step 1: table of the accepted French pairs.

Joins the French rows of the current submission (w20/final_classes.parquet, with pair classes name/num/leg) to the
S1 and target names, addresses, numbers, legal forms and the model p (w20/test/scores_c.parquet). Writes
fr_acc.parquet, used by the other frfp scripts, and prints the class counts and a sample.
"""
import polars as pl
pl.Config.set_tbl_rows(80); pl.Config.set_tbl_cols(30); pl.Config.set_tbl_width_chars(250); pl.Config.set_fmt_str_lengths(60)
W = r"work"
fc = pl.read_parquet(rf"{W}\w20\final_classes.parquet")
print(fc.schema, fc.height)
fr = fc.filter(pl.col("country")=="France")
print("fr pairs", fr.height, "fr s1 with pairs", fr["s1"].n_unique())
s1 = pl.read_parquet(rf"{W}\test\s1.parquet")
print(s1.schema)
tg = pl.read_parquet(rf"{W}\test\tgt.parquet")
print(tg.schema)
print(s1.filter(pl.col("country")=="France").height)
x = fr.join(s1.select("s1","business_name","business_address","addr_nums","legal","core"), on="s1").join(tg.select("t","src","business_name","business_address","addr_nums","legal","core","addr_blank","country"), on="t", suffix="_t")
sc = pl.read_parquet(rf"{W}\w20\test\scores_c.parquet")
x = x.join(sc, on=["s1","t"], how="left")
x.write_parquet("fr_acc.parquet")
print(x.group_by("name").len().sort("len", descending=True))
print(x.group_by("num").len().sort("len", descending=True))
print(x.group_by("leg").len().sort("len", descending=True))
print(x.group_by("src").len())
print(x.sample(20, seed=1).select("business_name","business_address","business_name_t","business_address_t","src","name","num","leg","p"))
