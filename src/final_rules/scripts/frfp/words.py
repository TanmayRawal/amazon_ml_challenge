"""French decoys (v22), exploration: which words are added / swapped / dropped.

For French pool pairs whose names differ by words (added, swap_word, swap_list, dropped), extracts the new and gone
words and groups the pair into same number / blank / decoy offset / other number. Writes words.parquet (used by
clus.py) and prints pool and accepted counts per word and group. Prints and writes only.
"""
import polars as pl
pl.Config.set_tbl_rows(60); pl.Config.set_tbl_width_chars(250); pl.Config.set_fmt_str_lengths(45)
W = r"work"
x = pl.read_parquet("pool_d.parquet")
s1 = pl.read_parquet(rf"{W}\test\s1.parquet", columns=["s1","core"])
tg = pl.read_parquet(rf"{W}\test\tgt.parquet", columns=["t","core"])
x = x.filter(pl.col("name").is_in(["added","swap_word","swap_list","dropped"])).join(s1, on="s1").join(tg, on="t", suffix="_t")
w = lambda c: pl.col(c).fill_null("").str.split(" ").list.eval(pl.element().filter(pl.element() != ""))
x = x.with_columns(new=w("core_t").list.set_difference(w("core")).list.sort().list.join(" "), gone=w("core").list.set_difference(w("core_t")).list.sort().list.join(" "))
DEC = [1,2,3,4,5,7,9,11,13,21]
x = x.with_columns(grp=pl.when(pl.col("num")=="num_eq").then(pl.lit("eq")).when(pl.col("num")=="blank").then(pl.lit("blank")).when(pl.col("d").is_in(DEC)&(pl.col("st")=="st")).then(pl.lit("decoy")).otherwise(pl.lit("othernum")))
x.write_parquet("words.parquet")
for nm in ["added","swap_word","swap_list","dropped"]:
    y = x.filter(pl.col("name")==nm)
    key = "new" if nm!="dropped" else "gone"
    t = y.group_by(key).agg(
        dec_pool=(pl.col("grp")=="decoy").sum(), dec_acc=((pl.col("grp")=="decoy")&pl.col("acc")).sum(),
        eq_pool=(pl.col("grp")=="eq").sum(), eq_acc=((pl.col("grp")=="eq")&pl.col("acc")).sum(),
        bl_pool=(pl.col("grp")=="blank").sum(), bl_acc=((pl.col("grp")=="blank")&pl.col("acc")).sum()).sort("eq_acc", descending=True)
    print(nm); print(t.head(25))
