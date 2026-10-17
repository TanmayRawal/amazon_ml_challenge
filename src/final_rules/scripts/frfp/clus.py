"""French decoys (v22), exploration: clustering of candidate copies.

Counts how often candidates of the same S1 come in clusters (same S1, same changed word, same number offset) for
added / swapped-word / swapped-list-word names (words.parquet) and for exact names at a substituted house number
(pool_d.parquet), split into decoy offsets, other positive and negative offsets. Prints only.
"""
import polars as pl
pl.Config.set_tbl_rows(60); pl.Config.set_tbl_width_chars(250)
x = pl.read_parquet("words.parquet")
# cluster size: same s1, same new word, same name class, same num group+d
x = x.with_columns(cs=pl.len().over("s1","name","new","gone","num","d"))
for nm, key in [("added","new"),("swap_word","new"),("swap_list","new")]:
    y = x.filter((pl.col("name")==nm) & pl.col("grp").is_in(["eq","decoy"]))
    print(nm)
    print(y.group_by(key,"grp").agg(n=pl.len(), acc=pl.col("acc").sum(), cs2=(pl.col("cs")>=2).mean(), cs2_acc=(pl.col("cs").filter(pl.col("acc"))>=2).mean())
          .filter(pl.col("n")>=300).sort(key,"grp"))
d = pl.read_parquet("pool_d.parquet").filter((pl.col("name")=="exact") & pl.col("num").is_in(["num_sub","num_trunc"]) & (pl.col("st")=="st"))
d = d.with_columns(cs=pl.len().over("s1","d"))
print(d.with_columns(g=pl.when(pl.col("d").is_in([1,2,3,4,5,7,9,11,13,21])).then(pl.lit("dec")).when(pl.col("d")<0).then(pl.lit("neg")).otherwise(pl.lit("pos"))).group_by("g","leg").agg(n=pl.len(), cs2=(pl.col("cs")>=2).mean(), acc=pl.col("acc").sum()).sort("g","leg"))
