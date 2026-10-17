"""Guarded France ensemble (v23): raw delta between the chain on the ensemble p and on V20 p.

Checks the reproduction first: U0 = uploaded French rows (final_pairs minus the FRANCE-2 additions) vs R_v20 (chain on
V20 p). Then, for each ensemble variant R_<tag> written by fr_chain2.py (w = 0.5 / 0.7, full or V20 pair universe),
prints the effective additions to and removals from the uploaded rows, by score source (both / v20 only / v18 only)
and by pair class (name, number, street). fr_guard.py then filters this delta. No files are written.
"""
import polars as pl
W = r"work"; OUT = rf"{W}\w20\frens"
s1i = pl.read_parquet(rf"{W}\test\s1.parquet", columns=["s1", "entity_id", "country"])
tgi = pl.read_parquet(rf"{W}\test\tgt.parquet", columns=["t", "entity_id"])
fin = pl.read_parquet(rf"{W}\w20\final_pairs.parquet")
fr2 = (pl.read_csv(r"submissions/v21e3x_fr2/changes.csv", infer_schema_length=0).select("s_id", "t_id")
       .join(s1i.select("s1", s_id="entity_id"), on="s_id").join(tgi.select("t", t_id="entity_id"), on="t_id").select("s1", "t"))
U0 = fin.join(s1i.filter(pl.col("country") == "France").select("s1"), on="s1", how="semi").join(fr2, on=["s1", "t"], how="anti")
cls = pl.read_parquet(rf"{OUT}\classes.parquet")
pool = pl.read_parquet(rf"{W}\w20\fr_ens_pool.parquet")
A = pl.read_parquet(rf"{OUT}\R_v20.parquet").select("s1", "t")
print(f"reproduction: U0 {U0.height}, R[v20] {A.height}, U0-A {U0.join(A, on=['s1','t'], how='anti').height}, A-U0 {A.join(U0, on=['s1','t'], how='anti').height}")
pl.Config.set_tbl_rows(30); pl.Config.set_tbl_width_chars(200)
for tag in ("w0.5", "w0.7", "w0.5_v20univ", "w0.7_v20univ"):
    B = pl.read_parquet(rf"{OUT}\R_{tag}.parquet").select("s1", "t")
    add = B.join(A, on=["s1", "t"], how="anti").join(U0, on=["s1", "t"], how="anti")      # effective additions to the uploaded rows
    rem = A.join(B, on=["s1", "t"], how="anti").join(U0, on=["s1", "t"], how="semi")      # effective removals
    add = add.join(cls, on=["s1", "t"], how="left").join(pool, on=["s1", "t"], how="left")
    rem = rem.join(cls, on=["s1", "t"], how="left").join(pool, on=["s1", "t"], how="left")
    print(f"\n== {tag}: add {add.height}, remove {rem.height}, net {add.height - rem.height:+d}")
    for nm, d in (("ADD", add), ("REMOVE", rem)):
        g = d.with_columns(src=pl.when(pl.col("p20").is_null()).then(pl.lit("v18only")).when(pl.col("p18").is_null()).then(pl.lit("v20only")).otherwise(pl.lit("both")))
        print(nm, "by source:", dict(g.group_by("src").len().iter_rows()), "| median p20", round(d["p20"].median() or 0, 3), "p18", round(d["p18"].median() or 0, 3))
        print(d.group_by("name", "num", "st").len().sort("len", descending=True).head(8))
