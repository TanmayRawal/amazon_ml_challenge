"""Guarded France ensemble delta: accept additions only in true-copy classes (and blank ties the formatting posterior
confirms); accept removals only in decoy-signature classes. Everything LB-validated as true is protected."""
import sys
import polars as pl
W = r"work"; OUT = rf"{W}\w20\frens"
tag = sys.argv[1]
DEC = [1, 2, 3, 4, 5, 7, 9, 11, 13, 21]
s1i = pl.read_parquet(rf"{W}\test\s1.parquet", columns=["s1", "entity_id", "country", "core_key"])
tgi = pl.read_parquet(rf"{W}\test\tgt.parquet", columns=["t", "entity_id", "core_key"])
fin = pl.read_parquet(rf"{W}\w20\final_pairs.parquet")
fr2 = (pl.read_csv(r"submissions/v21e3x_fr2/changes.csv", infer_schema_length=0).select("s_id", "t_id")
       .join(s1i.select("s1", s_id="entity_id"), on="s_id").join(tgi.select("t", t_id="entity_id"), on="t_id").select("s1", "t"))
frs = s1i.filter(pl.col("country") == "France")
U0 = fin.join(frs.select("s1"), on="s1", how="semi").join(fr2, on=["s1", "t"], how="anti")
protect = pl.concat([fr2] + [pl.read_parquet(rf"{W}\w20\{f}.parquet").select(pl.col("s1").cast(pl.UInt32), pl.col("t").cast(pl.UInt32))
                             for f in ("acro_add", "fr_tie_add", "fr_tie2_add")]).unique()
cls = pl.read_parquet(rf"{OUT}\classes.parquet")
pool = pl.read_parquet(rf"{W}\w20\fr_ens_pool.parquet")
post = pl.read_parquet(r"work/scratch/fr_tie2/post_sw.parquet").select(
    pl.col("t").cast(pl.UInt32), ps1=pl.col("s1").cast(pl.UInt32), post="post")
nsame = frs.filter(pl.col("core_key") != "").group_by("core_key").agg(n_s1=pl.len())
A = pl.read_parquet(rf"{OUT}\R_v20.parquet").select("s1", "t")
B = pl.read_parquet(rf"{OUT}\R_{tag}.parquet").select("s1", "t")
add = B.join(A, on=["s1", "t"], how="anti").join(U0, on=["s1", "t"], how="anti").join(protect, on=["s1", "t"], how="anti")
rem = A.join(B, on=["s1", "t"], how="anti").join(U0, on=["s1", "t"], how="semi").join(protect, on=["s1", "t"], how="anti")
def enrich(d):
    """Attach to pairs `d` their class columns, both models' p, the tie posterior owner (ps1, post) and the number of
    French S1s sharing the target's core name (n_s1)."""
    return (d.join(cls, on=["s1", "t"], how="left").join(pool, on=["s1", "t"], how="left").join(post, on="t", how="left")
             .join(tgi.select("t", tk="core_key"), on="t").join(nsame.rename({"core_key": "tk"}), on="tk", how="left")
             .with_columns(pl.col("n_s1").fill_null(0)))
add, rem = enrich(add), enrich(rem)
same_addr = pl.col("st") == "st"
ok_add = (
    (pl.col("name").is_in(["dropped", "swap_list", "swap_typo", "added"]) & pl.col("num").is_in(["num_eq", "nonum"]) & same_addr & (pl.col("leg") != "leg_swap"))
    | ((pl.col("name") == "web") & pl.col("num").is_in(["num_eq", "nonum"]) & same_addr)
    | ((pl.col("name") == "exact") & pl.col("num").is_in(["num_eq", "nonum"]) & same_addr & (pl.col("leg") != "leg_swap"))
    | ((pl.col("name") == "exact") & (pl.col("num") == "num_sub") & same_addr & (pl.col("leg") == "leg_eq") & ~pl.col("off").is_in(DEC))
    | ((pl.col("num") == "blank") & (pl.col("name") == "exact") & (pl.col("n_s1") == 1) & (pl.col("leg") == "leg_add"))   # single same-name S1, legal added (train analog 0.90)
    | ((pl.col("num") == "blank") & (pl.col("name") == "exact") & (pl.col("ps1") == pl.col("s1")) & (pl.col("post") >= 0.9) & (pl.col("leg") != "leg_swap")))  # confirmed tie owner
ok_rem = (
    ((pl.col("num") == "blank") & (pl.col("name") == "exact") & pl.col("ps1").is_not_null() & (pl.col("ps1") != pl.col("s1")) & (pl.col("post") >= 0.75))
    | (pl.col("name").is_in(["added", "dropped", "swap_word", "swap_typo", "multi"]) & (pl.col("st") == "st_diff") & (pl.col("num") != "blank"))
    | ((pl.col("name") == "exact") & (pl.col("num") == "num_sub") & pl.col("leg").is_in(["leg_add", "leg_swap"])))
A2, R2 = add.filter(ok_add.fill_null(False)), rem.filter(ok_rem.fill_null(False))
pl.Config.set_tbl_rows(20)
print(f"[{tag}] raw add {add.height} -> guarded {A2.height}; raw remove {rem.height} -> guarded {R2.height}")
print("ADD kept:"); print(A2.group_by("name", "num", "st").len().sort("len", descending=True).head(10))
print("REMOVE kept:"); print(R2.group_by("name", "num", "st").len().sort("len", descending=True).head(10))
A2.select("s1", "t").write_parquet(rf"{W}\w20\ens_add_{tag}.parquet"); R2.select("s1", "t").write_parquet(rf"{W}\w20\ens_rem_{tag}.parquet")
