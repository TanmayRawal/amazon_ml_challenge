"""France rule chain R(p) (approximation of the v21e3b France decision) and the ensemble delta.
Kill sets are deliberately BROAD: a pair killed under both p20 and p_ens yields no delta, so the uploaded decision stands."""
import sys, os, math
import polars as pl
sys.argv = ["x"]
exec(open(r"work/w20/audit_v20.py", encoding="utf-8").read().split("def per_s1")[0])   # classify, W
sys.path.insert(0, r"work/w20/code_V21/business_entity_resolution/src")
from matcher_b import apply_policy
POLICY = {"name": "dta_own", "missed": 0.031}
OUT = rf"{W}\w20\frens"
os.makedirs(OUT, exist_ok=True)
SUF = {"fils", "france", "associes", "services", "service", "developpement", "freres", "cie", "groupe"}
DEC = [1, 2, 3, 4, 5, 7, 9, 11, 13, 21]

j = pl.read_parquet(rf"{W}\w20\fr_ens_pool.parquet")
s1i = pl.read_parquet(rf"{W}\test\s1.parquet", columns=["s1", "entity_id"])
tgi = pl.read_parquet(rf"{W}\test\tgt.parquet", columns=["t", "entity_id"])
# uploaded France rows of the 0.989165 base minus FRANCE-2 additions = v21e3b France (U0)
fin = pl.read_parquet(rf"{W}\w20\final_pairs.parquet")
fr2 = (pl.read_csv(r"submissions/v21e3x_fr2/changes.csv", infer_schema_length=0)
       .select(s_id="s_id", t_id="t_id").join(s1i.rename({"entity_id": "s_id"}), on="s_id").join(tgi.rename({"entity_id": "t_id"}), on="t_id").select("s1", "t"))
frs = pl.read_parquet(rf"{W}\test\s1.parquet", columns=["s1", "country"]).filter(pl.col("country") == "France").select("s1")
U0 = fin.join(frs, on="s1", how="semi").join(fr2, on=["s1", "t"], how="anti")
print("U0 (uploaded French rows without FRANCE-2 adds):", U0.height)

cand = pl.concat([j.filter(pl.max_horizontal(pl.col("p20").fill_null(0), pl.col("p18").fill_null(0)) > 0.02).select("s1", "t"), U0]).unique()
print("pairs to classify:", cand.height)
c = classify(cand, "test")
s1 = pl.read_parquet(rf"{W}\test\s1.parquet", columns=["s1", "core", "addr_nums"])
tg = pl.read_parquet(rf"{W}\test\tgt.parquet", columns=["t", "core", "addr_nums"]).join(c.select("t").unique(), on="t", how="semi")
w = lambda col: pl.col(col).fill_null("").str.split(" ").list.eval(pl.element().filter(pl.element() != ""))
f1 = lambda col: pl.col(col).fill_null("").str.split(" ").list.first().cast(pl.Int64, strict=False)
c = (c.join(s1.select("s1", a=w("core"), sn=f1("addr_nums")), on="s1").join(tg.select("t", b=w("core"), tn=f1("addr_nums")), on="t")
       .with_columns(gone=pl.col("a").list.set_difference("b"), new=pl.col("b").list.set_difference("a"), off=pl.col("tn") - pl.col("sn")))
# French type-word vocabulary: frequent core words among French target names
tgf = pl.read_parquet(rf"{W}\test\tgt.parquet", columns=["core", "country"]).filter(pl.col("country") == "France")
df = tgf.select(x=w("core")).explode("x").drop_nulls().group_by("x").len()
common = set(df.filter(pl.col("len") >= 300)["x"].to_list())
print("common French words (df>=300):", len(common))
c = c.with_columns(
    g1=pl.col("gone").list.first(), n1=pl.col("new").list.first(),
    add_suf=(pl.col("gone").list.len() == 0) & (pl.col("new").list.len() >= 1) & pl.col("new").list.eval(pl.element().is_in(list(SUF))).list.all())
c = c.with_columns(
    vocab=(pl.col("name") == "swap_word") & pl.col("n1").is_in(list(common)) & pl.col("g1").is_in(list(common)),
    r1=(pl.col("name") == "exact") & (pl.col("leg") == "leg_swap"),
    r3=(pl.col("name") == "exact") & (pl.col("num") == "num_eq") & (pl.col("st") == "st_diff"),
    r4=pl.col("name").is_in(["exact", "added", "swap_typo"]) & pl.col("num").is_in(["num_sub", "num_trunc"]) & pl.col("off").is_in(DEC),
    sufB=pl.col("add_suf") & (pl.col("num") == "num_eq") & (pl.col("st") == "st") & (pl.col("leg") != "leg_swap"),
    floor=(pl.col("add_suf") | (pl.col("name") == "swap_list")) & (pl.col("num") == "num_eq") & (pl.col("st") == "st") & (pl.col("leg") != "leg_swap"))
c = c.with_columns(kill=pl.col("vocab") | pl.col("r1") | pl.col("r3") | pl.col("r4"))
c.select("s1", "t", "name", "num", "st", "leg", "off", "vocab", "r1", "r3", "r4", "sufB", "floor", "kill").write_parquet(rf"{OUT}\classes.parquet")
print(c.select([pl.col(k).sum() for k in ["vocab", "r1", "r3", "r4", "sufB", "floor", "kill"]]))
