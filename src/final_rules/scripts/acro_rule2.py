"""Tightened street agreement for the acronym rule (shared rare street token; every other target street token must be a
typo of an S1 token)."""
import polars as pl, sys
from rapidfuzz import fuzz
sys.argv = ["x"]
exec(open(r"work/w20/acro_rule.py").read().split(chr(10) + "if __name__")[0])
GEN = set("rue r av ave avenue bd boulevard blvd all allee pl place cours crs chemin che imp impasse rte route quai de du des la le les d l no n b bis ter st saint sainte ste parv parvis square sq cite sentier voie residence res lieu dit lot zi za zac".split())
ADM = set("gironde nord loire atlantique pays hauts france nouvelle aquitaine et".split())
US_GEN = set("street st road rd avenue ave drive dr lane ln boulevard blvd court ct way place pl suite ste apt unit floor fl po box north south east west n s e w nagar road marg".split())
def street_ok(ta, sa, dfmap, dfmax):
    """Tight street check between target tokens `ta` and S1 tokens `sa` (space-separated strings).

    Generic street/admin words and digits are ignored. Every other target-only token must be a typo (ratio >= 70) of an
    S1-only token, and at least one rare token (document frequency `dfmap` <= `dfmax`) must be shared or typo-matched.
    Returns True when both hold.
    """
    A = set((sa or "").split()); B = set((ta or "").split())
    skip = GEN | ADM | US_GEN
    Au = [a for a in A - B if a not in skip and not a.isdigit()]
    ok_shared = any(dfmap.get(x, 0) <= dfmax for x in (A & B) if x not in skip and not x.isdigit())
    for u in B - A:
        if u in skip or u.isdigit():
            continue
        m = [a for a in Au if fuzz.ratio(u, a) >= 70]
        if not m:
            return False
        if dfmap.get(u, 0) <= dfmax or any(dfmap.get(a, 0) <= dfmax for a in m):
            ok_shared = True
    return ok_shared
def tighten(split, j, countries):
    """Keep only the acronym candidate pairs `j` (from acro_rule.run) that pass street_ok.

    Street-token document frequencies are computed over the split's S1s of `countries`; the rarity threshold is 2000
    on test and 3000 on train. Returns the filtered pairs with k (candidates per target) recomputed.
    """
    s1 = pl.read_parquet(rf"{W}\{split}\s1.parquet", columns=["addr_toks", "country"]).filter(pl.col("country").is_in(countries))
    df = (s1.select(w=pl.col("addr_toks").fill_null("").str.split(" ")).explode("w").filter(pl.col("w") != "")
          .group_by("w").len())
    dfmap = dict(zip(df["w"].to_list(), df["len"].to_list()))
    dfmax = 2000 if split == "test" else 3000
    ok = [street_ok(a, b, dfmap, dfmax) for a, b in zip(j["ta"].to_list(), j["sa"].to_list())]
    j = j.with_columns(ok=pl.Series(ok)).filter(pl.col("ok"))
    return j.with_columns(k=pl.len().over("t"))
if __name__ == "__main__":
    pl.Config.set_tbl_rows(30); pl.Config.set_tbl_width_chars(250); pl.Config.set_fmt_str_lengths(50)
    tr = pl.read_parquet(rf"{W}\w20\acro_train.parquet").drop("k")
    tr = tighten("train", tr, ["US", "India"])
    u = tr.filter(pl.col("k") == 1)
    print("TRAIN unique:", u.group_by("country").agg(n=pl.len(), prec=pl.col("y").mean()))
    print(u.with_columns(b=pl.col("p").fill_null(-1).cut([-0.5, 0.05, 0.5, 0.77])).group_by("b").agg(n=pl.len(), prec=pl.col("y").mean()).sort("b"))
    tg, j = run("test", ["France"])
    j = tighten("test", j.drop("k"), ["France"])
    fin = pl.read_parquet(rf"{W}\w20\final_pairs.parquet")
    pool = pl.scan_parquet(rf"{W}\w20\test\scores_c.parquet").select("s1", "t", "p").collect().join(tg.select("t"), on="t", how="semi")
    j = (j.join(pool, on=["s1", "t"], how="left").join(fin.with_columns(acc=pl.lit(True)), on=["s1", "t"], how="left")
         .with_columns(pl.col("acc").fill_null(False)).join(fin.select("t").unique().with_columns(tcl=pl.lit(True)), on="t", how="left")
         .with_columns(pl.col("tcl").fill_null(False)))
    u = j.filter(pl.col("k") == 1)
    print("FRANCE unique:", u.height, "agreement on claimed:", u.filter(pl.col("tcl")).select(pl.col("acc").mean(), pl.len()).row(0))
    new = u.filter(~pl.col("tcl"))
    print("FRANCE adds:", new.height)
    print(new.with_columns(b=pl.col("p").fill_null(-1).cut([-0.5, 0.05, 0.2, 0.5, 0.77])).group_by("b").len().sort("b"))
    s1n = pl.read_parquet(rf"{W}\test\s1.parquet", columns=["s1", "business_name", "business_address", "entity_id"])
    tgn = pl.read_parquet(rf"{W}\test\tgt.parquet", columns=["t", "business_name", "business_address", "entity_id"])
    x = new.join(s1n, on="s1").join(tgn, on="t", suffix="_t")
    print(x.sample(25, seed=7).select("business_name", "business_name_t", "business_address", "business_address_t", "p"))
    x.write_parquet(rf"{W}\w20\acro_fr_new2.parquet")
