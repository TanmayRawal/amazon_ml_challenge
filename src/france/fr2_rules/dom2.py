"""Domain rule refinement: unique same-street claimant + street tokens must fuzzy-match (guards against the
'same number, other street' decoy).  Train true rates and France additions."""
import polars as pl
from rapidfuzz import fuzz

OUT = r"work/scratch/france2"
pl.Config.set_tbl_cols(30); pl.Config.set_tbl_width_chars(300); pl.Config.set_fmt_str_lengths(70); pl.Config.set_tbl_rows(60)
REG = {"nouvelle", "aquitaine", "hauts", "de", "france", "pays", "la", "loire", "atlantique", "gironde", "nord", "pas", "calais"}


def street_ok(st, tt):
    """True when every target street token `tt` (region words ignored) is shared with, or a typo (ratio >= 70) of, an
    S1 token in `st`."""
    s = [w for w in (st or "").split() if w not in REG]
    t = [w for w in (tt or "").split() if w not in REG]
    ss, ts = set(s), set(t)
    s_un = [w for w in s if w not in ts]
    t_un = [w for w in t if w not in ss]
    if not t_un:
        return True
    # every unmatched target token must fuzzy-match some unmatched S1 token (typo), else another street
    for w in t_un:
        if not any(fuzz.ratio(w, v) >= 70 for v in s_un):
            return False
    return True


def refine(x):
    """Keep unique same-street claimants (arel same/street_nonum, n_street == 1) and add the street_ok flag (sok)."""
    x = x.filter(pl.col("arel").is_in(["same", "street_nonum"]) & (pl.col("n_street") == 1))
    ok = [street_ok(a, b) for a, b in zip(x["s_toks"].to_list(), x["t_toks"].to_list())]
    return x.with_columns(sok=pl.Series(ok))


tr = refine(pl.read_parquet(f"{OUT}/dom_train.parquet"))
print("TRAIN", tr.group_by("country", "arel", "sok").agg(n=pl.len(), true=pl.col("y").mean(), n_un=(~pl.col("t_claim")).sum(),
      true_un=pl.col("y").filter(~pl.col("t_claim")).mean(), n_nullp=(pl.col("p") == 0).sum(), true_nullp=pl.col("y").filter(pl.col("p") == 0).mean()).sort("country", "arel", "sok"))
te = refine(pl.read_parquet(f"{OUT}/dom_test.parquet"))
print("TEST", te.group_by("country", "arel", "sok").agg(n=pl.len(), acc=pl.col("acc").mean(), un=(~pl.col("t_claimed")).sum(),
      pnull_un=pl.col("p").filter(~pl.col("t_claimed")).is_null().sum()).sort("country", "arel", "sok"))
fr = te.filter((pl.col("country") == "France") & ~pl.col("t_claimed"))
print(fr.filter(~pl.col("sok")).select("domain", "business_name", "arel", "p", "t_toks", "s_toks").head(20))
fr.write_parquet(f"{OUT}/dom_fr_cands.parquet")
