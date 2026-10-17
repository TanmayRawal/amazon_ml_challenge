"""Posterior with legal x word-order (core_eq) factors, all label-free per country.
f1 = P(core_eq | owner) from single-S1 groups; f0 = P(core_eq) between two different S1s of one core_key group."""
import polars as pl

W = r"work"
OUT = r"work/scratch/france2"
pl.Config.set_tbl_cols(30); pl.Config.set_tbl_width_chars(260); pl.Config.set_tbl_rows(80)


def f0_s1(split, countries):
    """f0 per country: probability that two distinct S1s of one core_key group (2-60 S1s) have the same word order."""
    s = (pl.scan_parquet(f"{W}/{split}/s1.parquet").filter(pl.col("country").is_in(countries) & (pl.col("core_key") != ""))
         .select("s1", "country", "core", "core_key").collect())
    s = s.with_columns(k=pl.len().over("country", "core_key")).filter((pl.col("k") >= 2) & (pl.col("k") <= 60))
    # P(core_eq) for random pair of distinct S1s in a group = sum_c n_c(n_c-1) / (k(k-1))
    g = s.group_by("country", "core_key", "core").agg(nc=pl.len(), k=pl.first("k"))
    g = g.group_by("country", "core_key").agg(num=(pl.col("nc") * (pl.col("nc") - 1)).sum(), k=pl.first("k"))
    return g.group_by("country").agg(f0=(pl.col("num").sum() / (pl.col("k") * (pl.col("k") - 1)).sum()))


def f1_tg(x):
    """f1 per country: probability that a blank copy keeps its owner's word order (single-S1 groups, same legal form)."""
    return x.filter((pl.col("ng") == 1) & (pl.col("leg") == "same")).group_by("country").agg(f1=pl.col("core_eq").mean())


def post3(x, f):
    """Posterior with legal-form and word-order factors: w = q * (f1/f0 if core_eq else (1-f1)/(1-f0)), normalised over
    the target's S1s (post3), with its rank rk3."""
    x = x.join(f, on="country").with_columns(
        r=pl.when(pl.col("core_eq")).then(pl.col("f1") / pl.col("f0")).otherwise((1 - pl.col("f1")) / (1 - pl.col("f0"))))
    x = x.with_columns(w=pl.col("q") * pl.col("r")).with_columns(post3=pl.col("w") / pl.col("w").sum().over("t"))
    return x.with_columns(rk3=pl.col("post3").rank("ordinal", descending=True).over("t"))


tr = pl.read_parquet(f"{OUT}/p1c_train.parquet")
te = pl.read_parquet(f"{OUT}/p1c_test.parquet")
ftr = f0_s1("train", ["US", "India"]).join(f1_tg(tr), on="country")
fte = f0_s1("test", ["France", "US", "India"]).join(f1_tg(te), on="country")
print(ftr); print(fte)
tr = post3(tr, ftr); te = post3(te, fte)
bins = [0.5, 0.7, 0.8, 0.85, 0.9, 0.95, 0.98]
for nm, z in (("all", tr), ("unclaimed", tr.filter(~pl.col("t_claim"))), ("unclaimed p>=0.3", tr.filter(~pl.col("t_claim") & (pl.col("p") >= 0.3)))):
    z = z.filter((pl.col("ng") >= 2) & (pl.col("rk3") == 1))
    print("TRAIN post3 calibration:", nm)
    print(z.with_columns(b=pl.col("post3").cut(bins)).group_by("country", "b").agg(n=pl.len(), post=pl.col("post3").mean(), true=pl.col("y").mean(),
          own=pl.col("t_owned_in").mean()).sort("country", "b"))
z = te.filter((pl.col("ng") >= 2) & (pl.col("rk3") == 1) & (pl.col("country") == "France"))
print("FRANCE post3: volume")
print(z.with_columns(b=pl.col("post3").cut(bins)).group_by("b").agg(n=pl.len(), acc=pl.col("acc").mean(), un=(~pl.col("t_claimed_any")).sum(),
      un_p03=((~pl.col("t_claimed_any")) & (pl.col("p") >= 0.3)).sum(), un_p05=((~pl.col("t_claimed_any")) & (pl.col("p") >= 0.5)).sum(),
      p_un=pl.col("p").filter(~pl.col("t_claimed_any")).mean()).sort("b"))
tr.write_parquet(f"{OUT}/p1f_train.parquet"); te.write_parquet(f"{OUT}/p1f_test.parquet")
