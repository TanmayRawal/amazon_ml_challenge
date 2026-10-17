"""Label-free Bayesian owner posterior for tied blank targets.
Legal transition P(t_legal | s_legal) is measured per country on single-S1 groups (ng==1, ~all owned by that S1).
Posterior over tie members w_i = P(t_legal | s_legal_i) (uniform prior).  Validate calibration on train (labels),
then apply to France (unclaimed tie targets)."""
import polars as pl

OUT = r"work/scratch/france2"
pl.Config.set_tbl_cols(30); pl.Config.set_fmt_str_lengths(60); pl.Config.set_tbl_width_chars(260); pl.Config.set_tbl_rows(80)


def trans(x):
    """Smoothed legal-form transition table q = P(t_legal | s_legal) per country, from single-S1 groups (ng == 1)."""
    s = x.filter(pl.col("ng") == 1)
    m = s.group_by("country", "s_legal", "t_legal").agg(c=pl.len())
    m = m.with_columns(tot=pl.col("c").sum().over("country", "s_legal"))
    # smoothing: add 0.5 to every (s_legal,t_legal) seen anywhere in the country
    return m.with_columns(q=(pl.col("c") + 0.5) / (pl.col("tot") + 5.0)).select("country", "s_legal", "t_legal", "q", "tot")


def fmt_ratio(x):
    """P(fold_eq | same legal, owner) from single groups -- used as a second factor"""
    s = x.filter((pl.col("ng") == 1) & (pl.col("leg") == "same"))
    return s.group_by("country").agg(fe=pl.col("fold_eq").mean())


def posterior(x, tr):
    """Owner posterior over each tie group: post = q / sum(q) over the target's S1s (unseen transitions get the
    smoothing mass), and rk = rank of the posterior within the target."""
    x = x.join(tr, on=["country", "s_legal", "t_legal"], how="left").with_columns(
        q=pl.col("q").fill_null(0.5 / (pl.col("tot").fill_null(0) + 5.0)))
    x = x.with_columns(post=pl.col("q") / pl.col("q").sum().over("t"))
    x = x.with_columns(rk=pl.col("post").rank("ordinal", descending=True).over("t"))
    return x


tr_x = pl.read_parquet(f"{OUT}/p1b_train.parquet")
te_x = pl.read_parquet(f"{OUT}/p1_test.parquet")
T_tr, T_te = trans(tr_x), trans(te_x)
print("France legal transitions from single-S1 blank groups (top)")
print(T_te.filter(pl.col("country") == "France").sort("tot", "q", descending=True).head(40))
tr = posterior(tr_x, T_tr)
bins = [0.5, 0.6, 0.7, 0.8, 0.85, 0.9, 0.95, 0.98]
for nm, z in (("all", tr), ("unclaimed", tr.filter(~pl.col("t_claim")))):
    z = z.filter((pl.col("ng") >= 2) & (pl.col("rk") == 1))
    print("TRAIN tie top-posterior calibration:", nm)
    print(z.with_columns(b=pl.col("post").cut(bins)).group_by("country", "b").agg(n=pl.len(), post=pl.col("post").mean(), true=pl.col("y").mean(),
          own=pl.col("t_owned_in").mean()).sort("country", "b"))
te = posterior(te_x, T_te)
z = te.filter((pl.col("ng") >= 2) & (pl.col("rk") == 1))
print("TEST tie top-posterior: accepted share and unclaimed volume")
print(z.with_columns(b=pl.col("post").cut(bins)).group_by("country", "b").agg(n=pl.len(), post=pl.col("post").mean(), acc=pl.col("acc").mean(),
      un=(~pl.col("t_claimed_any")).sum(), p_un=pl.col("p").filter(~pl.col("t_claimed_any")).mean()).sort("country", "b"))
tr.write_parquet(f"{OUT}/p1c_train.parquet"); te.write_parquet(f"{OUT}/p1c_test.parquet")
