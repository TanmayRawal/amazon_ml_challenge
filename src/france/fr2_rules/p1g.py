"""Final rule definitions for tied blank targets; train true rates (test-like) and France volumes."""
import polars as pl

OUT = r"work/scratch/france2"
pl.Config.set_tbl_cols(30); pl.Config.set_tbl_width_chars(260); pl.Config.set_tbl_rows(80)

tr = pl.read_parquet(f"{OUT}/p1f_train.parquet")
te = pl.read_parquet(f"{OUT}/p1f_test.parquet")


def mark(x):
    """Flag tie-rule candidates: A (top legal posterior >= 0.9, p >= 0.3, word order not contradicting; the rule used),
    B (posterior >= 0.85), C (>= 0.95) and D (legal x word-order posterior post3 >= 0.9)."""
    x = x.with_columns(n_core_eq=pl.col("core_eq").sum().over("t"))
    base = (pl.col("ng") >= 2) & (pl.col("rk") == 1) & (pl.col("p") >= 0.3)
    nocontra = pl.col("core_eq") | (pl.col("n_core_eq") == 0)
    return x.with_columns(
        A=base & (pl.col("post") >= 0.9) & nocontra,
        B=base & (pl.col("post") >= 0.85) & nocontra,
        C=base & (pl.col("post") >= 0.95) & nocontra,
        D=(pl.col("ng") >= 2) & (pl.col("rk3") == 1) & (pl.col("p") >= 0.3) & (pl.col("post3") >= 0.9),  # with order factor
    )


tr, te = mark(tr), mark(te)
for r in "ABCD":
    z = tr.filter(pl.col(r))
    print(r, "TRAIN", z.group_by("country").agg(
        n=pl.len(), true_all=pl.col("y").mean(),
        n_band=(pl.col("p") < 0.77).sum(), true_band=pl.col("y").filter(pl.col("p") < 0.77).mean(),
        n_un=(~pl.col("t_claim")).sum(), true_un=pl.col("y").filter(~pl.col("t_claim")).mean()).sort("country"))
    zf = te.filter(pl.col(r) & (pl.col("country") == "France"))
    print(r, "FRANCE", zf.select(n=pl.len(), acc=pl.col("acc").mean(), add=(~pl.col("t_claimed_any")).sum(),
                                  p_add=pl.col("p").filter(~pl.col("t_claimed_any")).mean(),
                                  p_band=((pl.col("p") < 0.77) & ~pl.col("t_claimed_any")).sum()))
te.filter(pl.col("country") == "France").write_parquet(f"{OUT}/p1g_france.parquet")
