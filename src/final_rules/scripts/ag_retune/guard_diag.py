"""Singleton guard (v24), diagnostics on fold 4.

Reuses the setup of f4_ideas.py (fold-4 pool, own, single_guard). For thresholds 0.5/0.6/0.7 it reports, per country
(US, India), how many S1s the guard changes, how many gain (ownerless S1s left empty) and lose (owned S1s emptied),
and the F0.5 change reweighted to a test-like prior of 45% ownerless S1s. Supports the choice of threshold 0.6.
"""
import sys, numpy as np, polars as pl
exec(open("f4_ideas.py", encoding="utf-8").read().split("s1_ = own(1.0)")[0])
s1_ = own(1.0)
bp = dta_select(s1_.filter(pl.col("p") > 1e-4), 0.031).select("s1", "t")
b = per_entity(ids.select("s1"), bp, gt).join(ids, on="s1")
for thr in (0.5, 0.6, 0.7):
    g = per_entity(ids.select("s1"), single_guard(bp, s1_, thr), gt).select("s1", fg="f")
    d = b.join(g, on="s1").with_columns(d=pl.col("fg") - pl.col("f"), own0=pl.col("n_true") == 0)
    for c in ("US", "India"):
        x = d.filter(pl.col("country") == c)
        ch = x.filter(pl.col("d") != 0)
        sh = x["own0"].mean()
        # reweight to 45% ownerless (test-like prior)
        w = pl.when(pl.col("own0")).then(0.45 / sh).otherwise(0.55 / (1 - sh))
        xw = x.with_columns(w=w)
        mw = (xw["d"] * xw["w"]).mean() * 1e5
        sew = (xw["d"] * xw["w"]).std() / np.sqrt(x.height) * 1e5
        print(thr, c, "ownerless share", round(sh, 3), "changed", ch.height, "gain(ownerless)", int((ch["d"] > 0).sum()), "loss(owned)", int((ch["d"] < 0).sum()),
              "| reweighted45 d_e5", round(mw, 2), "se", round(sew, 2), "z", round(mw / sew, 2) if sew else 0)
