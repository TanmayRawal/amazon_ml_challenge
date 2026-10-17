"""France blank ties: owner posterior from formatting inheritance learned on single-owner French blank copies.
L(c) = prod_k P(target attr_k | claimant attr_k, c is owner); posterior = L / sum L over same-core claimants."""
import re, unicodedata, collections, math
import polars as pl
W = r"work"
s1 = pl.read_parquet(rf"{W}\test\s1.parquet", columns=["s1", "country", "business_name", "core", "core_key", "legal"]).filter(pl.col("country") == "France")
tg = pl.read_parquet(rf"{W}\test\tgt.parquet", columns=["t", "country", "business_name", "core", "core_key", "legal", "addr_blank", "is_domain"]).filter(
    (pl.col("country") == "France") & pl.col("addr_blank") & ~pl.col("is_domain") & (pl.col("core_key") != ""))
def attrs(name, core, legal):
    """Attribute tuple of a name: (core word order, has accent, "&", parentheses, hyphen, legal form)."""
    n = name or ""
    return (core or "", any(unicodedata.combining(c) for c in unicodedata.normalize("NFKD", n)), "&" in n, "(" in n, "-" in n, legal or "")
cnt = s1.group_by("core_key").agg(k=pl.len(), s1s=pl.col("s1"))
x = tg.join(cnt, on="core_key")
SA = {r[0]: attrs(r[1], r[2], r[3]) for r in s1.select("s1", "business_name", "core", "legal").iter_rows()}
TA = {r[0]: attrs(r[1], r[2], r[3]) for r in x.select("t", "business_name", "core", "legal").iter_rows()}
# learn per-attribute agreement and legal transition on single-owner targets
single = x.filter(pl.col("k") == 1).with_columns(s1=pl.col("s1s").list.first())
agree = [0] * 5; n = 0; legT = collections.Counter(); legS = collections.Counter()
for t, s in single.select("t", "s1").iter_rows():
    a, b = TA[t], SA[s]; n += 1
    for i in range(5):
        agree[i] += a[i] == b[i]
    legT[(b[5], a[5])] += 1; legS[b[5]] += 1
A = [c / n for c in agree]
print("agreement:", dict(zip(["order", "acc", "amp", "par", "hyph"], [round(v, 4) for v in A])))
forms = sorted({k[1] for k in legT} | {k[0] for k in legT})
def p_leg(sl, tl):
    """Smoothed probability that an owner with legal form `sl` has a copy with legal form `tl`."""
    return (legT[(sl, tl)] + 0.5) / (legS[sl] + 0.5 * len(forms))
def loglik(ta, sa):
    """Naive-Bayes log-likelihood that an S1 with attributes `sa` owns a target with attributes `ta`."""
    ll = 0.0
    for i in range(5):
        ll += math.log(A[i] if ta[i] == sa[i] else max(1 - A[i], 1e-3))
    return ll + math.log(p_leg(sa[5], ta[5]))
fin = pl.read_parquet(rf"{W}\w20\final_pairs.parquet")
pool = pl.scan_parquet(rf"{W}\w20\test\scores_c.parquet").select("s1", "t", "p").collect()
ties = x.filter(pl.col("k") >= 2)
rows = []
for t, s1s in ties.select("t", "s1s").iter_rows():
    ll = [loglik(TA[t], SA[s]) for s in s1s]
    m = max(ll); w = [math.exp(v - m) for v in ll]; z = sum(w)
    j = max(range(len(s1s)), key=lambda i: w[i])
    srt = sorted(w, reverse=True)
    rows.append((t, s1s[j], w[j] / z, len(s1s), srt[1] / z if len(srt) > 1 else 0.0))
r = pl.DataFrame(rows, schema=["t", "s1", "post", "k", "post2"], orient="row")
cl = fin.join(r.select("t"), on="t", how="semi")  # claimed tie targets: who claims
r = r.join(cl.rename({"s1": "claimant"}), on="t", how="left").join(pool, on=["s1", "t"], how="left")
r = r.with_columns(claimed=pl.col("claimant").is_not_null(), agree=pl.col("claimant") == pl.col("s1"))
r = r.with_columns(b=pl.col("post").cut([0.5, 0.75, 0.9, 0.95, 0.99]))
pl.Config.set_tbl_rows(30)
print("tie targets:", r.height, "claimed:", r["claimed"].sum())
print(r.group_by("b").agg(n=pl.len(), claimed=pl.col("claimed").mean(), agree_if_claimed=pl.col("agree").filter(pl.col("claimed")).mean(),
                          unclaimed=(~pl.col("claimed")).sum(), p_med=pl.col("p").median()).sort("b"))
r.write_parquet(rf"{W}\w20\fr_tie_nb.parquet")
