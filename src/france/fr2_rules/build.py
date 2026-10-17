"""Build France-only variants of v21e3x: add rule-A (tied blank targets, legal posterior) and/or rule-D (domain targets,
accent-deleting domain matcher) pairs.  Non-French lines stay byte-identical; S1 order unchanged."""
import os

import polars as pl

W = r"work"
OUT = r"work/scratch/france2"
BEST = f"{W}/w20/submissions/v21e3x/matching_results.tsv"
Q = {"A": 0.90, "D": 0.97}

fr = pl.read_parquet(f"{OUT}/p1g_france.parquet")
a = fr.filter(pl.col("A") & ~pl.col("t_claimed_any")).select("s1", "t", "p", s_name="s_name", t_name="t_name").with_columns(rule=pl.lit("A"))
d = pl.read_parquet(f"{OUT}/dom3_fr_add.parquet").select("s1", "t", "p", s_name="business_name", t_name="domain").with_columns(rule=pl.lit("D"))
add = pl.concat([a, d])
assert add["t"].n_unique() == add.height, "a target added twice"
s1 = pl.read_parquet(f"{W}/test/s1.parquet", columns=["s1", "entity_id", "country"])
tg = pl.scan_parquet(f"{W}/test/tgt.parquet").select("t", t_id="entity_id").join(add.select("t").lazy(), on="t", how="semi").collect()
add = add.join(s1.rename({"entity_id": "s_id"}), on="s1").join(tg, on="t")
assert (add["country"] == "France").all() and add.height == a.height + d.height
best = pl.read_csv(BEST, separator="\t", quote_char=None, infer_schema_length=0)
bp = (best.with_columns(m=pl.col("matched_entity_ids").fill_null("").str.split(",")).explode("m").filter(pl.col("m") != "")
      .select(s_id="source1_entity_id", t_id="m"))
assert add.join(bp.select("t_id").unique(), on="t_id", how="semi").height == 0, "target already claimed"
cnt = bp.group_by("s_id").agg(n=pl.len())
add = add.join(cnt, on="s_id", how="left").with_columns(pl.col("n").fill_null(0))


def f05(tp, pred, true):
    """F0.5 of one S1 from true positives, predicted and true counts (1.0 when both sets are empty)."""
    if true == 0 and pred == 0:
        return 1.0
    if tp == 0:
        return 0.0
    p, r = tp / pred, tp / true
    return 1.25 * p * r / (0.25 * p + r)


# per-S1 effect of k added pairs with n current (assumed correct) predictions
add = add.with_columns(k=pl.len().over("s_id"))
rows = []
for rule_set in (["A"], ["D"], ["A", "D"]):
    z = add.filter(pl.col("rule").is_in(rule_set))
    per = z.group_by("s_id").agg(n=pl.first("n"), k=pl.len(), q=pl.col("rule").replace_strict(Q, return_dtype=pl.Float64).mean())
    tot = 0.0
    for n, k, q in per.select("n", "k", "q").iter_rows():
        # binomial over the k added pairs
        from math import comb
        base = f05(n, n, n)
        e = 0.0
        for j in range(k + 1):
            pj = comb(k, j) * q ** j * (1 - q) ** (k - j)
            e += pj * (f05(n + j, n + k, n + j) - f05(n, n, n + j))
        tot += e
    dF = tot / 259452
    rows.append(("+".join(rule_set), z.height, dF, 0.15 * dF))
    print(f"rules {'+'.join(rule_set)}: pairs {z.height}, expected France dF0.5 {dF:+.5f}, LB {0.15 * dF:+.6f}")
# break-even per rule
for r in ("A", "D"):
    z = add.filter(pl.col("rule") == r)
    g = z["n"].map_elements(lambda n: f05(n + 1, n + 1, n + 1) - f05(n, n, n + 1), return_dtype=pl.Float64).mean()
    l = z["n"].map_elements(lambda n: f05(n, n + 1, n) - f05(n, n, n), return_dtype=pl.Float64).mean()
    print(f"rule {r}: mean TP gain {g:.4f}, mean FP loss {l:.4f}, break-even true rate {-l / (g - l):.3f}")
    for q in (0.8, 0.85, 0.9, 0.95, 0.99):
        print(f"   q={q}: France dF {(z.height * (q * g + (1 - q) * l)) / 259452:+.5f}")

raw = open(BEST, "rb").read()
lines = raw.split(b"\n")
for name, rule_set in (("v21e3x_fr2", ["A", "D"]), ("v21e3x_fr2_D", ["D"]), ("v21e3x_fr2_A", ["A"])):
    z = add.filter(pl.col("rule").is_in(rule_set))
    m = {r["s_id"]: r["ts"] for r in z.group_by("s_id", maintain_order=True).agg(ts=pl.col("t_id")).iter_rows(named=True)}
    out, changed = [], 0
    for i, ln in enumerate(lines):
        if i == 0 or not ln:
            out.append(ln); continue
        cr = ln.endswith(b"\r")
        body = ln[:-1] if cr else ln
        sid, _, rest = body.partition(b"\t")
        k = sid.decode()
        if k in m:
            cur = [v for v in rest.decode().split(",") if v]
            body = sid + b"\t" + ",".join(cur + [t for t in m[k] if t not in cur]).encode()
            changed += 1
        out.append(body + (b"\r" if cr else b""))
    os.makedirs(f"{OUT}/{name}", exist_ok=True)
    open(f"{OUT}/{name}/matching_results.tsv", "wb").write(b"\n".join(out))
    z.select("rule", "s_id", "t_id", "s_name", "t_name", "p", "n").with_columns(action=pl.lit("add")).sort("rule", "s_id").write_csv(f"{OUT}/{name}/changes.csv")
    print(name, "S1 lines changed:", changed, "pairs added:", z.height)
