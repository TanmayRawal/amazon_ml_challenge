"""French ties by stop words (v22), exploration: which formatting attributes copies inherit.

For every feats() attribute, compares the agreement rate between a blank target and its known owner (single same-name
S1) with the agreement between two random S1s sharing a core name (null). Also prints the most common case, legal-form
spelling and skeleton transitions from owner to copy. Prints only.
"""
exec(open("feats.py").read())
SF = {r[0]: feats(r[1], r[2], r[3]) for r in s1.select("s1","business_name","core_key","legal").iter_rows()}
TF = {r[0]: feats(r[1], r[2], r[3]) for r in tg.select("t","business_name","core_key","legal").iter_rows()}
TS = dict(tg.select("t","src").iter_rows())
cnt = s1.group_by("core_key").agg(k=pl.len(), s1s=pl.col("s1"))
x = tg.join(cnt, on="core_key")
single = x.filter(pl.col("k")==1).with_columns(s1=pl.col("s1s").list.first())
keys = list(next(iter(SF.values())).keys())
ag = collections.Counter(); n = 0
for t, s in single.select("t","s1").iter_rows():
    n += 1
    for k in keys: ag[k] += TF[t][k] == SF[s][k]
# null: target vs random other same-core S1 in tie groups (not the argmax... use all S1 pairs same-core)
random.seed(0)
ag0 = collections.Counter(); n0 = 0
for ck, s1s in cnt.filter(pl.col("k")>=2).select("core_key","s1s").iter_rows():
    for _ in range(min(3, len(s1s))):
        a, b = random.sample(s1s, 2); n0 += 1
        for k in keys: ag0[k] += SF[a][k] == SF[b][k]
print("n_owner", n, "n_null_pairs", n0)
for k in keys: print(f"{k:8s} owner {ag[k]/n:.4f}  s1-s1 same-core {ag0[k]/n0:.4f}")
# by src for case
for src in (2,3):
    c = collections.Counter()
    for t, s in single.select("t","s1").iter_rows():
        if TS[t]==src: c[(SF[s]["case"], TF[t]["case"])] += 1
    print(src, c.most_common(10))
c = collections.Counter((SF[s]["legraw"], TF[t]["legraw"]) for t, s in single.select("t","s1").iter_rows())
print(c.most_common(20))
c = collections.Counter((SF[s]["skel"], TF[t]["skel"]) for t, s in single.select("t","s1").iter_rows() if SF[s]["skel"] != TF[t]["skel"])
print(c.most_common(25))
