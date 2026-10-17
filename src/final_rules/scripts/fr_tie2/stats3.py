"""French ties by stop words (v22), exploration: the stop-word sequence attribute.

Measures owner agreement vs same-name null for the stop-word sequence, the accented-word set and the stop-word set
(result: sequence agreement 0.969 vs null 0.808), lists the most common owner -> copy stop-word transitions, and
glances at entity IDs of owner/copy pairs. Prints only.
"""
exec(open("feats.py").read())
SW = {"de","du","des","la","le","les","l","d","et","en","au","aux"}
def sw(name):
    """Sequence of French stop words (incl. l/d/et; accents removed, lower case) in `name`, in order."""
    return tuple(x for x in (strip(t).lower() for t in re.findall(r"[^\W_]+", name or "")) if x in SW)
SF = {r[0]: (sw(r[1]), feats(r[1], r[2], r[3])["acc"]) for r in s1.select("s1","business_name","core_key","legal").iter_rows()}
TF = {r[0]: (sw(r[1]), feats(r[1], r[2], r[3])["acc"]) for r in tg.select("t","business_name","core_key","legal").iter_rows()}
cnt = s1.group_by("core_key").agg(k=pl.len(), s1s=pl.col("s1"))
x = tg.join(cnt, on="core_key")
single = x.filter(pl.col("k")==1).with_columns(s1=pl.col("s1s").list.first())
ag = [0,0,0]; n=0; trans = collections.Counter()
for t, s in single.select("t","s1").iter_rows():
    n+=1; ag[0]+= TF[t][0]==SF[s][0]; ag[1]+= TF[t][1]==SF[s][1]; ag[2] += set(TF[t][0])==set(SF[s][0])
    if TF[t][0]!=SF[s][0]: trans[(SF[s][0],TF[t][0])]+=1
random.seed(0); ag0=[0,0,0]; n0=0
for ck, s1s in cnt.filter(pl.col("k")>=2).select("core_key","s1s").iter_rows():
    for _ in range(min(3,len(s1s))):
        a,b = random.sample(s1s,2); n0+=1
        ag0[0]+= SF[a][0]==SF[b][0]; ag0[1]+=SF[a][1]==SF[b][1]; ag0[2]+=set(SF[a][0])==set(SF[b][0])
print("sw owner", ag[0]/n, "null", ag0[0]/n0); print("acc owner", ag[1]/n, "null", ag0[1]/n0); print("swset owner", ag[2]/n, "null", ag0[2]/n0)
print(trans.most_common(15))
# entity_id look
s1e = pl.read_parquet(rf"{W}\test\s1.parquet", columns=["s1","entity_id"]); tge = pl.read_parquet(rf"{W}\test\tgt.parquet", columns=["t","entity_id"])
print(s1e.head(5)); print(tge.head(5))
j = single.select("t","s1").join(s1e, on="s1").join(tge, on="t", suffix="_t")
print(j.head(10))
