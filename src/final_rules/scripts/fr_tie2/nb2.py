"""French ties by stop words (v22 fr_tie2_add): naive-Bayes owner posterior with the stop-word attribute.

Extends fr_tie_nb.py. Attributes: joint (word order, stop-word sequence de/du/des/la/le/les/en/au/aux), accents, "&",
parentheses, hyphen, plus a legal-form transition table, all learned on French blank targets with a single same-name
S1. For every tie target (2+ same-name S1s) the posterior of each claimant is computed and the argmax kept; four
variants are compared (baseline, +stop words, per-token accents, +copy-count prior) by agreement with model-claimed
ties. Writes post_sw.parquet (+stop words, used by final.py and fr_guard.py) and post_sw_cnt.parquet.
"""
exec(open("feats.py").read())
import math
SW = {"de","du","des","la","le","les","en","au","aux"}
def sw(name):
    """Sequence of French stop words (accents removed, lower case) in `name`, in order."""
    return tuple(x for x in (strip(t).lower() for t in re.findall(r"[^\W_]+", name or "")) if x in SW)
def A(name, core_key, legal):
    """Attribute tuple of a name: (order, stop words, any accent, "&", parentheses, hyphen, legal form, accented words)."""
    f = feats(name, core_key, legal)
    return (f["order"], sw(name), f["anyacc"], f["amp"], f["par"], f["hyph"], f["legal"], f["acc"])
SA = {r[0]: A(r[1], r[2], r[3]) for r in s1.select("s1","business_name","core_key","legal").iter_rows()}
TA = {r[0]: A(r[1], r[2], r[3]) for r in tg.select("t","business_name","core_key","legal").iter_rows()}
TS = dict(tg.select("t","src").iter_rows())
cnt = s1.group_by("core_key").agg(k=pl.len(), s1s=pl.col("s1"))
x = tg.join(cnt, on="core_key")
single = x.filter(pl.col("k")==1).with_columns(s1=pl.col("s1s").list.first())
# attributes: 0 joint(order,sw), 1 anyacc, 2 amp, 3 par, 4 hyph ; legal transition ; variant B: acc per-token instead of anyacc
def key(a, i, accfine):
    """Value of attribute i (0 = order+stop words, 1 = accents, 2-4 = "&", parentheses, hyphen) of tuple `a`;
    `accfine` compares accented words instead of the any-accent flag."""
    if i == 0: return (a[0], a[1])
    if i == 1: return a[7] if accfine else a[2]
    return a[i+1]
def learn(accfine, joint):
    """Learn on single-owner targets the owner agreement rate of each of the 5 attributes and the legal-form
    transition counts. `joint`: attribute 0 includes the stop-word sequence. Returns (rates, legT, legS)."""
    ag = [0]*5; n=0; legT=collections.Counter(); legS=collections.Counter()
    for t, s in single.select("t","s1").iter_rows():
        a, b = TA[t], SA[s]; n+=1
        for i in range(5):
            if i==0 and not joint: ag[i] += a[0]==b[0]
            else: ag[i] += key(a,i,accfine)==key(b,i,accfine)
        legT[(b[6],a[6])]+=1; legS[b[6]]+=1
    return [c/n for c in ag], legT, legS
# count prior: owner has other predicted copy in same src?
fin = pl.concat([d.select(pl.col("s1").cast(pl.Int64), pl.col("t").cast(pl.Int64)) for d in [pl.read_parquet(rf"{W}\w20\final_pairs.parquet"), pl.read_parquet(rf"{W}\w20\acro_add.parquet"), pl.read_parquet(rf"{W}\w20\fr_tie_add.parquet")]]).unique()
tsrc = pl.read_parquet(rf"{W}\test\tgt.parquet", columns=["t","src"]).with_columns(pl.col("t").cast(pl.Int64))
frp = fin.join(s1.select(pl.col("s1").cast(pl.Int64)), on="s1").join(tsrc, on="t")
bysrc = collections.defaultdict(set)
for s, t, src in frp.select("s1","t","src").iter_rows(): bysrc[(s, src)].add(t)
def other(s, t):
    """True when S1 `s` already has another predicted French target from the same source as target `t`."""
    return len(bysrc.get((s, TS[t]), set()) - {t}) > 0
o1 = sum(other(s,t) for t,s in single.select("t","s1").iter_rows())/single.height
ties = x.filter(pl.col("k")>=2)
nn=0; o0=0
for t, s1s in ties.select("t","s1s").iter_rows():
    for s in s1s: nn+=1; o0+=other(s,t)
print(f"count prior: owner has other same-src pred {o1:.4f}; tie-claimant avg {o0/nn:.4f}")
claimed = dict(fin.join(ties.select(pl.col("t").cast(pl.Int64)), on="t").select("t","s1").iter_rows())
prev = set(pl.read_parquet(rf"{W}\w20\fr_tie_add.parquet")["t"].to_list())
pool = pl.read_parquet(rf"{W}\w20\test\scores_c.parquet", columns=["s1","t","p"]).join(ties.select("t"), on="t")
P = {(a,b): c for a,b,c in pool.iter_rows()}
def run(accfine, joint, usecount, tag):
    """Compute the tie posterior for one model variant and print agreement with claimed ties at 0.75/0.9/0.95.

    Args:
        accfine: per-word accents instead of the any-accent flag.
        joint: include the stop-word sequence with word order.
        usecount: add the copy-count prior (claimant already has a same-source copy).
        tag: label for the printout.

    Returns:
        DataFrame t, s1 (argmax claimant), post, claimant, p, prev, claimed, agree.
    """
    Aa, legT, legS = learn(accfine, joint)
    forms = sorted({k[1] for k in legT} | {k[0] for k in legT})
    def ll(ta, sa, t, s):
        """Log-likelihood that S1 `s` (attributes `sa`) is the owner of target `t` (attributes `ta`)."""
        v = 0.0
        for i in range(5):
            eq = (ta[0]==sa[0]) if (i==0 and not joint) else (key(ta,i,accfine)==key(sa,i,accfine))
            v += math.log(Aa[i] if eq else max(1-Aa[i],1e-3))
        v += math.log((legT[(sa[6],ta[6])]+0.5)/(legS[sa[6]]+0.5*len(forms)))
        if usecount:
            h = other(s, t) if claimed.get(t) != s else len(bysrc.get((s,TS[t]),set())-{t})>0
            v += math.log(o1/(o0/nn)) if h else math.log((1-o1)/(1-o0/nn))
        return v
    rows=[]
    for t, s1s in ties.select("t","s1s").iter_rows():
        L = [ll(TA[t], SA[s], t, s) for s in s1s]; m=max(L); w=[math.exp(v-m) for v in L]; z=sum(w)
        j = max(range(len(s1s)), key=lambda i: w[i])
        rows.append((t, s1s[j], w[j]/z))
    r = pl.DataFrame(rows, schema=["t","s1","post"], orient="row")
    r = r.with_columns(claimant=pl.col("t").cast(pl.Int64).replace_strict(claimed, default=None),
                       p=pl.struct("s1","t").map_elements(lambda d: P.get((d["s1"],d["t"]), None), return_dtype=pl.Float64),
                       prev=pl.col("t").is_in(list(prev)))
    r = r.with_columns(claimed=pl.col("claimant").is_not_null(), agree=pl.col("claimant")==pl.col("s1"))
    print(f"== {tag}: A={[round(a,4) for a in Aa]}")
    for th in (0.75, 0.9, 0.95):
        q = r.filter(pl.col("post")>=th)
        cq = q.filter(pl.col("claimed"))
        un = q.filter(~pl.col("claimed") & ~pl.col("prev"))
        print(f"  post>={th}: n={q.height} claimed={cq.height} agree={cq['agree'].mean() if cq.height else None:.4f} disagree_n={(~cq['agree']).sum()} new_unclaimed={un.height} new_unclaimed_p>=0.2={un.filter(pl.col('p')>=0.2).height} p_null={un['p'].is_null().sum()}")
    return r
r0 = run(False, False, False, "baseline (orig attrs)")
r1 = run(False, True, False, "+stopwords joint w/ order")
r2 = run(True, True, False, "+stopwords, per-token accents")
r3 = run(False, True, True, "+stopwords +count prior")
r1.write_parquet("post_sw.parquet"); r3.write_parquet("post_sw_cnt.parquet")
