"""US/India rule D for domain-name targets (exhaustive candidate generation for the uniqueness check).
qualify(S1,t): domain label = concat of pieces (full S1 word | initial of distinct S1 word), any order, >=1 full word len>=3;
  and (t blank address) or [(t first number == S1 first number or t has no number) and street_ok (>=1 shared token,
  every unmatched t token fuzzy>=70 to an unmatched S1 token)].  Fire when exactly one S1 in the country qualifies."""
import re, sys, time, unicodedata
from functools import lru_cache
import polars as pl
from rapidfuzz import fuzz

W = r"work"
D = r"data/dataset"
pl.Config.set_tbl_cols(30); pl.Config.set_tbl_width_chars(250); pl.Config.set_fmt_str_lengths(45); pl.Config.set_tbl_rows(40)
CS = ["US", "India"]
split = sys.argv[1]; mode = sys.argv[2] if len(sys.argv) > 2 else "s100"; frac = mode
T0 = time.time()
def lap(m):
    """Print message `m` with the elapsed time since start."""
    print(f"[{time.time()-T0:6.1f}s] {m}", flush=True)
APOS = ("-", "'", "\u2019")

def translit(s):
    """Transliterate `s` by removing accents (NFKD, combining marks dropped)."""
    return "".join(c for c in unicodedata.normalize("NFKD", s) if not unicodedata.combining(c))

@lru_cache(maxsize=4_000_000)
def word_lists(name):
    """Distinct word tuples of a lower-cased name under 4 normalisations: non-alphanumerics deleted; apostrophes and
    hyphens as spaces; non-alphanumerics as spaces; accents transliterated. Cached."""
    low = name.lower()
    out = []
    for mode in (0, 1, 2, 3):
        s = translit(low) if mode == 3 else low
        if mode == 1:
            for a in APOS:
                s = s.replace(a, " ")
        s = re.sub(r"[^a-z0-9 ]", " " if mode in (2, 3) else "", s)
        w = tuple(x for x in s.split() if x)
        if w and w not in out:
            out.append(w)
    return out

def seg(domain, ws):
    """True when `domain` is a concatenation of pieces, each a full word or the initial of a distinct word of `ws`
    (first 8 words, any order, each used at most once), with at least one full word of 3+ letters."""
    ws = tuple(ws)[:8]; n = len(domain)
    @lru_cache(None)
    def go(i, used, full):
        """Can domain[i:] be segmented given the bitmask `used` of words taken and whether a full word was used."""
        if i == n:
            return full
        for j, w in enumerate(ws):
            if used >> j & 1:
                continue
            if domain.startswith(w, i) and go(i + len(w), used | (1 << j), full or len(w) >= 3):
                return True
            if domain[i] == w[0] and go(i + 1, used | (1 << j), full):
                return True
        return False
    return go(0, 0, False)

@lru_cache(maxsize=2_000_000)
def match(domain, name):
    """True when the domain label can be built from S1 name `name` under any of its word_lists normalisations."""
    if not domain or not name:
        return False
    return any(seg(domain, ws) for ws in word_lists(name))

def street_ok(st, tt):
    """Street agreement: S1 tokens `st` and target tokens `tt` share at least one token, and every unmatched target
    token has fuzz ratio >= 70 with some unmatched S1 token."""
    s = [w for w in (st or "").split()]; t = [w for w in (tt or "").split()]
    ss, ts = set(s), set(t)
    if len(ts & ss) == 0:
        return False
    s_un = [w for w in s if w not in ts]; t_un = [w for w in t if w not in ss]
    return all(any(fuzz.ratio(w, v) >= 70 for v in s_un) for w in t_un)

def join_words(name):
    """All words of 3+ letters of `name` over its normalisations; used as blocking keys against domain substrings."""
    ws = set()
    for w in word_lists(name):
        ws.update(x for x in w if len(x) >= 3)
    return list(ws)

def subs(d):
    """All distinct substrings of domain `d` of length 3 to 30."""
    n = len(d)
    return list({d[i:j] for i in range(n) for j in range(i + 3, min(n, i + 30) + 1)})

# ---------- targets in scope
tg = (pl.scan_parquet(rf"{W}\{split}\tgt.parquet").filter(pl.col("is_domain") & pl.col("country").is_in(CS))
      .select("t", "entity_id", "country", "domain", "business_name", t_toks="addr_toks", t_nums="addr_nums", t_blank="addr_blank").collect())
if split == "train":
    raw = pl.read_csv(rf"{D}\train\train_ground_truth.tsv", separator="\t", quote_char=None, infer_schema_length=0)
    hid = (raw.filter((pl.col("source1_entity_id").str.extract(r"(\d+)$").cast(pl.Int64) % 100) < 20)
           .with_columns(e=pl.col("matched_entity_ids").str.split(",")).explode("e").select(entity_id="e").unique())
    n0 = tg.height
    tg = tg.join(hid, on="entity_id", how="anti")
    if mode == "un":
        clm = pl.scan_parquet(rf"{W}/train/scores_c.parquet").filter(pl.col("p") >= 0.5).select("t").join(tg.select("t").lazy(), on="t", how="semi").unique().collect()
        tg = tg.join(clm, on="t", how="anti")
    else:
        tg = tg.filter(pl.col("t").hash(11) % 100 < int(mode[1:]))
    lap(f"train targets {n0} -> after hid/sample {tg.height}")
else:
    v = pl.read_parquet("v22c.parquet", columns=["t"]).unique()
    tg = tg.join(v, on="t", how="anti")
    lap(f"test unclaimed domain targets {tg.height}")
tg = tg.drop("entity_id").with_columns(t_blank=pl.col("t_blank") | ((pl.col("t_toks").fill_null("") == "") & (pl.col("t_nums").fill_null("") == "")),
                                        t_num0=pl.col("t_nums").fill_null("").str.split(" ").list.first().replace("", None))
tg = tg.filter(pl.col("domain").fill_null("").str.len_chars() >= 3)
tg = tg.with_columns(cls=pl.when(pl.col("t_blank")).then(pl.lit("blank")).when(pl.col("t_num0").is_null()).then(pl.lit("nonum")).otherwise(pl.lit("num")))
print(tg.group_by("country", "cls").len().sort("country", "cls"))
ts = tg.select("t", "country", "cls", "t_num0", "t_toks", sub=pl.col("domain").map_elements(subs, return_dtype=pl.List(pl.String))).explode("sub")
lap(f"target substrings {ts.height}")
# ---------- S1
s1 = (pl.scan_parquet(rf"{W}\{split}\s1.parquet").filter(pl.col("country").is_in(CS))
      .select("s1", "country", "business_name", s_toks="addr_toks", s_nums="addr_nums").collect())
s1 = s1.with_columns(s_num0=pl.col("s_nums").fill_null("").str.split(" ").list.first().replace("", None))
vocab = ts.select("country", "sub").unique()
sw = s1.select("s1", "country", "s_num0", "s_toks", word=pl.col("business_name").map_elements(join_words, return_dtype=pl.List(pl.String))).explode("word")
sw = sw.join(vocab, left_on=["country", "word"], right_on=["country", "sub"], how="semi")
lap(f"S1 word rows in domain vocab {sw.height}")
# (a) numbered targets
ca = ts.filter(pl.col("cls") == "num").join(sw.filter(pl.col("s_num0").is_not_null()), left_on=["country", "t_num0", "sub"], right_on=["country", "s_num0", "word"]).select("t", "s1").unique()
lap(f"cand num {ca.height}")
# (b) no-number non-blank: share word and >=1 address token
tb = ts.filter(pl.col("cls") == "nonum").with_columns(tok=pl.col("t_toks").str.split(" ")).explode("tok").filter(pl.col("tok").fill_null("") != "").select("t", "country", "sub", "tok").unique()
swb = sw.join(tb.select("country", "sub").unique(), left_on=["country", "word"], right_on=["country", "sub"], how="semi").with_columns(tok=pl.col("s_toks").str.split(" ")).explode("tok")
swb = swb.join(tb.select("country", "tok").unique(), on=["country", "tok"], how="semi").select("s1", "country", "word", "tok").unique()
cb = tb.join(swb, left_on=["country", "sub", "tok"], right_on=["country", "word", "tok"]).select("t", "s1").unique()
lap(f"cand nonum {cb.height}")
# (c) blank: whole country, share word
cc = ts.filter(pl.col("cls") == "blank").join(sw, left_on=["country", "sub"], right_on=["country", "word"]).select("t", "s1").unique()
lap(f"cand blank {cc.height}")
c = pl.concat([ca, cb, cc]).unique()
x = c.join(tg.select("t", "country", "domain", "cls", "t_num0", "t_toks"), on="t").join(s1.select("s1", "business_name", "s_num0", "s_toks"), on="s1")
lap(f"candidates {x.height}")
m = [match(d, n) for d, n in zip(x["domain"].to_list(), x["business_name"].to_list())]
x = x.with_columns(seg=pl.Series(m, dtype=pl.Boolean)).filter(pl.col("seg"))
lap(f"seg true {x.height}")
x = x.with_columns(addr_ok=(pl.col("cls") == "blank") | pl.col("t_num0").is_null() | (pl.col("t_num0") == pl.col("s_num0")).fill_null(False))
so = [True if cl == "blank" else street_ok(a, b) for cl, a, b in zip(x["cls"].to_list(), x["s_toks"].to_list(), x["t_toks"].to_list())]
x = x.with_columns(sok=pl.Series(so, dtype=pl.Boolean))
x = x.with_columns(q=pl.col("seg") & pl.col("addr_ok") & pl.col("sok"))
x = x.with_columns(n_q=pl.col("q").sum().over("t"), n_seg=pl.len().over("t"))
fire = x.filter(pl.col("q") & (pl.col("n_q") == 1))
lap(f"fired {fire.height} (targets {fire['t'].n_unique()})")
if split == "train":
    gt = pl.read_parquet(rf"{W}\train\gt.parquet").join(tg.select("t"), on="t", how="semi").with_columns(y=pl.lit(True))
    fire = fire.join(gt, on=["s1", "t"], how="left").with_columns(pl.col("y").fill_null(False))
    cl = pl.scan_parquet(rf"{W}\train\scores_c.parquet").filter(pl.col("p") >= 0.5).select("t").join(fire.select("t").lazy(), on="t", how="semi").unique().collect().with_columns(t_claim=pl.lit(True))
    pp = pl.scan_parquet(rf"{W}\train\scores_c.parquet").select("s1", "t", "p").join(fire.select("s1", "t").lazy(), on=["s1", "t"], how="semi").collect()
    fire = fire.join(cl, on="t", how="left").with_columns(pl.col("t_claim").fill_null(False)).join(pp, on=["s1", "t"], how="left")
    print(fire.group_by("country", "cls").agg(n=pl.len(), prec=pl.col("y").mean(), n_un=(~pl.col("t_claim")).sum(), prec_un=pl.col("y").filter(~pl.col("t_claim")).mean(),
          n_pnull=pl.col("p").is_null().sum()).sort("country", "cls"))
    print("OVERALL n", fire.height, "prec", round(fire["y"].mean(), 4), "| unclaimed n", (~fire["t_claim"]).sum(), "prec", round(fire.filter(~pl.col("t_claim"))["y"].mean(), 4))
    un = fire.filter(~pl.col("t_claim"))
    print(un.group_by("country").agg(n=pl.len(), prec=pl.col("y").mean()))
    print("unclaimed FPs:"); print(un.filter(~pl.col("y")).select("domain", "business_name", "cls", "t_num0", "s_num0", "t_toks", "s_toks", "p").head(25))
    print("unclaimed TP sample:"); print(un.filter(pl.col("y")).select("domain", "business_name", "cls", "t_toks", "s_toks", "p").head(10))
    fire.write_parquet(f"dom_train_fire_{frac}.parquet")
else:
    a = pl.scan_parquet(rf"{W}\w20\test\scores_c.parquet").select("s1", "t", p_them="p").join(fire.select("s1", "t").lazy(), on=["s1", "t"], how="semi").collect()
    b = pl.scan_parquet(rf"{W}\test\scores_c.parquet").select("s1", "t", p_ours="p").join(fire.select("s1", "t").lazy(), on=["s1", "t"], how="semi").collect()
    fire = fire.join(a, on=["s1", "t"], how="left").join(b, on=["s1", "t"], how="left")
    print(fire.group_by("country", "cls").agg(n=pl.len(), pthem_null=pl.col("p_them").is_null().sum(), pours_null=pl.col("p_ours").is_null().sum(), pthem_med=pl.col("p_them").median()).sort("country", "cls"))
    print(fire.select("domain", "business_name", "cls", "t_toks", "s_toks", "p_them", "p_ours").sample(min(25, fire.height), seed=5))
    fire.write_parquet("dom_test_fire.parquet")
    assert fire["t"].n_unique() == fire.height
    fire.select(pl.col("s1").cast(pl.UInt32), pl.col("t").cast(pl.UInt32)).write_parquet("dom_usin_add_candidate.parquet")
lap("done")
