"""General domain matcher: domain = concatenation of pieces, each a full word or the initial of a distinct word of the
S1 name (non-ASCII letters deleted, any order, stopwords optional), >=1 full word of length>=3 and covering the domain.
Rule: unclaimed domain target, candidates from pool (+ key join), same street (fuzzy) and same number or target without
number, unique segmentation claimant.  Train validation via scores_c candidates."""
import re
import sys
from functools import lru_cache

import polars as pl
from rapidfuzz import fuzz

W = r"work"
OUT = r"work/scratch/france2"
D = r"data/dataset"
pl.Config.set_tbl_cols(30); pl.Config.set_tbl_width_chars(300); pl.Config.set_fmt_str_lengths(60); pl.Config.set_tbl_rows(60)
REG = {"nouvelle", "aquitaine", "hauts", "de", "france", "pays", "la", "loire", "atlantique", "gironde", "nord", "pas", "calais"}


def words(name, mode):
    """Words of the lower-cased name with non-[a-z0-9 ] characters deleted (mode 0), after hyphens/apostrophes become
    spaces (mode 1), or with those characters turned into spaces (mode 2)."""
    s = name.lower()
    s = s.replace("-", " ").replace("'", " ").replace("’", " ") if mode else s
    s = re.sub(r"[^a-z0-9 ]", " " if mode == 2 else "", s)
    return [w for w in s.split() if w]


def seg(domain, ws):
    """True when `domain` is a concatenation of pieces, each a full word or the initial of a distinct word of `ws`
    (first 8 words, any order, each used at most once), with at least one full word of 3+ letters."""
    ws = tuple(ws)[:8]
    n = len(domain)

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


def match(domain, name):
    """True when the domain can be segmented from the S1 name under any of the three word normalisations."""
    if not domain or not name:
        return False
    for mode in (0, 1, 2):
        if seg(domain, words(name, mode)):
            return True
    return False


def street_ok(st, tt):
    """Street agreement (region words ignored): at least one shared token, and every unmatched target token is a typo
    (ratio >= 70) of an unmatched S1 token."""
    s = [w for w in (st or "").split() if w not in REG]
    t = [w for w in (tt or "").split() if w not in REG]
    ss, ts = set(s), set(t)
    s_un = [w for w in s if w not in ts]
    t_un = [w for w in t if w not in ss]
    if len(ts & ss) == 0:
        return False
    return all(any(fuzz.ratio(w, v) >= 70 for v in s_un) for w in t_un)


def feats(x):
    """Add rule-D columns to candidate pairs: num_eq, t_nonum, seg (domain matches name), sok (street_ok), cl (pair
    qualifies: seg & sok & (same number or target without number)) and n_cl (qualifying S1s per target)."""
    x = x.with_columns(
        num_eq=(pl.col("s_nums").fill_null("").str.split(" ").list.set_intersection(pl.col("t_nums").fill_null("").str.split(" ")).list.len() > 0)
        & (pl.col("t_nums").fill_null("") != ""),
        t_nonum=pl.col("t_nums").fill_null("") == "")
    m = [match(d, n) for d, n in zip(x["domain"].to_list(), x["business_name"].to_list())]
    so = [street_ok(a, b) for a, b in zip(x["s_toks"].to_list(), x["t_toks"].to_list())]
    x = x.with_columns(seg=pl.Series(m), sok=pl.Series(so))
    x = x.with_columns(cl=pl.col("seg") & pl.col("sok") & (pl.col("num_eq") | pl.col("t_nonum")))
    return x.with_columns(n_cl=pl.col("cl").sum().over("t"))


part = sys.argv[1]
if part == "test":
    acc = pl.read_parquet(f"{OUT}/best_pairs.parquet")
    tg = (pl.scan_parquet(f"{W}/test/tgt.parquet").filter(pl.col("is_domain") & (pl.col("country") == "France"))
          .select("t", "domain", t_toks="addr_toks", t_nums="addr_nums").collect()).join(acc, on="t", how="anti")
    pool = pl.scan_parquet(f"{W}/w20/v20_pool.parquet").join(tg.select("t").lazy(), on="t", how="semi").collect()
    keyc = pl.read_parquet(f"{OUT}/dom_test.parquet", columns=["s1", "t", "country"]).filter(pl.col("country") == "France").join(tg.select("t"), on="t", how="semi").select("s1", "t")
    c = pl.concat([pool.select("s1", "t"), keyc]).unique().join(pool, on=["s1", "t"], how="left")
    s1 = pl.read_parquet(f"{W}/test/s1.parquet", columns=["s1", "business_name", "addr_toks", "addr_nums", "country"]).rename({"addr_toks": "s_toks", "addr_nums": "s_nums"})
    x = c.join(tg, on="t").join(s1, on="s1").filter(pl.col("country") == "France")
    x = feats(x)
    u = x.filter(pl.col("cl") & (pl.col("n_cl") == 1))
    print("unclaimed FR domain targets:", tg.height, " candidates:", x.height, " unique claimant:", u.height)
    print(u.with_columns(pb=pl.col("p").cut([0.1, 0.3, 0.5, 0.77])).group_by("pb", "num_eq").agg(n=pl.len()).sort("pb", "num_eq"))
    print(u.select("domain", "business_name", "p", "num_eq", "t_toks", "s_toks").sample(30, seed=4))
    u.write_parquet(f"{OUT}/dom3_fr_add.parquet")
else:
    cs = ["US", "India"]
    raw = pl.read_csv(f"{D}/train/train_ground_truth.tsv", separator="\t", quote_char=None, infer_schema_length=0)
    hid = (raw.filter((pl.col("source1_entity_id").str.extract(r"(\d+)$").cast(pl.Int64) % 100) < 20)
           .with_columns(e=pl.col("matched_entity_ids").str.split(",")).explode("e").select(entity_id="e").unique())
    tg = (pl.scan_parquet(f"{W}/train/tgt.parquet").filter(pl.col("is_domain") & pl.col("country").is_in(cs))
          .select("t", "entity_id", "domain", t_toks="addr_toks", t_nums="addr_nums").collect())
    tg = tg.join(hid, on="entity_id", how="anti").drop("entity_id")
    # sample 40% of targets for speed
    tg = tg.filter(pl.col("t").hash(7) % 10 < 4)
    sc = pl.scan_parquet(f"{W}/train/scores_c.parquet").select("s1", "t", "p").join(tg.select("t").lazy(), on="t", how="semi").filter(pl.col("p") > 0.001).collect()
    s1 = pl.read_parquet(f"{W}/train/s1.parquet", columns=["s1", "business_name", "addr_toks", "addr_nums", "country"]).rename({"addr_toks": "s_toks", "addr_nums": "s_nums"})
    x = sc.join(tg, on="t").join(s1, on="s1")
    print("train candidate rows", x.height)
    x = feats(x)
    gt = pl.read_parquet(f"{W}/train/gt.parquet").join(tg.select("t"), on="t", how="semi").with_columns(y=pl.lit(1))
    x = x.join(gt, on=["s1", "t"], how="left").with_columns(pl.col("y").fill_null(0))
    x = x.with_columns(t_claim=(pl.col("p") >= 0.5).max().over("t"))
    u = x.filter(pl.col("cl") & (pl.col("n_cl") == 1))
    print(u.with_columns(pb=pl.col("p").cut([0.1, 0.3, 0.5, 0.77])).group_by("country", "pb").agg(n=pl.len(), true=pl.col("y").mean(),
          n_un=(~pl.col("t_claim")).sum(), true_un=pl.col("y").filter(~pl.col("t_claim")).mean()).sort("country", "pb"))
    print(u.group_by("country", "num_eq").agg(n=pl.len(), true=pl.col("y").mean(), n_un=(~pl.col("t_claim")).sum(), true_un=pl.col("y").filter(~pl.col("t_claim")).mean()).sort("country", "num_eq"))
