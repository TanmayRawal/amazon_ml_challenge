"""Acronym targets -> the unique S1 in the same country whose name initials spell the acronym and whose address agrees
(same house number + similar street). Validated on US/India train (labels), applied to France test."""
import re, sys, unicodedata
import polars as pl
from rapidfuzz import fuzz
W = r"work"
STOP = {"de", "du", "des", "la", "le", "les", "l", "d", "et", "en", "au", "aux", "the", "of", "and", "a", "for", "sur"}
LEGAL = {"sarl", "sas", "sasu", "eurl", "sa", "sci", "ei", "snc", "scop", "sca", "selarl", "gie", "eirl", "scm", "scp", "sem",
         "inc", "llc", "ltd", "corp", "co", "pc", "lp", "llp", "pllc", "plc", "pvt", "private", "limited", "public", "incorporated",
         "corporation", "company", "group"}
def fold(s):
    """Lower-case `s` and strip accents (NFKD, combining marks removed); None -> ""."""
    return "".join(c for c in unicodedata.normalize("NFKD", s or "") if not unicodedata.combining(c)).lower()
def variants(name):
    """Candidate acronyms of an S1 name: initials of all words, without stop words, without legal forms, and without both.

    Only word lists of 2+ words produce an acronym. Returns a list of distinct lower-case strings.
    """
    w = re.findall(r"[a-z0-9]+", fold(name))
    out = set()
    for ws in (w, [x for x in w if x not in STOP], [x for x in w if x not in LEGAL], [x for x in w if x not in STOP and x not in LEGAL]):
        if len(ws) >= 2:
            out.add("".join(x[0] for x in ws))
    return list(out)
def run(split, countries):
    """Build acronym-target candidate pairs for one split.

    Args:
        split: "train" or "test" (parsed parquet folder under W).
        countries: countries to keep (S1 and target must share the country).

    Returns:
        (tg, j): the acronym targets (2-5 letters, not a domain, with a house number) and the candidate pairs
        (t, s1) whose S1 initials spell the acronym, share the target's first house number and have street
        token_set_ratio >= 70; column k = number of candidate S1s per target.
    """
    s1 = (pl.read_parquet(rf"{W}\{split}\s1.parquet", columns=["s1", "business_name", "country", "addr_nums", "addr_toks"])
          .filter(pl.col("country").is_in(countries)))
    tg = (pl.read_parquet(rf"{W}\{split}\tgt.parquet", columns=["t", "business_name", "country", "addr_nums", "addr_toks", "addr_blank", "is_domain"])
          .filter(pl.col("country").is_in(countries)))
    tg = tg.with_columns(acro=pl.col("business_name").str.replace_all(r"[.\s]", "")).filter(
        pl.col("acro").str.contains(r"^[A-Za-z]{2,5}$") & ~pl.col("is_domain")).with_columns(acro=pl.col("acro").str.to_lowercase())
    s1 = s1.with_columns(v=pl.Series([variants(n) for n in s1["business_name"].to_list()], dtype=pl.List(pl.String))).explode("v").drop_nulls("v")
    s1 = s1.filter(pl.col("v").str.len_chars().is_between(2, 5))
    sp = lambda c: pl.col(c).fill_null("").str.split(" ").list.eval(pl.element().filter(pl.element() != ""))
    s1 = s1.with_columns(sn=sp("addr_nums")).explode("sn").drop_nulls("sn")
    tg = tg.with_columns(tn=sp("addr_nums").list.first()).filter(pl.col("tn").is_not_null() & ~pl.col("addr_blank"))
    j = tg.select("t", "country", "acro", "tn", ta="addr_toks").join(
        s1.select("s1", "country", acro="v", tn="sn", sa="addr_toks"), on=["country", "acro", "tn"]).unique(["t", "s1"])
    j = j.with_columns(st=pl.Series([fuzz.token_set_ratio(a or "", b or "") for a, b in zip(j["ta"].to_list(), j["sa"].to_list())]))
    j = j.filter(pl.col("st") >= 70)
    j = j.with_columns(k=pl.len().over("t"))
    return tg, j
if __name__ == "__main__":
    pl.Config.set_tbl_rows(30); pl.Config.set_tbl_width_chars(200)
    tg, j = run("train", ["US", "India"])
    gt = pl.read_parquet(rf"{W}\train\gt.parquet").with_columns(y=pl.lit(1))
    owned = gt.select("t").unique()
    tr_sc = pl.scan_parquet(rf"{W}\train\scores_c.parquet").select("s1", "t", "p").collect().join(tg.select("t"), on="t", how="semi")
    j = j.join(gt, on=["s1", "t"], how="left").with_columns(pl.col("y").fill_null(0)).join(tr_sc, on=["s1", "t"], how="left")
    print("train acronym targets (with number):", tg.height, "owned share", round(tg.join(owned, on="t", how="semi").height / tg.height, 4))
    u = j.filter(pl.col("k") == 1)
    print(u.group_by("country").agg(n=pl.len(), prec=pl.col("y").mean(), cover=pl.len()))
    print("by model p:"); print(u.with_columns(b=pl.col("p").fill_null(0).cut([0.05, 0.5, 0.77])).group_by("b").agg(n=pl.len(), prec=pl.col("y").mean()).sort("b"))
    print("k>=2 groups: owner among them", j.filter(pl.col("k") >= 2).group_by("t").agg(pl.col("y").max()).select(pl.col("y").mean()))
    j.write_parquet(rf"{W}\w20\acro_train.parquet")
