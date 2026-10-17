"""France label-free tie-breaker calibration: on French blank targets whose core_key has exactly ONE French S1 (owner known),
measure how often raw formatting attributes of the target agree with the owner's; compare with agreement against a random
other French S1 of a different core (null). Then look at tie groups."""
import re, unicodedata
import polars as pl
W = r"work"
s1 = pl.read_parquet(rf"{W}\test\s1.parquet", columns=["s1", "country", "business_name", "core", "core_key", "legal"]).filter(pl.col("country") == "France")
tg = pl.read_parquet(rf"{W}\test\tgt.parquet", columns=["t", "country", "business_name", "core", "core_key", "legal", "addr_blank", "is_domain", "src"]).filter(
    (pl.col("country") == "France") & pl.col("addr_blank") & ~pl.col("is_domain") & (pl.col("core_key") != ""))
def attrs(name, core, legal):
    """Raw formatting attributes of a name: word order, accents, "&", "et", hyphen, parentheses, apostrophe, legal form
    and its position, dotted initials, double spaces. Returns a dict."""
    n = name or ""
    return {
        "order": core or "",
        "acc": any(unicodedata.combining(c) for c in unicodedata.normalize("NFKD", n)),
        "amp": "&" in n, "et": bool(re.search(r"\bet\b", n, re.I)),
        "hyph": "-" in n, "par": "(" in n, "apos": ("'" in n) or ("’" in n),
        "legal": legal or "", "legal_pos": ("end" if re.search(r"(SARL|SAS|SASU|EURL|SA|SCI|EI|SNC)\W*$", n, re.I) else ("start" if re.search(r"^\W*(SARL|SAS|SASU|EURL|SA|SCI|EI|SNC)\b", n, re.I) else "none")),
        "dots": bool(re.search(r"[A-Z]\.[A-Z]", n)), "dspace": "  " in n,
    }
cnt = s1.group_by("core_key").agg(k=pl.len(), s1s=pl.col("s1"))
x = tg.join(cnt, on="core_key")
single = x.filter(pl.col("k") == 1).with_columns(s1=pl.col("s1s").list.first())
S = s1.select("s1", sname="business_name", score="core", slegal="legal")
single = single.join(S, on="s1")
import random
random.seed(0)
rows = []
sl = S.to_dicts()
for r in single.iter_rows(named=True):
    a = attrs(r["business_name"], r["core"], r["legal"]); b = attrs(r["sname"], r["score"], r["slegal"])
    o = random.choice(sl); c = attrs(o["sname"], o["score"], o["slegal"])
    rows.append({k: (a[k] == b[k]) for k in a} | {f"null_{k}": (a[k] == c[k]) for k in a} | {"src": r["src"]})
d = pl.DataFrame(rows)
print("single-owner blank targets:", d.height)
ks=[c for c in d.columns if c!="src" and not c.startswith("null_")]
for k in ks: print(f"{k:10s} owner {d[k].mean():.3f}  null {d["null_"+k].mean():.3f}")
print("tie groups (k>=2):", x.filter(pl.col("k") >= 2).height)
