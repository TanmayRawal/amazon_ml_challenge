"""French ties by stop words (v22 fr_tie2_add), shared setup: formatting features of French names.

Loads the French test S1s and the French blank-address, non-domain targets with a non-empty core name, and defines
feats(), the raw formatting attributes a copy may inherit from its owner (word order, accents, legal-form spelling,
case, punctuation, whitespace, "&", parentheses, hyphens). Executed (exec) at the top of stats2.py, stats3.py and nb2.py.
"""
import re, unicodedata, collections, random
import polars as pl
W = r"work"
s1 = pl.read_parquet(rf"{W}\test\s1.parquet", columns=["s1","country","business_name","core","core_key","legal"]).filter(pl.col("country")=="France")
tg = pl.read_parquet(rf"{W}\test\tgt.parquet", columns=["t","country","src","business_name","core","core_key","legal","addr_blank","is_domain"]).filter(
    (pl.col("country")=="France") & pl.col("addr_blank") & ~pl.col("is_domain") & (pl.col("core_key")!=""))
LEG = set(s1["legal"].unique().to_list()) | set(tg["legal"].unique().to_list()); LEG.discard("")
def strip(s):
    """Remove accents (combining marks) from `s`, keeping case."""
    return "".join(c for c in unicodedata.normalize("NFKD", s) if not unicodedata.combining(c))
def feats(name, core_key, legal):
    """Formatting attributes of a raw business name.

    Args:
        name: raw business_name.
        core_key: space-separated core name words (the words shared by all copies).
        legal: parsed legal form ("" if none).

    Returns:
        dict with skel (non-core, non-legal words), order (core words in order), acc (accented words), legraw
        (legal-form spelling style | position | bracketed), case, punct, ws (whitespace quirks), dig, fr ("(france)"),
        amp, par, hyph, anyacc, legal.
    """
    n = name or ""; ck = set((core_key or "").split())
    toks = re.findall(r"[^\W_]+", n)
    nt = [strip(x).lower() for x in toks]
    lg = [i for i, x in enumerate(nt) if x in LEG or (legal and x == legal.replace(".", "").replace(" ", ""))]
    # dotted legal like S.A.R.L -> letters split; detect by raw regex
    skel = tuple(x for i, x in enumerate(nt) if i not in lg and x not in ck and len(x) > 0)
    order = tuple(x for x in nt if x in ck)
    acc = tuple(sorted(strip(t).lower() for t in toks if strip(t) != t))
    legraw = ""
    m = re.search(r"(?i)\b(s\.?\s?a\.?\s?r\.?\s?l\.?|s\.?\s?a\.?\s?s\.?\s?u?\.?|e\.?\s?u\.?\s?r\.?\s?l\.?|s\.?\s?a\.?|e\.?\s?i\.?|s\.?\s?c\.?\s?i\.?|s\.?\s?n\.?\s?c\.?)(?=\W|$)", n)
    if legal and m:
        r = m.group(0)
        style = ("dot" if "." in r else "nodot") + ("U" if r.isupper() else "T" if r[:1].isupper() else "L")
        pos = "start" if m.start() == 0 else ("end" if n[m.end():].strip(" )]") == "" else "mid")
        br = "br" if (m.start() > 0 and n[m.start()-1] in "([") else ""
        legraw = f"{style}|{pos}|{br}"
    alpha = [t for t in toks if any(ch.isalpha() for ch in t) and strip(t).lower() in ck]
    def cs(t):
        """Case class of token `t`: "U" upper (2+ chars), "T" title, "L" lower."""
        return "U" if t.isupper() and len(t) > 1 else ("T" if t[:1].isupper() else "L")
    case = "".join(sorted(set(cs(t) for t in alpha)))
    punct = "".join(sorted(set(ch for ch in n if not ch.isalnum() and not ch.isspace() and ch not in "&()-")))
    ws = ("dbl" if "  " in n else "") + ("lead" if n[:1] == " " else "") + ("trail" if n[-1:] == " " else "")
    dig = bool(re.search(r"\d", n))
    fr = "(france)" in n.lower()
    return dict(skel=skel, order=order, acc=acc, legraw=legraw, case=case, punct=punct, ws=ws, dig=dig, fr=fr,
                amp="&" in n, par="(" in n, hyph="-" in n, anyacc=len(acc) > 0, legal=legal or "")
