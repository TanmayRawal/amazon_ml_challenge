"""Guarded France ensemble (v23): tighten the pair classes used by the approximate France rule chain.

Rewrites frens/classes.parquet (written by fr_chain.py) in place so that the chain reproduces the uploaded French rows:
- audit rule 3 becomes strict: exact name, same house number, and no street-name token pair with ratio >= 75;
- the vocabulary-swap flag is narrowed to one swapped word on each side, both frequent French type words
  (document frequency >= 1000), Levenshtein distance > 2, and not an abbreviation of each other;
- the list-word floor applies to appended list words only (floor = sufB); kill = r1 | r3 | r4.
Run before fr_chain2.py.
"""
import polars as pl
from rapidfuzz import fuzz
from rapidfuzz.distance import Levenshtein
W = r"work"; OUT = rf"{W}\w20\frens"
cls = pl.read_parquet(rf"{OUT}\classes.parquet")
s1 = pl.read_parquet(rf"{W}\test\s1.parquet", columns=["s1", "addr_toks", "core", "country"])
tg = pl.read_parquet(rf"{W}\test\tgt.parquet", columns=["t", "addr_toks", "core"])
GEN = set("rue r av ave avenue bd boulevard blvd all allee pl place cours crs chemin che ch imp impasse rte route quai de du des la le les d l no n b bis ter quater st saint sainte ste parv parvis square sq cite sentier voie residence res lieu dit lot zi za zac et".split())
fr_s1 = s1.filter(pl.col("country") == "France")
adf = fr_s1.select(w=pl.col("addr_toks").fill_null("").str.split(" ")).explode("w").filter(pl.col("w") != "").group_by("w").len()
BIG = set(adf.filter(pl.col("len") > 3000)["w"].to_list())      # cities, regions, departments
def street(s):
    """Street-name tokens of an address token string: drops generic street words, frequent place names, digits and
    single letters."""
    return [x for x in (s or "").split() if x not in GEN and x not in BIG and not x.isdigit() and len(x) > 1]
# r3 strict: exact + num_eq + street-name tokens on both sides and NO token pair with ratio >= 75
c3 = cls.filter((pl.col("name") == "exact") & (pl.col("num") == "num_eq")).join(s1.select("s1", sa="addr_toks"), on="s1").join(tg.select("t", ta="addr_toks"), on="t")
def disagree(a, b):
    """True when both addresses have street-name tokens and no pair of them has fuzz ratio >= 75."""
    A, B = street(a), street(b)
    if not A or not B:
        return False
    return not any(fuzz.ratio(x, y) >= 75 for x in A for y in B)
c3 = c3.with_columns(r3s=pl.Series([disagree(a, b) for a, b in zip(c3["sa"].to_list(), c3["ta"].to_list())]))
print("r3 old", cls["r3"].sum(), "-> strict", c3["r3s"].sum())
# vocab narrow: swap_word, both words frequent French type words (df>=1000 among French target cores), lev>2, not abbreviations
tgf = pl.read_parquet(rf"{W}\test\tgt.parquet", columns=["core", "country"]).filter(pl.col("country") == "France")
df = tgf.select(x=pl.col("core").fill_null("").str.split(" ")).explode("x").filter(pl.col("x") != "").group_by("x").len()
TYPE = set(df.filter(pl.col("len") >= 1000)["x"].to_list())
cv = cls.filter(pl.col("name") == "swap_word").join(s1.select("s1", a="core"), on="s1").join(tg.select("t", b="core"), on="t")
def isvocab(a, b):
    """True when core names `a` (S1) and `b` (target) differ by exactly one swapped word, both frequent type words,
    at Levenshtein distance > 2 and neither a prefix of the other (a vocabulary-swap decoy, e.g. Club <-> Amicale)."""
    A, B = set((a or "").split()), set((b or "").split())
    g, n = list(A - B), list(B - A)
    if len(g) != 1 or len(n) != 1:
        return False
    g, n = g[0], n[0]
    if g not in TYPE or n not in TYPE or Levenshtein.distance(g, n) <= 2 or g.startswith(n) or n.startswith(g):
        return False
    return True
cv = cv.with_columns(vocab2=pl.Series([isvocab(a, b) for a, b in zip(cv["a"].to_list(), cv["b"].to_list())]))
print("vocab old", cls["vocab"].sum(), "-> narrow", cv["vocab2"].sum(), "| TYPE words", len(TYPE))
cls = (cls.join(c3.select("s1", "t", "r3s"), on=["s1", "t"], how="left").join(cv.select("s1", "t", "vocab2"), on=["s1", "t"], how="left")
       .with_columns(r3=pl.col("r3s").fill_null(False), vocab=pl.col("vocab2").fill_null(False)).drop("r3s", "vocab2"))
# floor/sufB: appended list words only (drop swap_list replacements)
cls = cls.with_columns(floor=pl.col("sufB"))
cls = cls.with_columns(kill=pl.col("r1") | pl.col("r3") | pl.col("r4"))
cls.write_parquet(rf"{OUT}\classes.parquet")
print(cls.select([pl.col(k).sum() for k in ["vocab", "r1", "r3", "r4", "kill", "sufB", "floor"]]))
