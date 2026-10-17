"""v21e France + audit rules. (A) audit134: rules 1,3,4 -> p=0; (B) audit1234: + rule 2 floor p>=0.9. Full v21e files."""
import re, sys, polars as pl
from normalize import fold, LEGAL
from build_variant import vocab_pairs, apply_policy, scaled, POLICY, write, verify
V21E = "v21e_bundle/submission_v21e/matching_results.tsv"
N = 259452
sc = pl.read_parquet("fr_scores_c.parquet").filter(pl.col("p") > 1e-4)
x = pl.read_parquet("fr_pp2.parquet", columns=["s1", "t", "swk", "c_num", "c_st", "c_leg", "c_ed", "legal", "legal_t", "core", "core_t", "business_name_t"])
sufB = x.filter((pl.col("swk") == "suffix") & (pl.col("c_num") == "house_eq") & (pl.col("c_st") == "st_hi")
                & pl.col("c_leg").is_in(["leg_eq", "leg_t_dropped", "leg_t_added"])).select("s1", "t")
cats = pl.read_parquet("c02_cats.parquet")
# rule 1: exact core name + legal-form swap between two real (non co/group) forms, every address class
r1 = cats.filter(pl.col("name") == "exact_name+legalswap").select("s1", "t")
# rule 3: same house number, explicit street disagreement, exact-name cells
r3 = cats.filter((pl.col("addr") == "same_num_other_street") & pl.col("name").str.starts_with("exact_name")).select("s1", "t")
# rule 4: exact name, different house number at the one-sided decoy offsets, legal same or add/drop (swaps are rule 1)
h = pl.read_parquet("c01_hdiff.parquet")
r4 = h.filter(pl.col("off").is_in([1, 2, 3, 4, 5, 7, 9, 11]) & pl.col("leg").is_in(["same", "adddrop"])).select("s1", "t")
kill = pl.concat([r1, r3, r4]).unique()
# rule 2: appended list-word copies (elig_pos: suffix family, num_eq, legal agree, street not low) + appended 'developpement'
pos = pl.read_parquet("elig_pos.parquet").filter(pl.col("pos").is_in(["after_legal", "nolegal_end_conn"])).select("s1", "t")
REAL = {k for k, v in LEGAL.items() if v not in ("co", "group")}
IGN = {"co", "group"}
real = lambda l: " ".join(sorted(set((l or "").split()) - IGN))  # noqa: E731
dv = x.filter((pl.col("c_ed") == "added") & (pl.col("c_num") == "house_eq") & (pl.col("c_st") != "st_lo"))
dv = dv.filter(pl.Series([set(b.split()) - set(a.split()) == {"developpement"} for a, b in zip(dv["core"].to_list(), dv["core_t"].to_list())]))
dv = dv.filter(pl.Series([real(a) == real(b) for a, b in zip(dv["legal"].to_list(), dv["legal_t"].to_list())]))
def appended(name):
    """True when "developpement" is appended in target name `name`: its last occurrence comes after the first real
    legal form, or is the last word when there is no legal form."""
    toks = re.findall(r"[a-z0-9]+", fold(name or ""))
    if "developpement" not in toks: return False
    i = len(toks) - 1 - toks[::-1].index("developpement")
    li = [j for j, t in enumerate(toks) if t in REAL]
    return i > li[0] if li else (i == len(toks) - 1)
dv = dv.filter(pl.Series([appended(n) for n in dv["business_name_t"].to_list()])).select("s1", "t")
floor = pl.concat([pos, dv]).unique().join(kill, on=["s1", "t"], how="anti")


def decide(with_floor: bool) -> pl.DataFrame:
    """French decision with the audit rules: vocabulary swaps -> p = 0, suffix copies (sufB) -> odds x5, rules 1/3/4
    -> p = 0, optionally the rule-2 floor p >= 0.9, then v20's dta_own policy. Returns the predicted pairs (s1, t)."""
    s = scaled(scaled(scaled(sc, vocab_pairs(), 0.0), sufB, 5.0), kill, 0.0)
    if with_floor:
        s = (s.join(floor.with_columns(_f=pl.lit(True)), on=["s1", "t"], how="left")
             .with_columns(p=pl.when(pl.col("_f")).then(pl.max_horizontal("p", pl.lit(0.9))).otherwise(pl.col("p"))).drop("_f"))
    return apply_policy(s.filter(pl.col("p") > 1e-4), POLICY)


base = pl.read_parquet("pred_novocab_sufB5.parquet")

print(f"rule pairs: r1 {r1.height}, r3 {r3.height}, r4 {r4.height} (union {kill.height}); floor-eligible {floor.height} (list-word {pos.height}, developpement {dv.height})")
print(f"   in v21e France: r1 {r1.join(base, on=['s1','t'], how='semi').height}, r3 {r3.join(base, on=['s1','t'], how='semi').height}, "
      f"r4 {r4.join(base, on=['s1','t'], how='semi').height}; floor not yet in v21e {floor.join(base, on=['s1','t'], how='anti').height}")
Q = {"r1": (0.02, 0.05, 0.10), "r3": (0.1, 0.2, 0.35), "r4": (0.2, 0.33, 0.5), "floor": (0.95, 0.85, 0.75), "other": (0.9, 0.7, 0.5)}
tag = pl.concat([r1.with_columns(k=pl.lit("r1")), r3.with_columns(k=pl.lit("r3")), r4.with_columns(k=pl.lit("r4")),
                 floor.with_columns(k=pl.lit("floor"))]).unique(["s1", "t"], keep="first")
for name, wf in (("audit134", False), ("audit1234", True)):
    p = decide(wf)
    rem = base.join(p, on=["s1", "t"], how="anti").join(tag, on=["s1", "t"], how="left").with_columns(pl.col("k").fill_null("other"))
    add = p.join(base, on=["s1", "t"], how="anti").join(tag, on=["s1", "t"], how="left").with_columns(pl.col("k").fill_null("other"))
    rows = pl.concat([rem.select("s1"), add.select("s1")]).unique().height
    gains = []
    for i in range(3):   # optimistic / central / pessimistic truth rates
        g = sum(((1 - Q[k][i]) * 0.20 - Q[k][i] * 0.06) * n for k, n in rem.group_by("k").len().rows()) \
            + sum((Q[k][i] * 0.06 - (1 - Q[k][i]) * 0.20) * n for k, n in add.group_by("k").len().rows())
        gains.append(g / N)
    print(f"{name}: {p.height} pairs = {p.height / N:.4f}/S1; removed {rem.height} {rem.group_by('k').len().sort('k').rows()}, "
          f"added {add.height} {add.group_by('k').len().sort('k').rows()}, French rows changed {rows}")
    print(f"   expected France F0.5: optimistic {gains[0]:+.4f}, central {gains[1]:+.4f}, pessimistic {gains[2]:+.4f} "
          f"-> LB {0.149752 * gains[0]:+.5f} / {0.149752 * gains[1]:+.5f} / {0.149752 * gains[2]:+.5f}")
    if "write" in sys.argv:
        p.write_parquet(f"pred_v21e_{name}.parquet")
        write(name, p, name, src=V21E, prefix="v21e_fr_")
        verify(name, src=V21E, prefix="v21e_fr_")
