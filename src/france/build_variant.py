"""France-only variants of v20: modify French stack p by a rule, re-run the v20 decision (dta_own, missed 0.031) on
French pairs only, and write submissions/v20_fr_<name>/matching_results.tsv with US/India lines byte-identical to v20.
Usage: build_variant.py <name> <rule>   rules: novocab | veto | novocab_veto"""
import sys, os, json
import polars as pl
from matcher_b import apply_policy
ROOT = "C:/Users/Lenovo/Desktop/Ml challange/"
POLICY = {"name": "dta_own", "missed": 0.031}
SUF = {"fils", "france", "associes", "services", "service", "developpement", "freres"}


def vocab_pairs() -> pl.DataFrame:
    """French pairs whose names differ by one word replaced with another common French name word (type-word swap)."""
    x = pl.read_parquet("fr_pp2.parquet", columns=["s1", "t", "swk"])
    return x.filter(pl.col("swk") == "vocab").select("s1", "t")


def veto_pairs() -> pl.DataFrame:
    """web_research2 idea 4: 0.05<p<0.95 and all three US/IN cross-encoders give prob < 0.1."""
    x = pl.read_parquet("fr_xenc.parquet")
    sig = lambda c: 1 / (1 + (-pl.col(c)).exp())  # noqa: E731
    sc = pl.read_parquet("fr_scores_c.parquet").join(x, on=["s1", "t"])
    return sc.filter((pl.col("p") > 0.05) & (pl.col("p") < 0.95) & (sig("xenc") < 0.1) & (sig("xenc2") < 0.1) & (sig("xenc3") < 0.1)).select("s1", "t")


def scaled(sc: pl.DataFrame, pairs: pl.DataFrame, r: float) -> pl.DataFrame:
    """Multiply the odds of p by `r` for the (s1, t) pairs listed in `pairs` (r = 0 sets p to 0); other rows unchanged."""
    odds = pl.col("p") / (1.0 - pl.col("p")).clip(1e-12, None) * r
    return (sc.join(pairs.with_columns(_h=pl.lit(True)), on=["s1", "t"], how="left")
            .with_columns(p=pl.when(pl.col("_h")).then(odds / (1.0 + odds)).otherwise(pl.col("p"))).drop("_h"))


def write(name: str, pred: pl.DataFrame, note: str, src: str = "submissions/v20/matching_results.tsv", prefix: str = "v20_fr_") -> None:
    """Write submissions/<prefix><name>/matching_results.tsv: a byte copy of `src` in which only the French S1 rows are
    replaced by the prediction `pred` (s1, t; target IDs sorted as strings, empty when none). `note` is unused.
    Returns the number of French rows that changed."""
    s1 = pl.read_parquet("fr_s1.parquet", columns=["s1", "entity_id"])
    tg = pl.read_parquet(ROOT + "work_v20/test/tgt.parquet", columns=["t", "entity_id"]).join(pred.select("t").unique(), on="t", how="semi")
    m = (pred.join(s1, on="s1").join(tg, on="t", suffix="_t").group_by("entity_id")
         .agg(pl.col("entity_id_t").sort().str.join(",")))   # v20 writes ids sorted as strings
    new = dict(zip(m["entity_id"].to_list(), m["entity_id_t"].to_list()))
    fr = set(s1["entity_id"].to_list())
    src = ROOT + src
    out_dir = ROOT + f"submissions/{prefix}{name}"
    os.makedirs(out_dir, exist_ok=True)
    changed = same_fr = other = 0
    with open(src, "rb") as f, open(out_dir + "/matching_results.tsv", "wb") as g:
        for i, line in enumerate(f):
            if i == 0:
                g.write(line); continue
            eid, rest = line.rstrip(b"\r\n").split(b"\t", 1)
            e = eid.decode()
            if e not in fr:
                g.write(line); other += 1; continue
            nl = (e + "\t" + new.get(e, "")).encode() + line[len(line.rstrip(b"\r\n")):]
            if nl != line:
                changed += 1
            else:
                same_fr += 1
            g.write(nl)
    print(f"{name}: French rows changed {changed}, unchanged {same_fr}, other rows copied {other}")
    return changed


def verify(name: str, src: str = "submissions/v20/matching_results.tsv", prefix: str = "v20_fr_") -> None:
    """Check the written variant against `src`: same row count and S1 order, and no non-French row differs (asserts)."""
    a = open(ROOT + src, "rb").read().splitlines(keepends=True)
    b = open(ROOT + f"submissions/{prefix}{name}/matching_results.tsv", "rb").read().splitlines(keepends=True)
    fr = set(pl.read_parquet("fr_s1.parquet", columns=["entity_id"])["entity_id"].to_list())
    assert len(a) == len(b), "row count differs"
    diff_other = sum(1 for x, y in zip(a, b) if x != y and x.split(b"\t", 1)[0].decode() not in fr)
    diff_fr = sum(1 for x, y in zip(a, b) if x != y)
    order = all(x.split(b"\t", 1)[0] == y.split(b"\t", 1)[0] for x, y in zip(a, b))
    print(f"verify {name}: rows {len(b)}, same S1 order {order}, differing US/India rows {diff_other}, differing French rows {diff_fr}")
    assert diff_other == 0 and order


if __name__ == "__main__":
    name, rule = sys.argv[1], sys.argv[2]
    if rule == "from_pred":   # write an already decided French prediction (pred_<source>.parquet, e.g. b27_suffix2.py output)
        write(name, pl.read_parquet(f"pred_{sys.argv[3]}.parquet"), sys.argv[3]); verify(name); sys.exit()
    r = float(sys.argv[3]) if len(sys.argv) > 3 else 0.0
    sc = pl.read_parquet("fr_scores_c.parquet").filter(pl.col("p") > 1e-4)
    base = apply_policy(sc, POLICY)
    pairs = {"novocab": vocab_pairs, "veto": veto_pairs}.get(rule)
    if rule == "novocab_veto":
        pairs = lambda: pl.concat([vocab_pairs(), veto_pairs()]).unique()  # noqa: E731
    P = pairs()
    sc2 = scaled(sc, P, r)
    pred = apply_policy(sc2.filter(pl.col("p") > 1e-4), POLICY)
    j = base.with_columns(a=pl.lit(1)).join(pred.with_columns(b=pl.lit(1)), on=["s1", "t"], how="full", coalesce=True)
    rem, add = j.filter(pl.col("b").is_null()), j.filter(pl.col("a").is_null())
    rows = pl.concat([rem.select("s1"), add.select("s1")]).unique().height
    print(f"rule {rule} x{r}: {P.height} pairs scaled; v20 French pairs {base.height} -> {pred.height}: removed {rem.height}, added {add.height}, rows {rows}")
    pred.write_parquet(f"pred_{name}.parquet")
    write(name, pred, rule)
    verify(name)
