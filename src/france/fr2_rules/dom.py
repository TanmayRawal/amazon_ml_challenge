"""France domain-name targets: domain generator deletes non-ASCII letters (école -> cole). Key = concat of the first k words
of the lowercased name after deleting every char outside [a-z0-9 ] (hyphen/apostrophe variants).  Check how often the key
explains true/claimed domain pairs; then find unique same-address claimants for unclaimed France domain targets."""
import re
import sys

import polars as pl

W = r"work"
OUT = r"work/scratch/france2"
pl.Config.set_tbl_cols(30); pl.Config.set_tbl_width_chars(300); pl.Config.set_fmt_str_lengths(70); pl.Config.set_tbl_rows(60)


def keys(name):
    """Domain keys of an S1 name: concatenations of the first 1..6 words of the lower-cased name after deleting every
    character outside [a-z0-9 ] (with and without hyphens/apostrophes turned into spaces)."""
    out = set()
    low = name.lower()
    for sep_mode in (0, 1):
        s = low.replace("-", " ").replace("'", " ").replace("’", " ") if sep_mode else low
        s = re.sub(r"[^a-z0-9 ]", "", s)
        w = s.split()
        acc = ""
        for i, x in enumerate(w[:6]):
            acc += x
            out.add(acc)
    return list(out)


def s1_keys(split, countries):
    """S1s of `countries` in `split` with address tokens/numbers, exploded to one row per domain key (column k)."""
    s = (pl.scan_parquet(f"{W}/{split}/s1.parquet").filter(pl.col("country").is_in(countries))
         .select("s1", "country", "business_name", s_toks="addr_toks", s_nums="addr_nums").collect())
    s = s.with_columns(k=pl.col("business_name").map_elements(keys, return_dtype=pl.List(pl.String))).explode("k")
    return s


def addr_rel(x):
    """Address relation of each (S1, target) pair from token overlap (jac) and shared numbers: arel = same /
    street_nonum / street_diffnum / weak / none."""
    x = x.with_columns(
        st=pl.col("s_toks").str.split(" "), tt=pl.col("t_toks").str.split(" "),
        sn=pl.col("s_nums").str.split(" "), tn=pl.col("t_nums").str.split(" "))
    x = x.with_columns(
        jac=(pl.col("st").list.set_intersection("tt").list.len() / pl.min_horizontal(pl.col("st").list.len(), pl.col("tt").list.len()).clip(1)),
        num_eq=pl.col("sn").list.set_intersection("tn").list.len() > 0,
        t_nonum=(pl.col("t_nums").fill_null("") == ""),
    )
    return x.with_columns(arel=pl.when((pl.col("jac") >= 0.5) & pl.col("num_eq")).then(pl.lit("same"))
                          .when((pl.col("jac") >= 0.5) & pl.col("t_nonum")).then(pl.lit("street_nonum"))
                          .when(pl.col("jac") >= 0.5).then(pl.lit("street_diffnum"))
                          .when(pl.col("jac") > 0).then(pl.lit("weak")).otherwise(pl.lit("none"))).drop("st", "tt", "sn", "tn")


part = sys.argv[1]
if part == "peek":
    s = pl.read_parquet(f"{W}/test/s1.parquet", n_rows=3, columns=["business_address", "addr_toks", "addr_nums"])
    print(s)
elif part == "train":
    cs = ["US", "India"]
    tg = (pl.scan_parquet(f"{W}/train/tgt.parquet").filter(pl.col("is_domain") & pl.col("country").is_in(cs))
          .select("t", "country", "domain", t_toks="addr_toks", t_nums="addr_nums").collect())
    D = r"data/dataset"
    raw = pl.read_csv(f"{D}/train/train_ground_truth.tsv", separator="\t", quote_char=None, infer_schema_length=0)
    hid = (raw.filter((pl.col("source1_entity_id").str.extract(r"(\d+)$").cast(pl.Int64) % 100) < 20)
           .with_columns(e=pl.col("matched_entity_ids").str.split(",")).explode("e").select(entity_id="e").unique())
    tid = pl.scan_parquet(f"{W}/train/tgt.parquet").filter(pl.col("is_domain")).select("t", "entity_id").collect()
    tg = tg.join(tid.join(hid, on="entity_id", how="semi").select("t"), on="t", how="anti")
    s = s1_keys("train", cs)
    x = tg.join(s, left_on=["country", "domain"], right_on=["country", "k"]).unique(["t", "s1"])
    x = addr_rel(x)
    gt = pl.read_parquet(f"{W}/train/gt.parquet").join(tg.select("t"), on="t", how="semi").with_columns(y=pl.lit(1))
    x = x.join(gt, on=["s1", "t"], how="left").with_columns(pl.col("y").fill_null(0))
    # coverage: share of true domain pairs explained by the key
    print("true domain pairs:", gt.height, " explained by key:", x["y"].sum())
    sc = pl.scan_parquet(f"{W}/train/scores_c.parquet").select("s1", "t", "p").join(x.select("s1", "t").lazy(), on=["s1", "t"], how="semi").collect()
    x = x.join(sc, on=["s1", "t"], how="left").with_columns(pl.col("p").fill_null(0.0))
    x = x.with_columns(n_c=pl.len().over("t"), n_same=(pl.col("arel") == "same").sum().over("t"),
                       n_street=pl.col("arel").is_in(["same", "street_nonum"]).sum().over("t"),
                       t_claim=(pl.col("p") >= 0.5).max().over("t"))
    x.write_parquet(f"{OUT}/dom_train.parquet")
    print(x.group_by("country", "arel").agg(n=pl.len(), true=pl.col("y").mean(), p=pl.col("p").mean()).sort("country", "arel"))
    u = x.filter(pl.col("arel").is_in(["same", "street_nonum"]) & (pl.col("n_street") == 1))
    print("unique street claimant", u.group_by("country", "arel").agg(n=pl.len(), true=pl.col("y").mean(), n_un=(~pl.col("t_claim")).sum(),
          true_un=pl.col("y").filter(~pl.col("t_claim")).mean(), true_lowp=pl.col("y").filter(pl.col("p") < 0.5).mean(), n_lowp=(pl.col("p") < 0.5).sum()).sort("country", "arel"))
else:
    cs = ["France", "US", "India"]
    acc = pl.read_parquet(f"{OUT}/best_pairs.parquet")
    tg = (pl.scan_parquet(f"{W}/test/tgt.parquet").filter(pl.col("is_domain"))
          .select("t", "country", "domain", t_toks="addr_toks", t_nums="addr_nums").collect())
    s = s1_keys("test", cs)
    x = tg.join(s, left_on=["country", "domain"], right_on=["country", "k"]).unique(["t", "s1"])
    x = addr_rel(x)
    x = x.join(acc.with_columns(acc=pl.lit(True)), on=["s1", "t"], how="left").with_columns(pl.col("acc").fill_null(False))
    claimed = acc.select("t").unique().with_columns(t_claimed=pl.lit(True))
    x = x.join(claimed, on="t", how="left").with_columns(pl.col("t_claimed").fill_null(False))
    pool = pl.scan_parquet(f"{W}/w20/v20_pool.parquet").join(x.select("s1", "t").lazy(), on=["s1", "t"], how="semi").collect()
    x = x.join(pool, on=["s1", "t"], how="left")
    x = x.with_columns(n_c=pl.len().over("t"), n_same=(pl.col("arel") == "same").sum().over("t"),
                       n_street=pl.col("arel").is_in(["same", "street_nonum"]).sum().over("t"))
    x.write_parquet(f"{OUT}/dom_test.parquet")
    # coverage of claimed domain pairs by the key
    ct = acc.join(tg.select("t", "country"), on="t")
    print("claimed domain pairs by country:", ct.group_by("country").len().sort("country"))
    print("   ... explained by key:", x.filter(pl.col("acc")).group_by("country").len().sort("country"))
    print(x.group_by("country", "arel").agg(n=pl.len(), acc=pl.col("acc").mean(), t_claimed=pl.col("t_claimed").mean()).sort("country", "arel"))
    u = x.filter(pl.col("arel").is_in(["same", "street_nonum"]) & (pl.col("n_street") == 1))
    print("unique street claimant", u.group_by("country", "arel").agg(n=pl.len(), acc=pl.col("acc").mean(), un=(~pl.col("t_claimed")).sum(),
          p_un=pl.col("p").filter(~pl.col("t_claimed")).mean(), pnull_un=pl.col("p").filter(~pl.col("t_claimed")).is_null().sum()).sort("country", "arel"))
    ex = u.filter((pl.col("country") == "France") & ~pl.col("t_claimed"))
    print(ex.select("domain", "business_name", "arel", "jac", "p", "t_toks", "s_toks").sample(min(30, ex.height), seed=2))
