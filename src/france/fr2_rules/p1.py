"""Priority 1: blank-address targets tied on core_key across several S1s.
Can legal form + raw formatting pick a unique owner?  Train (US/India, labels) -> true rate; test France -> volume."""
import re
import sys
import unicodedata

import polars as pl

W = r"work"
OUT = r"work/scratch/france2"
pl.Config.set_tbl_cols(30); pl.Config.set_fmt_str_lengths(60); pl.Config.set_tbl_width_chars(260); pl.Config.set_tbl_rows(60)


def fold(s):
    """Accent-free, lower-case, alphanumeric-only form of `s` (words joined by single spaces)."""
    s = unicodedata.normalize("NFKD", s)
    s = "".join(c for c in s if not unicodedata.combining(c)).lower()
    return " ".join(re.findall(r"[a-z0-9]+", s))


def lowsp(s):
    """Lower-case `s` with whitespace collapsed."""
    return " ".join(s.lower().split())


def pairs(split, countries):
    """All (blank-address target, same-country S1 with the same core_key) pairs, for groups of at most 60 S1s.

    Adds the legal-form relation (same / drop / add / swap), raw / lower-case / folded / core name equality, a tier
    score sc, and per-target counts (n_same, n_raw, n_low, n_fold, n_swap, top, n_top); ng = S1s in the tie group.
    """
    s1 = (pl.scan_parquet(f"{W}/{split}/s1.parquet").filter(pl.col("country").is_in(countries))
          .select("s1", "country", s_name="business_name", s_core="core", core_key="core_key", s_legal="legal").collect())
    tg = (pl.scan_parquet(f"{W}/{split}/tgt.parquet").filter(pl.col("addr_blank") & pl.col("country").is_in(countries) & (pl.col("core_key") != ""))
          .select("t", "country", "src", t_name="business_name", t_core="core", core_key="core_key", t_legal="legal").collect())
    x = tg.join(s1, on=["country", "core_key"])
    x = x.with_columns(ng=pl.len().over("t")).filter(pl.col("ng") <= 60)
    names = pl.concat([x["s_name"], x["t_name"]]).unique()
    m = pl.DataFrame({"n": names, "fold": [fold(v) for v in names.to_list()], "lowsp": [lowsp(v) for v in names.to_list()]})
    x = (x.join(m.rename({"n": "s_name", "fold": "s_fold", "lowsp": "s_low"}), on="s_name", how="left")
         .join(m.rename({"n": "t_name", "fold": "t_fold", "lowsp": "t_low"}), on="t_name", how="left"))
    x = x.with_columns(
        leg=pl.when(pl.col("s_legal") == pl.col("t_legal")).then(pl.lit("same"))
        .when(pl.col("t_legal") == "").then(pl.lit("drop"))
        .when(pl.col("s_legal") == "").then(pl.lit("add")).otherwise(pl.lit("swap")),
        raw_eq=pl.col("s_name") == pl.col("t_name"),
        low_eq=pl.col("s_low") == pl.col("t_low"),
        fold_eq=pl.col("s_fold") == pl.col("t_fold"),
        core_eq=pl.col("s_core") == pl.col("t_core"),
    )
    # tier score: legal same 3 > add/drop 1 > swap 0; then fold_eq, low_eq, raw_eq, core order
    x = x.with_columns(
        legs=pl.col("leg").replace_strict({"same": 3, "drop": 1, "add": 1, "swap": 0}, return_dtype=pl.Int32),
    ).with_columns(
        sc=pl.col("legs") * 16 + pl.col("fold_eq").cast(pl.Int32) * 8 + pl.col("low_eq").cast(pl.Int32) * 4
        + pl.col("raw_eq").cast(pl.Int32) * 2 + pl.col("core_eq").cast(pl.Int32),
    )
    x = x.with_columns(
        n_same=(pl.col("leg") == "same").sum().over("t"),
        n_raw=pl.col("raw_eq").sum().over("t"),
        n_low=pl.col("low_eq").sum().over("t"),
        n_fold=pl.col("fold_eq").sum().over("t"),
        n_swap=(pl.col("leg") == "swap").sum().over("t"),
        top=pl.col("sc").max().over("t"),
    ).with_columns(n_top=(pl.col("sc") == pl.col("top")).sum().over("t"))
    return x


def rules(x):
    """boolean columns: this pair is the unique claimant under rule r"""
    return x.with_columns(
        R_leg=(pl.col("leg") == "same") & (pl.col("n_same") == 1),
        R_raw=pl.col("raw_eq") & (pl.col("n_raw") == 1),
        R_low=pl.col("low_eq") & (pl.col("n_low") == 1),
        R_fold=pl.col("fold_eq") & (pl.col("n_fold") == 1),
        R_top=(pl.col("sc") == pl.col("top")) & (pl.col("n_top") == 1),
        R_top_leg=(pl.col("sc") == pl.col("top")) & (pl.col("n_top") == 1) & (pl.col("leg") == "same"),
    )


RULES = ["R_leg", "R_raw", "R_low", "R_fold", "R_top", "R_top_leg"]

if __name__ == "__main__":
    part = sys.argv[1]
    if part == "train":
        x = rules(pairs("train", ["US", "India"]))
        gt = pl.read_parquet(f"{W}/train/gt.parquet").join(x.select("s1").unique(), on="s1", how="semi").with_columns(y=pl.lit(1, pl.Int8))
        x = x.join(gt, on=["s1", "t"], how="left").with_columns(pl.col("y").fill_null(0))
        # approximately out-of-fold model probability (scores_c), used to mimic "unclaimed"
        sc = pl.scan_parquet(f"{W}/train/scores_c.parquet").select("s1", "t", "p").join(x.select("s1", "t").lazy(), on=["s1", "t"], how="semi").collect()
        x = x.join(sc, on=["s1", "t"], how="left").with_columns(pl.col("p").fill_null(0.0))
        x = x.with_columns(t_owned_in=pl.col("y").max().over("t"), t_claim=(pl.col("p") >= 0.5).max().over("t"))
        x.write_parquet(f"{OUT}/p1_train.parquet")
        print("targets by group size, owner-in-group rate")
        tt = x.group_by("t").agg(pl.first("country"), ng=pl.first("ng"), own=pl.first("t_owned_in"), claim=pl.first("t_claim"))
        print(tt.with_columns(g=pl.col("ng").clip(1, 4)).group_by("country", "g").agg(n=pl.len(), own=pl.col("own").mean(), claim=pl.col("claim").mean()).sort("country", "g"))
        for r in RULES:
            for tie in (False, True):
                z = x.filter(pl.col(r) & ((pl.col("ng") >= 2) == tie))
                print(r, "tie" if tie else "single", z.group_by("country").agg(n=pl.len(), true=pl.col("y").mean(), p=pl.col("p").mean(),
                      unclaimed_n=(~pl.col("t_claim")).sum(), true_unclaimed=pl.col("y").filter(~pl.col("t_claim")).mean()).sort("country"))
    else:
        x = rules(pairs("test", ["France", "US", "India"]))
        best = pl.read_csv(f"{W}/w20/submissions/v21e3x/matching_results.tsv", separator="\t", quote_char=None, infer_schema_length=0)
        e = (best.with_columns(m=pl.col("matched_entity_ids").str.split(",")).explode("m").filter(pl.col("m").is_not_null() & (pl.col("m") != ""))
             .select(s_id="source1_entity_id", t_id="m"))
        s1 = pl.read_parquet(f"{W}/test/s1.parquet", columns=["s1", "entity_id"]).rename({"entity_id": "s_id"})
        tid = pl.scan_parquet(f"{W}/test/tgt.parquet").select("t", t_id="entity_id").join(e.select("t_id").unique().lazy(), on="t_id", how="semi").collect()
        acc = e.join(s1, on="s_id").join(tid, on="t_id").select("s1", "t").with_columns(acc=pl.lit(True))
        acc.write_parquet(f"{OUT}/best_pairs.parquet")
        pool = pl.scan_parquet(f"{W}/w20/v20_pool.parquet").join(x.select("s1", "t").lazy(), on=["s1", "t"], how="semi").collect()
        x = (x.join(acc, on=["s1", "t"], how="left").with_columns(pl.col("acc").fill_null(False))
             .join(pool, on=["s1", "t"], how="left").with_columns(pl.col("p").fill_null(0.0)))
        tclaim = acc.group_by("t").agg(t_claimed_any=pl.len())
        x = x.join(tclaim, on="t", how="left").with_columns(t_claimed_any=pl.col("t_claimed_any").fill_null(0) > 0,
                                                             t_claim=pl.col("acc").max().over("t"))
        x.write_parquet(f"{OUT}/p1_test.parquet")
        tt = x.group_by("t").agg(pl.first("country"), ng=pl.first("ng"), claim=pl.first("t_claim"), claimed_any=pl.first("t_claimed_any"))
        print(tt.with_columns(g=pl.col("ng").clip(1, 4)).group_by("country", "g").agg(n=pl.len(), claim=pl.col("claim").mean(), claimed_any=pl.col("claimed_any").mean()).sort("country", "g"))
        for r in RULES:
            for tie in (False, True):
                z = x.filter(pl.col(r) & ((pl.col("ng") >= 2) == tie))
                print(r, "tie" if tie else "single", z.group_by("country").agg(n=pl.len(), acc=pl.col("acc").mean(), p=pl.col("p").mean(),
                      unclaimed_n=(~pl.col("t_claimed_any")).sum(), p_unclaimed=pl.col("p").filter(~pl.col("t_claimed_any")).mean()).sort("country"))
