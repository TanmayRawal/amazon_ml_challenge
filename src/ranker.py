"""Stage 2 candidate generation: raw pass union + learned ranker (supervised meta-blocking).

Plan A cut each pass to a top-k and ranked the union with a hand-weighted score; on a
20k-entity sample that kept 0.868 recall of a 0.942 union. Here every pass is kept
whole (within block-size caps), a rare-address-word pass is added, and a LightGBM
model scores each (S1, target) pair from blocking-graph features only: which passes
fired, how rare the shared keys are, per-S1 ranks and a few record statistics. The
top rk_keep pairs per S1 are stored; the matcher uses the first cand_k of them.

Cross-fitting keeps the matcher's training candidates out-of-sample: model A is
trained on folds {0,1} and scores folds {2,3}; model B the reverse; the sealed fold
and the test set get the mean of both. The sealed fold never trains anything.
"""
from __future__ import annotations

import json
import time

import lightgbm as lgb
import numpy as np
import polars as pl

from blocking import (STRUCT_LIST, _addr_keys, _addr_words, _cap_blocks, _concat_keys, _exact_keys, _key,
                      _name_tokens)
from config import Config
from prepare import load
from splits import own_holdout

RANK_FEATURES = [
    "f_exact", "e_bs", "rare_idf", "rare_n", "rare_max", "addr_n", "addr_bs", "a2_n", "a2_idf",
    "a2_max", "f_concat", "cc_bs", "n_pass", "n_union", "rare_rank", "addr_rank", "a2_rank",
    "rare_rel", "a2_rel", "sk_n", "sk_idf", "sk_max", "sk_rank", "sk_rel",
    "s_n_ctok", "s_n_atok", "s_n_nums", "s_blank", "s_name_rep", "s_name_tdf",
    "t_n_ctok", "t_n_atok", "t_n_nums", "t_blank", "t_domain", "t_indic", "t_src3", "t_acro",
    # Stage 6c: record similarity. Without it the ranker cannot tell a typo'd true copy with
    # an identical address from the many same-key candidates, and ranked 2.1% of dev true
    # pairs below 60 (e.g. "A-400610 Cnsultancy Limited", same address: 277th of 277).
    "sim_name_jac", "sim_addr_jac", "sim_addr_cont", "num0_eq",
    # V12 channels: name word pairs, S1 house numbers minus their last digit, website-label 4-grams;
    # plus skeleton-word overlap and truncated first number.
    "np_n", "np_bs", "at_n", "dm_n", "dm_idf", "sim_sk_jac", "num0_tr",
    # V13: address word pairs, typo keys (word + first letter + length of another word).
    "ap_n", "ap_bs", "fz_n", "fz_bs",
    # V17: name word x house number, skeleton word x house number
    "nn_n", "nn_bs", "kn_n", "kn_bs",
]
CHANNEL_COLS = ["f_exact", "e_bs", "rare_idf", "rare_n", "rare_max", "addr_n", "addr_bs",
                "a2_n", "a2_idf", "a2_max", "f_concat", "cc_bs", "sk_n", "sk_idf", "sk_max",
                "np_n", "np_bs", "at_n", "dm_n", "dm_idf", "ap_n", "ap_bs", "fz_n", "fz_bs", "nn_n", "nn_bs", "kn_n", "kn_bs"]
# Negatives carry weight 1/rk_neg_rate = 50. Without L2 and with lr 0.1 the leaves of nearly pure
# positives get huge outputs after a few trees and the ordering collapses: V13 model A kept 1 tree
# (held-out AUC 0.958), V12 stopped at 53/37 trees. With L2 = 10, lr 0.05 and big leaves the same
# rows reach AUC 0.999 by 40-80 trees (experiments/ranker_exp.py, 10-chunk sample).
LGB_PARAMS = {
    "objective": "binary", "learning_rate": 0.05, "num_leaves": 31, "min_data_in_leaf": 3000,
    "feature_fraction": 0.7, "lambda_l2": 10.0, "verbosity": -1, "seed": 2026, "deterministic": True,
    "force_row_wise": True,
}
# Early stopping watches AUC (the ranker only orders candidates); capped at 60 rounds to bound ranking
# time. A model that still stops before MIN_TREES is retrained at half the learning rate.
ES_PARAMS = {"metric": ["auc", "binary_logloss"], "first_metric_only": True}
ES_ROUNDS, ES_PATIENCE, MIN_TREES = 60, 20, 10
MODEL_DIR = "ranker"


# ----------------------------------------------------------------------------- indexes

def _idf_table(tok_df: pl.DataFrame, n_c: pl.DataFrame, df_max: int) -> pl.DataFrame:
    """Keep tokens with document frequency <= df_max and add their per-country IDF and blocking key.

    Args: tok_df: country, tok, df; n_c: country, N (target records per country).
    Returns: country, tok, idf (log N/df), key."""
    return (
        tok_df.filter(pl.col("df") <= df_max)
        .join(n_c, on="country")
        .with_columns(idf=(pl.col("N") / pl.col("df")).log().cast(pl.Float32),
                      key=_key(pl.col("country"), pl.col("tok")))
        .select("country", "tok", "idf", "key")
    )


def _skeleton_keys(df: pl.DataFrame, idc: str) -> pl.DataFrame:
    """Skeleton keys per record: "F:<sorted unique words>" and "P:<a>|<b>" for every
    unordered pair among the first 6 sorted words. df needs idc, country, name_sk."""
    words = df.select(idc, "country", w=pl.col("name_sk").str.split(" ")
                      .list.eval(pl.element().filter(pl.element().str.len_chars() >= 2)).list.unique().list.sort())
    full = words.filter(pl.col("w").list.len() > 0).select(
        idc, "country", tok=pl.lit("F:") + pl.col("w").list.join(" "))
    single = words.select(idc, "country", a=pl.col("w").list.head(6)).explode("a", empty_as_null=True).drop_nulls("a")
    pair = (
        single.join(single.select(idc, b="a"), on=idc)
        .filter(pl.col("a") < pl.col("b"))
        .select(idc, "country", tok=pl.lit("P:") + pl.col("a") + "|" + pl.col("b"))
    )
    return pl.concat([full, pair]).unique([idc, "tok"])


def _word_pair_keys(df: pl.DataFrame, idc: str, n: int = 5) -> pl.DataFrame:
    """V12: unordered pairs among the first n distinct core words (>= 2 letters). Names of common
    words with one word dropped, added or garbled ("Vision Associates of" vs "Vision Associates
    of Tucson") share no exact key and no rare word, but share a word pair."""
    w = df.select(idc, "country", w=pl.col("core").str.split(" ")
                  .list.eval(pl.element().filter(pl.element().str.len_chars() >= 2))
                  .list.unique(maintain_order=True).list.head(n))
    w = w.filter(pl.col("w").list.len() >= 2)
    keys = []
    for i in range(n):
        for j in range(i + 1, n):
            a, b = pl.col("w").list.get(i, null_on_oob=True), pl.col("w").list.get(j, null_on_oob=True)
            lo, hi = pl.min_horizontal(a, b), pl.max_horizontal(a, b)
            keys.append(w.filter(pl.col("w").list.len() > j).select(idc, key=_key(pl.col("country"), lo, hi)))
    return pl.concat(keys).unique()


def _word_num_keys(df: pl.DataFrame, idc: str, col: str, minlen: int) -> pl.DataFrame:
    """V17: (name word, house number) keys. Generic names ("Alpha", "Perfect", "White") put the name words
    in huge blocks that the caps drop, while the address kept the number ("No 116 1st Floor" vs "No 116-
    1st Floor"). A common word with one house number is a small block. On V14's sealed fold, core word x
    number (blocks <= 50) reaches 18% of the true pairs no channel found, skeleton word x number 15%."""
    w = pl.col(col).str.split(" ").list.eval(pl.element().filter(pl.element().str.len_chars() >= minlen)).list.unique()
    n = pl.col("addr_nums").str.split(" ").list.eval(pl.element().filter(pl.element() != "")).list.unique()
    return (df.select(idc, "country", w=w, n=n).explode("w").explode("n").drop_nulls()
            .select(idc, key=_key(pl.col("country"), pl.col("w"), pl.col("n"))).unique())


def _truncated_numbers(s1: pl.DataFrame) -> pl.DataFrame:
    """S1 house numbers of 3+ digits with the last digit dropped (V12: 3204 -> 320, 909 -> 90) or the
    first digit dropped (V13: 1684 -> 684, 8515 -> 515); target copies often lose one digit."""
    long = pl.element().filter(pl.element().str.len_chars() >= 3)
    nums = pl.col("addr_nums").str.split(" ")
    return s1.with_columns(addr_nums=pl.concat_list(nums.list.eval(long.str.head(-1)),
                                                    nums.list.eval(long.str.tail(-1).str.strip_chars_start("0")))
                           .list.eval(pl.element().filter(pl.element() != "")).list.unique().list.join(" "))


def _addr_pair_keys(df: pl.DataFrame, idc: str, n: int = 6) -> pl.DataFrame:
    """V13: unordered pairs among the first n distinct non-structural address words (3+ letters).
    Garbled-name copies with a changed or missing house number ("CLOVER HILL DRIVE, PORTSMOUTH")
    reach no number key; the street and city words still pair up."""
    w = df.filter(~pl.col("addr_blank")).select(idc, "country", w=pl.col("addr_toks").str.split(" ").list.eval(
        pl.element().filter((pl.element().str.len_chars() >= 3) & ~pl.element().is_in(STRUCT_LIST)))
        .list.unique(maintain_order=True).list.head(n))
    keys = []
    for i in range(n):
        for j in range(i + 1, n):
            a, b = pl.col("w").list.get(i, null_on_oob=True), pl.col("w").list.get(j, null_on_oob=True)
            keys.append(w.filter(pl.col("w").list.len() > j)
                        .select(idc, key=_key(pl.col("country"), pl.min_horizontal(a, b), pl.max_horizontal(a, b))))
    return pl.concat(keys).unique()


def _fuzzy_keys(df: pl.DataFrame, idc: str, spread: bool, n: int = 4) -> pl.DataFrame:
    """V13: typo keys. For ordered pairs (a, b) of the first n distinct core words with len(b) >= 4:
    (a, first letter of b, length of b). Heavy typos keep the first letter and roughly the length
    ("Atlantic Galaoat" / "Atlantic Galata", "Desert Afligacne" / "Desert Alliance"); with spread the
    S1 side also emits lengths +-1."""
    w = df.select(idc, "country", w=pl.col("core").str.split(" ")
                  .list.eval(pl.element().filter(pl.element().str.len_chars() >= 2))
                  .list.unique(maintain_order=True).list.head(n))
    w = w.filter(pl.col("w").list.len() >= 2)
    keys = []
    for i in range(n):
        for j in range(n):
            if i == j:
                continue
            a, b = pl.col("w").list.get(i, null_on_oob=True), pl.col("w").list.get(j, null_on_oob=True)
            sub = w.filter((pl.col("w").list.len() > max(i, j)) & (b.str.len_chars() >= 4))
            for d in ((-1, 0, 1) if spread else (0,)):
                keys.append(sub.select(idc, key=_key(pl.col("country"), a, b.str.slice(0, 1),
                                                     (b.str.len_chars().cast(pl.Int32) + d).cast(pl.String))))
    return pl.concat(keys).unique()


def _domain_index(s1: pl.DataFrame, tg: pl.DataFrame, n_c: pl.DataFrame, df_max: int) -> tuple:
    """V12: website labels split by their 4-grams ("hypnosislucky" contains "hypn" and "luck");
    S1 side: first 4 letters of each Latin core word. Returns (t keys, s keys with idf and df).
    V13: a one-word name of 10+ letters is treated as a label too ("urologypartnerscom")."""
    label = (pl.when(pl.col("domain").str.len_chars() >= 6).then(pl.col("domain"))
             .when(~pl.col("core").str.contains(" ") & (pl.col("core").str.len_chars() >= 10)
                   & pl.col("core").str.contains(r"^[a-z0-9]+$")).then(pl.col("core")))
    d = tg.select("t", "country", d=label).drop_nulls("d")
    grams = (d.with_columns(g=pl.int_ranges(0, (pl.col("d").str.len_chars().cast(pl.Int64) - 3).clip(0)))
             .explode("g").select("t", "country", tok=pl.col("d").str.slice(pl.col("g"), 4)).unique())
    df = (grams.group_by("country", "tok").agg(df=pl.len()).filter(pl.col("df") <= df_max).join(n_c, on="country")
          .with_columns(idf=(pl.col("N") / pl.col("df")).log().cast(pl.Float32), key=_key(pl.col("country"), pl.col("tok"))))
    t_keys = grams.join(df, on=["country", "tok"]).select("t", "key")
    s_tok = (_name_tokens(s1, "s1").filter((pl.col("tok").str.len_chars() >= 4) & pl.col("tok").str.contains(r"^[a-z]"))
             .select("s1", "country", tok=pl.col("tok").str.slice(0, 4)).unique())
    s_keys = s_tok.join(df, on=["country", "tok"]).select("s1", "key", "idf", "df")
    return t_keys, s_keys


def _with_block_size(t_keys: pl.DataFrame, cap: int) -> pl.DataFrame:
    """Attach each key's block size bs (targets sharing it) and drop keys whose block exceeds cap.
    t_keys: t, key."""
    sizes = t_keys.group_by("key").agg(bs=pl.len().cast(pl.UInt32))
    return t_keys.join(sizes.filter(pl.col("bs") <= cap), on="key")


def build_index(cfg: Config, split: str, log=print) -> dict:
    """Build every blocking channel's S1-side and target-side key tables for a split.

    Returns: dict with s_<name> / t_<name> frames per channel (exact, rare, addr, a2, cc, sk, at, np, dm, ap, fz,
    nn, kn; target keys restricted to keys some S1 has), plus per-record stats and hashed strings for pair features."""
    t0 = time.time()
    s1 = load(cfg, split, "s1", ["s1", "country", "core", "core_key", "concat", "addr_nums", "addr_toks", "addr_blank"])
    tg = load(cfg, split, "tgt", ["t", "country", "core", "core_key", "concat", "domain", "addr_nums", "addr_toks", "addr_blank"])
    n_c = tg.group_by("country").agg(N=pl.len())
    ix = {}
    ix["s_exact"] = _exact_keys(s1, "s1")
    ix["t_exact"] = _with_block_size(_exact_keys(tg, "t"), cfg.rk_exact_cap)

    t_tok = _name_tokens(tg, "t")
    rare = _idf_table(t_tok.group_by("country", "tok").agg(df=pl.len()), n_c, cfg.rk_name_df_max)
    ix["t_rare"] = t_tok.join(rare, on=["country", "tok"]).select("t", "key")
    ix["s_rare"] = _name_tokens(s1, "s1").join(rare, on=["country", "tok"]).select("s1", "key", "idf")
    del t_tok

    ix["s_addr"] = _addr_keys(s1, "s1", cfg.rk_addr_nums, cfg.rk_addr_toks)
    ix["t_addr"] = _with_block_size(_addr_keys(tg, "t", cfg.rk_addr_nums, cfg.rk_addr_toks), cfg.rk_addr_cap)
    ix["s_at"] = _addr_keys(_truncated_numbers(s1), "s1", cfg.rk_addr_nums, cfg.rk_addr_toks)
    ix["t_at"] = ix["t_addr"]

    ix["s_np"] = _word_pair_keys(s1, "s1")
    ix["t_np"] = _with_block_size(_word_pair_keys(tg, "t"), cfg.rk_np_cap)
    ix["t_dm"], ix["s_dm"] = _domain_index(s1, tg, n_c, cfg.rk_dm_df_max)
    ix["s_ap"] = _addr_pair_keys(s1, "s1")
    ix["t_ap"] = _with_block_size(_addr_pair_keys(tg, "t"), cfg.rk_ap_cap)
    ix["s_fz"] = _fuzzy_keys(s1, "s1", spread=True)
    ix["t_fz"] = _with_block_size(_fuzzy_keys(tg, "t", spread=False), cfg.rk_fz_cap)

    t_aw = _addr_words(tg, "t")
    words = _idf_table(t_aw.group_by("country", "tok").agg(df=pl.len()), n_c, cfg.rk_a2_df_max)
    ix["t_a2"] = t_aw.join(words, on=["country", "tok"]).select("t", "key")
    ix["s_a2"] = _addr_words(s1, "s1").join(words, on=["country", "tok"]).select("s1", "key", "idf")
    del t_aw

    ix["s_cc"] = _concat_keys(s1, "s1", ["concat"], cfg.concat_len)
    ix["t_cc"] = _with_block_size(_concat_keys(tg, "t", ["concat", "domain"], cfg.concat_len), cfg.rk_concat_cap)

    # Stage 5: cross-script pass against Indic-script target names (pairs no Latin key can
    # reach). Single skeleton words are too common to block on (merged consonants collide),
    # so the keys are the exact full skeleton and unordered skeleton-word pairs. On train
    # these recover ~40% of the Indic true pairs the other passes miss.
    t_script = load(cfg, split, "tgt", ["t", "country", "script"]).filter(pl.col("script").is_in(["indic", "mixed"]))
    t_sk = _skeleton_keys(load(cfg, split, "tgt_sk").join(t_script, on="t"), "t")
    counts = t_sk.group_by("country", "tok").agg(df=pl.len())
    counts = counts.filter(pl.col("df") <= pl.when(pl.col("tok").str.starts_with("F:"))
                           .then(cfg.rk_sk_full_df_max).otherwise(cfg.rk_sk_pair_df_max))
    sk_idf = _idf_table(counts, n_c, max(cfg.rk_sk_full_df_max, cfg.rk_sk_pair_df_max))
    ix["t_sk"] = t_sk.join(sk_idf, on=["country", "tok"]).select("t", "key")
    ix["s_sk"] = (
        _skeleton_keys(load(cfg, split, "s1_sk").join(s1.select("s1", "country"), on="s1"), "s1")
        .join(sk_idf, on=["country", "tok"]).select("s1", "key", "idf")
    )
    del t_sk, t_script, counts

    # V17: name word x house number (Latin core words, and skeleton words, which also reach Indic scripts)
    s_sk_all = load(cfg, split, "s1_sk").join(s1.select("s1", "country", "addr_nums"), on="s1")
    t_sk_all = load(cfg, split, "tgt_sk").join(tg.select("t", "country", "addr_nums"), on="t")
    ix["s_nn"] = _word_num_keys(s1, "s1", "core", 3)
    ix["t_nn"] = _with_block_size(_word_num_keys(tg, "t", "core", 3), cfg.rk_wn_cap)
    ix["s_kn"] = _word_num_keys(s_sk_all, "s1", "name_sk", 2)
    ix["t_kn"] = _with_block_size(_word_num_keys(t_sk_all, "t", "name_sk", 2), cfg.rk_wn_cap)
    del s_sk_all, t_sk_all

    for name in ("exact", "rare", "addr", "a2", "cc", "sk", "at", "np", "dm", "ap", "fz", "nn", "kn"):
        ix[f"t_{name}"] = ix[f"t_{name}"].join(ix[f"s_{name}"].select("key").unique(), on="key", how="semi")
    log(f"[{split}] ranker index: " + ", ".join(f"{k}={v.height:,}" for k, v in ix.items())
        + f" ({time.time() - t0:.0f}s)")
    ix["s_stats"], ix["t_stats"] = record_stats(cfg, split)
    ix["s_str"], ix["t_str"] = record_strings(cfg, split)
    return ix


def record_strings(cfg: Config, split: str) -> tuple[pl.DataFrame, pl.DataFrame]:
    """Per record (sorted by index): hashed unique core-name and address tokens, their counts,
    and the hashed first house number. Integer sets keep the per-pair work cheap."""
    def hashed(c: str) -> pl.Expr:
        """Split column c into unique non-empty tokens hashed to UInt32."""
        return (pl.col(c).str.split(" ").list.set_difference(pl.lit([""]))
                .list.eval((pl.element().hash(seed=5) % 4_294_967_296).cast(pl.UInt32)))

    def one(name: str, idc: str) -> pl.DataFrame:
        """Hashed token lists, their counts and the hashed first house number (plus truncated variants) of one table."""
        df = (load(cfg, split, name, [idc, "core", "addr_toks", "addr_nums"])
              .join(load(cfg, split, f"{name}_sk"), on=idc, how="left").sort(idc).select(
            hc=hashed("core"), ha=hashed("addr_toks"), hs=hashed("name_sk"),
            num0=pl.col("addr_nums").str.split(" ").list.first().fill_null("")))
        num_hash = lambda e: pl.when(e.str.len_chars() == 0).then(None).otherwise(e.hash(seed=6))  # noqa: E731
        return df.with_columns(nc=pl.col("hc").list.len().cast(pl.Float32), na=pl.col("ha").list.len().cast(pl.Float32),
                               ns=pl.col("hs").list.len().cast(pl.Float32),
                               num0_tr=num_hash(pl.when(pl.col("num0").str.len_chars() >= 3)
                                                .then(pl.col("num0").str.head(-1)).otherwise(pl.lit(""))),
                               num0_tl=num_hash(pl.when(pl.col("num0").str.len_chars() >= 3)   # V13: first digit dropped
                                                .then(pl.col("num0").str.tail(-1).str.strip_chars_start("0"))
                                                .otherwise(pl.lit(""))),
                               num0=num_hash(pl.col("num0")))
    return one("s1", "s1"), one("tgt", "t")


def _sim_features(u: pl.DataFrame, s_str: pl.DataFrame, t_str: pl.DataFrame, batch: int = 2_000_000) -> pl.DataFrame:
    """Token Jaccard of core names and of address words, address containment, same first number.
    Computed in batches: gathered token lists for a whole 12M-pair chunk would cost several GB."""
    if u.height > batch:
        return pl.concat([_sim_features(u.slice(lo, batch), s_str, t_str, batch) for lo in range(0, u.height, batch)])
    s, t = s_str[u["s1"].to_numpy()], t_str[u["t"].to_numpy()]
    df = pl.DataFrame({"sc": s["hc"], "tc": t["hc"], "sa": s["ha"], "ta": t["ha"], "ns": s["nc"], "nt": t["nc"],
                       "na": s["na"], "nb": t["na"], "sn": s["num0"], "tn": t["num0"],
                       "ss": s["hs"], "ts": t["hs"], "ks": s["ns"], "kt": t["ns"],
                       "sn_tr": s["num0_tr"], "tn_tr": t["num0_tr"], "sn_tl": s["num0_tl"], "tn_tl": t["num0_tl"]})
    ni = pl.col("sc").list.set_intersection("tc").list.len().cast(pl.Float32)
    ai = pl.col("sa").list.set_intersection("ta").list.len().cast(pl.Float32)
    si = pl.col("ss").list.set_intersection("ts").list.len().cast(pl.Float32)
    return df.select(
        sim_name_jac=(ni / (pl.col("ns") + pl.col("nt") - ni)).fill_nan(0.0),
        sim_addr_jac=(ai / (pl.col("na") + pl.col("nb") - ai)).fill_nan(0.0),
        sim_addr_cont=(ai / pl.min_horizontal("na", "nb")).fill_nan(0.0),
        num0_eq=(pl.col("sn") == pl.col("tn")).fill_null(False).cast(pl.Float32),
        sim_sk_jac=(si / (pl.col("ks") + pl.col("kt") - si)).fill_nan(0.0),
        num0_tr=((pl.col("sn_tr") == pl.col("tn")) | (pl.col("tn_tr") == pl.col("sn"))
                 | (pl.col("sn_tl") == pl.col("tn")) | (pl.col("tn_tl") == pl.col("sn"))).fill_null(False).cast(pl.Float32),
    )


def record_stats(cfg: Config, split: str) -> tuple[pl.DataFrame, pl.DataFrame]:
    """Per-record statistics gathered into pair features (sorted by index)."""
    from features import name_frequencies

    def counts(prefix: str) -> list[pl.Expr]:
        """Token counts of core name, address words and address numbers, and the blank-address flag, named <prefix>_*."""
        def n(col):
            """Number of space-separated tokens in column col (0 for an empty string)."""
            return pl.when(pl.col(col) == "").then(0).otherwise(pl.col(col).str.count_matches(" ") + 1)
        return [n("core").cast(pl.Float32).alias(f"{prefix}_n_ctok"),
                n("addr_toks").cast(pl.Float32).alias(f"{prefix}_n_atok"),
                n("addr_nums").cast(pl.Float32).alias(f"{prefix}_n_nums"),
                pl.col("addr_blank").cast(pl.Float32).alias(f"{prefix}_blank")]

    tg = load(cfg, split, "tgt", ["t", "country", "core", "core_key", "addr_toks", "addr_nums", "addr_blank",
                                  "is_domain", "script", "src", "acro_self"])
    s1 = name_frequencies(load(cfg, split, "s1", ["s1", "country", "core", "core_key", "addr_toks", "addr_nums",
                                                   "addr_blank"]), tg.select("t", "country", "core_key"))
    s_stats = s1.sort("s1").select(
        *counts("s"),
        s_name_rep=pl.col("name_rep").cast(pl.Float32).log1p(),
        s_name_tdf=pl.col("name_tdf").cast(pl.Float32).log1p(),
    )
    t_stats = tg.sort("t").select(
        *counts("t"),
        t_domain=pl.col("is_domain").cast(pl.Float32),
        t_indic=pl.col("script").is_in(["indic", "mixed"]).cast(pl.Float32),
        t_src3=(pl.col("src") == 3).cast(pl.Float32),
        t_acro=(pl.col("acro_self") != "").cast(pl.Float32),
    )
    return s_stats, t_stats


# ------------------------------------------------------------------------------- union

def union_chunk(ix: dict, lo: int, hi: int) -> pl.DataFrame:
    """All candidate pairs for S1 rows [lo, hi) with blocking-graph features."""
    sel = pl.col("s1").is_between(lo, hi - 1)
    parts = [
        ix["s_exact"].filter(sel).join(ix["t_exact"], on="key").group_by("s1", "t")
        .agg(f_exact=pl.lit(1.0), e_bs=pl.col("bs").min().cast(pl.Float32)),
        ix["s_rare"].filter(sel).join(ix["t_rare"], on="key").group_by("s1", "t")
        .agg(rare_idf=pl.col("idf").sum(), rare_n=pl.len().cast(pl.Float32), rare_max=pl.col("idf").max()),
        ix["s_addr"].filter(sel).join(ix["t_addr"], on="key").group_by("s1", "t")
        .agg(addr_n=pl.len().cast(pl.Float32), addr_bs=pl.col("bs").min().cast(pl.Float32)),
        ix["s_a2"].filter(sel).join(ix["t_a2"], on="key").group_by("s1", "t")
        .agg(a2_n=pl.len().cast(pl.Float32), a2_idf=pl.col("idf").sum(), a2_max=pl.col("idf").max()),
        ix["s_cc"].filter(sel).join(ix["t_cc"], on="key").group_by("s1", "t")
        .agg(f_concat=pl.lit(1.0), cc_bs=pl.col("bs").min().cast(pl.Float32)),
        ix["s_sk"].filter(sel).join(ix["t_sk"], on="key").group_by("s1", "t")
        .agg(sk_n=pl.len().cast(pl.Float32), sk_idf=pl.col("idf").sum(), sk_max=pl.col("idf").max()),
        ix["s_at"].filter(sel).join(ix["t_at"], on="key").group_by("s1", "t")
        .agg(at_n=pl.len().cast(pl.Float32)),
        ix["s_np"].filter(sel).join(ix["t_np"], on="key").group_by("s1", "t")
        .agg(np_n=pl.len().cast(pl.Float32), np_bs=pl.col("bs").min().cast(pl.Float32)),
        # Website grams: one shared gram of a common word says little, so keep a pair only when
        # 2+ S1 words are found in the label or the one shared gram is rare.
        ix["s_dm"].filter(sel).join(ix["t_dm"], on="key").group_by("s1", "t")
        .agg(dm_n=pl.len().cast(pl.Float32), dm_idf=pl.col("idf").sum(), _dm_df=pl.col("df").min())
        .filter((pl.col("dm_n") >= 2) | (pl.col("_dm_df") <= 50)).drop("_dm_df"),
        ix["s_ap"].filter(sel).join(ix["t_ap"], on="key").group_by("s1", "t")
        .agg(ap_n=pl.len().cast(pl.Float32), ap_bs=pl.col("bs").min().cast(pl.Float32)),
        ix["s_fz"].filter(sel).join(ix["t_fz"], on="key").group_by("s1", "t")
        .agg(fz_n=pl.len().cast(pl.Float32), fz_bs=pl.col("bs").min().cast(pl.Float32)),
        ix["s_nn"].filter(sel).join(ix["t_nn"], on="key").group_by("s1", "t")
        .agg(nn_n=pl.len().cast(pl.Float32), nn_bs=pl.col("bs").min().cast(pl.Float32)),
        ix["s_kn"].filter(sel).join(ix["t_kn"], on="key").group_by("s1", "t")
        .agg(kn_n=pl.len().cast(pl.Float32), kn_bs=pl.col("bs").min().cast(pl.Float32)),
    ]
    cols = CHANNEL_COLS
    u = (
        pl.concat(parts, how="diagonal_relaxed")
        .group_by("s1", "t")
        .agg(pl.col(cols).max())
        .with_columns(pl.col(cols).fill_null(0.0).cast(pl.Float32))
    )
    u = u.with_columns(
        n_pass=((pl.col("f_exact") > 0).cast(pl.Float32) + (pl.col("rare_n") > 0).cast(pl.Float32)
                + (pl.col("addr_n") > 0).cast(pl.Float32) + (pl.col("a2_n") > 0).cast(pl.Float32)
                + (pl.col("f_concat") > 0).cast(pl.Float32) + (pl.col("sk_n") > 0).cast(pl.Float32)
                + (pl.col("at_n") > 0).cast(pl.Float32) + (pl.col("np_n") > 0).cast(pl.Float32)
                + (pl.col("dm_n") > 0).cast(pl.Float32) + (pl.col("ap_n") > 0).cast(pl.Float32)
                + (pl.col("fz_n") > 0).cast(pl.Float32) + (pl.col("nn_n") > 0).cast(pl.Float32)
                + (pl.col("kn_n") > 0).cast(pl.Float32)),
        e_bs=pl.col("e_bs").log1p(), addr_bs=pl.col("addr_bs").log1p(), cc_bs=pl.col("cc_bs").log1p(),
        np_bs=pl.col("np_bs").log1p(), ap_bs=pl.col("ap_bs").log1p(), fz_bs=pl.col("fz_bs").log1p(),
        nn_bs=pl.col("nn_bs").log1p(), kn_bs=pl.col("kn_bs").log1p(),
        n_union=pl.len().over("s1").cast(pl.Float32).log1p(),
        rare_rank=pl.col("rare_idf").rank("dense", descending=True).over("s1").cast(pl.Float32),
        addr_rank=pl.col("addr_n").rank("dense", descending=True).over("s1").cast(pl.Float32),
        a2_rank=pl.col("a2_idf").rank("dense", descending=True).over("s1").cast(pl.Float32),
        rare_rel=(pl.col("rare_idf") / pl.col("rare_idf").max().over("s1")).fill_nan(0.0),
        a2_rel=(pl.col("a2_idf") / pl.col("a2_idf").max().over("s1")).fill_nan(0.0),
        sk_rank=pl.col("sk_idf").rank("dense", descending=True).over("s1").cast(pl.Float32),
        sk_rel=(pl.col("sk_idf") / pl.col("sk_idf").max().over("s1")).fill_nan(0.0),
    )
    s_part = ix["s_stats"][u["s1"].to_numpy()]
    t_part = ix["t_stats"][u["t"].to_numpy()]
    sim = _sim_features(u, ix["s_str"], ix["t_str"])
    return pl.concat([u, s_part, t_part, sim], how="horizontal")


def _chunks(n: int, size: int):
    """Split range(n) into consecutive (lo, hi) chunks of at most size rows."""
    return [(lo, min(n, lo + size)) for lo in range(0, n, size)]


# ------------------------------------------------------------------------------ train

def _model_dir(cfg: Config):
    """Directory of the ranker models (created if missing)."""
    path = cfg.work_dir / "models" / MODEL_DIR
    path.mkdir(parents=True, exist_ok=True)
    return path


def train_ranker(cfg: Config, log=print) -> None:
    """Train the two cross-fitted candidate rankers (model A on folds {0,1}, model B on {2,3}).

    Samples the union of all blocking channels over dev-fold S1s (every positive, a hashed share of negatives,
    reweighted), early-stops on an own holdout (retrying at half the learning rate if too few trees) and writes
    ranker_A.txt, ranker_B.txt and meta.json."""
    t0 = time.time()
    ix = build_index(cfg, "train", log)
    gt = load(cfg, "train", "gt").with_columns(label=pl.lit(1, pl.UInt8))
    folds = load(cfg, "train", "folds", ["s1", "fold"])
    n = ix["s_stats"].height
    rows = []
    keep_neg = (pl.struct("s1", "t").hash(seed=cfg.seed) % 1_000_000) < int(cfg.rk_neg_rate * 1_000_000)
    for lo, hi in _chunks(n, cfg.rk_chunk)[: cfg.rk_sample_chunks]:
        u = (union_chunk(ix, lo, hi).join(gt, on=["s1", "t"], how="left")
             .with_columns(pl.col("label").fill_null(0)).join(folds, on="s1"))
        rows.append(u.filter((pl.col("fold") != cfg.sealed_fold) & ((pl.col("label") == 1) | keep_neg)))
        log(f"[ranker] sampled S1 {lo:,}-{hi - 1:,}: union {u.height:,} ({time.time() - t0:.0f}s)")
    data = pl.concat(rows)
    weight = pl.when(pl.col("label") == 1).then(1.0).otherwise(1.0 / cfg.rk_neg_rate)
    log(f"[ranker] training rows {data.height:,}, positives {int(data['label'].sum()):,}")

    groups = {"A": [0, 1], "B": [2, 3]}
    meta = {"features": RANK_FEATURES, "params": {**LGB_PARAMS, **ES_PARAMS}, "groups": groups, "config": cfg.fingerprint()}
    for name, train_folds in groups.items():
        own = data.filter(pl.col("fold").is_in(train_folds))
        hold = own.select(own_holdout(cfg.seed)).to_series()
        tr, va = own.filter(~hold), own.filter(hold)
        dtr = lgb.Dataset(tr.select(RANK_FEATURES).to_numpy(), tr["label"].to_numpy(),
                          weight=tr.select(weight).to_series().to_numpy(), feature_name=RANK_FEATURES)
        dva = lgb.Dataset(va.select(RANK_FEATURES).to_numpy(), va["label"].to_numpy(),
                          weight=va.select(weight).to_series().to_numpy(), reference=dtr)
        lr = LGB_PARAMS["learning_rate"]
        for _ in range(3):
            booster = lgb.train({**LGB_PARAMS, **ES_PARAMS, "learning_rate": lr}, dtr, num_boost_round=ES_ROUNDS,
                                valid_sets=[dva],
                                callbacks=[lgb.early_stopping(ES_PATIENCE, first_metric_only=True, verbose=False)])
            if booster.best_iteration >= MIN_TREES:
                break
            log(f"[ranker] model {name}: only {booster.best_iteration} trees at lr {lr}; retrying at lr {lr / 2}")
            lr /= 2
        else:
            raise RuntimeError(f"ranker model {name} keeps stopping before {MIN_TREES} trees")
        meta[f"learning_rate_{name}"] = lr
        booster.save_model(str(_model_dir(cfg) / f"ranker_{name}.txt"), num_iteration=booster.best_iteration)
        gain = sorted(zip(RANK_FEATURES, booster.feature_importance("gain")), key=lambda kv: -kv[1])[:10]
        meta[f"best_iter_{name}"] = booster.best_iteration
        meta[f"valid_logloss_{name}"] = booster.best_score["valid_0"]["binary_logloss"]
        meta[f"valid_auc_{name}"] = booster.best_score["valid_0"]["auc"]
        log(f"[ranker] model {name}: best_iter={booster.best_iteration} valid auc={meta[f'valid_auc_{name}']:.5f} "
            f"logloss={meta[f'valid_logloss_{name}']:.4f}; top gain: " + ", ".join(k for k, _ in gain))
    (_model_dir(cfg) / "meta.json").write_text(json.dumps(meta, indent=2))
    log(f"[ranker] trained in {time.time() - t0:.0f}s")


# ------------------------------------------------------------------------------- rank

def _predict(booster: lgb.Booster, df: pl.DataFrame, batch: int = 4_000_000) -> np.ndarray:
    """Ranker probabilities for the RANK_FEATURES of df, predicted in batches to bound memory."""
    out = np.empty(df.height, dtype=np.float32)
    for lo in range(0, df.height, batch):
        x = df.slice(lo, batch).select(RANK_FEATURES).to_numpy()
        out[lo:lo + batch] = booster.predict(x, num_threads=0)
    return out


def rank_split(cfg: Config, split: str, log=print) -> None:
    """Score every union pair of a split and keep the top rk_keep per S1 in cand_b/part_*.parquet.

    Train dev folds are scored by the ranker that did not train on them (cross-fitting); the sealed fold and test
    get the mean of both. Kept rows carry channel columns, the ranker score `cheap` and the rank `rrank`."""
    t0 = time.time()
    models = {name: lgb.Booster(model_file=str(_model_dir(cfg) / f"ranker_{name}.txt")) for name in ("A", "B")}
    ix = build_index(cfg, split, log)
    fold_of = None
    if split == "train":
        fold_of = load(cfg, "train", "folds", ["s1", "fold"]).sort("s1")["fold"].to_numpy()
    out_dir = cfg.split_dir(split) / "cand_b"
    out_dir.mkdir(exist_ok=True)
    for old in out_dir.glob("part_*.parquet"):
        old.unlink()
    n = ix["s_stats"].height
    union_total = kept_total = 0
    for i, (lo, hi) in enumerate(_chunks(n, cfg.rk_chunk)):
        u = union_chunk(ix, lo, hi)
        if fold_of is None:
            score = 0.5 * (_predict(models["A"], u) + _predict(models["B"], u))
        else:
            fold = fold_of[u["s1"].to_numpy()]
            score = np.empty(u.height, dtype=np.float32)
            in_a = np.isin(fold, [2, 3])      # scored by model A (trained on 0,1)
            in_b = np.isin(fold, [0, 1])      # scored by model B (trained on 2,3)
            both = ~(in_a | in_b)             # sealed fold: mean of both
            for mask, name in ((in_a, "A"), (in_b, "B")):
                if mask.any():
                    score[mask] = _predict(models[name], u.filter(pl.Series(mask)))
            if both.any():
                sub = u.filter(pl.Series(both))
                score[both] = 0.5 * (_predict(models["A"], sub) + _predict(models["B"], sub))
        top = (
            u.select("s1", "t", "f_exact", "rare_idf", "rare_n", "addr_n", "a2_n", "a2_idf", "f_concat",
                     "sk_n", "sk_idf", "at_n", "np_n", "dm_n", "sim_sk_jac", "num0_tr", "ap_n", "fz_n")
            .with_columns(cheap=pl.Series(score, dtype=pl.Float32))
            .sort(["s1", "cheap", "t"], descending=[False, True, False])
            .group_by("s1", maintain_order=True)
            .head(cfg.rk_keep)
            .with_columns(rrank=pl.int_range(pl.len()).over("s1").cast(pl.UInt8) + 1)
        )
        top.write_parquet(out_dir / f"part_{i:03d}.parquet")
        union_total += u.height
        kept_total += top.height
        if i % 5 == 0:
            log(f"[{split}] rank chunk {i}: union {u.height:,} -> kept {top.height:,} ({time.time() - t0:.0f}s)")
    log(f"[{split}] union {union_total:,} pairs ({union_total / n:.0f}/S1) -> kept {kept_total:,} "
        f"({kept_total / n:.1f}/S1) in {time.time() - t0:.0f}s")


HOP_KEYS = ("k_addr", "k_num_tok")   # hop value 1, 2 (0 = ranked candidate)
HOP_FEATURES = ["hop", "hop_p", "hop_n", "hop_bs"]   # key; best seed p; seed siblings sharing the key; key block size
HOP_MIN_TSR = 80   # V20: added pairs need name token_set_ratio >= 80 (V19's hop: 3.7 added/S1 at 0.17% true; with it
                   # 0.03/S1 at 34% true, half the recovered misses; experiments/retrieval)
HOP_SK_MAX = 3     # V21: hops 1-3 (address keys) also pass on the name-skeleton ratio. `core` keeps Indic script, so an
                   # Indic target scored 0 against the Latin S1 and was always dropped: sealed fold +5.1% of India misses
                   # at +0.022 added/S1, 4.5% true (hop 4 = nw_aw is reached through core words, left on core)
_E = pl.element()


def _words(col: str, minlen: int, n: int, struct: bool = False) -> pl.Expr:
    """Expression: first n distinct words of col with at least minlen characters (optionally without
    structural address words such as street types)."""
    f = _E.str.len_chars() >= minlen
    if struct:
        f = f & ~_E.is_in(STRUCT_LIST)
    return pl.col(col).str.split(" ").list.eval(_E.filter(f)).list.unique(maintain_order=True).list.head(n)


def _num_ap_keys(df: pl.DataFrame, idc: str) -> pl.DataFrame:
    """House number (first 2) x unordered pair of the first 4 non-structural address words."""
    w = df.filter(~pl.col("addr_blank")).select(idc, "country", n=_words("addr_nums", 1, 2), a=_words("addr_toks", 3, 4, True))
    x = w.explode("a").drop_nulls("a")
    p = x.join(x.select(idc, b="a"), on=idc).filter(pl.col("a") < pl.col("b"))
    return p.explode("n").drop_nulls("n").select(idc, key=_key(pl.col("country"), pl.col("n"), pl.col("a"), pl.col("b"))).unique()


def _nw_aw_keys(df: pl.DataFrame, idc: str) -> pl.DataFrame:
    """Name word (first 4, 3+ letters) x address word (first 4 non-structural)."""
    return (df.filter(~pl.col("addr_blank")).select(idc, "country", a=_words("core", 3, 4), b=_words("addr_toks", 3, 4, True))
            .explode("a").explode("b").drop_nulls().select(idc, key=_key(pl.col("country"), pl.col("a"), pl.col("b"))).unique())


def _skf_addr_keys(df: pl.DataFrame, idc: str) -> pl.DataFrame:
    """V21: full name skeleton (sorted unique skeleton words) x each house number / each of the first 12 non-structural
    address words. Reaches Indic-script targets ("మా ప్రొడక్ట్స్" = "Maa Products") whose core shares no word with the S1."""
    w = pl.col("name_sk").str.split(" ").list.eval(_E.filter(_E.str.len_chars() >= 2)).list.unique().list.sort().list.join(" ")
    nums = pl.col("addr_nums").str.split(" ").list.eval(_E.filter(_E != "")).list.unique()
    return (df.select(idc, "country", a=w, b=pl.concat_list(nums, _words("addr_toks", 3, 12, True))).filter(pl.col("a") != "")
            .explode("b").drop_nulls("b").select(idc, key=_key(pl.col("country"), pl.col("a"), pl.col("b"))).unique())


def _label_keys(df: pl.DataFrame, idc: str) -> pl.DataFrame:
    """V21: S1 concatenated name == target website label ("Colonial Controls" / "colonialcontrols.com"). The ranker's
    concat key is a 6-letter prefix whose blocks are mostly over the cap; the whole label is a small block."""
    c = pl.col("concat" if idc == "s1" else "domain")
    return df.filter(c.str.len_chars() >= 5).select(idc, key=_key(pl.col("country"), c))


_WL = lambda c: pl.col(c).str.split(" ").list.set_difference(pl.lit([""]))  # noqa: E731
_NUM_SH = _WL("sn").list.set_intersection(_WL("tn")).list.len()
_AW_SH = _WL("sa").list.set_intersection(_WL("ta")).list.len()
_NUM0 = (pl.col("sn").str.split(" ").list.first() == pl.col("tn").str.split(" ").list.first()) & (pl.col("tn") != "")
_INDIC_T = pl.col("script").is_in(["indic", "mixed"])

# name -> (key fn, block cap, target filter, address rule). Hop value = position + 3.
# V20 direct channels (hop 3, 4; name tsr >= 80): sealed 25% sample: num_ap (cap 20) 0.03 added/S1 at 21% true;
# nw_aw (cap 10) 0.5/S1 at 1.3% true, +18% of retrieval misses.
# V21 channels (hop 5, 6) are kept on address agreement instead of the name filter (their names already agree, and
# an Indic or website name has no Latin core to compare). V20 residual misses, sealed fold 4 (dev fold 0), distinct
# pairs beyond V20 cand_b: skf_ad +0.10 added/S1 at 2.0% true, +11.3% (11.9%) of India misses; dom +0.013/S1 at 8.6%
# true, +4.8% India / +2.7% US (experiments/retrieval2).
EXTRA_CHANNELS = {"num_ap": (_num_ap_keys, 20, None, None), "nw_aw": (_nw_aw_keys, 10, None, None),
                  "skf_ad": (_skf_addr_keys, 50, _INDIC_T, (_NUM_SH >= 2) | (_NUM0 & (_AW_SH >= 1))),
                  "dom": (_label_keys, 50, None, (_NUM_SH >= 1) & (_AW_SH >= 1))}


def hop_pairs(cfg: Config, split: str, src, log=print, p_min: float = 0.9, cap: int = 200) -> tuple:
    """(pairs, s_names, t_names): every second-hop and direct-channel pair (s1, t, hop = lowest key number, hop_p,
    hop_n, hop_bs, v21 = passed a V21 channel's address rule), before the ranked anti-join and the name filter, and
    the core / skeleton names that filter compares."""
    from pathlib import Path
    conf = pl.read_parquet(Path(src) / split / "scores_c.parquet", columns=["s1", "t", "p"]).filter(pl.col("p") > p_min)

    def with_sk(d: pl.DataFrame, idc: str) -> pl.DataFrame:
        """Left-join the name skeleton (name_sk, empty string when missing) onto a table keyed by idc."""
        sk = load(cfg, split, "tgt_sk" if idc == "t" else "s1_sk")
        return d.join(sk, on=idc, how="left").with_columns(pl.col("name_sk").fill_null(""))

    tg = with_sk(load(cfg, split, "tgt", ["t", "country", "core", "domain", "addr_toks", "addr_nums", "addr_blank", "script"]), "t")
    s1c = with_sk(load(cfg, split, "s1", ["s1", "country", "core", "concat", "addr_toks", "addr_nums", "addr_blank"]), "s1")
    num1 = pl.col("addr_nums").str.split(" ").list.eval(pl.element().filter(pl.element().str.len_chars() >= 2)).list.first()
    keys = tg.select("t",
                     k_addr=pl.when(~pl.col("addr_blank") & (pl.col("addr_toks") != ""))
                     # audit: number order-free, so "Unit 1416, 6515 Belcrest Rd" meets "6515 Belcrest Rd, Unit 1416"
                     .then(pl.concat_str("country", "addr_toks", pl.col("addr_nums").str.split(" ").list.sort().list.join(" "),
                                         separator="|")),
                     k_num_tok=pl.when(num1.is_not_null() & (pl.col("addr_toks") != ""))
                     .then(pl.concat_str("country", num1, pl.col("addr_toks").str.split(" ").list.slice(0, 2).list.join(" "),
                                         separator="|")))
    f32 = pl.Float32
    new = []
    for h, k in enumerate(HOP_KEYS, 1):   # provenance: key, best seed p, number of seed siblings, key block size
        kt = keys.select("t", key=k).drop_nulls()
        kt = kt.join(kt.group_by("key").len("sz").filter(pl.col("sz") <= cap), on="key")
        seeds = conf.join(kt.drop("sz"), on="t").rename({"t": "t_seed"})
        reach = seeds.join(kt, on="key").group_by("s1", "t").agg(hop_p=pl.col("p").max(), hop_n=pl.col("t_seed").n_unique(),
                                                                  hop_bs=pl.col("sz").min())
        new.append(reach.select("s1", "t", hop=pl.lit(float(h), f32), hop_p=pl.col("hop_p").cast(f32),
                                hop_n=pl.col("hop_n").cast(f32), hop_bs=pl.col("hop_bs").cast(f32), v21=pl.lit(False)))
    del keys
    for h, (fn, bcap, t_only, rule) in enumerate(EXTRA_CHANNELS.values(), len(HOP_KEYS) + 1):
        sk = fn(s1c, "s1")
        kset = sk.select("key").unique()
        tsub = tg if t_only is None else tg.filter(t_only)
        tk = pl.concat([fn(tsub.slice(lo, 500_000), "t").join(kset, on="key", how="semi") for lo in range(0, tsub.height, 500_000)])
        tk = tk.join(tk.group_by("key").len("sz").filter(pl.col("sz") <= bcap), on="key")
        for lo in range(0, s1c.height, 200_000):   # bounded memory
            pairs = (sk.filter(pl.col("s1").is_between(lo, lo + 199_999)).join(tk, on="key")
                     .group_by("s1", "t").agg(hop_bs=pl.col("sz").min()))
            if rule is not None:   # V21 channels: kept on address agreement
                pairs = (pairs.join(s1c.select("s1", sn="addr_nums", sa="addr_toks"), on="s1")
                         .join(tg.select("t", tn="addr_nums", ta="addr_toks"), on="t").filter(rule))
            new.append(pairs.select("s1", "t", hop=pl.lit(float(h), f32), hop_p=pl.lit(0.0, f32), hop_n=pl.lit(0.0, f32),
                                    hop_bs=pl.col("hop_bs").cast(f32), v21=pl.lit(rule is not None)))
    s_names = s1c.select("s1", s_core=pl.col("core").fill_null(""), s_sk="name_sk")
    t_names = tg.select("t", t_core=pl.col("core").fill_null(""), t_sk="name_sk")
    del tg, s1c
    new = pl.concat(new).group_by("s1", "t").agg(pl.col("hop").min(), pl.col("hop_p").max(), pl.col("hop_n").max(),
                                                 pl.col("hop_bs").min(), pl.col("v21").any())
    log(f"[{split}] hop pairs {new.height:,} from {conf.height:,} confident pairs")
    return new, s_names, t_names


def hop_keep(add: pl.DataFrame) -> pl.Series:
    """Name filter on hop pairs (with s_core, t_core, s_sk, t_sk): token_set_ratio of the cores >= HOP_MIN_TSR, or of
    the name skeletons for hops <= HOP_SK_MAX; pairs a V21 channel kept on address agreement pass as they are."""
    from rapidfuzz import fuzz, process
    ok = add["v21"].to_numpy() | (process.cpdist(add["s_core"].to_list(), add["t_core"].to_list(), scorer=fuzz.token_set_ratio,
                                                 workers=8) >= HOP_MIN_TSR)
    rest = ~ok & (add["hop"].to_numpy() <= HOP_SK_MAX)
    if rest.any():
        sub = add.filter(pl.Series(rest))
        ok[rest] = process.cpdist(sub["s_sk"].to_list(), sub["t_sk"].to_list(), scorer=fuzz.token_set_ratio, workers=8) >= HOP_MIN_TSR
    return pl.Series(ok)


def second_hop(cfg: Config, split: str, src, log=print, p_min: float = 0.9, cap: int = 200) -> None:
    """V19: add the targets that share an exact key with one of the S1's confident matches (stack p > p_min in
    <src>/<split>/scores_c.parquet, an earlier run on the same prepared data). Records of one entity in one
    source share that source's view of the address, so a copy the S1's own (differently noised) address never
    reached is found from its sibling. On V17's 60 sealed candidates: exact address key +0.8 candidates/S1
    recovers 26% of retrieval misses, house number + first street words +3.2 recovers 27%
    (experiments/second_hop.py). V20/V21: plus the direct EXTRA_CHANNELS, all filtered by hop_keep.
    Added pairs get rrank = rk_cand_k, zero channel columns, the S1's lowest ranker score, and hop = key number;
    rerunning replaces earlier hop rows."""
    t0 = time.time()
    new, s_names, t_names = hop_pairs(cfg, split, src, log, p_min, cap)
    parts = sorted((cfg.split_dir(split) / "cand_b").glob("part_*.parquet"))
    added = 0
    for i, part in enumerate(parts):
        c = pl.read_parquet(part)
        c = c.filter(pl.col("hop") == 0) if "hop" in c.columns else c.with_columns(
            [pl.lit(0.0, pl.Float32).alias(f) for f in HOP_FEATURES])
        lo, hi = i * cfg.rk_chunk, (i + 1) * cfg.rk_chunk          # rank_split writes S1 index ranges in order
        add = (new.filter((pl.col("s1") >= lo) & (pl.col("s1") < hi)).join(c.select("s1", "t"), on=["s1", "t"], how="anti")
               .join(c.group_by("s1").agg(pl.col("cheap").min()), on="s1", how="left"))
        add = add.join(s_names, on="s1").join(t_names, on="t")
        if add.height:
            add = add.filter(hop_keep(add))
        add = add.drop("s_core", "t_core", "s_sk", "t_sk", "v21")
        fill = [pl.lit(0, c.schema[col]).alias(col) for col in c.columns if col not in ["s1", "t", "cheap", "rrank"] + HOP_FEATURES]
        add = add.with_columns(*fill, pl.col("cheap").fill_null(0.0).cast(c.schema["cheap"]),
                               rrank=pl.lit(min(cfg.rk_cand_k, 255), pl.UInt8)).select(c.columns)
        pl.concat([c, add]).sort("s1", "rrank").write_parquet(part)
        added += add.height
    n_s1 = load(cfg, split, "s1", ["s1"]).height
    log(f"[{split}] second hop: +{added:,} candidates ({added / n_s1:.2f}/S1) in {time.time() - t0:.0f}s")


def ranked_report(cfg: Config, log=print) -> dict:
    """Recall and oracle F0.5 at several cut-offs, dev vs sealed, by country and Indic targets."""
    from score import candidate_metrics

    parts = sorted((cfg.split_dir("train") / "cand_b").glob("part_*.parquet"))
    cand = pl.concat([pl.read_parquet(p, columns=["s1", "t", "rrank"]) for p in parts])
    gt = load(cfg, "train", "gt")
    s1 = load(cfg, "train", "s1", ["s1", "country"]).join(load(cfg, "train", "folds", ["s1", "fold"]), on="s1")
    indic_t = load(cfg, "train", "tgt", ["t", "script"]).filter(pl.col("script").is_in(["indic", "mixed"])).select("t")
    res = {}
    for k in (10, 20, 30, 40, 50, 60):
        ck = cand.filter(pl.col("rrank") <= k)
        for part, ids in (("dev", s1.filter(pl.col("fold") != cfg.sealed_fold)),
                          ("sealed", s1.filter(pl.col("fold") == cfg.sealed_fold))):
            m = candidate_metrics(ids, ck, gt)
            gti = gt.join(ids.select("s1"), on="s1", how="semi").join(indic_t, on="t", how="semi")
            m["indic_recall"] = gti.join(ck, on=["s1", "t"], how="semi").height / max(gti.height, 1)
            for c in ("US", "India"):
                mc = candidate_metrics(ids.filter(pl.col("country") == c), ck, gt)
                m[f"recall_{c}"] = mc["pair_recall"]
            res[f"K{k}_{part}"] = m
            log(f"[ranked K={k:>2} {part:6s}] recall={m['pair_recall']:.4f} US={m['recall_US']:.4f} "
                f"India={m['recall_India']:.4f} indic={m['indic_recall']:.4f} oracle={m['oracle_macro_f05']:.4f} "
                f"cand/S1={m['cand_per_s1_mean']:.1f}")
    (cfg.split_dir("train") / "report_cand_b.json").write_text(json.dumps(res, indent=2))
    return res
