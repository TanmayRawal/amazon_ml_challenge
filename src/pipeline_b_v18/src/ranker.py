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

from blocking import (_addr_keys, _addr_words, _cap_blocks, _concat_keys, _exact_keys, _key,
                      _name_tokens)
from config import Config
from fuzzy import chunk_pairs
from prepare import load

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
    # Stage 6g: (name word x address word) pass.
    "na_n", "na_bs",
    # v12: char-trigram TF-IDF pass (fuzzy.py): cosine and rank among the S1's fuzzy neighbours.
    "fz_cos", "fz_rank",
]
# v12: without regularization the rankers produced extreme leaves (logits of -30 to -100 for some
# true pairs): cross-fitted dev scoring then ranked obvious matches last (v11: dev union recall
# 0.977, top-60 0.970). L2, a leaf-output cap and a hessian floor keep the scores finite.
LGB_PARAMS = {
    "objective": "binary", "learning_rate": 0.1, "num_leaves": 63, "min_data_in_leaf": 200,
    "feature_fraction": 0.9, "lambda_l2": 10.0, "max_delta_step": 2.0, "min_sum_hessian_in_leaf": 1.0,
    "verbosity": -1, "seed": 2026, "deterministic": True, "force_row_wise": True,
}
MODEL_DIR = "ranker"


# ----------------------------------------------------------------------------- indexes

def _idf_table(tok_df: pl.DataFrame, idf_ref: float, df_max: int) -> pl.DataFrame:
    """Rarity log(idf_ref / df): depends on how many records share the token, not on the size of
    the pool (test US has 62% of train's targets, France 23%)."""
    return (
        tok_df.filter(pl.col("df") <= df_max)
        .with_columns(idf=(idf_ref / pl.col("df")).log().cast(pl.Float32),
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


def _name_addr_keys(df: pl.DataFrame, idc: str) -> pl.DataFrame:
    """Every (name word of >= 3 letters, address word) pair of a record, hashed with its country.
    Built one country at a time: the string cross product is ~90M rows for train targets."""
    out = []
    for country in df["country"].unique().to_list():
        sub = df.filter(pl.col("country") == country)
        nt = _name_tokens(sub, idc).filter(pl.col("tok").str.len_chars() >= 3).select(idc, nt="tok")
        aw = _addr_words(sub, idc).select(idc, aw="tok")
        out.append(nt.join(aw, on=idc).select(idc, key=_key(pl.lit(country), pl.col("nt"), pl.col("aw"))).unique())
    return pl.concat(out)


def _with_block_size(t_keys: pl.DataFrame, cap: int) -> pl.DataFrame:
    """Attach the block size `bs` to each target key and drop keys with more than `cap` targets."""
    sizes = t_keys.group_by("key").agg(bs=pl.len().cast(pl.UInt32))
    return t_keys.join(sizes.filter(pl.col("bs") <= cap), on="key")


def build_index(cfg: Config, split: str, log=print) -> dict:
    """Build every blocking index of the stage-2 union for a split.

    Returns a dict of S1/target key tables per pass (exact, rare, addr, a2, cc, sk, na), per-record
    stats and hashed strings, and the fuzzy-pass directory and chunk size.
    """
    t0 = time.time()
    s1 = load(cfg, split, "s1", ["s1", "country", "core", "core_key", "concat", "addr_nums", "addr_toks", "addr_blank"])
    tg = load(cfg, split, "tgt", ["t", "country", "core", "core_key", "concat", "domain", "addr_nums", "addr_toks", "addr_blank"])
    ix = {}
    ix["s_exact"] = _exact_keys(s1, "s1")
    ix["t_exact"] = _with_block_size(_exact_keys(tg, "t"), cfg.rk_exact_cap)

    t_tok = _name_tokens(tg, "t")
    rare = _idf_table(t_tok.group_by("country", "tok").agg(df=pl.len()), cfg.idf_ref, cfg.rk_name_df_max)
    ix["t_rare"] = t_tok.join(rare, on=["country", "tok"]).select("t", "key")
    ix["s_rare"] = _name_tokens(s1, "s1").join(rare, on=["country", "tok"]).select("s1", "key", "idf")
    del t_tok

    ix["s_addr"] = _addr_keys(s1, "s1", cfg.rk_addr_nums, cfg.rk_addr_toks)
    ix["t_addr"] = _with_block_size(_addr_keys(tg, "t", cfg.rk_addr_nums, cfg.rk_addr_toks), cfg.rk_addr_cap)

    t_aw = _addr_words(tg, "t")
    words = _idf_table(t_aw.group_by("country", "tok").agg(df=pl.len()), cfg.idf_ref, cfg.rk_a2_df_max)
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
    sk_idf = _idf_table(counts, cfg.idf_ref, max(cfg.rk_sk_full_df_max, cfg.rk_sk_pair_df_max))
    ix["t_sk"] = t_sk.join(sk_idf, on=["country", "tok"]).select("t", "key")
    ix["s_sk"] = (
        _skeleton_keys(load(cfg, split, "s1_sk").join(s1.select("s1", "country"), on="s1"), "s1")
        .join(sk_idf, on=["country", "tok"]).select("s1", "key", "idf")
    )
    del t_sk, t_script, counts

    # Stage 6g: (name word x address word) keys, at most rk_na_df_max targets per key. They reach
    # true pairs that share a common name word and an address word but no rare key (~23% of the
    # dev true pairs no other pass reached, for ~30 more union pairs per S1).
    ix["t_na"] = _with_block_size(_name_addr_keys(tg, "t"), cfg.rk_na_df_max)
    ix["s_na"] = _name_addr_keys(s1, "s1")

    for name in ("exact", "rare", "addr", "a2", "cc", "sk", "na"):
        ix[f"t_{name}"] = ix[f"t_{name}"].join(ix[f"s_{name}"].select("key").unique(), on="key", how="semi")
    log(f"[{split}] ranker index: " + ", ".join(f"{k}={v.height:,}" for k, v in ix.items())
        + f" ({time.time() - t0:.0f}s)")
    ix["s_stats"], ix["t_stats"] = record_stats(cfg, split)
    ix["s_str"], ix["t_str"] = record_strings(cfg, split)
    ix["fz_dir"], ix["chunk"] = cfg.split_dir(split) / "fuzzy", cfg.rk_chunk
    return ix


def record_strings(cfg: Config, split: str) -> tuple[pl.DataFrame, pl.DataFrame]:
    """Per record (sorted by index): hashed unique core-name and address tokens, their counts,
    and the hashed first house number. Integer sets keep the per-pair work cheap."""
    def hashed(c: str) -> pl.Expr:
        """Unique non-empty tokens of column `c`, hashed to UInt32."""
        return (pl.col(c).str.split(" ").list.set_difference(pl.lit([""]))
                .list.eval((pl.element().hash(seed=5) % 4_294_967_296).cast(pl.UInt32)))

    def one(name: str, idc: str) -> pl.DataFrame:
        """Hashed token sets, token counts and hashed first number for table `name` ("s1"/"tgt")."""
        df = load(cfg, split, name, [idc, "core", "addr_toks", "addr_nums"]).sort(idc).select(
            hc=hashed("core"), ha=hashed("addr_toks"),
            num0=pl.col("addr_nums").str.split(" ").list.first().fill_null(""))
        return df.with_columns(nc=pl.col("hc").list.len().cast(pl.Float32), na=pl.col("ha").list.len().cast(pl.Float32),
                               num0=pl.when(pl.col("num0") == "").then(None).otherwise(pl.col("num0").hash(seed=6)))
    return one("s1", "s1"), one("tgt", "t")


def _sim_features(u: pl.DataFrame, s_str: pl.DataFrame, t_str: pl.DataFrame, batch: int = 2_000_000) -> pl.DataFrame:
    """Token Jaccard of core names and of address words, address containment, same first number.
    Computed in batches: gathered token lists for a whole 12M-pair chunk would cost several GB."""
    if u.height > batch:
        return pl.concat([_sim_features(u.slice(lo, batch), s_str, t_str, batch) for lo in range(0, u.height, batch)])
    s, t = s_str[u["s1"].to_numpy()], t_str[u["t"].to_numpy()]
    df = pl.DataFrame({"sc": s["hc"], "tc": t["hc"], "sa": s["ha"], "ta": t["ha"], "ns": s["nc"], "nt": t["nc"],
                       "na": s["na"], "nb": t["na"], "sn": s["num0"], "tn": t["num0"]})
    ni = pl.col("sc").list.set_intersection("tc").list.len().cast(pl.Float32)
    ai = pl.col("sa").list.set_intersection("ta").list.len().cast(pl.Float32)
    return df.select(
        sim_name_jac=(ni / (pl.col("ns") + pl.col("nt") - ni)).fill_nan(0.0),
        sim_addr_jac=(ai / (pl.col("na") + pl.col("nb") - ai)).fill_nan(0.0),
        sim_addr_cont=(ai / pl.min_horizontal("na", "nb")).fill_nan(0.0),
        num0_eq=(pl.col("sn") == pl.col("tn")).fill_null(False).cast(pl.Float32),
    )


def record_stats(cfg: Config, split: str) -> tuple[pl.DataFrame, pl.DataFrame]:
    """Per-record statistics gathered into pair features (sorted by index)."""
    from features import name_frequencies

    def counts(prefix: str) -> list[pl.Expr]:
        """Token counts of name, address words and numbers plus the blank-address flag, named with `prefix`."""
        def n(col):
            """Number of space-separated tokens in `col` (0 for an empty string)."""
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
        ix["s_na"].filter(sel).join(ix["t_na"], on="key").group_by("s1", "t")
        .agg(na_n=pl.len().cast(pl.Float32), na_bs=pl.col("bs").min().cast(pl.Float32)),
    ]
    cols = ["f_exact", "e_bs", "rare_idf", "rare_n", "rare_max", "addr_n", "addr_bs",
            "a2_n", "a2_idf", "a2_max", "f_concat", "cc_bs", "sk_n", "sk_idf", "sk_max", "na_n", "na_bs"]
    if ix.get("fz_dir") is not None and ix["fz_dir"].exists():
        parts.append(chunk_pairs(ix["fz_dir"], ix["chunk"], lo, hi))
        cols += ["fz_cos", "fz_rank"]
    else:
        raise FileNotFoundError(f"fuzzy pass missing: run `run.py fuzzy --split <split>` first ({ix.get('fz_dir')})")
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
                + (pl.col("na_n") > 0).cast(pl.Float32) + (pl.col("fz_cos") > 0).cast(pl.Float32)),
        e_bs=pl.col("e_bs").log1p(), addr_bs=pl.col("addr_bs").log1p(), cc_bs=pl.col("cc_bs").log1p(),
        na_bs=pl.col("na_bs").log1p(),
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
    """Split [0, n) into consecutive (lo, hi) ranges of at most `size`."""
    return [(lo, min(n, lo + size)) for lo in range(0, n, size)]


# ------------------------------------------------------------------------------ train

def _model_dir(cfg: Config):
    """Ranker model directory under <work_dir>/models (created if needed)."""
    path = cfg.work_dir / "models" / MODEL_DIR
    path.mkdir(parents=True, exist_ok=True)
    return path


def train_ranker(cfg: Config, log=print) -> None:
    """Train the cross-fitted LightGBM rankers A and B on sampled union pairs of the train dev folds.

    Positives are all kept, negatives sampled at rk_neg_rate (re-weighted); models and meta.json saved.
    """
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
    meta = {"features": RANK_FEATURES, "params": LGB_PARAMS, "groups": groups, "config": cfg.fingerprint()}
    for name, train_folds in groups.items():
        tr = data.filter(pl.col("fold").is_in(train_folds))
        va = data.filter(~pl.col("fold").is_in(train_folds))
        dtr = lgb.Dataset(tr.select(RANK_FEATURES).to_numpy(), tr["label"].to_numpy(),
                          weight=tr.select(weight).to_series().to_numpy(), feature_name=RANK_FEATURES)
        dva = lgb.Dataset(va.select(RANK_FEATURES).to_numpy(), va["label"].to_numpy(),
                          weight=va.select(weight).to_series().to_numpy(), reference=dtr)
        booster = lgb.train(LGB_PARAMS, dtr, num_boost_round=800, valid_sets=[dva],
                            callbacks=[lgb.early_stopping(30, verbose=False)])
        booster.save_model(str(_model_dir(cfg) / f"ranker_{name}.txt"), num_iteration=booster.best_iteration)
        gain = sorted(zip(RANK_FEATURES, booster.feature_importance("gain")), key=lambda kv: -kv[1])[:10]
        meta[f"best_iter_{name}"] = booster.best_iteration
        meta[f"valid_logloss_{name}"] = booster.best_score["valid_0"]["binary_logloss"]
        log(f"[ranker] model {name}: best_iter={booster.best_iteration} valid logloss="
            f"{meta[f'valid_logloss_{name}']:.4f}; top gain: " + ", ".join(k for k, _ in gain))
    (_model_dir(cfg) / "meta.json").write_text(json.dumps(meta, indent=2))
    log(f"[ranker] trained in {time.time() - t0:.0f}s")


# ------------------------------------------------------------------------------- rank

def _predict(booster: lgb.Booster, df: pl.DataFrame, batch: int = 4_000_000) -> np.ndarray:
    """Ranker probabilities for the RANK_FEATURES of df, predicted in batches."""
    out = np.empty(df.height, dtype=np.float32)
    for lo in range(0, df.height, batch):
        x = df.slice(lo, batch).select(RANK_FEATURES).to_numpy()
        out[lo:lo + batch] = booster.predict(x, num_threads=0)
    return out


def rank_split(cfg: Config, split: str, log=print) -> None:
    """Score the union pairs of a split and keep the top rk_keep per S1 in <split>/cand_b/part_*.parquet.

    Train dev folds are scored out-of-fold; the sealed fold and test get the mean of both models.
    """
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
                     "sk_n", "sk_idf", "fz_cos")
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
