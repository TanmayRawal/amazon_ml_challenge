"""v4: stacked second-stage matcher (plan D-lite).

Stage 1 (matcher B) scores every candidate pair; its out-of-fold probability p1 feeds a
second LightGBM together with features that need p1 of *other* pairs:

  target competition   best p1 among other S1s claiming the same target, gap to it,
                       their summed p1 and number of confident claimants. (Stage 1's
                       competition features used the ranker score, which cannot tell
                       same-name entities apart; 38% of stage-1 misses on the sealed fold
                       were targets out-claimed that way.)
  list context         this S1's best competing p1, summed p1, confident count, rank
  sibling similarity   name/address similarity to the S1's most confident other
                       candidate, its p1, and exact-duplicate flag (duplicates always
                       share an owner in train) - helps blank-address and trade-name copies

Stage 6 adds features aimed at decoys (95% of v5's false matches point at ownerless
records). Decoys copy an S1's name, swap its main legal form (Pvt <-> Public, LLC <-> Inc)
and shift its house number a little; true copies keep the number, and their records agree
with each other. So:

  legal relation       swap / added / extra / lost / dropped legal forms ("group", "co" ignored)
  number relation      numbers found on one side only, log numeric offset between them
  consensus            p1-weighted support for this target's first number, core name and
                       legal form among the S1's other candidates; whether its number is
                       the list's p1-weighted mode; share of list mass on the S1's own number

Stage 2 is cross-fitted exactly like stage 1 (group A = folds {0,1}, B = {2,3}; the
sealed fold and test use the mean of both) and calibrated on its out-of-fold scores.
"""
from __future__ import annotations

import itertools
import json
import time

import lightgbm as lgb
import numpy as np
import polars as pl
from rapidfuzz import fuzz, process

from config import Config
from features import feature_parts
from matcher_b import GROUPS, apply_policy, calibrated, fit_isotonic, split_scores
from prepare import fresh, load
from report import matching_report, strata_report
from score import per_entity
from splits import own_holdout

RAW_KEEP = [
    "name_tset", "name_ratio", "name_pratio", "addr_tset", "addr_jac", "num_match", "num_conflict",
    "num_near", "num_suffix", "t_blank", "s_blank", "t_indic", "t_domain", "log_name_rep",
    "log_name_tdf", "rk_p", "t_n_s1", "t_best_gap", "legal_conflict", "concat_jw", "acro", "src3",
    "sk_tset", "sk_jw",
    "np_n", "sim_sk_jac", "num_trunc", "dom_cover",   # V12
    "ap_n", "fz_n", "num_shift",                      # V13
    "sfx_conflict", "st_tset", "st_cont", "addr_wjac", "nm_shared_idf",   # V15
    "hop", "hop_p", "hop_n", "hop_bs",                # V19
    "num_delta", "nm_edit", "leg_pvt_only", "leg_ltd_only",   # V20 decoy signals
]
STACKED = [
    "p1", "logit_p1", "c_other_max", "c_gap", "c_sum_others", "c_n_conf_others",
    "s_best_other", "s_p_sum", "s_n_conf", "s_rank", "p1_rel", "sib_p", "sib_name", "sib_addr", "sib_dup",
]
STAGE6 = [
    "leg_swap", "leg_added", "leg_extra", "leg_lost", "leg_drop", "num_s_only", "num_t_only", "num_off",
    "sup_num_p", "sup_num_share", "t_num_mode", "s_num_share", "sup_core_p", "sup_leg_p",
]
# Stage 6e (v9): which name words changed and how common they are. Typo noise creates one-off
# tokens ("advisors" -> "advieors", DF 1); decoys swap in a word used elsewhere ("peridiq" ->
# "peridex", DF 219). Among v8 pairs scored 0.8-0.95, one-word substitutions were true 97% of
# the time with a rare new token but 66% with a mid-frequency one.
VOCAB = ["nw_n_new", "nw_n_gone", "nw_new_df_min", "nw_new_df_max", "nw_gone_df_min"]
# Stage 6f (v10): source completeness. 85% of S1s with matches own records in both S2 and S3.
# When the S1 already has a confident match in the other source and none in this one, the top
# candidate of this source was true far more often than stage 2 said (p .5-.8: 0.80 vs 0.63).
SOURCE = ["src_mass_same", "src_mass_other", "src_n_conf_same", "src_n_conf_other", "src_rank", "src_gap"]
# V13: tie-breaking between S1s that share a name. For blank-address records claimed by 2+ S1s, the
# owner had the highest lowercase raw-name similarity (legal words, punctuation kept) as often as it had
# the highest p (0.49 vs chance 0.35), and the two combine (confident picks 5.6k -> 7.1k, 80% -> 87% true).
# d_* = this pair's value minus the best other claimant's (p > 0.05 pairs of all S1s).
RAWNAME = ["raw_sim", "raw_exact", "d_raw_sim", "d_raw_exact", "d_leg_eq"]
# V13b: agreement with the S1's own confident candidates. Records of one entity in one source share
# that source's view of the address ("HN 570", "58rd St", "##904") even where the S1 differs, while
# decoys perturb the S1 instead. Honest sealed re-scoring test on V12 scores: +0.0006 F0.5.
SIBLING = ["n_sib_same", "n_sib_other", "sa_same", "sa_other", "sn_same", "sn_other", "num_view", "num_new_min",
           "tok_view", "tok_new_min", "addr_eq_same"]
FEATURES_D = STACKED + RAW_KEEP + STAGE6 + VOCAB + SOURCE + RAWNAME + SIBLING
LEGAL_IGNORE = ["", "group", "co"]   # generic words that noise adds or drops freely
PARAMS = {
    "objective": "binary", "learning_rate": 0.1, "num_leaves": 63, "min_data_in_leaf": 500,
    "feature_fraction": 0.8, "bagging_fraction": 0.8, "bagging_freq": 1, "lambda_l2": 1.0,
    "verbosity": -1, "seed": 2027, "deterministic": True, "force_row_wise": True,
}
NEG_RATE = 0.2


# Level 2 stacks on stage-1 scores (v4-v6). Level 3 repeats the same model on level-2
# scores, so list and consensus features are weighted by cleaner probabilities.
LEVELS = {
    2: {"feat": "feat_d", "model": "matcher_c", "out": "scores_c"},
    3: {"feat": "feat_e", "model": "matcher_e", "out": "scores_e"},
}


def _dir(cfg: Config, level: int = 2):
    """Model directory of a stacked level (created if missing)."""
    path = cfg.work_dir / "models" / LEVELS[level]["model"]
    path.mkdir(parents=True, exist_ok=True)
    return path


def _parts(cfg: Config, split: str, level: int = 2) -> list:
    """Sorted feature part files of a stacked level for a split."""
    return sorted((cfg.split_dir(split) / LEVELS[level]["feat"]).glob("part_*.parquet"))


def level_scores(cfg: Config, split: str, level: int, log=print) -> pl.DataFrame:
    """Calibrated scores of a stacked level. Train scores are out-of-fold and written by
    train_and_tune; other splits are computed once and cached next to them."""
    path = cfg.split_dir(split) / f"{LEVELS[level]['out']}.parquet"
    if path.exists() and (split == "train" or fresh(path, _dir(cfg, level) / "meta.json", *_parts(cfg, split, level))):
        return pl.read_parquet(path)
    if split == "train":
        raise FileNotFoundError(f"{path} is written by train-c --level {level}")
    meta = json.loads((_dir(cfg, level) / "meta.json").read_text())
    scores = calibrated(raw_scores(cfg, split, level), np.array(meta["isotonic"]["x"]), np.array(meta["isotonic"]["y"]))
    scores.write_parquet(path)
    log(f"[stack L{level}] {split}: scored {scores.height:,} pairs -> {path}")
    return scores


def _input_scores(cfg: Config, split: str, level: int, log=print) -> pl.DataFrame:
    """Scores a level stacks on: stage-1 (matcher B) scores for level 2, the previous level's scores above."""
    return split_scores(cfg, split, log) if level == 2 else level_scores(cfg, split, level - 1, log)


# ----------------------------------------------------------------------------- features

def _target_context(scores: pl.DataFrame, n_targets: int) -> dict:
    """Dense per-target arrays (indexed by t): best and second-best p1 over all claiming
    S1s, their sum and the number of confident claims. Unclaimed slots stay 0."""
    ctx = (
        scores.select("t", "p").sort(["t", "p"], descending=[False, True])
        .group_by("t", maintain_order=True)
        .agg(c_p1=pl.col("p").first(), c_p2=pl.col("p").slice(1, 1).first(),
             c_sum=pl.col("p").sum(), c_nconf=(pl.col("p") > 0.5).sum().cast(pl.Float64))
        .with_columns(pl.col("c_p2").fill_null(0.0))
    )
    idx = ctx["t"].to_numpy()
    out = {}
    for col in ("c_p1", "c_p2", "c_sum", "c_nconf"):
        arr = np.zeros(n_targets, dtype=np.float64)
        arr[idx] = ctx[col].to_numpy()
        out[col] = arr
    return out


def raw_name_context(scores: pl.DataFrame, s1_names: pl.DataFrame, t_names: pl.DataFrame) -> pl.DataFrame:
    """V13 RAWNAME features for every pair with p > 0.05 (others get defaults when joined).
    scores: s1, t, p. s1_names: s1, name, legal. t_names: t, name, legal."""
    x = (scores.filter(pl.col("p") > 0.05).select("s1", "t")
         .join(s1_names.rename({"name": "s_name", "legal": "s_leg"}), on="s1")
         .join(t_names.rename({"name": "t_name", "legal": "t_leg"}), on="t"))
    a = x["s_name"].str.to_lowercase().str.strip_chars()
    b = x["t_name"].str.to_lowercase().str.strip_chars()
    x = x.with_columns(
        raw_sim=pl.Series(process.cpdist(a.to_list(), b.to_list(), scorer=fuzz.ratio, workers=-1, dtype=np.float32) / 100.0),
        raw_exact=(a == b).cast(pl.Float32),
        leg_eq=(pl.col("s_leg") == pl.col("t_leg")).cast(pl.Float32),
    )
    rel = []
    for c in ("raw_sim", "raw_exact", "leg_eq"):
        top2 = pl.col(c).top_k(2).over("t", mapping_strategy="join")
        other = pl.when(pl.len().over("t") < 2).then(0.0).when(pl.col(c) >= pl.col(c).max().over("t")) \
            .then(top2.list.get(1, null_on_oob=True)).otherwise(pl.col(c).max().over("t"))
        rel.append((pl.col(c) - other.fill_null(0.0)).cast(pl.Float32).alias(f"d_{c}"))
    return x.with_columns(rel).select("s1", "t", "raw_sim", "raw_exact", "d_raw_sim", "d_raw_exact", "d_leg_eq")


def _ratio(a: list, b: list, empty: np.ndarray) -> np.ndarray:
    """rapidfuzz token_set_ratio of paired strings, scaled to [0, 1]; positions flagged in empty get 0."""
    out = process.cpdist(a, b, scorer=fuzz.token_set_ratio, workers=-1, dtype=np.float32) / 100.0
    out[empty] = 0.0
    return out


def _first_number(e: pl.Expr) -> pl.Expr:
    """Expression: first integer found in the first element of a list of strings (null if none)."""
    return e.list.first().str.extract(r"(\d+)").cast(pl.Float64, strict=False)


def decoy_features(x: pl.DataFrame) -> pl.DataFrame:
    """Stage 6 legal, number and consensus features.

    x: complete S1 candidate lists with s1, p (stage-1), s_p_sum, s_leg/t_leg (normalized legal
    forms), s_nums/t_nums (space-separated house numbers) and t_core.
    """
    ign = pl.lit(LEGAL_IGNORE)
    x = x.with_columns(
        _sl=pl.col("s_leg").str.split(" ").list.set_difference(ign),
        _tl=pl.col("t_leg").str.split(" ").list.set_difference(ign),
        _sn=pl.col("s_nums").str.split(" ").list.set_difference(pl.lit([""])),
        _tn=pl.col("t_nums").str.split(" ").list.set_difference(pl.lit([""])),
        t_n0=pl.col("t_nums").str.split(" ").list.first().fill_null(""),
        s_n0=pl.col("s_nums").str.split(" ").list.first().fill_null(""),
    ).with_columns(
        _s_x=pl.col("_sl").list.set_difference("_tl").list.len(),
        _t_x=pl.col("_tl").list.set_difference("_sl").list.len(),
        _s_only=pl.col("_sn").list.set_difference("_tn"),
        _t_only=pl.col("_tn").list.set_difference("_sn"),
    )
    s_has, t_has = pl.col("_sl").list.len() > 0, pl.col("_tl").list.len() > 0
    off = (_first_number(pl.col("_s_only")) - _first_number(pl.col("_t_only"))).abs()
    has_n0 = pl.col("t_n0") != ""
    mass_n0 = pl.col("p").sum().over("s1", "t_n0")
    x = x.with_columns(
        leg_swap=(pl.col("_s_x") > 0) & (pl.col("_t_x") > 0),
        leg_added=~s_has & t_has,
        leg_extra=s_has & (pl.col("_s_x") == 0) & (pl.col("_t_x") > 0),
        leg_lost=s_has & ~t_has,
        leg_drop=t_has & (pl.col("_s_x") > 0) & (pl.col("_t_x") == 0),
        num_s_only=pl.col("_s_only").list.len(),
        num_t_only=pl.col("_t_only").list.len(),
        num_off=pl.when(off.is_null()).then(-1.0).otherwise(off.log1p()),
        sup_num_p=pl.when(has_n0).then(mass_n0 - pl.col("p")).otherwise(0.0),
        sup_core_p=pl.when(pl.col("t_core") != "").then(pl.col("p").sum().over("s1", "t_core") - pl.col("p"))
        .otherwise(0.0),
        sup_leg_p=pl.col("p").sum().over("s1", "t_leg") - pl.col("p"),
        _mass=pl.when(has_n0).then(mass_n0),
        _s_agree=pl.when((pl.col("s_n0") != "") & (pl.col("t_n0") == pl.col("s_n0"))).then(pl.col("p")).otherwise(0.0),
    )
    return x.with_columns(
        sup_num_share=pl.col("sup_num_p") / (pl.col("s_p_sum") - pl.col("p")).clip(1e-9, None),
        t_num_mode=(pl.col("_mass") >= pl.col("_mass").max().over("s1")).fill_null(False),
        s_num_share=pl.col("_s_agree").sum().over("s1") / pl.col("s_p_sum").clip(1e-9, None),
    ).drop("_sl", "_tl", "_sn", "_tn", "_s_x", "_t_x", "_s_only", "_t_only", "_mass", "_s_agree")


def source_features(x: pl.DataFrame) -> pl.DataFrame:
    """x: s1, p, t_src. Per-source list context: confident mass and count the S1 already has in
    the target's source (excluding the target) and in the other source, and the target's rank
    and gap to the best candidate within its source."""
    conf = (pl.col("p") > 0.5).cast(pl.Float32)
    same_mass = pl.col("p").sum().over("s1", "t_src")
    same_conf = conf.sum().over("s1", "t_src")
    return x.with_columns(
        src_mass_same=same_mass - pl.col("p"),
        src_mass_other=pl.col("p").sum().over("s1") - same_mass,
        src_n_conf_same=same_conf - conf,
        src_n_conf_other=conf.sum().over("s1") - same_conf,
        src_rank=pl.col("p").rank("ordinal", descending=True).over("s1", "t_src").cast(pl.Float32),
        src_gap=pl.col("p") - pl.col("p").max().over("s1", "t_src"),
    )


def sibling_features(x: pl.DataFrame, tg: pl.DataFrame, s1t: pl.DataFrame) -> pl.DataFrame:
    """x: complete S1 lists with s1, t, p, t_src. tg/s1t: positional (row = id) with addr_low,
    name_low, addr_toks, addr_nums (tg) and addr_toks, addr_nums (s1t). For uncertain pairs
    (0.01 < p < 0.99), compares the target with each confident (p > 0.9) other candidate of the S1:
    best address / name similarity per same and other source, and the most target numbers/words
    missing from the S1 that a sibling has ("view") vs the fewest that no sibling has ("new").
    Everything else gets -1."""
    band = x.filter((pl.col("p") > 0.01) & (pl.col("p") < 0.99)).select("s1", "t", "t_src")
    conf = x.filter(pl.col("p") > 0.9).select("s1", t2="t", src2="t_src")
    pr = band.join(conf, on="s1").filter(pl.col("t2") != pl.col("t"))
    if pr.height == 0:
        return x.with_columns([pl.lit(-1.0, pl.Float32).alias(c) for c in SIBLING])
    ti, ti2, si = pr["t"].to_numpy(), pr["t2"].to_numpy(), pr["s1"].to_numpy()
    a, a2 = tg["addr_low"].gather(ti), tg["addr_low"].gather(ti2)
    blank = ((a.str.strip_chars() == "") | (a2.str.strip_chars() == "")).to_numpy()
    asim = _ratio(a.to_list(), a2.to_list(), blank)
    asim[blank] = -1.0
    nsim = _ratio(tg["name_low"].gather(ti).to_list(), tg["name_low"].gather(ti2).to_list(), np.zeros(len(ti), bool))
    sp = lambda col, frame, idx: frame[col].gather(idx).str.split(" ")  # noqa: E731
    e = pl.lit([""])
    t_only_n = pl.col("nums").list.set_difference("s_nums").list.set_difference(e)
    t_only_w = pl.col("toks").list.set_difference("s_toks").list.set_difference(e)
    pr = pr.with_columns(
        asim=pl.Series(asim), nsim=pl.Series(nsim), same=pl.col("t_src") == pl.col("src2"),
        aeq=pl.Series((a == a2).to_numpy() & ~blank),
        nums=sp("addr_nums", tg, ti), nums2=sp("addr_nums", tg, ti2), s_nums=sp("addr_nums", s1t, si),
        toks=sp("addr_toks", tg, ti), toks2=sp("addr_toks", tg, ti2), s_toks=sp("addr_toks", s1t, si),
    ).with_columns(
        num_view=t_only_n.list.set_intersection("nums2").list.len(),
        num_new=t_only_n.list.set_difference("nums2").list.len(),
        tok_view=t_only_w.list.set_intersection("toks2").list.len(),
        tok_new=t_only_w.list.set_difference("toks2").list.len(),
    )
    same, oth = pl.col("same"), ~pl.col("same")
    agg = pr.group_by("s1", "t").agg(
        n_sib_same=same.sum(), n_sib_other=oth.sum(),
        sa_same=pl.col("asim").filter(same).max(), sa_other=pl.col("asim").filter(oth).max(),
        sn_same=pl.col("nsim").filter(same).max(), sn_other=pl.col("nsim").filter(oth).max(),
        num_view=pl.col("num_view").max(), num_new_min=pl.col("num_new").min(),
        tok_view=pl.col("tok_view").max(), tok_new_min=pl.col("tok_new").min(),
        addr_eq_same=(pl.col("aeq") & same).any())
    return x.join(agg, on=["s1", "t"], how="left", maintain_order="left").with_columns(
        [pl.col(c).cast(pl.Float32).fill_null(-1.0) for c in SIBLING])


def token_df(tg: pl.DataFrame) -> pl.DataFrame:
    """Per country: in how many target records each core-name token appears (label-free)."""
    return (tg.select("t", "country", tok=pl.col("core").str.split(" ")).explode("tok")
            .filter(pl.col("tok").is_not_null() & (pl.col("tok") != "")).unique(["t", "tok"])
            .group_by("country", "tok").agg(df=pl.len().cast(pl.Float32)))


def vocab_features(x: pl.DataFrame, tdf: pl.DataFrame) -> pl.DataFrame:
    """x: s_core, t_core, country (one row per pair). Adds VOCAB: counts of target words missing
    from the S1 name ("new") and S1 words missing from the target ("gone"), and log1p of the
    rarest/commonest new word's DF and of the rarest gone word's DF (-1 when there is none)."""
    a = pl.col("s_core").str.split(" ").list.set_difference(pl.lit([""]))
    b = pl.col("t_core").str.split(" ").list.set_difference(pl.lit([""]))
    x = x.with_row_index("_r").with_columns(_new=b.list.set_difference(a), _gone=a.list.set_difference(b))

    def stats(col: str, aggs: dict) -> pl.DataFrame:
        """Per row of x: log1p document frequencies of the tokens in list column col, reduced with aggs."""
        return (x.select("_r", "country", tok=col).explode("tok").drop_nulls("tok")
                .join(tdf, on=["country", "tok"], how="left").with_columns(pl.col("df").fill_null(0.0).log1p())
                .group_by("_r").agg(**aggs))

    new = stats("_new", {"nw_new_df_min": pl.col("df").min(), "nw_new_df_max": pl.col("df").max()})
    gone = stats("_gone", {"nw_gone_df_min": pl.col("df").min()})
    return (x.join(new, on="_r", how="left").join(gone, on="_r", how="left")
            .with_columns(pl.col("nw_new_df_min", "nw_new_df_max", "nw_gone_df_min").fill_null(-1.0),
                          nw_n_new=pl.col("_new").list.len(), nw_n_gone=pl.col("_gone").list.len())
            .sort("_r").drop("_r", "_new", "_gone"))


def _stack_part(df: pl.DataFrame, tctx: dict, tg: pl.DataFrame, s1t: pl.DataFrame,
                tdf: pl.DataFrame | None = None) -> pl.DataFrame:
    """df: one part's rows (complete S1 lists) with p and RAW_KEEP (+label/fold)."""
    t_rows = df["t"].to_numpy()
    df = df.with_columns([pl.Series(col, arr[t_rows]) for col, arr in tctx.items()])
    ranked = (
        df.sort(["s1", "p", "t"], descending=[False, True, False])
        .with_columns(r=pl.int_range(pl.len()).over("s1"))
    )
    top2 = ranked.filter(pl.col("r") < 2).group_by("s1").agg(
        t_1=pl.col("t").first(), t_2=pl.col("t").slice(1, 1).first(),
        p_1=pl.col("p").first(), p_2=pl.col("p").slice(1, 1).first(),
    )
    x = (
        ranked.join(top2, on="s1")
        .with_columns(
            sib_t=pl.when(pl.col("r") == 0).then(pl.col("t_2")).otherwise(pl.col("t_1")),
            sib_p=pl.when(pl.col("r") == 0).then(pl.col("p_2")).otherwise(pl.col("p_1")).fill_null(0.0),
            s_best_other=pl.when(pl.col("r") == 0).then(pl.col("p_2")).otherwise(pl.col("p_1")).fill_null(0.0),
            c_other_max=pl.when(pl.col("p") >= pl.col("c_p1")).then(pl.col("c_p2")).otherwise(pl.col("c_p1")),
        )
        .with_columns(
            p1=pl.col("p"),
            logit_p1=(pl.col("p").clip(1e-6, 1 - 1e-6) / (1 - pl.col("p").clip(1e-6, 1 - 1e-6))).log(),
            c_gap=pl.col("p") - pl.col("c_other_max"),
            c_sum_others=pl.col("c_sum") - pl.col("p"),
            c_n_conf_others=pl.col("c_nconf").cast(pl.Float64) - (pl.col("p") > 0.5).cast(pl.Float64),
            s_p_sum=pl.col("p").sum().over("s1"),
            s_n_conf=(pl.col("p") > 0.5).sum().over("s1").cast(pl.Float64),
            s_rank=(pl.col("r") + 1).cast(pl.Float64),
            p1_rel=pl.col("p") / pl.col("p_1").clip(1e-9, None),
        )
    )
    has_sib = x["sib_t"].is_not_null().to_numpy()
    sib_idx = x["sib_t"].fill_null(0).to_numpy()
    t_idx = x["t"].to_numpy()
    core, addr, key = tg["core"], tg["addr_toks"], tg["dupkey"]
    a_core, b_core = core.gather(t_idx), core.gather(sib_idx)
    a_addr, b_addr = addr.gather(t_idx), addr.gather(sib_idx)
    no_name = ~has_sib | (a_core == "").to_numpy() | (b_core == "").to_numpy()
    no_addr = ~has_sib | (a_addr == "").to_numpy() | (b_addr == "").to_numpy()
    dup = (key.gather(t_idx) == key.gather(sib_idx)).to_numpy() & has_sib
    s_idx = x["s1"].to_numpy()
    x = x.with_columns(
        sib_name=pl.Series(_ratio(a_core.to_list(), b_core.to_list(), no_name)),
        sib_addr=pl.Series(_ratio(a_addr.to_list(), b_addr.to_list(), no_addr)),
        sib_dup=pl.Series(dup.astype(np.float32)),
        s_leg=s1t["legal"].gather(s_idx), s_nums=s1t["addr_nums"].gather(s_idx),
        t_leg=tg["legal"].gather(t_idx), t_nums=tg["addr_nums"].gather(t_idx), t_core=a_core,
        s_core=s1t["core"].gather(s_idx), country=s1t["country"].gather(s_idx),
        t_src=tg["src"].gather(t_idx),
    )
    x = decoy_features(x)
    x = source_features(x)
    x = sibling_features(x, tg, s1t)
    x = vocab_features(x, tdf if tdf is not None else token_df(tg))
    keep = ["s1", "t"] + [c for c in ("label", "fold") if c in x.columns]
    return x.select(keep + [pl.col(f).cast(pl.Float32) for f in FEATURES_D])


def build_stack_features(cfg: Config, split: str, log=print, level: int = 2) -> None:
    """Build stacked features for a split and write them as feat_d (level 2) or feat_e (level 3) parts.

    Joins the input level's scores onto each ranker chunk's feature part and adds target context, raw-name context
    and the list / decoy / source / sibling / vocabulary features (see _stack_part)."""
    t0 = time.time()
    scores = _input_scores(cfg, split, level, log).select("s1", "t", "p")
    tg = (
        load(cfg, split, "tgt", ["t", "country", "src", "core", "core_key", "addr_toks", "legal", "addr_nums",
                                 "business_name", "business_address"]).sort("t")
        .with_columns(dupkey=pl.concat_str(["core_key", "addr_toks"], separator="|"),
                      name_low=pl.col("business_name").fill_null("").str.to_lowercase(),
                      addr_low=pl.col("business_address").fill_null("").str.to_lowercase())
    )
    s1t = load(cfg, split, "s1", ["s1", "country", "core", "legal", "addr_nums", "addr_toks"]).sort("s1")
    if not (s1t["s1"].to_numpy() == np.arange(s1t.height)).all():
        raise RuntimeError("S1 ids must be 0..n-1 for positional lookups")
    tdf = token_df(tg)
    tctx = _target_context(scores, tg.height)
    rawctx = raw_name_context(
        scores,
        load(cfg, split, "s1", ["s1", "business_name", "legal"]).rename({"business_name": "name"}),
        load(cfg, split, "tgt", ["t", "business_name", "legal"]).rename({"business_name": "name"}))
    log(f"[stack L{level}] {split} raw-name context: {rawctx.height:,} pairs ({time.time() - t0:.0f}s)")
    # Feature parts are the ranker's S1 chunks, in order: slice stage-1 scores the same way
    # instead of joining every part against the full table.
    by_chunk = scores.with_columns(chunk=(pl.col("s1") // cfg.rk_chunk).cast(pl.Int64)).partition_by(
        "chunk", as_dict=True, include_key=False)
    del scores
    out_dir = cfg.split_dir(split) / LEVELS[level]["feat"]
    out_dir.mkdir(exist_ok=True)
    for old in out_dir.glob("part_*.parquet"):
        old.unlink()
    for i, part in enumerate(feature_parts(cfg, split, "c")):
        cols = ["s1", "t"] + RAW_KEEP + (["label", "fold"] if split == "train" else [])
        df = pl.read_parquet(part, columns=cols)
        lo, hi = df["s1"].min(), df["s1"].max()
        if lo // cfg.rk_chunk != i or hi // cfg.rk_chunk != i:
            raise RuntimeError(f"{part.name}: S1 range {lo}-{hi} is not ranker chunk {i}")
        df = df.join(by_chunk[(i,)], on=["s1", "t"], how="left")
        if df["p"].null_count():
            raise RuntimeError(f"{part.name}: pairs without a stage-1 score")
        df = df.join(rawctx, on=["s1", "t"], how="left").with_columns(
            pl.col("raw_sim", "raw_exact").fill_null(0.0), pl.col("d_raw_sim", "d_raw_exact", "d_leg_eq").fill_null(-1.0))
        _stack_part(df, tctx, tg, s1t, tdf).write_parquet(out_dir / f"part_{i:03d}.parquet")
        if i % 10 == 0:
            log(f"[stack L{level}] {split} part {i}: {df.height:,} rows ({time.time() - t0:.0f}s)")
    log(f"[stack L{level}] {split} features done ({time.time() - t0:.0f}s)")


# ------------------------------------------------------------------------ train & tune

def _weights(df: pl.DataFrame) -> np.ndarray:
    """Sample weights undoing negative subsampling: 1 for positives, 1 / NEG_RATE for negatives."""
    return np.where(df["label"].to_numpy() == 1, 1.0, 1.0 / NEG_RATE)


def train_and_tune(cfg: Config, log=print, level: int = 2) -> dict:
    """Train the stacked matcher at a level and choose its decision policy.

    Fits two cross-fitted LightGBM models on subsampled dev-fold features, writes calibrated out-of-fold train
    scores, tunes the set-selection policy (threshold or expected-F0.5) on dev folds, reports dev and sealed-fold
    F0.5 and writes meta.json. Returns the meta dict."""
    t0 = time.time()
    tag = f"[stack L{level}]"
    # New models make any cached test scores of this level (and the levels built on it) stale.
    for lv, spec in LEVELS.items():
        if lv >= level:
            (cfg.split_dir("test") / f"{spec['out']}.parquet").unlink(missing_ok=True)
    keep_neg = (pl.struct("s1", "t").hash(seed=cfg.seed + 2) % 1_000_000) < int(NEG_RATE * 1_000_000)
    dev = pl.col("fold") != cfg.sealed_fold
    data = pl.concat([pl.read_parquet(p).filter(dev & ((pl.col("label") == 1) | keep_neg))
                      for p in _parts(cfg, "train", level)])
    log(f"{tag} sample {data.height:,} rows ({time.time() - t0:.0f}s)")
    meta = {"level": level, "features": FEATURES_D, "params": PARAMS, "neg_rate": NEG_RATE, "groups": GROUPS}
    for name, folds in GROUPS.items():
        own = data.filter(pl.col("fold").is_in(folds))
        hold = own.select(own_holdout(cfg.seed + 2)).to_series()
        tr, va = own.filter(~hold), own.filter(hold)
        dtr = lgb.Dataset(tr.select(FEATURES_D).to_numpy(), tr["label"].to_numpy(), weight=_weights(tr),
                          feature_name=FEATURES_D)
        dva = lgb.Dataset(va.select(FEATURES_D).to_numpy(), va["label"].to_numpy(), weight=_weights(va), reference=dtr)
        booster = lgb.train(PARAMS, dtr, num_boost_round=1000, valid_sets=[dva],
                            callbacks=[lgb.early_stopping(30, verbose=False)])
        booster.save_model(str(_dir(cfg, level) / f"stack_{name}.txt"), num_iteration=booster.best_iteration)
        gain = sorted(zip(FEATURES_D, booster.feature_importance("gain")), key=lambda kv: -kv[1])
        meta[f"best_iter_{name}"] = booster.best_iteration
        meta[f"valid_logloss_{name}"] = booster.best_score["valid_0"]["binary_logloss"]
        log(f"{tag} model {name}: best_iter={booster.best_iteration} logloss={meta[f'valid_logloss_{name}']:.4f} "
            f"({time.time() - t0:.0f}s); top gain: " + ", ".join(k for k, _ in gain[:10]))

    raw = raw_scores(cfg, "train", level)
    iso = fit_isotonic(raw, cfg.sealed_fold, cfg.seed)
    scores = calibrated(raw, iso.X_thresholds_, iso.y_thresholds_)
    scores.write_parquet(cfg.split_dir("train") / f"{LEVELS[level]['out']}.parquet")
    meta["isotonic"] = {"x": iso.X_thresholds_.tolist(), "y": iso.y_thresholds_.tolist()}
    log(f"{tag} scored + calibrated {scores.height:,} pairs ({time.time() - t0:.0f}s)")

    gt = load(cfg, "train", "gt")
    s1 = load(cfg, "train", "s1", ["s1", "country"]).join(load(cfg, "train", "folds", ["s1", "fold"]), on="s1")
    dev_ids = s1.filter(pl.col("fold") != cfg.sealed_fold)
    sealed_ids = s1.filter(pl.col("fold") == cfg.sealed_fold)
    covered = gt.join(scores.select("s1", "t"), on=["s1", "t"], how="semi").join(dev_ids, on="s1", how="semi").height
    missed = (gt.join(dev_ids, on="s1", how="semi").height - covered) / dev_ids.height

    pool = scores.filter(pl.col("p") > 1e-4)
    results = []
    for tau, rho in itertools.product([0.4, 0.5, 0.6], [0.0]):
        pol = {"name": "eum", "tau": tau, "rho": rho}
        results.append((float(per_entity(dev_ids.select("s1"), apply_policy(pool, pol), gt)["f"].mean()), pol))
    for name, m in itertools.product(["dta", "dta_own"], [0.0, missed]):
        pol = {"name": name, "missed": round(m, 4)}
        results.append((float(per_entity(dev_ids.select("s1"), apply_policy(pool, pol), gt)["f"].mean()), pol))
    for f, pol in sorted(results, key=lambda r: -r[0])[:6]:
        log(f"{tag} dev F0.5={f:.4f}  {pol}")
    best_f, best = max(results, key=lambda r: r[0])
    log(f"{tag} chosen policy {best} (dev {best_f:.4f})")
    # A density-matched train split (prepare.match_density) already has test-like decoy density.
    shift = 1.0 if (cfg.split_dir("train") / "density.json").exists() else cfg.decision_shift
    if shift != 1.0:
        # The shift is a robustness hedge chosen on distractor injection (dev entities), not on
        # this grid: report what it costs on the plain sealed fold.
        plain = matching_report(sealed_ids, apply_policy(pool, best), gt, "stack sealed fold, no shift", log)
        meta["sealed_no_shift"] = plain
        best = {**best, "shift": shift}
    meta["policy"] = best

    pred = apply_policy(pool, best)
    meta["dev"] = matching_report(dev_ids, pred, gt, "stack dev", log)
    meta["sealed"] = matching_report(sealed_ids, pred, gt, "stack sealed fold", log)
    meta["strata_sealed"] = strata_report(cfg, pred, cfg.sealed_fold, log)
    (_dir(cfg, level) / "meta.json").write_text(json.dumps(meta, indent=2, default=str))
    log(f"{tag} done in {time.time() - t0:.0f}s")
    return meta


def raw_scores(cfg: Config, split: str, level: int = 2) -> pl.DataFrame:
    """Uncalibrated probabilities of both stacked models (pa, pb) for every pair in a split's feature parts."""
    boosters = {n: lgb.Booster(model_file=str(_dir(cfg, level) / f"stack_{n}.txt")) for n in GROUPS}
    out = []
    for part in _parts(cfg, split, level):
        df = pl.read_parquet(part)
        x = df.select(FEATURES_D).to_numpy()
        keep = [c for c in ("s1", "t", "label", "fold") if c in df.columns]
        out.append(df.select(keep).with_columns(
            pa=pl.Series(boosters["A"].predict(x, num_threads=0).astype(np.float32)),
            pb=pl.Series(boosters["B"].predict(x, num_threads=0).astype(np.float32)),
        ))
    return pl.concat(out)


def predict_split(cfg: Config, split: str, log=print, level: int = 2) -> tuple[pl.DataFrame, pl.DataFrame, dict]:
    """Apply a stacked level's tuned policy to a split.

    Returns: (predicted s1, t matches; all candidate pairs s1, t; model info with policy and sealed metrics)."""
    meta = json.loads((_dir(cfg, level) / "meta.json").read_text())
    scores = level_scores(cfg, split, level, log)
    pred = apply_policy(scores.filter(pl.col("p") > 1e-4), meta["policy"])
    log(f"[stack L{level}] {split}: {scores.height:,} candidate pairs -> {pred.height:,} predicted matches")
    return pred, scores.select("s1", "t"), {"policy": meta["policy"], "sealed": meta.get("sealed", {})}
