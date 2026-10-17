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
from features import VOCAB, feature_parts, pair_decoy, token_df, vocab_features  # noqa: F401 (re-exported)
from matcher_b import GROUPS, apply_policy, calibrated, fit_isotonic, split_scores
from prepare import load
from report import matching_report, strata_report
from score import per_entity

RAW_KEEP = [
    "name_tset", "name_ratio", "name_pratio", "addr_tset", "addr_jac", "num_match", "num_conflict",
    "num_near", "num_suffix", "t_blank", "s_blank", "t_indic", "t_domain", "log_name_rep",
    "log_name_tdf", "rk_p", "t_n_s1", "t_best_gap", "legal_conflict", "concat_jw", "acro", "src3",
    "sk_tset", "sk_jw", "fz_cos",
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
# the time with a rare new token but 66% with a mid-frequency one. (VOCAB and vocab_features live
# in features.py; from v11 stage 1 uses them too.)
# Stage 6f (v10): source completeness. 85% of S1s with matches own records in both S2 and S3.
# When the S1 already has a confident match in the other source and none in this one, the top
# candidate of this source was true far more often than stage 2 said (p .5-.8: 0.80 vs 0.63).
SOURCE = ["src_mass_same", "src_mass_other", "src_n_conf_same", "src_n_conf_other", "src_rank", "src_gap"]
# v13: pair judge (pair_judge.py), a cross-encoder reading both records' "name | address" for pairs
# stage 1 is unsure about: its probability and its gap to the S1's best-judged candidate. NaN
# outside the scored band. On 20k held-out band pairs it was right 96% of the time (stage 1: 89%).
PJ = ["pj", "pj_gap"]
FEATURES_D = STACKED + RAW_KEEP + STAGE6 + VOCAB + SOURCE + PJ
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
    """Model directory of a stacked level (created if needed)."""
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
    if path.exists():
        return pl.read_parquet(path)
    if split == "train":
        raise FileNotFoundError(f"{path} is written by train-c --level {level}")
    meta = json.loads((_dir(cfg, level) / "meta.json").read_text())
    scores = calibrated(raw_scores(cfg, split, level), np.array(meta["isotonic"]["x"]), np.array(meta["isotonic"]["y"]))
    scores.write_parquet(path)
    log(f"[stack L{level}] {split}: scored {scores.height:,} pairs -> {path}")
    return scores


def _input_scores(cfg: Config, split: str, level: int, log=print) -> pl.DataFrame:
    """Scores feeding a level: stage-1 (matcher B) for level 2, the previous level's scores otherwise."""
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


def _ratio(a: list, b: list, empty: np.ndarray) -> np.ndarray:
    """Element-wise token-set ratio (0-1) of two string lists; 0 where `empty` is set."""
    out = process.cpdist(a, b, scorer=fuzz.token_set_ratio, workers=-1, dtype=np.float32) / 100.0
    out[empty] = 0.0
    return out


def decoy_features(x: pl.DataFrame) -> pl.DataFrame:
    """Stage 6 legal, number (features.pair_decoy) and consensus features.

    x: complete S1 candidate lists with s1, p (stage-1), s_p_sum, s_leg/t_leg (normalized legal
    forms), s_nums/t_nums (space-separated house numbers) and t_core.
    """
    x = pair_decoy(x).with_columns(
        t_n0=pl.col("t_nums").str.split(" ").list.first().fill_null(""),
        s_n0=pl.col("s_nums").str.split(" ").list.first().fill_null(""),
    )
    has_n0 = pl.col("t_n0") != ""
    mass_n0 = pl.col("p").sum().over("s1", "t_n0")
    x = x.with_columns(
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
    ).drop("_mass", "_s_agree")


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


def _stack_part(df: pl.DataFrame, tctx: dict, tg: pl.DataFrame, s1t: pl.DataFrame,
                tdf: pl.DataFrame | None = None) -> pl.DataFrame:
    """df: one part's rows (complete S1 lists) with p, RAW_KEEP and pj (+label/fold)."""
    if "pj" not in df.columns:               # stress tests build stage-2 features without a pair judge
        df = df.with_columns(pj=pl.lit(None, pl.Float32))
    df = df.with_columns(pj_gap=pl.col("pj") - pl.col("pj").max().over("s1"))
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
    x = vocab_features(x, tdf if tdf is not None else token_df(tg))
    keep = ["s1", "t"] + [c for c in ("label", "fold") if c in x.columns]
    return x.select(keep + [pl.col(f).cast(pl.Float32) for f in FEATURES_D])


def build_stack_features(cfg: Config, split: str, log=print, level: int = 2) -> None:
    """Build the stacked-stage features (FEATURES_D) of a split, one part per ranker chunk.

    Combines the input-level scores, target competition context, pair-judge scores and the
    list/decoy/source/vocabulary features; writes <split>/<LEVELS[level]["feat"]>/part_*.parquet.
    """
    t0 = time.time()
    scores = _input_scores(cfg, split, level, log).select("s1", "t", "p")
    drop = None
    if split == "train" and cfg.stack_drop_noisy_hidden:
        from prepare import noisy_hidden_copies
        drop = noisy_hidden_copies(cfg)
        n0 = scores.height
        scores = scores.join(drop, on="t", how="anti")
        log(f"[stack L{level}] train: dropped {n0 - scores.height:,} pairs whose target is a blank/domain copy of a hidden S1")
    tg = (
        load(cfg, split, "tgt", ["t", "country", "src", "core", "core_key", "addr_toks", "legal", "addr_nums"]).sort("t")
        .with_columns(dupkey=pl.concat_str(["core_key", "addr_toks"], separator="|"))
    )
    s1t = load(cfg, split, "s1", ["s1", "country", "core", "legal", "addr_nums"]).sort("s1")
    if not (s1t["s1"].to_numpy() == np.arange(s1t.height)).all():
        raise RuntimeError("S1 ids must be 0..n-1 for positional lookups")
    tdf = token_df(tg)
    tctx = _target_context(scores, tg.height)
    # Feature parts are the ranker's S1 chunks, in order: slice stage-1 scores the same way
    # instead of joining every part against the full table.
    by_chunk = scores.with_columns(chunk=(pl.col("s1") // cfg.rk_chunk).cast(pl.Int64)).partition_by(
        "chunk", as_dict=True, include_key=False)
    del scores
    from pair_judge import pj_scores
    pj = pj_scores(cfg, split)
    pj_chunk = pj.with_columns(chunk=(pl.col("s1") // cfg.rk_chunk).cast(pl.Int64)).partition_by(
        "chunk", as_dict=True, include_key=False)
    log(f"[stack L{level}] {split}: pair-judge scores for {pj.height:,} pairs")
    no_pj = pj.clear()
    out_dir = cfg.split_dir(split) / LEVELS[level]["feat"]
    out_dir.mkdir(exist_ok=True)
    for old in out_dir.glob("part_*.parquet"):
        old.unlink()
    for i, part in enumerate(feature_parts(cfg, split, "c")):
        cols = ["s1", "t"] + RAW_KEEP + (["label", "fold"] if split == "train" else [])
        df = pl.read_parquet(part, columns=cols)
        if drop is not None:
            df = df.join(drop, on="t", how="anti")
        lo, hi = df["s1"].min(), df["s1"].max()
        if lo // cfg.rk_chunk != i or hi // cfg.rk_chunk != i:
            raise RuntimeError(f"{part.name}: S1 range {lo}-{hi} is not ranker chunk {i}")
        df = df.join(by_chunk[(i,)], on=["s1", "t"], how="left")
        if df["p"].null_count():
            raise RuntimeError(f"{part.name}: pairs without a stage-1 score")
        df = df.join(pj_chunk.get((i,), no_pj), on=["s1", "t"], how="left")
        _stack_part(df, tctx, tg, s1t, tdf).write_parquet(out_dir / f"part_{i:03d}.parquet")
        if i % 10 == 0:
            log(f"[stack L{level}] {split} part {i}: {df.height:,} rows ({time.time() - t0:.0f}s)")
    log(f"[stack L{level}] {split} features done ({time.time() - t0:.0f}s)")


# ------------------------------------------------------------------------ train & tune

def _weights(df: pl.DataFrame) -> np.ndarray:
    """Sample weights undoing negative sampling (1 for positives, 1/NEG_RATE for negatives)."""
    return np.where(df["label"].to_numpy() == 1, 1.0, 1.0 / NEG_RATE)


def train_and_tune(cfg: Config, log=print, level: int = 2) -> dict:
    """Train the cross-fitted stacked LightGBM models, calibrate, pick the decision policy on dev folds.

    Writes the out-of-fold train scores and meta.json (policy incl. decision_shift, dev/sealed
    reports); returns the metadata.
    """
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
        tr = data.filter(pl.col("fold").is_in(folds))
        other = data.filter(~pl.col("fold").is_in(folds))
        va = other.sample(n=min(3_000_000, other.height), seed=2, shuffle=True)
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
    if cfg.decision_shift != 1.0:
        # The shift is a robustness hedge chosen on distractor injection (dev entities), not on
        # this grid: report what it costs on the plain sealed fold.
        plain = matching_report(sealed_ids, apply_policy(pool, best), gt, "stack sealed fold, no shift", log)
        meta["sealed_no_shift"] = plain
        best = {**best, "shift": cfg.decision_shift}
    meta["policy"] = best

    pred = apply_policy(pool, best)
    meta["dev"] = matching_report(dev_ids, pred, gt, "stack dev", log)
    meta["sealed"] = matching_report(sealed_ids, pred, gt, "stack sealed fold", log)
    meta["strata_sealed"] = strata_report(cfg, pred, cfg.sealed_fold, log)
    (_dir(cfg, level) / "meta.json").write_text(json.dumps(meta, indent=2, default=str))
    log(f"{tag} done in {time.time() - t0:.0f}s")
    return meta


def raw_scores(cfg: Config, split: str, level: int = 2) -> pl.DataFrame:
    """Uncalibrated scores of both stacked models (pa, pb) for every pair of a split."""
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


def _one_word_swaps(cfg: Config, split: str, pool: pl.DataFrame, country: str, adds: bool = False) -> pl.DataFrame:
    """(s1, t, new) pairs of `pool` in `country` whose core names differ in exactly one word at the same
    position, with the same legal form, the same first house number and address-token Jaccard >= 0.5.
    With adds=True also pairs whose target name is the S1 name plus exactly one extra word (new = that word)."""
    if adds:
        return pl.concat([_one_word_swaps(cfg, split, pool, country), _one_word_adds(cfg, split, pool, country)]).unique(["s1", "t"])
    d = cfg.split_dir(split)
    s1 = (pl.scan_parquet(d / "s1.parquet").filter(pl.col("country") == country)
          .select("s1", s_core="core", s_legal="legal", s_nums="addr_nums", s_toks="addr_toks").collect())
    x = pool.select("s1", "t").join(s1, on="s1")
    tg = (pl.scan_parquet(d / "tgt.parquet").select("t", "core", "legal", "addr_nums", "addr_toks")
          .join(x.lazy().select("t").unique(), on="t", how="semi").collect())
    words = lambda c: pl.col(c).fill_null("").str.split(" ").list.eval(pl.element().filter(pl.element() != ""))
    x = x.join(tg, on="t").with_columns(_a=words("s_core"), _b=words("core"), _sa=words("s_toks"), _ta=words("addr_toks"),
                                        _sl=words("s_legal"), _tl=words("legal"))
    x = x.filter((pl.col("_a").list.len() == pl.col("_b").list.len()) & (pl.col("_a").list.len() >= 2)
                 & (pl.col("_b").list.set_difference("_a").list.len() == 1)
                 & (pl.col("_sl").list.sort() == pl.col("_tl").list.sort())
                 & (pl.col("s_nums").str.split(" ").list.first() == pl.col("addr_nums").str.split(" ").list.first())
                 & (pl.col("s_nums") != "")
                 & (pl.col("_sa").list.set_intersection("_ta").list.len()
                    >= 0.5 * pl.col("_sa").list.set_union("_ta").list.len()))
    same_pos = pl.struct("_a", "_b").map_elements(lambda r: sum(u != v for u, v in zip(r["_a"], r["_b"])) == 1,
                                                  return_dtype=pl.Boolean)
    return x.filter(same_pos).select("s1", "t", new=pl.col("_b").list.set_difference("_a").list.first())


def _one_word_adds(cfg: Config, split: str, pool: pl.DataFrame, country: str) -> pl.DataFrame:
    """(s1, t, new): target core = S1 core words plus exactly one extra word, same legal form, same first
    house number, address-token Jaccard >= 0.5."""
    d = cfg.split_dir(split)
    s1 = (pl.scan_parquet(d / "s1.parquet").filter(pl.col("country") == country)
          .select("s1", s_core="core", s_legal="legal", s_nums="addr_nums", s_toks="addr_toks").collect())
    x = pool.select("s1", "t").join(s1, on="s1")
    tg = (pl.scan_parquet(d / "tgt.parquet").select("t", "core", "legal", "addr_nums", "addr_toks")
          .join(x.lazy().select("t").unique(), on="t", how="semi").collect())
    words = lambda c: pl.col(c).fill_null("").str.split(" ").list.eval(pl.element().filter(pl.element() != ""))
    x = x.join(tg, on="t").with_columns(_a=words("s_core"), _b=words("core"), _sa=words("s_toks"), _ta=words("addr_toks"),
                                        _sl=words("s_legal"), _tl=words("legal"))
    x = x.filter((pl.col("_a").list.len() >= 1) & (pl.col("_b").list.len() == pl.col("_a").list.len() + 1)
                 & (pl.col("_a").list.set_difference("_b").list.len() == 0)
                 & (pl.col("_b").list.set_difference("_a").list.len() == 1)
                 & (pl.col("_sl").list.sort() == pl.col("_tl").list.sort())
                 & (pl.col("s_nums").str.split(" ").list.first() == pl.col("addr_nums").str.split(" ").list.first())
                 & (pl.col("s_nums") != "")
                 & (pl.col("_sa").list.set_intersection("_ta").list.len()
                    >= 0.5 * pl.col("_sa").list.set_union("_ta").list.len()))
    return x.select("s1", "t", new=pl.col("_b").list.set_difference("_a").list.first())


def predict_split(cfg: Config, split: str, log=print, level: int = 2) -> tuple[pl.DataFrame, pl.DataFrame, dict]:
    """Final predictions of a stacked level: calibrated scores, optional test-time adjustments
    (per-country shift, French list/vocabulary words, noisy-target and French swap boosts), then
    the tuned decision policy.

    Returns (predicted pairs, candidate pairs, model info with the applied policy).
    """
    meta = json.loads((_dir(cfg, level) / "meta.json").read_text())
    scores = level_scores(cfg, split, level, log)
    policy = dict(meta["policy"])
    countries = None
    if cfg.decision_shift_country:
        policy["shift_country"] = dict(cfg.decision_shift_country)
        countries = load(cfg, split, "s1", ["s1", "country"])
    if cfg.fr_list_words or cfg.fr_vocab_words:
        # French one-word swaps over ALL candidates (list-word copies can sit far below the 1e-4 pool cut)
        swap = _one_word_swaps(cfg, split, scores.select("s1", "t"), "France", adds=cfg.fr_list_adds)
        lst = swap.filter(pl.col("new").is_in(list(cfg.fr_list_words))).select("s1", "t", _l=pl.lit(True))
        voc = swap.filter(pl.col("new").is_in(list(cfg.fr_vocab_words))).select("s1", "t", _v=pl.lit(True))
        odds = cfg.fr_vocab_mult * pl.col("p") / (1 - pl.col("p")).clip(1e-9, None)
        scores = (scores.join(lst, on=["s1", "t"], how="left").join(voc, on=["s1", "t"], how="left")
                  .with_columns(p=pl.when(pl.col("_l").fill_null(False)).then(pl.max_horizontal("p", pl.lit(cfg.fr_list_floor)))
                                .when(pl.col("_v").fill_null(False)).then(odds / (1 + odds)).otherwise(pl.col("p")))
                  .drop("_l", "_v"))
        policy["fr_list"] = {"words": list(cfg.fr_list_words), "floor": cfg.fr_list_floor, "pairs": lst.height,
                             "adds": cfg.fr_list_adds}
        if cfg.fr_vocab_words:
            policy["fr_vocab"] = {"words": list(cfg.fr_vocab_words), "mult": cfg.fr_vocab_mult, "pairs": voc.height}
        log(f"[stack L{level}] {split}: French list-word swaps {lst.height:,} -> p >= {cfg.fr_list_floor}; "
            f"vocabulary-word swaps {voc.height:,} x{cfg.fr_vocab_mult}")
    pool = scores.filter(pl.col("p") > 1e-4)
    if cfg.noisy_boost != 1.0:
        noisy = load(cfg, split, "tgt", ["t", "addr_blank", "is_domain"]).filter(pl.col("addr_blank") | pl.col("is_domain"))
        odds = cfg.noisy_boost * pl.col("p") / (1 - pl.col("p")).clip(1e-9, None)
        pool = pool.with_columns(p=pl.when(pl.col("t").is_in(noisy["t"].implode())).then(odds / (1 + odds)).otherwise(pl.col("p")))
        policy["noisy_boost"] = cfg.noisy_boost
        log(f"[stack L{level}] {split}: odds x{cfg.noisy_boost} for pairs with a blank-address/domain target")
    if cfg.fr_swap_boost != 1.0:
        swap = _one_word_swaps(cfg, split, pool, "France").select("s1", "t")
        odds = cfg.fr_swap_boost * pl.col("p") / (1 - pl.col("p")).clip(1e-9, None)
        pool = (pool.join(swap.with_columns(_sw=pl.lit(True)), on=["s1", "t"], how="left")
                .with_columns(p=pl.when(pl.col("_sw").fill_null(False)).then(odds / (1 + odds)).otherwise(pl.col("p")))
                .drop("_sw"))
        policy["fr_swap_boost"] = cfg.fr_swap_boost
        log(f"[stack L{level}] {split}: odds x{cfg.fr_swap_boost} for {swap.height:,} French one-word-swap pairs")
    pred = apply_policy(pool, policy, countries)
    log(f"[stack L{level}] {split}: {scores.height:,} candidate pairs -> {pred.height:,} predicted matches (policy {policy})")
    return pred, scores.select("s1", "t"), {"policy": policy, "sealed": meta.get("sealed", {})}
