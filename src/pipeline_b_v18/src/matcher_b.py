"""Stage 3 matcher: cross-fitted LightGBM on extended pair features + calibrated decisions.

Training follows the ranker's cross-fitting: model A learns from folds {0,1} and scores
{2,3}; model B the reverse; the sealed fold and the test set get the mean of both
calibrated outputs. Isotonic calibration is fitted on the out-of-fold dev scores.

Decision policies compared on dev folds only (the sealed fold is reported, not tuned):
  eum      threshold tau (+ relative cut rho) after hard single-owner filtering
  dta      exact expected-F0.5 per S1 (decide.dta_select)
  dta_own  single-owner renormalization of p, then exact expected F0.5
"""
from __future__ import annotations

import itertools
import json
import time

import lightgbm as lgb
import numpy as np
import polars as pl
from sklearn.isotonic import IsotonicRegression

from config import Config
from decide import dta_select, hard_owner, prior_shift, single_owner, threshold_select
from features import FEATURES_C, feature_parts
from prepare import load
from report import matching_report, strata_report

PARAMS = {
    "objective": "binary", "learning_rate": 0.08, "num_leaves": 127, "min_data_in_leaf": 500,
    "feature_fraction": 0.8, "bagging_fraction": 0.8, "bagging_freq": 1, "lambda_l2": 1.0,
    "verbosity": -1, "seed": 2026, "deterministic": True, "force_row_wise": True,
}
NEG_RATE = 0.2
GROUPS = {"A": [0, 1], "B": [2, 3]}
VARIANT = "c"


def _dir(cfg: Config):
    """Model directory <work_dir>/models/matcher_b (created if needed)."""
    path = cfg.work_dir / "models" / "matcher_b"
    path.mkdir(parents=True, exist_ok=True)
    return path


def _load_sample(cfg: Config) -> pl.DataFrame:
    """Dev-fold training rows: all positives plus a hash-sampled NEG_RATE share of negatives."""
    keep_neg = (pl.struct("s1", "t").hash(seed=cfg.seed + 1) % 1_000_000) < int(NEG_RATE * 1_000_000)
    dev = pl.col("fold") != cfg.sealed_fold
    return pl.concat([
        pl.read_parquet(p).filter(dev & ((pl.col("label") == 1) | keep_neg))
        for p in feature_parts(cfg, "train", VARIANT)
    ])


def _weights(df: pl.DataFrame) -> np.ndarray:
    """Sample weights undoing negative sampling (1 for positives, 1/NEG_RATE for negatives)."""
    return np.where(df["label"].to_numpy() == 1, 1.0, 1.0 / NEG_RATE)


def train_models(cfg: Config, log=print) -> dict:
    """Train the two cross-fitted LightGBM models (A on folds 0-1, B on folds 2-3) and save them.

    Returns metadata (features, params, best iterations, validation log-loss).
    """
    t0 = time.time()
    data = _load_sample(cfg)
    log(f"[matcher B] sample {data.height:,} rows, positives {int(data['label'].sum()):,} ({time.time() - t0:.0f}s)")
    meta = {"features": FEATURES_C, "params": PARAMS, "neg_rate": NEG_RATE, "groups": GROUPS}
    for name, folds in GROUPS.items():
        tr = data.filter(pl.col("fold").is_in(folds))
        other = data.filter(~pl.col("fold").is_in(folds))
        va = other.sample(n=min(3_000_000, other.height), seed=1, shuffle=True)
        dtr = lgb.Dataset(tr.select(FEATURES_C).to_numpy(), tr["label"].to_numpy(), weight=_weights(tr),
                          feature_name=FEATURES_C, free_raw_data=True)
        dva = lgb.Dataset(va.select(FEATURES_C).to_numpy(), va["label"].to_numpy(), weight=_weights(va), reference=dtr)
        booster = lgb.train(PARAMS, dtr, num_boost_round=1500, valid_sets=[dva],
                            callbacks=[lgb.early_stopping(50, verbose=False)])
        booster.save_model(str(_dir(cfg) / f"matcher_{name}.txt"), num_iteration=booster.best_iteration)
        gain = sorted(zip(FEATURES_C, booster.feature_importance("gain")), key=lambda kv: -kv[1])
        meta[f"best_iter_{name}"] = booster.best_iteration
        meta[f"valid_logloss_{name}"] = booster.best_score["valid_0"]["binary_logloss"]
        log(f"[matcher B] model {name}: best_iter={booster.best_iteration} logloss={meta[f'valid_logloss_{name}']:.4f} "
            f"({time.time() - t0:.0f}s); top gain: " + ", ".join(k for k, _ in gain[:12]))
        del dtr, dva
    return meta


def _boosters(cfg: Config) -> dict:
    """Load the saved boosters {"A": ..., "B": ...}."""
    return {n: lgb.Booster(model_file=str(_dir(cfg) / f"matcher_{n}.txt")) for n in GROUPS}


def _predict(b: lgb.Booster, df: pl.DataFrame) -> np.ndarray:
    """Raw booster probabilities for the FEATURES_C columns of df (float32)."""
    return b.predict(df.select(FEATURES_C).to_numpy(), num_threads=0).astype(np.float32)


def raw_scores(cfg: Config, split: str) -> pl.DataFrame:
    """Uncalibrated scores: cross-fitted on dev folds; pA and pB kept for sealed/test."""
    boosters = _boosters(cfg)
    out = []
    for part in feature_parts(cfg, split, VARIANT):
        df = pl.read_parquet(part)
        pa = _predict(boosters["A"], df)
        pb = _predict(boosters["B"], df)
        keep = [c for c in ("s1", "t", "label", "fold") if c in df.columns]
        out.append(df.select(keep).with_columns(pa=pl.Series(pa), pb=pl.Series(pb)))
    return pl.concat(out)


def calibrated(raw: pl.DataFrame, xs: np.ndarray, ys: np.ndarray) -> pl.DataFrame:
    """Out-of-fold calibrated p on dev folds; mean of both calibrated models elsewhere.

    (xs, ys) are the isotonic breakpoints; np.interp reproduces IsotonicRegression.predict
    with clipping outside the fitted range.
    """
    ca = np.interp(raw["pa"].to_numpy(), xs, ys)
    cb = np.interp(raw["pb"].to_numpy(), xs, ys)
    if "fold" in raw.columns:
        fold = raw["fold"].to_numpy()
        p = np.where(np.isin(fold, GROUPS["B"]), ca,          # folds 2,3 were not seen by model A
             np.where(np.isin(fold, GROUPS["A"]), cb, 0.5 * (ca + cb)))
    else:
        p = 0.5 * (ca + cb)
    return raw.select([c for c in ("s1", "t", "label", "fold") if c in raw.columns]).with_columns(
        p=pl.Series(p.astype(np.float64)))


def fit_isotonic(raw: pl.DataFrame, sealed_fold: int, seed: int) -> IsotonicRegression:
    """Fit isotonic calibration on out-of-fold dev scores (up to 8M sampled rows)."""
    dev = raw.filter(pl.col("fold") != sealed_fold)
    oof = np.where(np.isin(dev["fold"].to_numpy(), GROUPS["B"]), dev["pa"].to_numpy(), dev["pb"].to_numpy())
    rng = np.random.default_rng(seed)
    idx = rng.choice(len(oof), size=min(8_000_000, len(oof)), replace=False)
    return IsotonicRegression(out_of_bounds="clip", y_min=0.0, y_max=1.0).fit(oof[idx], dev["label"].to_numpy()[idx])


def apply_policy(scores: pl.DataFrame, policy: dict, countries: pl.DataFrame | None = None) -> pl.DataFrame:
    """policy["shift"] scales all odds; policy["shift_country"] ({country: r}) overrides it per
    country and needs `countries` (s1, country). France has no labels, so its factor is set
    from leaderboard probes rather than on dev folds."""
    s = scores.select("s1", "t", "p")
    by_country = policy.get("shift_country") or {}
    if by_country:
        if countries is None:
            raise ValueError("shift_country needs the S1 countries")
        r = pl.col("country").replace_strict(by_country, default=policy.get("shift", 1.0), return_dtype=pl.Float64)
        s = prior_shift(s.join(countries.select("s1", "country"), on="s1", how="left"), r).drop("country")
    else:
        s = prior_shift(s, policy.get("shift", 1.0))
    if policy["name"] == "eum":
        return threshold_select(hard_owner(s), policy["tau"], policy["rho"])
    if policy["name"] == "dta_own":
        s = single_owner(s)
    elif policy["name"] == "dta_hard":
        s = hard_owner(s)
    return dta_select(s.filter(pl.col("p") > 1e-4), policy["missed"]).select("s1", "t")


def train_and_tune(cfg: Config, log=print) -> dict:
    """Train, calibrate and score train, pick the best decision policy on dev folds, report the sealed fold.

    Writes scores_b.parquet and models/matcher_b/meta.json; returns the metadata.
    """
    t0 = time.time()
    meta = train_models(cfg, log)
    raw = raw_scores(cfg, "train")
    iso = fit_isotonic(raw, cfg.sealed_fold, cfg.seed)
    scores = calibrated(raw, iso.X_thresholds_, iso.y_thresholds_)
    scores.write_parquet(cfg.split_dir("train") / "scores_b.parquet")
    meta["isotonic"] = {"x": iso.X_thresholds_.tolist(), "y": iso.y_thresholds_.tolist()}
    log(f"[matcher B] scored + calibrated {scores.height:,} pairs ({time.time() - t0:.0f}s)")

    gt = load(cfg, "train", "gt")
    s1 = load(cfg, "train", "s1", ["s1", "country"]).join(load(cfg, "train", "folds", ["s1", "fold"]), on="s1")
    dev_ids = s1.filter(pl.col("fold") != cfg.sealed_fold)
    sealed_ids = s1.filter(pl.col("fold") == cfg.sealed_fold)
    # Calibration check on dev (out-of-fold): mean p vs match rate by p decile.
    dev_sc = scores.filter(pl.col("fold") != cfg.sealed_fold)
    rel = (dev_sc.with_columns(b=(pl.col("p") * 10).floor().clip(0, 9)).group_by("b")
           .agg(mean_p=pl.col("p").mean(), rate=pl.col("label").mean(), n=pl.len()).sort("b"))
    log(f"[matcher B] reliability on dev (out-of-fold):\n{rel}")

    covered = gt.join(scores.select("s1", "t"), on=["s1", "t"], how="semi").join(dev_ids, on="s1", how="semi").height
    missed = (gt.join(dev_ids, on="s1", how="semi").height - covered) / dev_ids.height
    log(f"[matcher B] blocking misses per dev S1: {missed:.3f}")

    from score import per_entity

    def macro(ids, pred):
        """Mean per-entity F0.5 of `pred` over the S1 ids in `ids`."""
        return float(per_entity(ids.select("s1"), pred, gt)["f"].mean())

    pool = scores.filter(pl.col("p") > 1e-4)
    results = []
    for tau, rho in itertools.product([0.3, 0.4, 0.5, 0.6, 0.7], [0.0, 0.5]):
        pol = {"name": "eum", "tau": tau, "rho": rho}
        results.append((macro(dev_ids, apply_policy(pool, pol)), pol))
    for name, m in itertools.product(["dta", "dta_own", "dta_hard"], [0.0, missed, 2 * missed]):
        pol = {"name": name, "missed": round(m, 4)}
        results.append((macro(dev_ids, apply_policy(pool, pol)), pol))
    for f, pol in sorted(results, key=lambda r: -r[0])[:8]:
        log(f"[matcher B] dev F0.5={f:.4f}  {pol}")
    best_f, best = max(results, key=lambda r: r[0])
    meta["policy"] = best
    meta["missed"] = missed
    log(f"[matcher B] chosen policy {best} (dev {best_f:.4f})")

    pred = apply_policy(pool, best)
    meta["dev"] = matching_report(dev_ids, pred, gt, "matcher B dev", log)
    meta["sealed"] = matching_report(sealed_ids, pred, gt, "matcher B sealed fold", log)
    meta["strata_sealed"] = strata_report(cfg, pred, cfg.sealed_fold, log)
    (_dir(cfg) / "meta.json").write_text(json.dumps(meta, indent=2, default=str))
    log(f"[matcher B] done in {time.time() - t0:.0f}s")
    return meta


def split_scores(cfg: Config, split: str, log=print) -> pl.DataFrame:
    """Calibrated matcher-B scores for a split, cached as <split>/scores_b.parquet."""
    path = cfg.split_dir(split) / "scores_b.parquet"
    if path.exists():
        return pl.read_parquet(path)
    meta = json.loads((_dir(cfg) / "meta.json").read_text())
    scores = calibrated(raw_scores(cfg, split), np.array(meta["isotonic"]["x"]), np.array(meta["isotonic"]["y"]))
    scores.write_parquet(path)
    log(f"[matcher B] {split}: scored {scores.height:,} pairs -> {path}")
    return scores


def predict_split(cfg: Config, split: str, log=print) -> tuple[pl.DataFrame, pl.DataFrame, dict]:
    """Apply the tuned policy to a split's calibrated scores.

    Returns (predicted pairs, candidate pairs, model info for the manifest).
    """
    meta = json.loads((_dir(cfg) / "meta.json").read_text())
    scores = split_scores(cfg, split, log)
    pred = apply_policy(scores.filter(pl.col("p") > 1e-4), meta["policy"])
    log(f"[matcher B] {split}: {scores.height:,} candidate pairs -> {pred.height:,} predicted matches")
    return pred, scores.select("s1", "t"), {"policy": meta["policy"], "sealed": meta.get("sealed", {})}
