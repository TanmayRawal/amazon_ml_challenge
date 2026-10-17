"""Plan A matcher: logistic regression on lexical pair features plus a tuned decision rule.

The model is deliberately linear: it is the fast baseline and also the linear
diagnostic for later tree models. It is fitted on the dev folds only; the sealed fold
is scored once with the chosen settings.

Decision rule (tuned on dev folds for macro F0.5):
  1. keep candidates with p >= tau;
  2. optional target exclusivity: a target stays only with its highest-p S1
     (every ground-truth target has exactly one owner);
  3. optional relative cut: keep p >= rho * (best p of that S1);
  4. at most 11 matches per S1 (the ground-truth maximum).
Everything else, including S1s with no surviving candidate, is predicted empty.
"""
from __future__ import annotations

import itertools
import json
import time

import numpy as np
import polars as pl
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler

from config import Config
from features import FEATURES, feature_parts
from prepare import load
from report import matching_report
from score import per_entity, summarize

MAX_MATCHES = 11
TRAIN_ROWS = 6_000_000
MODEL_FILE = "plan_a.json"


def _suffix(variant: str) -> str:
    """File-name suffix for a feature variant ("" for "a", "_<variant>" otherwise)."""
    return "" if variant == "a" else f"_{variant}"


def _model_path(cfg: Config, variant: str = "a"):
    """Path of the saved Plan A model JSON for a variant (models dir created if needed)."""
    path = cfg.work_dir / "models"
    path.mkdir(exist_ok=True)
    return path / MODEL_FILE.replace(".json", f"{_suffix(variant)}.json")


def scores_path(cfg: Config, variant: str = "a"):
    """Path of the cached train-split Plan A scores parquet for a variant."""
    return cfg.split_dir("train") / f"scores_a{_suffix(variant)}.parquet"


def _sample_dev(cfg: Config, variant: str) -> pl.DataFrame:
    """Deterministic hash sample of about TRAIN_ROWS dev-fold feature rows (sealed fold excluded)."""
    parts = feature_parts(cfg, "train", variant)
    dev = pl.col("fold") != cfg.sealed_fold
    n_dev = sum(pl.scan_parquet(p).filter(dev).select(pl.len()).collect().item() for p in parts)
    frac = min(1.0, TRAIN_ROWS / max(n_dev, 1))
    keep = (pl.struct("s1", "t").hash(seed=cfg.seed) % 1_000_000) < int(frac * 1_000_000)
    return pl.concat([pl.read_parquet(p).filter(dev & keep) for p in parts])


def _predict(model: dict, df: pl.DataFrame) -> np.ndarray:
    """Logistic-regression probability per row, from the standardizer and coefficients stored in `model`."""
    x = df.select(model["features"]).to_numpy().astype(np.float64)
    z = ((x - np.array(model["mean"])) / np.array(model["scale"])) @ np.array(model["coef"]) + model["intercept"]
    return 1.0 / (1.0 + np.exp(-z))


def score_split(cfg: Config, split: str, model: dict, variant: str = "a") -> pl.DataFrame:
    """Probability for every candidate pair of a split (plus label/fold on train)."""
    out = []
    for part in feature_parts(cfg, split, variant):
        df = pl.read_parquet(part)
        keep = [c for c in ("s1", "t", "label", "fold") if c in df.columns]
        out.append(df.select(keep).with_columns(p=pl.Series(_predict(model, df), dtype=pl.Float32)))
    return pl.concat(out)


def decide(scores: pl.DataFrame, tau: float, rho: float, excl: bool) -> pl.DataFrame:
    """Plan A decision rule (see module docstring) applied to scored pairs.

    Args:
        scores: DataFrame with s1, t and probability p.
        tau: absolute probability threshold.
        rho: relative cut; keep p >= rho * best p of the S1 (0 disables).
        excl: if True, each target stays only with its highest-p S1.
    Returns:
        DataFrame [s1, t] of predicted matches, at most MAX_MATCHES per S1.
    """
    d = scores.filter(pl.col("p") >= tau)
    if excl:
        d = d.filter(pl.col("p") == pl.col("p").max().over("t"))
    if rho > 0:
        d = d.filter(pl.col("p") >= rho * pl.col("p").max().over("s1"))
    d = (
        d.sort(["s1", "p", "t"], descending=[False, True, False])
        .group_by("s1", maintain_order=True)
        .head(MAX_MATCHES)
    )
    return d.select("s1", "t")


def _macro(ids: pl.DataFrame, pred: pl.DataFrame, gt: pl.DataFrame) -> float:
    """Macro F0.5: mean per-S1 F0.5 of `pred` against `gt` over the S1s in `ids`."""
    return float(per_entity(ids, pred, gt)["f"].mean())


def train_and_tune(cfg: Config, log=print, variant: str = "a") -> dict:
    """Fit the Plan A logistic regression on dev folds and grid-search the decision rule.

    Scores all train pairs, tunes (tau, rho, excl) for dev macro F0.5 (coarse then fine
    grid over tau), reports dev and sealed-fold results and saves the model JSON.
    Returns the model dict (standardizer, coefficients, chosen rule, reports).
    """
    t0 = time.time()
    df = _sample_dev(cfg, variant)
    x = df.select(FEATURES).to_numpy().astype(np.float64)
    y = df["label"].to_numpy()
    log(f"[plan A] training rows {len(y):,} (positives {y.mean():.3%})")
    scaler = StandardScaler().fit(x)
    lr = LogisticRegression(C=1.0, max_iter=1000).fit(scaler.transform(x), y)
    model = {
        "features": FEATURES,
        "mean": scaler.mean_.tolist(),
        "scale": scaler.scale_.tolist(),
        "coef": lr.coef_[0].tolist(),
        "intercept": float(lr.intercept_[0]),
        "train_rows": int(len(y)),
        "config": cfg.fingerprint(),
    }
    weights = sorted(zip(FEATURES, lr.coef_[0]), key=lambda kv: -abs(kv[1]))
    log("[plan A] standardized weights: " + ", ".join(f"{k}={v:+.2f}" for k, v in weights))

    model["variant"] = variant
    scores = score_split(cfg, "train", model, variant)
    scores.write_parquet(scores_path(cfg, variant))
    gt = load(cfg, "train", "gt")
    folds = load(cfg, "train", "folds", ["s1", "fold"])
    s1 = load(cfg, "train", "s1", ["s1", "country"]).join(folds, on="s1")
    dev_ids = s1.filter(pl.col("fold") != cfg.sealed_fold)
    sealed_ids = s1.filter(pl.col("fold") == cfg.sealed_fold)
    pool = scores.filter(pl.col("p") >= 0.15).select("s1", "t", "p")
    log(f"[plan A] scored {scores.height:,} pairs; {pool.height:,} with p >= 0.15 ({time.time() - t0:.0f}s)")

    # Coarse grid over the rule shape, then a fine grid over tau for the best shape.
    results = []
    for excl, rho, tau in itertools.product([False, True], [0.0, 0.5, 0.8], [0.3, 0.4, 0.5, 0.6, 0.7, 0.8]):
        results.append((_macro(dev_ids, decide(pool, tau, rho, excl), gt), tau, rho, excl))
    best = max(results)
    log("[plan A] coarse grid top 5: " + "; ".join(f"F={f:.4f} tau={t} rho={r} excl={e}" for f, t, r, e in sorted(results)[-5:][::-1]))
    _, tau0, rho, excl = best
    fine = [(_macro(dev_ids, decide(pool, t, rho, excl), gt), t, rho, excl)
            for t in np.round(np.arange(max(0.2, tau0 - 0.08), min(0.95, tau0 + 0.081), 0.02), 3)]
    f_dev, tau, rho, excl = max(fine)
    model.update(tau=float(tau), rho=float(rho), excl=bool(excl), max_matches=MAX_MATCHES)
    log(f"[plan A] chosen tau={tau} rho={rho} excl={excl} dev macro F0.5={f_dev:.4f}")

    pred = decide(pool, tau, rho, excl)
    model["dev"] = matching_report(dev_ids, pred, gt, "plan A dev", log)
    model["sealed"] = matching_report(sealed_ids, pred, gt, "plan A sealed fold", log)
    _model_path(cfg, variant).write_text(json.dumps(model, indent=2))
    log(f"[plan A] model saved to {_model_path(cfg, variant)} ({time.time() - t0:.0f}s)")
    return model


def load_model(cfg: Config, variant: str = "a") -> dict:
    """Load the saved Plan A model dict for a variant."""
    return json.loads(_model_path(cfg, variant).read_text())


def predict_split(cfg: Config, split: str, log=print, variant: str = "a") -> tuple[pl.DataFrame, pl.DataFrame, dict]:
    """Return (predicted pairs, candidate pairs, model) for a split, as s1/t indices."""
    model = load_model(cfg, variant)
    scores = score_split(cfg, split, model, variant)
    pred = decide(scores, model["tau"], model["rho"], model["excl"])
    log(f"[plan A] {split}: {scores.height:,} candidate pairs -> {pred.height:,} predicted matches")
    return pred, scores.select("s1", "t"), model
