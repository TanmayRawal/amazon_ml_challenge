"""v13 pair judge: a small multilingual cross-encoder that reads both records and scores uncertain pairs.

The feature-based matchers see a pair through ~60 hand-made similarities. For the pairs they are
unsure about, a transformer reading "name | address" of both records can use what those features
flatten: which exact words changed, digit edits, scripts and French/Indian spellings. Its score
becomes one more stage-2 feature ("pj"); pairs outside the scored band get NaN.

Model: sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2 (Apache-2.0, 118M parameters,
50+ languages incl. Hindi and French), fine-tuned only on the provided training pairs, run offline.
Trained on stage-1 p in [PJ_LO, PJ_HI] and scored on [SCORE_LO, SCORE_HI] (laptop GPU throughput
is ~1.4-2k pairs/s, so the scoring band can be narrowed). Cross-fitted like the other stages: model A learns from folds {0,1}
and scores folds {2,3}, model B the reverse; the sealed fold and test are scored by model A alone,
so every pj comes from one model, as in training. Training and scoring run in pair_judge_gpu.py
with the Python in BER_GPU_PYTHON (torch + transformers + polars); this module prepares the pair
files and reads the scores back.
"""
from __future__ import annotations

import os
import shutil
import subprocess
import time
from pathlib import Path

import polars as pl

from config import Config
from data_io import read_source, source_path
from matcher_b import GROUPS
from prepare import hidden_s1, load

MODEL = "sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2"
PJ_LO, PJ_HI = 0.01, 0.99          # training band
SCORE_LO, SCORE_HI = 0.01, 0.99    # scoring band: v12's sharper stage 1 leaves ~1.8-2.0 pairs per S1 here
TRAIN_CAP = 1_200_000              # pairs per cross-fitted model


def pj_dir(cfg: Config) -> Path:
    """Pair-judge working directory <work_dir>/pj (created if needed)."""
    path = cfg.work_dir / "pj"
    path.mkdir(parents=True, exist_ok=True)
    return path


def _text(df: pl.DataFrame, prefix: str) -> pl.Expr:
    """Model input text "name | address" from the raw columns with the given prefix ("s_" or "t_")."""
    return pl.concat_str([pl.col(f"{prefix}business_name").fill_null(""), pl.lit(" | "),
                          pl.col(f"{prefix}business_address").fill_null("")])


def _attach_text(cfg: Config, split: str, pairs: pl.DataFrame) -> pl.DataFrame:
    """Add columns a (S1 text) and b (target text) to (s1, t) pairs from the split's caches."""
    # read only the records these pairs use (runs next to memory-heavy pipeline steps)
    d = cfg.split_dir(split)
    s1 = (pl.scan_parquet(d / "s1.parquet").select("s1", s_business_name="business_name", s_business_address="business_address")
          .join(pairs.lazy().select("s1").unique(), on="s1", how="semi").collect())
    tg = (pl.scan_parquet(d / "tgt.parquet").select("t", t_business_name="business_name", t_business_address="business_address")
          .join(pairs.lazy().select("t").unique(), on="t", how="semi").collect())
    return (pairs.join(s1, on="s1")
            .join(tg, on="t")
            .with_columns(a=_text(pairs, "s_"), b=_text(pairs, "t_"))
            .drop("s_business_name", "s_business_address", "t_business_name", "t_business_address"))


def export_train(cfg: Config, old_scores: Path | None = None, log=print) -> Path:
    """Training pairs: the uncertain band of this run's out-of-fold stage-1 train scores
    (train/scores_b.parquet), or of an earlier run's scores given as `old_scores` (old S1 indexes =
    row order of train_source1.tsv, before test-density hiding; v13 used v10's), with folds and labels."""
    t0 = time.time()
    if old_scores is None:
        sc = (pl.scan_parquet(cfg.split_dir("train") / "scores_b.parquet")
              .filter(pl.col("p").is_between(PJ_LO, PJ_HI)).select("s1", "t").collect())
    else:
        raw = read_source(source_path(cfg.data_dir, "train", 1), "S1-").with_row_index("old").select(
            pl.col("old").cast(pl.UInt32), "entity_id")
        raw = raw.filter(~hidden_s1(pl.col("entity_id"), cfg.train_hide_frac)) if cfg.train_hide_frac > 0 else raw
        s1 = load(cfg, "train", "s1", ["s1", "entity_id"])
        old_to_new = raw.join(s1, on="entity_id").select("old", "s1")
        sc = (pl.scan_parquet(old_scores).filter(pl.col("p").is_between(PJ_LO, PJ_HI)).select(old="s1", t="t").collect()
              .join(old_to_new, on="old").drop("old"))
    folds = load(cfg, "train", "folds", ["s1", "fold"])
    gt = load(cfg, "train", "gt").with_columns(label=pl.lit(1, pl.UInt8))
    pairs = (sc.join(folds, on="s1").filter(pl.col("fold") != cfg.sealed_fold)
             .join(gt, on=["s1", "t"], how="left").with_columns(pl.col("label").fill_null(0)))
    parts = []
    for name, fl in GROUPS.items():
        g = pairs.filter(pl.col("fold").is_in(fl))
        parts.append(g.sample(n=min(TRAIN_CAP, g.height), seed=7, shuffle=True))
    out = _attach_text(cfg, "train", pl.concat(parts))
    path = pj_dir(cfg) / "train_pairs.parquet"
    out.write_parquet(path)
    log(f"[pair-judge] training pairs {out.height:,} (positive rate {out['label'].mean():.3f}) -> {path} "
        f"({time.time() - t0:.0f}s)")
    return path


def _scorer(split: str, fold: pl.Expr | None) -> pl.Expr:
    """Which model scores a pair: the one that did not train on its fold group; A elsewhere."""
    if split != "train" or fold is None:
        return pl.lit("A")
    return pl.when(fold.is_in(GROUPS["A"])).then(pl.lit("B")).otherwise(pl.lit("A"))


def export_score(cfg: Config, split: str, log=print) -> None:
    """Pairs to score: this run's stage-1 scores inside the scoring band, one file per scoring model."""
    cached = cfg.split_dir(split) / "scores_b.parquet"
    if cached.exists():
        sc = pl.scan_parquet(cached).filter(pl.col("p").is_between(SCORE_LO, SCORE_HI)).collect()
    else:
        from matcher_b import split_scores
        sc = split_scores(cfg, split, log).filter(pl.col("p").is_between(SCORE_LO, SCORE_HI))
    cols = ["s1", "t"] + (["fold"] if "fold" in sc.columns else [])
    out = _attach_text(cfg, split, sc.select(cols)).with_columns(
        m=_scorer(split, pl.col("fold") if "fold" in cols else None))
    for name in GROUPS:
        part = out.filter(pl.col("m") == name).drop("m")
        path = pj_dir(cfg) / f"{split}_score_{name}.parquet"
        path.unlink(missing_ok=True)
        # scores and half-finished chunks of an earlier run belong to other pairs
        (pj_dir(cfg) / f"{split}_pj_{name}.parquet").unlink(missing_ok=True)
        shutil.rmtree(pj_dir(cfg) / f"{split}_pj_{name}.parquet.parts", ignore_errors=True)
        if part.height:
            part.write_parquet(path)
        log(f"[pair-judge] {split}: {part.height:,} band pairs for model {name} -> {path}")


def _gpu(args: list[str], log) -> None:
    """Run pair_judge_gpu.py with `args` under BER_GPU_PYTHON (offline Hugging Face), streaming its output.

    Raises RuntimeError if the interpreter is not configured or the process fails.
    """
    py = os.environ.get("BER_GPU_PYTHON")
    if not py:
        raise RuntimeError("BER_GPU_PYTHON must point to a Python with torch (CUDA) + transformers + polars")
    env = {k: v for k, v in os.environ.items() if k not in ("__PYVENV_LAUNCHER__", "PYTHONHOME", "PYTHONPATH", "VIRTUAL_ENV")}
    # the base model must already be in the local Hugging Face cache: no network during training or scoring
    env.update(HF_HUB_OFFLINE="1", TRANSFORMERS_OFFLINE="1", HF_HUB_DISABLE_PROGRESS_BARS="1", TRANSFORMERS_VERBOSITY="error")
    script = Path(__file__).with_name("pair_judge_gpu.py")
    proc = subprocess.Popen([py, "-u", str(script)] + args, stdout=subprocess.PIPE, stderr=subprocess.STDOUT, text=True, env=env)
    for line in proc.stdout:
        log(f"    {line.rstrip()}")
    if proc.wait() != 0:
        raise RuntimeError(f"pair_judge_gpu.py {args[0]} failed")


def _gpu_retry(args: list[str], log, attempts: int = 500) -> None:
    """The laptop GPU (Windows/WDDM, shared with the desktop) fails every few minutes under host
    memory pressure with a CUDA error that breaks the process; the GPU script checkpoints (training)
    or saves chunks (scoring), so a restart resumes where it stopped."""
    for attempt in range(1, attempts + 1):
        try:
            _gpu(args, log)
            return
        except RuntimeError:
            if attempt == attempts:
                raise
            log(f"[pair-judge] {args[0]}: attempt {attempt} failed, resuming")
            time.sleep(5)


# Model sets: tag "" = model_A/model_B -> {split}_pj_{A,B}.parquet; tag "2" = a second set trained on the
# same pairs with another seed (model_A2/model_B2 -> {split}_pj2_{A,B}.parquet). pj_scores averages the
# sets listed in cfg.pj_tags.
SEEDS = {"": 2026, "2": 7}


def train(cfg: Config, log=print, tag: str = "") -> None:
    """Fine-tune the cross-fitted pair-judge models A and B (model set `tag`); skips trained ones."""
    d = pj_dir(cfg)
    os.environ["PJ_SEED"] = str(SEEDS.get(tag, 2026))
    for name, fl in GROUPS.items():
        out = d / f"model_{name}{tag}"
        if (out / "config.json").exists():
            log(f"[pair-judge] model {name}{tag} already trained ({out})")
            continue
        _gpu_retry(["train", str(d / "train_pairs.parquet"), str(out), ",".join(map(str, fl)), MODEL], log)


def score(cfg: Config, split: str, log=print, tag: str = "") -> None:
    """Score the exported band pairs of a split with model set `tag`, writing {split}_pj{tag}_{A,B}.parquet."""
    d = pj_dir(cfg)
    for name in GROUPS:
        pairs = d / f"{split}_score_{name}.parquet"
        if pairs.exists():
            _gpu_retry(["score", str(pairs), str(d / f"model_{name}{tag}"), str(d / f"{split}_pj{tag}_{name}.parquet")], log)


def pj_scores(cfg: Config, split: str) -> pl.DataFrame:
    """s1, t, pj for the scored band: out-of-fold on dev folds, model A on the sealed fold and test;
    the mean over the model sets in cfg.pj_tags."""
    sets = []
    for tag in cfg.pj_tags:
        parts = [pl.read_parquet(p, columns=["s1", "t", "pj"]) for p in sorted(pj_dir(cfg).glob(f"{split}_pj{tag}_*.parquet"))]
        if not parts:
            raise FileNotFoundError(f"no pair-judge scores (set '{tag}') for {split} in {pj_dir(cfg)}")
        sets.append(pl.concat(parts).rename({"pj": f"pj{tag or '1'}"}))
    out = sets[0]
    for s in sets[1:]:
        out = out.join(s, on=["s1", "t"], how="inner")
    cols = [c for c in out.columns if c.startswith("pj")]
    return out.select("s1", "t", pj=pl.mean_horizontal(cols).cast(pl.Float32))
