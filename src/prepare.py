"""Stage 0: load raw TSVs once, normalize names and addresses, cache as parquet.

Outputs per split (under work_dir/<split>/):
  s1.parquet   s1 (dense u32 index in file order), entity_id, country, raw + normalized fields
  tgt.parquet  t  (dense u32 index: Source 2 rows, then Source 3 rows), src (2/3), same fields
  gt.parquet   (train only) s1, t pairs of the ground truth
"""
from __future__ import annotations

import json
import time

import polars as pl

from config import Config
from data_io import read_ground_truth, read_source, source_path
from normalize import ADDR_FIELDS, NAME_FIELDS, parallel_normalize

NAME_SCHEMA = {"core": pl.String, "core_key": pl.String, "legal": pl.String, "concat": pl.String,
               "initials": pl.String, "acro_self": pl.String, "domain": pl.String,
               "is_domain": pl.Boolean, "script": pl.String}
ADDR_SCHEMA = {"addr_nums": pl.String, "addr_toks": pl.String, "addr_blank": pl.Boolean}
BATCH = 1_000_000


def _normalize(values: list[str], kind: str, schema: dict, n_jobs: int) -> pl.DataFrame:
    """Normalize strings in batches of BATCH rows with parallel_normalize.

    Args: values: unique raw strings; kind: "name" or "addr"; schema: output column types; n_jobs: workers.
    Returns: one row per input string with the normalized columns plus `_key` (the raw string) for joining back."""
    parts = []
    for i in range(0, len(values), BATCH):
        batch = values[i:i + BATCH]
        rows = parallel_normalize(kind, batch, n_jobs)
        part = pl.DataFrame(rows, schema=schema, orient="row")
        parts.append(part.with_columns(pl.Series("_key", batch, dtype=pl.String)))
    return pl.concat(parts)


def normalize_frame(df: pl.DataFrame, n_jobs: int, log) -> pl.DataFrame:
    """Attach normalized name/address fields, computing each unique string once."""
    t0 = time.time()
    names = df["business_name"].unique().to_list()
    name_tab = _normalize(names, "name", NAME_SCHEMA, n_jobs)
    log(f"    names: {len(names):,} unique in {time.time() - t0:.0f}s")
    t0 = time.time()
    addrs = df["business_address"].unique().to_list()
    addr_tab = _normalize(addrs, "addr", ADDR_SCHEMA, n_jobs)
    log(f"    addresses: {len(addrs):,} unique in {time.time() - t0:.0f}s")
    out = (
        df.join(name_tab, left_on="business_name", right_on="_key", how="left", maintain_order="left")
        .join(addr_tab, left_on="business_address", right_on="_key", how="left", maintain_order="left")
    )
    missing = out.filter(pl.col("core").is_null() | pl.col("addr_nums").is_null()).height
    if missing:
        raise RuntimeError(f"{missing} rows lost their normalized fields")
    return out


def prepare_split(cfg: Config, split: str, log=print) -> None:
    """Stage 0 for one split: read the three sources, assign dense indices, normalize and cache as parquet.

    Writes s1.parquet and tgt.parquet (Source 2 rows, then Source 3 rows, src = 2/3) under cfg.split_dir(split);
    for train also gt.parquet (s1, t) after checking the ground-truth ids against the source files."""
    out_dir = cfg.split_dir(split)
    t_all = time.time()
    s1 = read_source(source_path(cfg.data_dir, split, 1), "S1-")
    s2 = read_source(source_path(cfg.data_dir, split, 2), "S2-")
    s3 = read_source(source_path(cfg.data_dir, split, 3), "S3-")
    log(f"[{split}] read S1={s1.height:,} S2={s2.height:,} S3={s3.height:,}")

    s1 = s1.with_row_index("s1").with_columns(pl.col("s1").cast(pl.UInt32))
    tgt = pl.concat([
        s2.with_columns(src=pl.lit(2, pl.UInt8)),
        s3.with_columns(src=pl.lit(3, pl.UInt8)),
    ]).with_row_index("t").with_columns(pl.col("t").cast(pl.UInt32))

    log(f"[{split}] normalizing S1")
    s1 = normalize_frame(s1, cfg.n_jobs, log)
    log(f"[{split}] normalizing S2+S3")
    tgt = normalize_frame(tgt, cfg.n_jobs, log)

    s1.write_parquet(out_dir / "s1.parquet")
    tgt.write_parquet(out_dir / "tgt.parquet")

    if split == "train":
        gt_path = cfg.data_dir / "train" / "train_ground_truth.tsv"
        gt_s1, pairs = read_ground_truth(gt_path)
        if gt_s1.height != s1.height or gt_s1.join(s1, left_on="s1_id", right_on="entity_id", how="anti").height:
            raise RuntimeError("ground truth S1 ids do not match train_source1")
        gt = (
            pairs.join(s1.select("s1", s1_id="entity_id"), on="s1_id", how="left")
            .join(tgt.select("t", t_id="entity_id"), on="t_id", how="left")
        )
        if gt.filter(pl.col("s1").is_null() | pl.col("t").is_null()).height:
            raise RuntimeError("ground truth references ids missing from the source files")
        gt.select("s1", "t").write_parquet(out_dir / "gt.parquet")
        log(f"[{split}] ground truth pairs: {gt.height:,}")
    log(f"[{split}] prepared in {time.time() - t_all:.0f}s")


def prepare_skeletons(cfg: Config, split: str, log=print) -> None:
    """Stage 5: cross-script name skeletons, stored beside the caches as
    s1_sk.parquet (s1, name_sk) and tgt_sk.parquet (t, name_sk)."""
    for what, idc in (("s1", "s1"), ("tgt", "t")):
        t0 = time.time()
        df = load(cfg, split, what, [idc, "business_name"])
        names = df["business_name"].unique().to_list()
        sk = parallel_normalize("skel", names, cfg.n_jobs)
        table = pl.DataFrame({"business_name": names, "name_sk": sk}, schema={"business_name": pl.String, "name_sk": pl.String})
        out = df.join(table, on="business_name", how="left", maintain_order="left").select(idc, "name_sk")
        if out["name_sk"].null_count():
            raise RuntimeError(f"{split}/{what}: names without a skeleton")
        out.write_parquet(cfg.split_dir(split) / f"{what}_sk.parquet")
        log(f"[{split}] {what} skeletons: {len(names):,} unique names ({time.time() - t0:.0f}s)")


def density_keep(train: pl.DataFrame, test: pl.DataFrame) -> dict:
    """train/test: country, n_s1, n_t. Share of each train country's S1s to keep so that its S1-per-target
    ratio equals the test set's; 1.0 where test is not sparser or the country is absent from test."""
    x = train.join(test, on="country", how="left", suffix="_test")
    ratio = (pl.col("n_s1_test") / pl.col("n_t_test")) / (pl.col("n_s1") / pl.col("n_t"))
    return dict(x.select("country", keep=ratio.fill_null(1.0).clip(upper_bound=1.0)).iter_rows())


def hide_s1(s1: pl.DataFrame, others: dict, keep: dict, seed: int) -> tuple[pl.DataFrame, dict]:
    """Keep each country's share of S1 rows (seeded hash of entity_id) and re-index them 0..n-1.
    others: name -> frame keyed by s1 (gt, skeletons); their rows of hidden S1s are dropped, so hidden
    S1s' targets stay in the pool as ownerless distractors, like test decoys."""
    share = pl.col("country").replace_strict(keep, default=1.0, return_dtype=pl.Float64)
    kept = s1.filter((pl.col("entity_id").hash(seed=seed) % 1_000_000) < share * 1_000_000).sort("s1")
    remap = kept.select(old="s1").with_row_index("s1").with_columns(pl.col("s1").cast(pl.UInt32))
    fix = lambda df: df.rename({"s1": "old"}).join(remap, on="old").drop("old").select(df.columns)  # noqa: E731
    return fix(kept), {k: fix(v) for k, v in others.items()}


def match_density(cfg: Config, log=print) -> None:
    """Hide train S1s until each country's S1-per-target ratio matches test (test holds ~5.8 target
    records per S1 against ~4.7 in train, i.e. ~40% ownerless decoys against ~26%). Every model then
    trains, and the sealed fold is scored, at test-like decoy density, so no decision-shift hedge is
    needed. Rewrites train s1 / gt / s1_sk / folds in place; run once after prepare, translit and folds.
    Folds are kept (hidden S1s dropped), not rebuilt: re-ranking the hash inside each stratum would move
    S1s between folds, and models trained earlier (the V13x transformer) must still never have seen the
    sealed fold."""
    out = cfg.split_dir("train")
    marker = out / "density.json"
    if marker.exists():
        raise RuntimeError(f"train S1s are already density-matched: {marker.read_text()}")
    count = lambda df: df.group_by("country").len()  # noqa: E731
    if (cfg.split_dir("test") / "s1.parquet").exists():
        t_s1, t_tg = load(cfg, "test", "s1", ["country"]), load(cfg, "test", "tgt", ["country"])
    else:
        t_s1 = read_source(source_path(cfg.data_dir, "test", 1), "S1-").select("country")
        t_tg = pl.concat([read_source(source_path(cfg.data_dir, "test", k), f"S{k}-").select("country") for k in (2, 3)])
    s1 = load(cfg, "train", "s1")
    train = count(s1).rename({"len": "n_s1"}).join(count(load(cfg, "train", "tgt", ["country"])).rename({"len": "n_t"}), on="country")
    test = count(t_s1).rename({"len": "n_s1"}).join(count(t_tg).rename({"len": "n_t"}), on="country")
    keep = density_keep(train, test)
    others = {"gt": load(cfg, "train", "gt")}
    for name in ("s1_sk", "folds"):
        if (out / f"{name}.parquet").exists():
            others[name] = load(cfg, "train", name)
    if "folds" not in others:
        raise RuntimeError("run folds before match-density (folds are kept, not rebuilt)")
    s1_new, others = hide_s1(s1, others, keep, cfg.seed)
    s1_new.write_parquet(out / "s1.parquet")
    for name, df in others.items():
        df.sort("s1").write_parquet(out / f"{name}.parquet")
    marker.write_text(json.dumps({"keep": keep, "s1_before": s1.height, "s1_after": s1_new.height}, indent=2))
    sizes = dict(others["folds"].group_by("fold").len().sort("fold").iter_rows())
    log(f"[train] density match: kept {s1_new.height:,} of {s1.height:,} S1 ({keep}); fold sizes {sizes}")


def load(cfg: Config, split: str, what: str, columns: list[str] | None = None) -> pl.DataFrame:
    """Read one cached parquet table of a split (s1, tgt, gt, folds, s1_sk, ...), optionally only some columns."""
    return pl.read_parquet(cfg.split_dir(split) / f"{what}.parquet", columns=columns)


def fresh(path, *deps) -> bool:
    """Audit: a cached test-split score file is reused only when newer than the model and inputs it came from
    (a retrained model or rebuilt features must not meet an old model's test scores)."""
    return path.exists() and all(path.stat().st_mtime > d.stat().st_mtime for d in deps if d.exists())
