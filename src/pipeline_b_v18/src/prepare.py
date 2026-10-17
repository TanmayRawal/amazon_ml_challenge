"""Stage 0: load raw TSVs once, normalize names and addresses, cache as parquet.

Outputs per split (under work_dir/<split>/):
  s1.parquet   s1 (dense u32 index in file order), entity_id, country, raw + normalized fields
  tgt.parquet  t  (dense u32 index: Source 2 rows, then Source 3 rows), src (2/3), same fields
  gt.parquet   (train only) s1, t pairs of the ground truth
"""
from __future__ import annotations

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


def hidden_s1(entity_id: pl.Expr, frac: float) -> pl.Expr:
    """Deterministic choice of train S1 records to drop: the last two digits of the numeric id
    (ids are random, so this is a uniform sample that any run reproduces)."""
    return (entity_id.str.extract(r"(\d+)$").cast(pl.Int64) % 100) < int(round(frac * 100))


def _normalize(values: list[str], kind: str, schema: dict, n_jobs: int) -> pl.DataFrame:
    """Normalize unique strings in batches; returns the normalized fields plus the input as `_key`."""
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
    """Read and normalize one split's sources and write s1.parquet, tgt.parquet (and gt.parquet for train).

    For train, S1 records selected by hidden_s1 are dropped (test-density simulation).
    """
    out_dir = cfg.split_dir(split)
    t_all = time.time()
    s1 = read_source(source_path(cfg.data_dir, split, 1), "S1-")
    s2 = read_source(source_path(cfg.data_dir, split, 2), "S2-")
    s3 = read_source(source_path(cfg.data_dir, split, 3), "S3-")
    log(f"[{split}] read S1={s1.height:,} S2={s2.height:,} S3={s3.height:,}")
    if split == "train" and cfg.train_hide_frac > 0:
        n0 = s1.height
        s1 = s1.filter(~hidden_s1(pl.col("entity_id"), cfg.train_hide_frac))
        log(f"[{split}] test-density training: kept {s1.height:,} of {n0:,} S1 records; the rest stay as "
            f"ownerless targets")

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
        if cfg.train_hide_frac > 0:
            gt_s1 = gt_s1.filter(~hidden_s1(pl.col("s1_id"), cfg.train_hide_frac))
            pairs = pairs.filter(~hidden_s1(pl.col("s1_id"), cfg.train_hide_frac))
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


def noisy_hidden_copies(cfg: Config) -> pl.DataFrame:
    """Train targets (t) that are blank-address or domain-name copies of a hidden S1 (train_hide_frac).
    These decoys are artifacts of the test-density simulation: the test's extra decoys are clean."""
    if cfg.train_hide_frac <= 0:
        return pl.DataFrame({"t": []}, schema={"t": pl.UInt32})
    _, pairs = read_ground_truth(cfg.data_dir / "train" / "train_ground_truth.tsv")
    ids = pairs.filter(hidden_s1(pl.col("s1_id"), cfg.train_hide_frac)).select(entity_id="t_id")
    tg = load(cfg, "train", "tgt", ["t", "entity_id", "addr_blank", "is_domain"])
    return (tg.filter(pl.col("addr_blank") | pl.col("is_domain")).join(ids, on="entity_id", how="semi").select("t"))


def load(cfg: Config, split: str, what: str, columns: list[str] | None = None) -> pl.DataFrame:
    """Read the cached <split>/<what>.parquet (e.g. "s1", "tgt", "gt", "folds"), optionally only `columns`."""
    return pl.read_parquet(cfg.split_dir(split) / f"{what}.parquet", columns=columns)
