"""Read the challenge TSVs exactly as written and write submission files.

Quote processing is disabled on purpose: some fields start with CSV-style escaped
double quotes, and a quote-aware parser would silently rewrite them. Every read is
checked against a raw newline count.
"""
from __future__ import annotations

from pathlib import Path

import polars as pl

COLUMNS = ["entity_id", "business_name", "business_address", "country"]
GT_COLUMNS = ["source1_entity_id", "matched_entity_ids"]
MATCH_HEADER = ("source1_entity_id", "matched_entity_ids")
CAND_HEADER = ("source1_entity_id", "candidate_entity_ids")


def source_path(data_dir: Path, split: str, source: int) -> Path:
    """Path of <data_dir>/<split>/<split>_source<source>.tsv."""
    return data_dir / split / f"{split}_source{source}.tsv"


def count_data_lines(path: Path) -> int:
    """Number of lines after the header, counted on raw bytes."""
    n, last = 0, b"\n"
    with open(path, "rb") as fh:
        while chunk := fh.read(1 << 24):
            n += chunk.count(b"\n")
            last = chunk[-1:]
    if last != b"\n":
        n += 1
    return n - 1


def _read_tsv(path: Path) -> pl.DataFrame:
    """Read a tab-separated file as all-string columns with quote processing disabled."""
    return pl.read_csv(
        path,
        separator="\t",
        quote_char=None,
        has_header=True,
        infer_schema=False,
        missing_utf8_is_empty_string=True,
        encoding="utf8",
    )


def read_source(path: Path, prefix: str) -> pl.DataFrame:
    """Load one source file and verify shape, ID uniqueness and ID prefix."""
    df = _read_tsv(path)
    if df.columns != COLUMNS:
        raise ValueError(f"{path}: unexpected columns {df.columns}")
    expected = count_data_lines(path)
    if df.height != expected:
        raise ValueError(f"{path}: parsed {df.height} rows but file has {expected} data lines")
    if df["entity_id"].n_unique() != df.height:
        raise ValueError(f"{path}: duplicate entity_id values")
    bad = df.filter(~pl.col("entity_id").str.starts_with(prefix)).height
    if bad:
        raise ValueError(f"{path}: {bad} entity_id values without prefix {prefix}")
    return df


def read_ground_truth(path: Path) -> tuple[pl.DataFrame, pl.DataFrame]:
    """Return (all Source-1 ids in file order, exploded (s1_id, t_id) match pairs)."""
    gt = _read_tsv(path)
    if gt.columns != GT_COLUMNS:
        raise ValueError(f"{path}: unexpected columns {gt.columns}")
    if gt.height != count_data_lines(path):
        raise ValueError(f"{path}: row count mismatch")
    if gt["source1_entity_id"].n_unique() != gt.height:
        raise ValueError(f"{path}: duplicate source1_entity_id rows")
    pairs = (
        gt.select(
            s1_id=pl.col("source1_entity_id"),
            t_id=pl.col("matched_entity_ids").str.split(","),
        )
        .explode("t_id")
        .filter(pl.col("t_id") != "")
    )
    if pairs.unique().height != pairs.height:
        raise ValueError(f"{path}: duplicate ids inside a match list")
    return gt.select(s1_id=pl.col("source1_entity_id")), pairs


def write_id_lists(
    path: Path, header: tuple[str, str], s1_ids: pl.Series, pairs: pl.DataFrame
) -> None:
    """Write one row per S1 id (in the given order) with a comma-joined, sorted id list.

    `pairs` has string columns s1_id and t_id. Rows with no pairs get an empty list.
    """
    lists = (
        pairs.select("s1_id", "t_id")
        .unique()
        .group_by("s1_id")
        .agg(pl.col("t_id").sort())
        .with_columns(pl.col("t_id").list.join(","))
    )
    out = (
        pl.DataFrame({"s1_id": s1_ids})
        .join(lists, on="s1_id", how="left", maintain_order="left")
        .with_columns(pl.col("t_id").fill_null(""))
        .rename({"s1_id": header[0], "t_id": header[1]})
    )
    path.parent.mkdir(parents=True, exist_ok=True)
    out.write_csv(path, separator="\t", quote_style="never", line_terminator="\n")
