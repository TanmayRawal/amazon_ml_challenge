"""Plan A candidate generation: four lexical indexes inside each country, chunked over S1.

Every key includes the country string, so countries never mix (0 of 7.64M true pairs
cross countries in train); country values are read from the data, never hard-coded.

  exact   identical order-insensitive core name; blocks with more than
          exact_block_max targets are skipped (very common names need the address)
  rare    shared rare name token (target DF <= name_df_max); keep the name_topk
          targets with the largest summed IDF per S1
  addr    shared (house number, first 5 letters of an address word); keys with more
          than addr_key_df_max targets are skipped; keep addr_topk per S1
  concat  equal first concat_len characters of the concatenated name or website label

The union is ranked by a cheap score and cut to cand_k per S1. That final set is
exactly what the matcher scores and what candidate_pairs.tsv reports.
"""
from __future__ import annotations

import time

import polars as pl

from config import Config
from normalize import STRUCTURAL
from prepare import load

SEP = "\x1f"
STRUCT_LIST = sorted(STRUCTURAL)
PASS_COLS = ["f_exact", "rare_idf", "rare_n", "addr_n", "f_concat"]


def _key(*exprs: pl.Expr) -> pl.Expr:
    """Hash the given string expressions (joined with a separator) into one blocking key."""
    return pl.concat_str(list(exprs), separator=SEP).hash(seed=17)


def _exact_keys(df: pl.DataFrame, idc: str) -> pl.DataFrame:
    """Exact-name keys: (country, order-insensitive core name) per record with a non-empty core.

    Returns a frame with the id column `idc` and a hashed `key`.
    """
    return df.filter(pl.col("core_key") != "").select(idc, key=_key(pl.col("country"), pl.col("core_key")))


def _name_tokens(df: pl.DataFrame, idc: str) -> pl.DataFrame:
    """Distinct non-empty core-name tokens per record, with its country."""
    return (
        df.select(idc, "country", tok=pl.col("core").str.split(" "))
        .explode("tok")
        .filter(pl.col("tok").is_not_null() & (pl.col("tok") != ""))
        .unique([idc, "tok"])
    )


def _addr_keys(df: pl.DataFrame, idc: str, n_nums: int, n_toks: int) -> pl.DataFrame:
    """Address keys: (country, house number, first 5 letters of an address word).

    Uses the first `n_nums` numbers and the first `n_toks` non-structural words (>= 3 letters)
    of each non-blank address. Returns distinct (`idc`, `key`) rows.
    """
    toks = (
        pl.col("addr_toks").str.split(" ")
        .list.eval(pl.element().filter((pl.element().str.len_chars() >= 3) & ~pl.element().is_in(STRUCT_LIST)))
        .list.head(n_toks)
    )
    return (
        df.filter(~pl.col("addr_blank"))
        .select(idc, "country", num=pl.col("addr_nums").str.split(" ").list.head(n_nums), tok=toks)
        .explode("num")
        .explode("tok")
        .filter(pl.col("num").is_not_null() & (pl.col("num") != "") & pl.col("tok").is_not_null())
        .select(idc, key=_key(pl.col("country"), pl.col("num"), pl.col("tok").str.slice(0, 5)))
        .unique()
    )


def _addr_words(df: pl.DataFrame, idc: str) -> pl.DataFrame:
    """Distinct non-structural address words (>= 3 letters) per record."""
    return (
        df.filter(~pl.col("addr_blank"))
        .select(idc, "country", tok=pl.col("addr_toks").str.split(" "))
        .explode("tok", empty_as_null=True)
        .filter(pl.col("tok").is_not_null() & (pl.col("tok").str.len_chars() >= 3) & ~pl.col("tok").is_in(STRUCT_LIST))
        .unique([idc, "tok"])
    )


def _concat_keys(df: pl.DataFrame, idc: str, cols: list[str], length: int) -> pl.DataFrame:
    """Prefix keys: (country, first `length` chars) of each column in `cols` with >= 5 chars."""
    parts = [
        df.filter(pl.col(c).str.len_chars() >= 5).select(idc, key=_key(pl.col("country"), pl.col(c).str.slice(0, length)))
        for c in cols
    ]
    return pl.concat(parts).unique()


def _cap_blocks(t_keys: pl.DataFrame, max_size: int) -> pl.DataFrame:
    """Drop target keys whose block holds more than `max_size` targets."""
    small =t_keys.group_by("key").agg(n=pl.len()).filter(pl.col("n") <= max_size).select("key")
    return t_keys.join(small, on="key", how="semi")


def _top_per_s1(df: pl.DataFrame, score_col: str, k: int) -> pl.DataFrame:
    """Deterministic top-k per S1 (ties broken by target index)."""
    return (
        df.sort(["s1", score_col, "t"], descending=[False, True, False])
        .group_by("s1", maintain_order=True)
        .head(k)
    )


def cheap_score() -> pl.Expr:
    """Hand-weighted score over the blocking pass columns, used to cut the union to cand_k."""
    return (
        2.0 * pl.col("f_exact")
        + pl.col("rare_idf") / 10.0
        + 0.8 * pl.min_horizontal(pl.col("addr_n"), pl.lit(3))
        + 1.5 * pl.col("f_concat")
    ).cast(pl.Float32)


def build_candidates(cfg: Config, split: str, log=print) -> None:
    """Run the four blocking passes for a split and write top cand_k candidates per S1.

    Processes S1 in chunks of cfg.s1_chunk and writes <split>/cand_a/part_*.parquet with
    columns s1, t, the pass columns (PASS_COLS) and the cheap score.
    """
    t0 =time.time()
    s1 = load(cfg, split, "s1", ["s1", "country", "core", "core_key", "concat", "addr_nums", "addr_toks", "addr_blank"])
    tg = load(cfg, split, "tgt", ["t", "country", "core", "core_key", "concat", "domain", "addr_nums", "addr_toks", "addr_blank"])

    s_exact = _exact_keys(s1, "s1")
    t_exact = _cap_blocks(_exact_keys(tg, "t"), cfg.exact_block_max)

    t_tok = _name_tokens(tg, "t")
    n_c = tg.group_by("country").agg(N=pl.len())
    rare = (
        t_tok.group_by("country", "tok").agg(df=pl.len())
        .filter(pl.col("df") <= cfg.name_df_max)
        .join(n_c, on="country")
        .with_columns(idf=(pl.col("N") / pl.col("df")).log().cast(pl.Float32), key=_key(pl.col("country"), pl.col("tok")))
        .select("country", "tok", "idf", "key")
    )
    t_rare = t_tok.join(rare, on=["country", "tok"]).select("t", "key")
    s_rare = _name_tokens(s1, "s1").join(rare, on=["country", "tok"]).select("s1", "key", "idf")
    del t_tok

    s_addr = _addr_keys(s1, "s1", cfg.addr_nums_per_rec, cfg.addr_toks_per_rec)
    t_addr = _cap_blocks(_addr_keys(tg, "t", cfg.addr_nums_per_rec, cfg.addr_toks_per_rec), cfg.addr_key_df_max)

    s_cc = _concat_keys(s1, "s1", ["concat"], cfg.concat_len)
    t_cc = _cap_blocks(_concat_keys(tg, "t", ["concat", "domain"], cfg.concat_len), cfg.concat_df_max)
    del tg

    # Targets only need the keys some S1 actually uses.
    t_exact = t_exact.join(s_exact.select("key").unique(), on="key", how="semi")
    t_rare = t_rare.join(s_rare.select("key").unique(), on="key", how="semi")
    t_addr = t_addr.join(s_addr.select("key").unique(), on="key", how="semi")
    t_cc = t_cc.join(s_cc.select("key").unique(), on="key", how="semi")
    log(
        f"[{split}] index rows  exact S1={s_exact.height:,} T={t_exact.height:,} | rare S1={s_rare.height:,} "
        f"T={t_rare.height:,} | addr S1={s_addr.height:,} T={t_addr.height:,} | concat S1={s_cc.height:,} "
        f"T={t_cc.height:,}  ({time.time() - t0:.0f}s)"
    )

    out_dir = cfg.split_dir(split) / "cand_a"
    out_dir.mkdir(exist_ok=True)
    for old in out_dir.glob("part_*.parquet"):
        old.unlink()

    n = s1.height
    total = 0
    for i, lo in enumerate(range(0, n, cfg.s1_chunk)):
        hi = min(n, lo + cfg.s1_chunk)
        sel = pl.col("s1").is_between(lo, hi - 1)
        e = (s_exact.filter(sel).join(t_exact, on="key").select("s1", "t").unique()
             .with_columns(f_exact=pl.lit(1, pl.UInt8)))
        r = (s_rare.filter(sel).join(t_rare, on="key").group_by("s1", "t")
             .agg(rare_idf=pl.col("idf").sum(), rare_n=pl.len().cast(pl.UInt8)))
        r = _top_per_s1(r, "rare_idf", cfg.name_topk)
        a = (s_addr.filter(sel).join(t_addr, on="key").group_by("s1", "t")
             .agg(addr_n=pl.len().cast(pl.UInt8)))
        a = _top_per_s1(a, "addr_n", cfg.addr_topk)
        c = (s_cc.filter(sel).join(t_cc, on="key").select("s1", "t").unique()
             .with_columns(f_concat=pl.lit(1, pl.UInt8)))
        u = (
            pl.concat([e, r, a, c], how="diagonal_relaxed")
            .group_by("s1", "t")
            .agg(pl.col(PASS_COLS).max())
            .with_columns(pl.col(PASS_COLS).fill_null(0))
            .with_columns(pl.col("rare_idf").cast(pl.Float32), cheap=cheap_score())
        )
        u = _top_per_s1(u, "cheap", cfg.cand_k)
        u.write_parquet(out_dir / f"part_{i:03d}.parquet")
        total += u.height
        log(f"[{split}] chunk {i}: S1 {lo:,}-{hi - 1:,} -> {u.height:,} candidates ({time.time() - t0:.0f}s)")
    log(f"[{split}] candidates total {total:,} ({total / max(n, 1):.1f} per S1)")


def load_candidates(cfg: Config, split: str, columns: list[str] | None = None) -> pl.DataFrame:
    """Concatenate the Plan A candidate parts of a split (optionally only `columns`)."""
    parts =sorted((cfg.split_dir(split) / "cand_a").glob("part_*.parquet"))
    return pl.concat([pl.read_parquet(p, columns=columns) for p in parts])
