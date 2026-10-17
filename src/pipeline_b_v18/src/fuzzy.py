"""v12 fuzzy blocking pass: char-trigram TF-IDF nearest targets per S1 (name + address).

The key-based passes need an exact shared token, so a typo in a common-word name ("Rexe, Inc"
for "Reex, Inc"), a merged word or a changed street word can leave a true copy with no shared
key. Cosine similarity over character trigrams of "concat core numbers address" tolerates all
of these. On a 57k-S1 US dev sample the top 50 by cosine found 58% of the true pairs the v11
union + ranker missed (recall 0.9776 -> 0.9906 together).

Per country, the vectorizer is fitted on that country's targets of the split (label-free test
statistics are allowed); trigrams in more than fz_max_df of the targets are dropped. The top-k
search is a sparse x sparse product: on a CUDA GPU through fuzzy_gpu.py, run with the Python in
BER_GPU_PYTHON (needs numpy, scipy, torch), otherwise on the CPU with sparse_dot_topn (same
result, several hours for train). Output: <split>/fuzzy/part_XXX.parquet with s1, t, fz_cos,
fz_rank, one file per ranker chunk (union_chunk reads them).
"""
from __future__ import annotations

import os
import subprocess
import sys
import tempfile
import time
from pathlib import Path

import numpy as np
import polars as pl
from scipy import sparse
from sklearn.feature_extraction.text import TfidfVectorizer

from config import Config
from prepare import load

_COLS = ["country", "core", "concat", "addr_nums", "addr_toks"]


def _text(df: pl.DataFrame) -> list[str]:
    """Per record, the text "concat core numbers address" fed to the trigram vectorizer."""
    return df.select(pl.concat_str(["concat", "core", "addr_nums", "addr_toks"], separator=" ")).to_series().to_list()


def _topk_cpu(S, T, k: int, min_cos: float, log) -> tuple:
    """Top-k targets per S1 row by cosine (>= min_cos) on the CPU with sparse_dot_topn.

    Returns (S row indices, T row indices, cosines).
    """
    from sparse_dot_topn import sp_matmul_topn
    tt = T.T.tocsr()
    rows, cols, vals = [], [], []
    for lo in range(0, S.shape[0], 5000):
        c = sp_matmul_topn(S[lo:lo + 5000], tt, top_n=k, threshold=min_cos, n_threads=os.cpu_count()).tocoo()
        rows.append(c.row + lo)
        cols.append(c.col)
        vals.append(c.data)
        if lo % 100_000 == 0:
            log(f"    cpu top-k {lo:,}/{S.shape[0]:,}")
    return np.concatenate(rows), np.concatenate(cols), np.concatenate(vals).astype(np.float32)


def _topk_gpu(S, T, k: int, min_cos: float, py: str, log) -> tuple:
    """Same as _topk_cpu but runs fuzzy_gpu.py with interpreter `py` via temporary .npz/.npy files."""
    script = Path(__file__).with_name("fuzzy_gpu.py")
    with tempfile.TemporaryDirectory(dir=os.environ.get("BER_TMP")) as tmp:
        sparse.save_npz(Path(tmp) / "T.npz", T, compressed=False)
        sparse.save_npz(Path(tmp) / "S.npz", S, compressed=False)
        # A different interpreter: drop the venv's launcher variables or it loads this venv's stdlib.
        env = {key: val for key, val in os.environ.items()
               if key not in ("__PYVENV_LAUNCHER__", "PYTHONHOME", "PYTHONPATH", "VIRTUAL_ENV")}
        proc = subprocess.run([py, str(script), tmp, str(k), str(min_cos)], capture_output=True, text=True, env=env)
        if proc.returncode != 0:
            raise RuntimeError(f"fuzzy_gpu.py failed:\n{proc.stdout}\n{proc.stderr}")
        for line in proc.stdout.strip().splitlines()[-2:]:
            log(f"    {line}")
        return (np.load(Path(tmp) / "rows.npy"), np.load(Path(tmp) / "cols.npy"), np.load(Path(tmp) / "vals.npy"))


def build(cfg: Config, split: str, log=print) -> None:
    """Run the fuzzy pass for a split, country by country, and write <split>/fuzzy/part_*.parquet.

    Each part holds the (s1, t, fz_cos, fz_rank) pairs of one ranker chunk of cfg.rk_chunk S1 rows.
    """
    t0 = time.time()
    s1 = load(cfg, split, "s1", ["s1"] + _COLS).sort("s1")
    tg = load(cfg, split, "tgt", ["t"] + _COLS).sort("t")
    gpu_py = os.environ.get("BER_GPU_PYTHON")
    parts = []
    for country in sorted(s1["country"].unique().to_list()):
        sc, tc = s1.filter(pl.col("country") == country), tg.filter(pl.col("country") == country)
        if sc.height == 0 or tc.height == 0:
            continue
        vec = TfidfVectorizer(analyzer="char_wb", ngram_range=(3, 3), min_df=2, max_df=cfg.fz_max_df,
                              sublinear_tf=True, dtype=np.float32)
        T = vec.fit_transform(_text(tc)).tocsr()
        S = vec.transform(_text(sc)).tocsr()
        log(f"[fuzzy] {split}/{country}: {sc.height:,} S1 x {tc.height:,} targets, vocab {len(vec.vocabulary_):,}, "
            f"nnz {T.nnz:,} ({time.time() - t0:.0f}s)")
        if gpu_py:
            r, c, v = _topk_gpu(S, T, cfg.fz_top, cfg.fz_min_cos, gpu_py, log)
        else:
            r, c, v = _topk_cpu(S, T, cfg.fz_top, cfg.fz_min_cos, log)
        parts.append(pl.DataFrame({"s1": sc["s1"].to_numpy()[r], "t": tc["t"].to_numpy()[c], "fz_cos": v}))
        log(f"[fuzzy] {split}/{country}: {len(v):,} pairs ({len(v) / sc.height:.1f}/S1) ({time.time() - t0:.0f}s)")
        del T, S, vec
    df = (pl.concat(parts).with_columns(fz_rank=pl.col("fz_cos").rank("ordinal", descending=True).over("s1")
                                        .cast(pl.Float32))
          .with_columns(chunk=(pl.col("s1") // cfg.rk_chunk).cast(pl.Int64)))
    out_dir = cfg.split_dir(split) / "fuzzy"
    out_dir.mkdir(exist_ok=True)
    for old in out_dir.glob("part_*.parquet"):
        old.unlink()
    by_chunk = df.partition_by("chunk", as_dict=True, include_key=False)
    n_chunks = (s1.height + cfg.rk_chunk - 1) // cfg.rk_chunk
    empty = df.drop("chunk").clear()
    for i in range(n_chunks):
        by_chunk.get((i,), empty).sort("s1", "fz_rank").write_parquet(out_dir / f"part_{i:03d}.parquet")
    log(f"[fuzzy] {split}: {df.height:,} pairs in {n_chunks} parts ({time.time() - t0:.0f}s)")


def chunk_pairs(fz_dir: Path, size: int, lo: int, hi: int) -> pl.DataFrame:
    """Fuzzy pairs of S1 rows [lo, hi); parts hold ranker chunks of `size` S1 rows."""
    out = []
    for i in range(lo // size, (hi - 1) // size + 1):
        f = fz_dir / f"part_{i:03d}.parquet"
        if f.exists():
            out.append(pl.read_parquet(f).filter(pl.col("s1").is_between(lo, hi - 1)))
    return pl.concat(out) if out else pl.DataFrame(schema={"s1": pl.UInt32, "t": pl.UInt32, "fz_cos": pl.Float32,
                                                           "fz_rank": pl.Float32})


if __name__ == "__main__":
    sys.exit("run through run.py fuzzy --split <split>")
