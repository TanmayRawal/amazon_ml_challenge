"""GPU top-k for the fuzzy pass (run by fuzzy.py with the Python in BER_GPU_PYTHON).

usage: fuzzy_gpu.py <dir with T.npz, S.npz> <k> <min_cos>
T: targets x trigrams, S: S1 x trigrams (both L2-normalized TF-IDF, CSR). Writes rows.npy (S row),
cols.npy (T row), vals.npy (cosine) for the k best targets of every S1 row with cosine >= min_cos.
Needs numpy, scipy and torch with CUDA; T stays on the GPU as a sparse CSR matrix and each batch
of S1 rows is multiplied as a dense block.
"""
import sys
import time
from pathlib import Path

import numpy as np
import scipy.sparse as sp
import torch


def main() -> None:
    """Load T/S from argv[1], compute the per-S1 top-k cosine targets on the GPU, save the .npy outputs."""
    d, k, min_cos = Path(sys.argv[1]), int(sys.argv[2]), float(sys.argv[3])
    t0 = time.time()
    T = sp.load_npz(d / "T.npz").tocsr().astype(np.float32)
    S = sp.load_npz(d / "S.npz").tocsr().astype(np.float32)
    dev = torch.device("cuda")
    tg = torch.sparse_csr_tensor(torch.from_numpy(T.indptr.astype(np.int32)), torch.from_numpy(T.indices.astype(np.int32)),
                                 torch.from_numpy(T.data), size=T.shape, device=dev)
    n = T.shape[0]
    k = min(k, n)
    batch = int(max(16, min(256, 1.2e9 / (n * 4))))
    rows, cols, vals = [], [], []
    with torch.no_grad():
        for lo in range(0, S.shape[0], batch):
            blk = torch.from_numpy(S[lo:lo + batch].toarray().T.copy()).to(dev)     # trigrams x b
            score = tg @ blk                                                     # targets x b
            # torch.topk over millions of rows is ~10x slower than the product itself; keep the
            # scores above min_cos, then rank those within each S1 column.
            t_idx, col = torch.nonzero(score >= min_cos, as_tuple=True)
            v = score[t_idx, col]
            order = torch.argsort(col.to(torch.float64) * 4.0 - v.to(torch.float64))
            t_idx, col, v = t_idx[order], col[order], v[order]
            counts = torch.bincount(col, minlength=blk.shape[1])
            starts = torch.cumsum(counts, 0) - counts
            rank = torch.arange(len(col), device=dev) - starts[col]
            keep = rank < k
            rows.append((col[keep] + lo).cpu().numpy().astype(np.int64))
            cols.append(t_idx[keep].cpu().numpy().astype(np.int64))
            vals.append(v[keep].cpu().numpy().astype(np.float32))
            del blk, score, t_idx, col, v, order
    np.save(d / "rows.npy", np.concatenate(rows))
    np.save(d / "cols.npy", np.concatenate(cols))
    np.save(d / "vals.npy", np.concatenate(vals))
    print(f"gpu top-k: {S.shape[0]:,} x {n:,}, batch {batch}, {sum(len(v) for v in vals):,} pairs in {time.time() - t0:.0f}s")


if __name__ == "__main__":
    main()
