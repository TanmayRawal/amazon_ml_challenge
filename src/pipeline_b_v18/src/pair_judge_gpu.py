"""GPU side of the pair judge (run by pair_judge.py with the Python in BER_GPU_PYTHON).

  train <pairs.parquet> <out_dir> <folds e.g. 0,1> <model name>   fine-tune on pairs of those folds
  score <pairs.parquet> <model_dir> <out.parquet>                  pj for every pair

Pairs files carry a (S1 "name | address"), b (target "name | address") and, for training, label.
Needs torch (CUDA), transformers and polars.
"""
import os
import sys
import time
from pathlib import Path

import numpy as np
import polars as pl
import torch
from transformers import AutoModelForSequenceClassification, AutoTokenizer

MAX_LEN = 128
SEED = int(os.environ.get("PJ_SEED", 2026))   # a second model set (pair_judge tag "2") uses another seed
CKPT_EVERY = int(os.environ.get("PJ_CKPT_EVERY", 500))           # training steps between checkpoints
SCORE_CHUNK = int(os.environ.get("PJ_SCORE_CHUNK", 1_000_000))   # pairs per saved scoring chunk


def _batches(n: int, size: int, lengths: np.ndarray, shuffle: bool, rng):
    """Length-bucketed batches: sort by length in blocks of 100 batches, shuffle batch order."""
    order = rng.permutation(n) if shuffle else np.arange(n)
    blocks = []
    for lo in range(0, n, size * 100):
        blk = order[lo:lo + size * 100]
        blk = blk[np.argsort(lengths[blk], kind="stable")]
        blocks += [blk[i:i + size] for i in range(0, len(blk), size)]
    if shuffle:
        rng.shuffle(blocks)
    return blocks


def train(pairs: str, out_dir: str, folds: str, model_name: str) -> None:
    """Fine-tune `model_name` as a 1-logit cross-encoder for one epoch on pairs of the given folds.

    Args:
        pairs: parquet with a, b, label, fold; folds: comma-separated fold ids, e.g. "0,1".
        out_dir: where the model and tokenizer are saved; checkpoints allow resuming after a crash.
    """
    t0 = time.time()
    torch.manual_seed(SEED)
    rng = np.random.default_rng(SEED)
    fl = [int(f) for f in folds.split(",")]
    df = pl.read_parquet(pairs).filter(pl.col("fold").is_in(fl))
    a, b, y = df["a"].to_list(), df["b"].to_list(), df["label"].to_numpy().astype(np.float32)
    tok = AutoTokenizer.from_pretrained(model_name)
    model = AutoModelForSequenceClassification.from_pretrained(model_name, num_labels=1).cuda()
    # The 250k-word multilingual vocabulary is 96M of the 118M parameters; its gradient and AdamW
    # state alone need ~1.2 GB, which does not fit a 6 GB laptop GPU next to the activations.
    # Freeze it and fine-tune the transformer layers and the head.
    model.get_input_embeddings().weight.requires_grad_(False)
    params = [p for p in model.parameters() if p.requires_grad]
    lengths = np.array([len(x) + len(z) for x, z in zip(a, b)])
    # batch 64 keeps the peak near 2 GB: other programs share the laptop GPU and a peak near
    # 4 GB of 6 failed with CUDA out-of-memory errors outside PyTorch's allocator
    bs, lr = 64, 4e-5
    batches = _batches(len(y), bs, lengths, True, rng)
    opt = torch.optim.AdamW(params, lr=lr, weight_decay=0.01)
    total, warm = len(batches), max(1, len(batches) // 20)
    sched = torch.optim.lr_scheduler.LambdaLR(opt, lambda s: min((s + 1) / warm, max(0.0, (total - s) / (total - warm))))
    scaler = torch.amp.GradScaler("cuda")
    loss_fn = torch.nn.BCEWithLogitsLoss()
    # The laptop GPU is shared with other programs and fails now and then with a CUDA error that
    # breaks the process's CUDA context. Checkpoint every CKPT_EVERY steps; pair_judge.py restarts
    # the process, which resumes here (the batch order is fixed by SEED).
    ckpt = Path(str(out_dir) + ".ckpt.pt")
    trainable = {n for n, p in model.named_parameters() if p.requires_grad}   # frozen vocabulary not saved
    start, run = 0, 0.0
    if ckpt.exists():
        state = torch.load(ckpt, map_location="cpu", weights_only=False)
        model.load_state_dict(state["model"], strict=False)
        opt.load_state_dict(state["opt"])
        sched.load_state_dict(state["sched"])
        scaler.load_state_dict(state["scaler"])
        start, run = state["step"] + 1, state["run"]
        print(f"train folds {folds}: resumed at step {start:,}/{total:,}", flush=True)
    model.train()
    for step in range(start, total):
        idx = batches[step]
        enc = tok([a[i] for i in idx], [b[i] for i in idx], truncation=True, max_length=MAX_LEN, padding=True,
                  return_tensors="pt").to("cuda")
        with torch.autocast("cuda", dtype=torch.float16):
            logits = model(**enc).logits.squeeze(-1)
        loss = loss_fn(logits.float(), torch.from_numpy(y[idx]).cuda())
        opt.zero_grad(set_to_none=True)
        scaler.scale(loss).backward()
        scaler.unscale_(opt)
        torch.nn.utils.clip_grad_norm_(params, 1.0)
        scaler.step(opt)
        scaler.update()
        sched.step()
        # On this Windows (WDDM) laptop GPU, long runs of queued asynchronous work failed with "CUDA
        # error: out of memory" every few hundred steps at ~2.5 GB of 6; draining the queue once
        # per step avoided it.
        torch.cuda.synchronize()
        run = 0.98 * run + 0.02 * loss.item() if step else loss.item()
        if step % CKPT_EVERY == 0 or step == total - 1:
            tmp = Path(str(ckpt) + ".tmp")
            weights = {k: v for k, v in model.state_dict().items() if k in trainable}
            torch.save({"step": step, "run": run, "model": weights, "opt": opt.state_dict(),
                        "sched": sched.state_dict(), "scaler": scaler.state_dict()}, tmp)
            os.replace(tmp, ckpt)
            peak = torch.cuda.max_memory_allocated() / 2**30
            print(f"train folds {folds}: step {step:,}/{total:,} loss {run:.4f} peak GPU {peak:.2f} GB "
                  f"({time.time() - t0:.0f}s)", flush=True)
    model.save_pretrained(out_dir)
    tok.save_pretrained(out_dir)
    ckpt.unlink(missing_ok=True)
    print(f"train folds {folds}: {len(y):,} pairs, 1 epoch, saved {out_dir} ({time.time() - t0:.0f}s)", flush=True)


@torch.no_grad()
def _predict(model_dir: str, a: list, b: list) -> np.ndarray:
    """Sigmoid match probability of the model in `model_dir` for each (a[i], b[i]) text pair."""
    tok = AutoTokenizer.from_pretrained(model_dir)
    model = AutoModelForSequenceClassification.from_pretrained(model_dir).cuda().eval()
    lengths = np.array([len(x) + len(z) for x, z in zip(a, b)])
    out = np.empty(len(a), dtype=np.float32)
    t0 = time.time()
    for k, idx in enumerate(_batches(len(a), 256, lengths, False, None)):
        enc = tok([a[i] for i in idx], [b[i] for i in idx], truncation=True, max_length=MAX_LEN, padding=True,
                  return_tensors="pt").to("cuda")
        with torch.autocast("cuda", dtype=torch.float16):
            logits = model(**enc).logits.squeeze(-1).float()
        out[idx] = torch.sigmoid(logits).cpu().numpy()
        if k % 2000 == 0:
            print(f"score {model_dir}: {k * 256:,}/{len(a):,} ({time.time() - t0:.0f}s)", flush=True)
    return out


def score(pairs: str, model_dir: str, out: str) -> None:
    """pj for every pair with one model, in SCORE_CHUNK slices saved next to `out`, so a restarted
    process skips finished ones."""
    df = pl.read_parquet(pairs)
    parts = Path(str(out) + ".parts")
    parts.mkdir(exist_ok=True)
    t0, chunks = time.time(), []
    for lo in range(0, df.height, SCORE_CHUNK):
        path = parts / f"pj_{lo // SCORE_CHUNK:04d}.npy"
        if not path.exists():
            sl = df.slice(lo, SCORE_CHUNK)
            np.save(path, _predict(model_dir, sl["a"].to_list(), sl["b"].to_list()))
            print(f"score {Path(model_dir).name}: {min(lo + SCORE_CHUNK, df.height):,}/{df.height:,} "
                  f"({time.time() - t0:.0f}s)", flush=True)
        chunks.append(np.load(path))
    pj = np.concatenate(chunks) if chunks else np.empty(0, dtype=np.float32)
    keep = [c for c in ("s1", "t", "fold") if c in df.columns]
    df.select(keep).with_columns(pj=pl.Series(pj)).write_parquet(out)
    for p in parts.glob("*.npy"):
        p.unlink()
    parts.rmdir()
    print(f"scored {df.height:,} pairs -> {out}", flush=True)


if __name__ == "__main__":
    cmd = sys.argv[1]
    if cmd == "train":
        train(*sys.argv[2:6])
    elif cmd == "score":
        score(*sys.argv[2:5])
    else:
        sys.exit(__doc__)
