"""Cross-encoder for the V13x link (see xlink.py). Runs under a Python with CUDA torch + transformers;
reads and writes parquet with pyarrow only.

  train <band.parquet> <model_dir> [text=text_b] [halves=0,1] [batch=128] [maxlen=128] [init=<model_dir>] [lr=5e-5] [epochs=1] [model=<hf id>]
      model h fine-tunes on dev-fold pairs (fold != 4) whose target is in half h -> <model_dir>/xenc_h{h}.pt
  score <band.parquet> <model_dir> <out.parquet> [text=text_b] [mode=oof|mean] [batch=256] [maxlen=128]
      oof:  a pair whose target is in half h is scored by model 1-h (never saw that target)
      mean: mean logit of both models (test)

Model: sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2 (Apache-2.0, 118M), one logit head,
one epoch, AdamW 5e-5, bf16 autocast, length-bucketed batches.
"""
import random
import subprocess
import sys
import time
from pathlib import Path

import numpy as np
import pyarrow as pa
import pyarrow.compute as pc
import pyarrow.parquet as pq
import torch
from transformers import AutoModelForSequenceClassification, AutoTokenizer

NAME = "sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2"


def batches(lens, size, shuffle, seed=0):
    """Length-bucketed batches: sort inside chunks of 50 batches so padding stays small."""
    idx = np.arange(len(lens))
    if shuffle:
        idx = idx[np.random.default_rng(seed).permutation(len(idx))]
    out = []
    for c in range(0, len(idx), size * 50):
        chunk = idx[c:c + size * 50]
        chunk = chunk[np.argsort(lens[chunk], kind="stable")]
        out += [chunk[i:i + size] for i in range(0, len(chunk), size)]
    if shuffle:
        random.Random(seed).shuffle(out)
    return out


def load_texts(tab, text):
    """Pair texts of a pyarrow table (text_a and the chosen text column) and their summed character lengths."""
    ta, tb = tab["text_a"].to_pylist(), tab[text].to_pylist()
    return ta, tb, np.array([len(a) + len(b) for a, b in zip(ta, tb)])


def model_name(model_dir) -> str:
    """The pretrained model a model_dir was fine-tuned from (model_name.txt, written by train)."""
    f = Path(model_dir) / "model_name.txt"
    return f.read_text().strip() if f.exists() else NAME


def train(band, model_dir, text="text_b", halves="0,1", batch="128", maxlen="128", init="", lr="5e-5", epochs="1", model=""):
    """Fine-tune one cross-encoder per target half on dev-fold band pairs; saves <model_dir>/xenc_h{h}.pt.

    Arguments arrive as command-line strings (see the module docstring); init continues from an earlier model_dir."""
    batch, maxlen = int(batch), int(maxlen)
    model_id = model or (model_name(init) if init else NAME)
    tok = AutoTokenizer.from_pretrained(model_id)
    full = pq.read_table(band, columns=["fold", "thalf", "label", "text_a", text])
    Path(model_dir).mkdir(parents=True, exist_ok=True)
    (Path(model_dir) / "model_name.txt").write_text(model_id)
    for h in (int(v) for v in halves.split(",")):
        torch.manual_seed(h)
        tab = full.filter(pc.and_(pc.not_equal(full["fold"], 4), pc.equal(full["thalf"], h)))
        ta, tb, lens = load_texts(tab, text)
        y = tab["label"].to_numpy().astype(np.float32)
        model = AutoModelForSequenceClassification.from_pretrained(model_id, num_labels=1)
        if init:   # continue from an earlier model of the same half (F5 French fine-tune)
            model.load_state_dict(torch.load(Path(init) / f"xenc_h{h}.pt", map_location="cpu"))
        model = model.cuda()
        bl = [b for ep in range(int(epochs)) for b in batches(lens, batch, True, seed=h + 100 * ep)]
        steps, warm = len(bl), max(1, int(0.03 * len(bl)))
        opt = torch.optim.AdamW(model.parameters(), lr=float(lr), weight_decay=0.01)
        sched = torch.optim.lr_scheduler.LambdaLR(opt, lambda s: min(1.0, (s + 1) / warm) * max(0.0, (steps - s) / max(1, steps - warm)))
        model.train()
        t0, run = time.time(), 0.0
        print(f"half {h}: {len(y):,} pairs, {steps:,} steps", flush=True)
        for step, b in enumerate(bl):
            e = tok([ta[i] for i in b], [tb[i] for i in b], truncation=True, max_length=maxlen, padding=True, return_tensors="pt").to("cuda")
            with torch.autocast("cuda", dtype=torch.bfloat16):
                logit = model(**e).logits.squeeze(-1)
            loss = torch.nn.functional.binary_cross_entropy_with_logits(logit.float(), torch.from_numpy(y[b]).cuda())
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            opt.step(), sched.step(), opt.zero_grad(set_to_none=True)
            run = 0.98 * run + 0.02 * loss.item() if step else loss.item()
            if step % 1000 == 0 or step == steps - 1:
                el = time.time() - t0
                print(f"half {h} step {step:,}/{steps:,} loss {run:.4f} eta {el / (step + 1) * (steps - step - 1) / 60:.0f} min", flush=True)
        torch.save(model.state_dict(), Path(model_dir) / f"xenc_h{h}.pt")
        del model, opt
        torch.cuda.empty_cache()
        print(f"half {h} trained in {(time.time() - t0) / 60:.1f} min", flush=True)


def wait_gpu(need_mb=2500, max_min=180):
    """Another torch job (e5 training) may hold the 8 GB GPU: wait for free memory instead of failing."""
    t0 = time.time()
    while time.time() - t0 < max_min * 60:
        try:
            free = int(subprocess.run(["nvidia-smi", "--query-gpu=memory.free", "--format=csv,noheader,nounits"],
                                      capture_output=True, text=True).stdout.split()[0])
        except (OSError, ValueError, IndexError):
            return
        if free >= need_mb:
            return
        print(f"waiting for GPU memory ({free} MB free)", flush=True) if int(time.time() - t0) % 600 < 30 else None
        time.sleep(30)


def write_halves(tab, logits, out):
    """Audit: train link features are single-model (out-of-fold) logits, test ones the mean of both models. Keep
    both test logits so the combiner can be fed the matching single-model score (model 1-h for target half h)."""
    path = str(Path(out).with_name(Path(out).stem + "_halves.parquet"))
    pq.write_table(pa.table({"s1": tab["s1"], "t": tab["t"], "thalf": tab["thalf"],
                             "x0": pa.array(logits[0]), "x1": pa.array(logits[1])}), path)
    return path


def score(band, model_dir, out, text="text_b", mode="oof", batch="256", maxlen="128"):
    """Score band pairs with the fine-tuned cross-encoders and write s1, t, xenc (logit) to out.

    mode=oof: each pair by the model that never saw its target half; mode=mean: mean of both models (test), with
    the per-model logits also saved by write_halves."""
    batch, maxlen = int(batch), int(maxlen)
    wait_gpu()
    model_id = model_name(model_dir)
    tok = AutoTokenizer.from_pretrained(model_id)
    tab = pq.read_table(band, columns=["s1", "t", "thalf", "text_a", text])
    ta, tb, lens = load_texts(tab, text)
    half = tab["thalf"].to_numpy()
    logits = np.zeros((2, tab.num_rows), dtype=np.float32)
    t0 = time.time()
    for h in (0, 1):
        rows = np.where(half != h)[0] if mode == "oof" else np.arange(tab.num_rows)  # model h never saw half h
        if len(rows) == 0:
            continue
        model = AutoModelForSequenceClassification.from_pretrained(model_id, num_labels=1)
        model.load_state_dict(torch.load(Path(model_dir) / f"xenc_h{h}.pt", map_location="cpu"))
        model = model.cuda().eval()
        with torch.no_grad():
            for i, b in enumerate(batches(lens[rows], batch, False)):
                r = rows[b]
                e = tok([ta[j] for j in r], [tb[j] for j in r], truncation=True, max_length=maxlen, padding=True, return_tensors="pt").to("cuda")
                with torch.autocast("cuda", dtype=torch.bfloat16):
                    logits[h, r] = model(**e).logits.squeeze(-1).float().cpu().numpy()
                if i % 1000 == 0:
                    print(f"model {h}: {i * batch:,}/{len(rows):,} ({time.time() - t0:.0f}s)", flush=True)
        del model
        torch.cuda.empty_cache()
    xenc = logits.sum(0) if mode == "oof" else logits.mean(0)   # oof: exactly one model scored each row
    pq.write_table(pa.table({"s1": tab["s1"], "t": tab["t"], "xenc": pa.array(xenc)}), out)
    if mode != "oof":
        write_halves(tab, logits, out)
    print(f"scored {tab.num_rows:,} pairs ({mode}) in {(time.time() - t0) / 60:.1f} min -> {out}", flush=True)


if __name__ == "__main__":
    cmd, *rest = sys.argv[1:]
    pos = [a for a in rest if "=" not in a]
    kw = dict(a.split("=", 1) for a in rest if "=" in a)
    {"train": train, "score": score}[cmd](*pos, **kw)
