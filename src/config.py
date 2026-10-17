"""Paths and tunable parameters for the pipeline, kept in one place."""
from __future__ import annotations

import hashlib
import json
import os
from dataclasses import asdict, dataclass
from pathlib import Path


@dataclass
class Config:
    data_dir: Path
    work_dir: Path
    n_jobs: int = max(1, (os.cpu_count() or 4) - 4)
    seed: int = 2026

    # Validation: folds over Source-1 entities; the last fold is sealed for stage gates.
    n_folds: int = 5
    sealed_fold: int = 4

    # Plan A blocking (all DF limits are counts of targets inside one country).
    exact_block_max: int = 300   # skip exact-name blocks larger than this
    name_df_max: int = 500       # a name token is "rare" when at most this many targets carry it
    name_topk: int = 25          # rare-token pass: best targets kept per S1
    addr_nums_per_rec: int = 2   # address key = (house number, street-token prefix)
    addr_toks_per_rec: int = 6
    addr_key_df_max: int = 300
    addr_topk: int = 25
    concat_len: int = 6          # concatenated-name prefix key (domains, merged words)
    concat_df_max: int = 300
    cand_k: int = 50             # final candidate list length per S1
    s1_chunk: int = 200_000      # S1 rows per blocking chunk (bounds peak memory)
    feat_chunk: int = 2_000_000  # candidate pairs per feature chunk

    # Stage 2 union + learned ranker (supervised meta-blocking). Caps chosen on a
    # 20k-entity sample: raw union recall 0.967 (Indic 0.847) at ~400 pairs/S1.
    rk_exact_cap: int = 1000
    rk_name_df_max: int = 1000
    rk_addr_nums: int = 3
    rk_addr_toks: int = 10
    rk_addr_cap: int = 300
    rk_a2_df_max: int = 300      # rare address word pass
    rk_concat_cap: int = 300
    rk_sk_full_df_max: int = 1000  # Stage 5 cross-script pass: exact full-skeleton key cap
    rk_sk_pair_df_max: int = 500   # Stage 5 cross-script pass: skeleton word-pair key cap
    rk_np_cap: int = 500           # V12 name word-pair pass: targets per pair key
    rk_dm_df_max: int = 2000       # V12 website 4-gram pass
    rk_ap_cap: int = 300           # V13 address word-pair pass
    rk_fz_cap: int = 300           # V13 typo-key pass
    rk_wn_cap: int = 50            # V17 name word x house number passes
    rk_chunk: int = 7_500        # S1 rows per union chunk (V13 union ~1,150 pairs/S1; 7.5k from v17 so a GPU job fits beside it)
    rk_keep: int = 60            # ranked candidates stored per S1
    rk_cand_k: int = 60          # ranked candidates passed to the matcher (40 up to v7; 50 from v8; 60 from v17,
                                 # where precision is high enough to afford harder candidates)
    rk_neg_rate: float = 0.02    # negative sampling rate for ranker training
    rk_sample_chunks: int = 80   # union chunks used to collect ranker training rows (~600k S1)

    # Decision: odds multiplier for the stacked models' final policy (decide.prior_shift).
    # Test holds ~41-48% ownerless records vs ~26% in train; chosen on distractor injection
    # over dev-fold entities (stress --variant c --pool dev).
    decision_shift: float = 0.6

    def split_dir(self, split: str) -> Path:
        """Return (creating if needed) the working directory for one data split."""
        path =self.work_dir / split
        path.mkdir(parents=True, exist_ok=True)
        return path

    def fingerprint(self) -> str:
        """Short hash of every setting except paths, recorded in run manifests."""
        items = {k: v for k, v in asdict(self).items() if k not in ("data_dir", "work_dir")}
        return hashlib.sha256(json.dumps(items, sort_keys=True).encode()).hexdigest()[:12]


def from_args(data_dir: str | None, work_dir: str | None, **overrides) -> Config:
    """Resolve paths from CLI flags, then environment variables, then defaults."""
    data = Path(data_dir or os.environ.get("BER_DATA_DIR", "dataset")).resolve()
    work = Path(work_dir or os.environ.get("BER_WORK_DIR", "work")).resolve()
    work.mkdir(parents=True, exist_ok=True)
    cfg = Config(data_dir=data, work_dir=work)
    for key, value in overrides.items():
        if value is not None:
            setattr(cfg, key, value)
    return cfg
