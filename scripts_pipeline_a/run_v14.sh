#!/usr/bin/env bash
# V14 end to end, one stage at a time (no CPU/GPU overlap, so memory never stacks up):
#   prepare + translit (train, test; Cie/Groupe legal forms) -> V13 folds -> density matching
#   -> ranker -> matcher B -> stack L2 (+ sibling features)            [train, sealed fold 4 = local score]
#   -> transformer link: export uncertain train pairs, score them out-of-fold, link-train
#   -> test: rank, features, stack, uncertain pairs, transformer (mean of both halves) -> submit-x
# Env: WORK DATA VERSION (required); FOLDS_FROM = work dir whose train folds to reuse (else run folds);
#      XENC_FROM = dir holding xenc_h0.pt / xenc_h1.pt; BER_UPLOAD_DIR = copy of the upload files.
# Each stage is skipped when its done-marker exists in $WORK/done/, so a rerun resumes after a failure.
set -uo pipefail
ROOT="C:/Users/Lenovo/Desktop/Ml challange"
CODE="$(cd "$(dirname "$0")" && pwd -W)"
P12=python
P13=python
SITE="$ROOT/baseline_us_india_70_15_15/venv/Lib/site-packages"
CLI="$CODE/business_entity_resolution/src/xenc_cli.py"
export PYTHONIOENCODING=utf-8 PYTHONDONTWRITEBYTECODE=1 HF_HUB_OFFLINE=1 HF_HUB_DISABLE_SYMLINKS_WARNING=1 TRANSFORMERS_VERBOSITY=error
export BER_WORK_DIR="$WORK" BER_DATA_DIR="$DATA"
mkdir -p "$WORK/done" "$WORK/models/xenc"
say() { echo "[v14 $(date +%T)] $*"; }
fail() { say "FAILED: $*"; exit 1; }
stage() {   # stage <name> <command...>: run once, mark done
    local name=$1; shift
    [ -e "$WORK/done/$name" ] && { say "skip $name (done)"; return; }
    say "=== $name"
    "$@" || fail "$name"
    touch "$WORK/done/$name"
}
run() { (cd "$CODE/business_entity_resolution" && PYTHONPATH="$SITE" "$P12" src/run.py "$@"); }
py12() { PYTHONPATH="$SITE" "$P12" -c "$1"; }
x13() { env -u PYTHONPATH "$P13" "$@"; }

use_folds() {   # reuse the folds the transformer was trained against (same S1 order required)
    if [ -z "${FOLDS_FROM:-}" ]; then run folds; return; fi
    py12 "
import polars as pl, shutil
a = pl.read_parquet(r'$WORK/train/s1.parquet', columns=['s1', 'entity_id']).sort('s1')
b = pl.read_parquet(r'$FOLDS_FROM/train/s1.parquet', columns=['s1', 'entity_id']).sort('s1')
assert a.equals(b), 'S1 order differs from FOLDS_FROM: folds cannot be reused'
shutil.copy2(r'$FOLDS_FROM/train/folds.parquet', r'$WORK/train/folds.parquet')
print('folds copied from', r'$FOLDS_FROM')"
}
wait_models() {   # XENC_WAIT=1: models are trained alongside (V15 F5); wait for READY
    [ "${XENC_WAIT:-}" = 1 ] || return 0
    until [ -e "$WORK/models/xenc/READY" ] || [ -e "$WORK/models/xenc/FAILED" ]; do sleep 30; done
    [ -e "$WORK/models/xenc/READY" ] || fail "transformer training (see $WORK/f5_train.log)"
}
xenc_models() { cp "$XENC_FROM/xenc_h0.pt" "$XENC_FROM/xenc_h1.pt" "$WORK/models/xenc/"; }

say "start: work=$WORK data=$DATA version=$VERSION code=$CODE"
stage xenc_models   xenc_models
stage prepare       run prepare --split all
stage translit      run translit --split all
stage folds         use_folds
stage density       run match-density
stage rank_train    run rank-train
stage rank_tr       run rank --split train
stage feat_tr       run features --split train --variant c
stage train_b       run train-b
stage stack_tr      run stack-features --split train
stage train_c       run train-c
stage band_tr       run xenc-export --split train
wait_models
stage xenc_tr       x13 "$CLI" score "$WORK/train/xenc_band.parquet" "$WORK/models/xenc" "$WORK/train/xenc_scores.parquet" text=text_b mode=oof batch=256 maxlen=128
if [ -n "${XENC2_FROM:-}" ]; then   # B: second transformer for the link ensemble
    stage xenc2_tr  x13 "$CLI" score "$WORK/train/xenc_band.parquet" "$XENC2_FROM" "$WORK/train/xenc_scores2.parquet" text=text_b mode=oof batch=256 maxlen=128
fi
stage link_train    run link-train
stage rank_te       run rank --split test
stage feat_te       run features --split test --variant c
stage stack_te      run stack-features --split test
stage band_te       run xenc-export --split test
stage xenc_te       x13 "$CLI" score "$WORK/test/xenc_band.parquet" "$WORK/models/xenc" "$WORK/test/xenc_scores.parquet" text=text_b mode=mean batch=256 maxlen=128
if [ -n "${XENC2_FROM:-}" ]; then
    stage xenc2_te  x13 "$CLI" score "$WORK/test/xenc_band.parquet" "$XENC2_FROM" "$WORK/test/xenc_scores2.parquet" text=text_b mode=mean batch=256 maxlen=128
fi
stage submit        run submit-x --version "$VERSION"
say "ALL DONE"
