#!/usr/bin/env bash
# A bigger cross-encoder for the link ensemble (V18): intfloat/multilingual-e5-base (MIT, 278M parameters),
# fine-tuned one epoch on V16's uncertain train pairs with the usual target-half split (model h learns targets
# of half h, dev folds only). The text model does not depend on the stack, so it can score V17's pairs, which
# it writes into work_v18, never into work_v17 while V17 runs. GPU scheduling: training starts after V16
# (GPU free); train scoring waits for V17's own transformer stages; test scoring starts once V17's uncertain
# test pairs exist. Log: work_v18/run_e5.log. Steps resume from done markers.
set -uo pipefail
ROOT="C:/Users/Lenovo/Desktop/Ml challange"
CODE="$(cd "$(dirname "$0")" && pwd -W)"
V16="${V16:-$ROOT/work_v16}"
V17="${V17:-$ROOT/work_v17}"
W="${W:-$ROOT/work_v18}"
M="$W/models/xenc_e5"
MODEL="${E5_MODEL:-intfloat/multilingual-e5-base}"
P13=python
CLI="$CODE/business_entity_resolution/src/xenc_cli.py"
export PYTHONIOENCODING=utf-8 PYTHONDONTWRITEBYTECODE=1 HF_HUB_OFFLINE=1 HF_HUB_DISABLE_SYMLINKS_WARNING=1 TRANSFORMERS_VERBOSITY=error
say() { echo "[e5 $(date +%T)] $*"; }
step() { local name=$1; shift; [ -e "$W/done/$name" ] && { say "skip $name (done)"; return; }; say "=== $name"; "$@" || { say "FAILED: $name"; exit 1; }; touch "$W/done/$name"; }
x13() { env -u PYTHONPATH "$P13" "$@"; }
wait_for() { until [ -e "$1" ]; do grep -qs "FAILED" "$2" && { say "FAILED upstream ($2)"; exit 1; }; sleep 60; done; }

mkdir -p "$W/train" "$W/test" "$W/done" "$M"
wait_for "$V16/done/package_fr" "$V16/run_v16.log"
step e5_train x13 "$CLI" train "$V16/train/xenc_band.parquet" "$M" text=text_b halves=0,1 batch=32 maxlen=128 epochs="${E5_EPOCHS:-1}" lr=3e-5 model="$MODEL"
wait_for "$V17/done/xenc2_tr" "$V17/run_v17.log"
step e5_tr    x13 "$CLI" score "$V17/train/xenc_band.parquet" "$M" "$W/train/xenc_scores3.parquet" text=text_b mode=oof batch=128 maxlen=128
wait_for "$V17/done/band_te" "$V17/run_v17.log"
step e5_te    x13 "$CLI" score "$V17/test/xenc_band.parquet" "$M" "$W/test/xenc_scores3.parquet" text=text_b mode=mean batch=128 maxlen=128
say "ALL DONE"
