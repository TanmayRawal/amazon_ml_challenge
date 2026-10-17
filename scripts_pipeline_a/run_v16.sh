#!/usr/bin/env bash
# V16 = V15 with a stronger transformer (U2): trained from scratch for 2 epochs on V15's uncertain train pairs
# (same target-half split: each half's model never sees the other half's targets), then the French F5
# fine-tune on top of it (pseudo pairs from V15's test scores). US/India pairs are scored, and the link trained,
# with U2 itself; only French pairs use the F5 model, which costs US/India a little (v15: sealed link gain
# +0.0029 with it vs +0.0036 without). Ranking, features, matcher B and the stack are V15's and are not rebuilt.
# Submissions: v16 (transformer re-scores US/India) and v16fr (France too, F5 model). Needs V15 finished.
# Log: work_v16/run_v16.log (+ u2_train.log, f5_train.log). Each step is skipped when its marker exists.
set -uo pipefail
ROOT="C:/Users/Lenovo/Desktop/Ml challange"
CODE="$(cd "$(dirname "$0")" && pwd -W)"
V15="${V15:-$ROOT/work_v15}"
W="${W:-$ROOT/work_v16}"
DATA="${DATA:-$ROOT/6ab10eb3b23ba_student_resource/student_resource/dataset}"
export BER_UPLOAD_DIR="${BER_UPLOAD_DIR-$ROOT/submissions}"
P12=python
P13=python
SITE="$ROOT/baseline_us_india_70_15_15/venv/Lib/site-packages"
CLI="$CODE/business_entity_resolution/src/xenc_cli.py"
export PYTHONIOENCODING=utf-8 PYTHONDONTWRITEBYTECODE=1 HF_HUB_OFFLINE=1 HF_HUB_DISABLE_SYMLINKS_WARNING=1 TRANSFORMERS_VERBOSITY=error
export BER_WORK_DIR="$W" BER_DATA_DIR="$DATA"
say() { echo "[v16 $(date +%T)] $*"; }
fail() { say "FAILED: $*"; exit 1; }
step() { local name=$1; shift; [ -e "$W/done/$name" ] && { say "skip $name (done)"; return; }; say "=== $name"; "$@" || fail "$name"; touch "$W/done/$name"; }
run() { (cd "$CODE/business_entity_resolution" && PYTHONPATH="$SITE" "$P12" src/run.py "$@"); }
x13() { env -u PYTHONPATH "$P13" "$@"; }

setup() {   # V15's split caches, stack model (policy) and uncertain pairs; the transformer outputs are redone
    mkdir -p "$W/train" "$W/test" "$W/models"
    for f in s1 tgt gt folds scores_c xenc_band; do cp "$V15/train/$f.parquet" "$W/train/"; done
    cp "$V15/train/density.json" "$W/train/"
    for f in s1 tgt scores_c xenc_band; do cp "$V15/test/$f.parquet" "$W/test/"; done
    cp -r "$V15/models/matcher_c" "$W/models/"
}
u2() { x13 "$CLI" train "$W/train/xenc_band.parquet" "$W/models/xenc_u2" text=text_b halves=0,1 batch=128 maxlen=128 epochs=2 lr=5e-5 > "$W/u2_train.log" 2>&1; }
f5() { x13 "$CLI" train "$W/models/xenc_f5/pairs.parquet" "$W/models/xenc" text=text_b halves=0,1 batch=128 maxlen=128 init="$W/models/xenc_u2" lr=2e-5 > "$W/f5_train.log" 2>&1; }

[ -e "$V15/done/submit_fr" ] || { say "V15 has not finished"; exit 1; }
mkdir -p "$W/done"
say "start: V15=$V15 work=$W code=$CODE"
step setup      setup
step u2_train   u2
step pseudo_fr  run pseudo-fr
step f5_train   f5
step xenc_tr    x13 "$CLI" score "$W/train/xenc_band.parquet" "$W/models/xenc_u2" "$W/train/xenc_scores.parquet" text=text_b mode=oof batch=256 maxlen=128
step link_train run link-train
step xenc_te    x13 "$CLI" score "$W/test/xenc_band.parquet" "$W/models/xenc_u2" "$W/test/xenc_scores.parquet" text=text_b mode=mean batch=256 maxlen=128
step submit     run submit-x --version v16
step fr_band    run xenc-fr-band
step xenc_fr    x13 "$CLI" score "$W/test/xenc_band_fr.parquet" "$W/models/xenc" "$W/test/xenc_scores_fr.parquet" text=text_b mode=mean batch=256 maxlen=128
step fr_merge   run xenc-fr-merge
step submit_fr  env BER_LINK_ALL=1 bash -c "cd '$CODE/business_entity_resolution' && PYTHONPATH='$SITE' '$P12' src/run.py submit-x --version v16fr"
step package    run package --version v16 --team "${TEAM:-team}" --doc "$CODE/Documentation_template.md"
step package_fr run package --version v16fr --team "${TEAM:-team}" --doc "$CODE/Documentation_template.md"
say "ALL DONE"
