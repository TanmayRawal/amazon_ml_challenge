#!/usr/bin/env bash
# V18 = V17 + the e5-base cross-encoder (run_e5.sh) as a third transformer in the link ensemble
# (xenc = U2, xenc2 = V15's transformer, xenc3 = e5, plus list context). Only the link and the submissions are
# redone; ranking, features, matcher B and the stack are V17's. Submissions: v18 (US/India re-scored) and
# v18 with the decoy shift (odds x0.5 for same-street different-number US/India pairs, xlink.decoy_shift) and
# v18ns without it (backup); v18fr dropped (the French F5 model lowers France on the leaderboard).
# Log: work_v18/run_v18.log. Needs V17 finished and run_e5.sh done.
set -uo pipefail
ROOT="C:/Users/Lenovo/Desktop/Ml challange"
CODE="$(cd "$(dirname "$0")" && pwd -W)"
V17="${V17:-$ROOT/work_v17}"
W="${W:-$ROOT/work_v18}"
DATA="${DATA:-$ROOT/6ab10eb3b23ba_student_resource/student_resource/dataset}"
export BER_UPLOAD_DIR="${BER_UPLOAD_DIR-$ROOT/submissions}"
P12=python
SITE="$ROOT/baseline_us_india_70_15_15/venv/Lib/site-packages"
export PYTHONIOENCODING=utf-8 PYTHONDONTWRITEBYTECODE=1
say() { echo "[v18 $(date +%T)] $*"; }
step() { local name=$1; shift; [ -e "$W/done/$name" ] && { say "skip $name (done)"; return; }; say "=== $name"; "$@" || { say "FAILED: $name"; exit 1; }; touch "$W/done/$name"; }
run() { (cd "$CODE/business_entity_resolution" && PYTHONPATH="$SITE" BER_WORK_DIR="$W" BER_DATA_DIR="$DATA" "$P12" src/run.py "$@"); }
run_shift() { (export BER_DECOY_SHIFT="${DECOY_SHIFT:-0.5}"; run "$@"); }   # xlink.decoy_shift (US/India test decoys)

setup() {
    mkdir -p "$W/models"
    for f in s1 tgt gt folds scores_c xenc_band xenc_scores xenc_scores2; do cp "$V17/train/$f.parquet" "$W/train/"; done
    cp "$V17/train/density.json" "$W/train/"
    for f in s1 tgt scores_c xenc_band xenc_scores2 xenc_scores_fr; do cp "$V17/test/$f.parquet" "$W/test/"; done
    cp "$V17/test/xenc_scores_base.parquet" "$W/test/xenc_scores.parquet"      # U2 scores, before the French merge
    rm -rf "$W/models/matcher_c" && cp -r "$V17/models/matcher_c" "$W/models/"
}
[ -e "$V17/done/package_v17fr" ] || { say "V17 has not finished"; exit 1; }
[ -e "$W/done/e5_te" ] || { say "e5 scores are not ready (run_e5.sh)"; exit 1; }
say "start: V17=$V17 work=$W"
step setup      setup
step link_train run link-train
step submit     run_shift submit-x --version v18
step submit_ns  run submit-x --version v18ns
step package    run package --version v18 --team "${TEAM:-team}" --doc "$CODE/Documentation_template.md"
say "ALL DONE"
