#!/usr/bin/env bash
# V15 = V14 + street / house-number-suffix / IDF-overlap / name-evidence features (F1-F4, U3)
#       + transformer fine-tuned on French pseudo-labels from V14's test scores (F5).
# Reuses V14's prepared data, folds, ranker and ranked candidates; reruns from features (run_v14.sh stages).
# First F5 (GPU, ~15 min, alone so memory never stacks): fine-tune V14's transformer -> work_v15/models/xenc;
# then the CPU chain: features -> matcher B -> stack -> link -> test.
# Submissions: v15 (transformer re-scores US/India only, as V14) and v15fr (France too, BER_LINK_ALL=1).
# Log: work_v15/run_v15.log (+ work_v15/f5_train.log). Needs V14 finished.
set -uo pipefail
ROOT="C:/Users/Lenovo/Desktop/Ml challange"
CODE="$(cd "$(dirname "$0")" && pwd -W)"
V14="${V14:-$ROOT/work_v14}"
W="${W:-$ROOT/work_v15}"
P12=python
P13=python
SITE="$ROOT/baseline_us_india_70_15_15/venv/Lib/site-packages"
DATA="${DATA:-$ROOT/6ab10eb3b23ba_student_resource/student_resource/dataset}"
export BER_UPLOAD_DIR="${BER_UPLOAD_DIR-$ROOT/submissions}"
export PYTHONIOENCODING=utf-8 PYTHONDONTWRITEBYTECODE=1 HF_HUB_OFFLINE=1 HF_HUB_DISABLE_SYMLINKS_WARNING=1 TRANSFORMERS_VERBOSITY=error
say() { echo "[v15 $(date +%T)] $*"; }
run() { (cd "$CODE/business_entity_resolution" && PYTHONPATH="$SITE" "$P12" src/run.py "$@"); }

[ -e "$V14/done/submit" ] || { say "V14 has not finished"; exit 1; }
if [ ! -e "$W/done/rank_te" ]; then
    say "setup: V14 data, folds, ranker and candidates -> $W"
    mkdir -p "$W/train" "$W/test" "$W/models/xenc" "$W/done"
    for f in s1 tgt gt folds s1_sk tgt_sk; do cp "$V14/train/$f.parquet" "$W/train/"; done
    cp "$V14/train/density.json" "$W/train/"
    for f in s1 tgt s1_sk tgt_sk; do cp "$V14/test/$f.parquet" "$W/test/"; done
    cp -r "$V14/train/cand_b" "$W/train/" && cp -r "$V14/test/cand_b" "$W/test/" && cp -r "$V14/models/ranker" "$W/models/"
    for s in xenc_models prepare translit folds density rank_train rank_tr rank_te; do touch "$W/done/$s"; done
fi

if [ ! -e "$W/models/xenc/READY" ]; then
    say "F5: French pseudo pairs from V14's test scores"
    BER_WORK_DIR="$V14" BER_DATA_DIR="$DATA" run pseudo-fr || { say "FAILED: pseudo-fr"; exit 1; }
    rm -f "$W/models/xenc/FAILED"
    env -u PYTHONPATH "$P13" "$CODE/business_entity_resolution/src/xenc_cli.py" train "$V14/models/xenc_f5/pairs.parquet" \
        "$W/models/xenc" text=text_b halves=0,1 batch=128 maxlen=128 init="$V14/models/xenc" lr=2e-5 \
        > "$W/f5_train.log" 2>&1 || { say "FAILED: F5 fine-tune (see $W/f5_train.log)"; exit 1; }
    touch "$W/models/xenc/READY"
    say "F5: fine-tune done"
fi

WORK="$W" DATA="$DATA" VERSION=v15 bash "$CODE/run_v14.sh" || { say "FAILED: V15 pipeline"; exit 1; }

if [ ! -e "$W/done/submit_fr" ]; then
    say "=== submit v15fr (transformer re-scores France too)"
    BER_WORK_DIR="$W" BER_DATA_DIR="$DATA" BER_LINK_ALL=1 run submit-x --version v15fr || { say "FAILED: submit v15fr"; exit 1; }
    touch "$W/done/submit_fr"
fi
for v in v15 v15fr; do   # full final-submission package per version: submissions/<v>/<TEAM>_submission.zip
    if [ ! -e "$W/done/package_$v" ]; then
        BER_WORK_DIR="$W" BER_DATA_DIR="$DATA" run package --version "$v" --team "${TEAM:-team}" --doc "$CODE/Documentation_template.md" \
            || { say "FAILED: package $v"; exit 1; }
        touch "$W/done/package_$v"
    fi
done
say "ALL DONE"
