#!/usr/bin/env bash
# V17 = V16 + two retrieval channels (name word x house number, skeleton word x house number) + 60 candidates
# per S1 instead of 50. Full rerun from the ranker on V14's prepared, density-matched data and folds. The
# transformers are V16's (U2 for US/India, U2 + French F5 for France): no retraining, the target-half split stays
# valid for new pairs.
# Submissions: v17 (transformer re-scores US/India) and v17fr (every country), each with its full package.
# Log: work_v17/run_v17.log. Needs V16 finished. Stages resume from their done markers.
set -uo pipefail
ROOT="C:/Users/Lenovo/Desktop/Ml challange"
CODE="$(cd "$(dirname "$0")" && pwd -W)"
V14="${V14:-$ROOT/work_v14}"
V16="${V16:-$ROOT/work_v16}"
W="${W:-$ROOT/work_v17}"
DATA="${DATA:-$ROOT/6ab10eb3b23ba_student_resource/student_resource/dataset}"
export BER_UPLOAD_DIR="${BER_UPLOAD_DIR-$ROOT/submissions}"
P12=python
P13=python
SITE="$ROOT/baseline_us_india_70_15_15/venv/Lib/site-packages"
export PYTHONIOENCODING=utf-8 PYTHONDONTWRITEBYTECODE=1
say() { echo "[v17 $(date +%T)] $*"; }
run() { (cd "$CODE/business_entity_resolution" && PYTHONPATH="$SITE" BER_WORK_DIR="$W" BER_DATA_DIR="$DATA" "$P12" src/run.py "$@"); }

[ -e "$V16/done/submit_fr" ] || { say "V16 has not finished"; exit 1; }
if [ ! -e "$W/done/density" ]; then
    say "setup: V14's prepared data and folds -> $W"
    mkdir -p "$W/train" "$W/test" "$W/models" "$W/done"
    for f in s1 tgt gt folds s1_sk tgt_sk; do cp "$V14/train/$f.parquet" "$W/train/"; done
    cp "$V14/train/density.json" "$W/train/"
    for f in s1 tgt s1_sk tgt_sk; do cp "$V14/test/$f.parquet" "$W/test/"; done
    for s in prepare translit folds density; do touch "$W/done/$s"; done
fi
WORK="$W" DATA="$DATA" VERSION=v17 XENC_FROM="$V16/models/xenc_u2" XENC2_FROM="${XENC2_FROM:-$ROOT/work_v15/models/xenc}" bash "$CODE/run_v14.sh" || { say "FAILED: V17 pipeline"; exit 1; }
if [ ! -e "$W/done/submit_fr" ]; then
    say "=== submit v17fr (French pairs re-scored by the F5 model)"
    run xenc-fr-band && env -u PYTHONPATH "$P13" "$CODE/business_entity_resolution/src/xenc_cli.py" score "$W/test/xenc_band_fr.parquet" \
        "$V16/models/xenc" "$W/test/xenc_scores_fr.parquet" text=text_b mode=mean batch=256 maxlen=128 && run xenc-fr-merge \
        || { say "FAILED: French transformer scores"; exit 1; }
    BER_LINK_ALL=1 run submit-x --version v17fr || { say "FAILED: submit v17fr"; exit 1; }
    touch "$W/done/submit_fr"
fi
for v in v17 v17fr; do
    if [ ! -e "$W/done/package_$v" ]; then
        run package --version "$v" --team "${TEAM:-team}" --doc "$CODE/Documentation_template.md" || { say "FAILED: package $v"; exit 1; }
        touch "$W/done/package_$v"
    fi
done
say "ALL DONE"
