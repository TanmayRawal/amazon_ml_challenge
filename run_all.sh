#!/usr/bin/env bash
# One command to regenerate output/matching_results.tsv and output/candidate_pairs.tsv (final submission v24).
#
#   BER_DATA_DIR=<student_resource/dataset>  bash run_all.sh
#
# Optional settings (defaults in brackets):
#   PY          Python with requirements.txt                              [python]
#   PY_TORCH    Python with requirements_transformer.txt (CUDA GPU)      [$PY]
#   WORK_A      work directory for pipeline A                            [./work_a]
#   WORK_B      work directory for pipeline B                            [./work_b]
#   OUT_DIR     where the two final TSVs are written                     [./output]
#   VALIDATOR   path to utils/validate_submission.py (run at the end if set)
#
# Stages (each pipeline skips stages it has already finished, so a rerun resumes after an interruption):
#   1. pipeline A (V21): reproduce.sh; data prep, ranker, matchers, three cross-encoders, link model
#   2. pipeline B (v18): src/pipeline_b_v18/src/run.py; data prep, TF-IDF search, ranker, matchers, pair judges
#   3. ensemble + second opinion + France variant B rows -> v21e3b (src/ensemble/ensemble_v21e.py)
#   4. final rule overlays -> v24 (src/final_rules/finalize.py)
#   5. official validator (optional)
# Runtime: roughly 15-20 hours on one 8-12 GB GPU and 32 GB RAM, dominated by the transformer stages.
set -euo pipefail
HERE="$(cd "$(dirname "$0")" && pwd)"
: "${BER_DATA_DIR:?set BER_DATA_DIR to the dataset folder that holds train/ and test/}"
PY="${PY:-python}"
PY_TORCH="${PY_TORCH:-$PY}"
WORK_A="${WORK_A:-$HERE/work_a}"
WORK_B="${WORK_B:-$HERE/work_b}"
OUT_DIR="${OUT_DIR:-$HERE/output}"
export PYTHONIOENCODING=utf-8
say() { echo "[run_all $(date +%T)] $*"; }

say "1/5 pipeline A -> $WORK_A"
PY="$PY" PY_TORCH="$PY_TORCH" BER_DATA_DIR="$BER_DATA_DIR" BER_WORK_DIR="$WORK_A" VERSION=v21 HOP=1 bash "$HERE/reproduce.sh"

say "2/5 pipeline B -> $WORK_B"
mkdir -p "$WORK_B/done"
b() { local name=$1; shift
    [ -e "$WORK_B/done/$name" ] && { say "skip B:$name"; return; }
    say "B:$name"
    (cd "$HERE/src/pipeline_b_v18" && BER_DATA_DIR="$BER_DATA_DIR" BER_WORK_DIR="$WORK_B" BER_GPU_PYTHON="$PY_TORCH" \
        "$PY" src/run.py "$@")
    touch "$WORK_B/done/$name"; }
b prepare     prepare --split all
b translit    translit --split all
b folds       folds
b indic       indic-dict
b fuzzy_tr    fuzzy --split train
b fuzzy_te    fuzzy --split test
b rank_train  rank-train
b rank_tr     rank --split train
b rank_te     rank --split test
b feat_tr     features --split train --variant c
b feat_te     features --split test --variant c
b train_b     train-b
b score_b     score-b --split test
b pj1_train   pj-train
b pj1_score   pj-score --split all
b pj2_train   pj-train --pj-tag 2
b pj2_score   pj-score --split all --pj-tag 2
b stack_tr    stack-features --split train --pj-sets ",2" --drop-noisy-hidden
b train_c     train-c --pj-sets ",2" --drop-noisy-hidden
b stack_te    stack-features --split test --pj-sets ",2"

say "3/5 ensemble + second opinion -> $WORK_A/submissions/v21e3b"
(cd "$HERE/src/ensemble" && "$PY" ensemble_v21e.py --code "$HERE/src" --work "$WORK_A" \
    --v18-scores "$WORK_B/test/scores_c.parquet" --data-dir "$BER_DATA_DIR" \
    --france "$HERE/src/france/france_variant_B_rows.tsv" \
    --second "$HERE/src/ensemble/ens_second_with_p18_p20.txt" --version v21e3b)

say "4/5 final rule overlays -> $OUT_DIR"
"$PY" "$HERE/src/final_rules/finalize.py" "$WORK_A/submissions/v21e3b" "$OUT_DIR"

if [ -n "${VALIDATOR:-}" ]; then
    say "5/5 official validator"
    "$PY" "$VALIDATOR" --matching "$OUT_DIR/matching_results.tsv" --candidate "$OUT_DIR/candidate_pairs.tsv" \
        --test-dir "$BER_DATA_DIR/test" --check-ids
else
    say "5/5 skipped (set VALIDATOR=<student_resource>/utils/validate_submission.py to run the official check)"
fi
say "done: $OUT_DIR/matching_results.tsv and $OUT_DIR/candidate_pairs.tsv"
