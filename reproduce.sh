#!/usr/bin/env bash
# End to end, raw TSVs -> output/matching_results.tsv + output/candidate_pairs.tsv, using only this folder.
#
#   PY=<python 3.12 with requirements.txt>  PY_TORCH=<python 3.13 with requirements_transformer.txt> \
#   BER_DATA_DIR=<dataset dir holding train/ and test/>  BER_WORK_DIR=<empty work dir>  VERSION=v19 \
#   bash reproduce.sh
#
# The submitted run (V19; V18 = the same without the second pass, HOP=0) is this recipe:
#   pass 1   prepare/normalize -> folds (fold 4 sealed) -> density matching -> learned ranker (60 candidates/S1)
#            -> pair features -> matcher B -> stack L2 -> uncertain pairs (0.01 < p < 0.99)
#   models   three cross-encoders, each as two models split by target half (model h trains on dev-fold pairs whose
#            target is in half h), fine-tuned only on the provided train pairs / French test pseudo-labels:
#              xenc   U2: paraphrase-multilingual-MiniLM-L12-v2 (Apache-2.0), 2 epochs, lr 5e-5
#              xenc2  the same model, 1 epoch, then fine-tuned on French pseudo pairs + equal train replay (lr 2e-5)
#              xenc3  intfloat/multilingual-e5-base (MIT), 1 epoch, lr 3e-5
#   pass 2   (HOP=1) second-hop candidates from pass 1's confident matches, then features .. uncertain pairs again
#   link     LightGBM on stack p + the three transformer logits (train: the out-of-fold model; test: mean of
#            both) + list context -> single-owner F0.5 decision (US/India re-scored, France stack-only)
#   output   $VERSION with the decoy-shift correction (odds x0.5 for same-street different-number US/India pairs)
#            and ${VERSION}ns without it; both with odds x3 for blank-address / website-name targets (US/India:
#            on test these are nearly all true copies, xlink.bw_boost)
# Difference to the submitted checkpoints: those were trained on the uncertain pairs of earlier runs of this same
# pipeline (U2 and e5 on V15's, xenc2 on V14's plus V14's French pseudo pairs); a rerun trains them on this run's
# pass-1 pairs, so scores match closely but not bit for bit. Each stage is skipped when its marker exists in
# $BER_WORK_DIR/done/ (a rerun resumes after a failure; a rerun stage's later markers are set aside as stale).
# About 12 h on 32 threads + one 8 GB GPU; peak ~30 GB RAM (stages run one at a time).
set -uo pipefail
HERE="$(cd "$(dirname "$0")" && pwd)"
PY="${PY:-python}"
PY_TORCH="${PY_TORCH:-$PY}"
: "${BER_DATA_DIR:?set BER_DATA_DIR}" "${BER_WORK_DIR:?set BER_WORK_DIR}"
VERSION="${VERSION:-v19}"
HOP="${HOP:-1}"
export BER_DATA_DIR BER_WORK_DIR PYTHONIOENCODING=utf-8 HF_HUB_DISABLE_SYMLINKS_WARNING=1 TRANSFORMERS_VERBOSITY=error
W="$BER_WORK_DIR"
M="$W/models"
mkdir -p "$W/done" "$M"
say() { echo "[reproduce $(date +%T)] $*"; }
SEEN=" "
stage() { local name=$1 m b; shift; SEEN="$SEEN$name "; [ -e "$W/done/$name" ] && { say "skip $name"; return; }
    for m in "$W"/done/*; do b=${m##*/}; [ -f "$m" ] && [[ "$SEEN" != *" $b "* ]] || continue
        mkdir -p "$W/done/stale" && mv -f "$m" "$W/done/stale/"; done
    say "=== $name"; "$@" || { say "FAILED: $name"; exit 1; }; touch "$W/done/$name"; }
run() { (cd "$HERE" && "$PY" src/run.py "$@"); }
run_shift() { (cd "$HERE" && BER_DECOY_SHIFT=0.5 "$PY" src/run.py "$@"); }   # xlink.decoy_shift
xenc() { (cd "$HERE" && "$PY_TORCH" src/xenc_cli.py "$@"); }
score3() {   # score3 <split> <mode>: the three transformers on the split's uncertain pairs
    xenc score "$W/$1/xenc_band.parquet" "$M/xenc" "$W/$1/xenc_scores.parquet" text=text_b mode=$2 batch=256 &&
    xenc score "$W/$1/xenc_band.parquet" "$M/xenc2" "$W/$1/xenc_scores2.parquet" text=text_b mode=$2 batch=256 &&
    xenc score "$W/$1/xenc_band.parquet" "$M/xenc_e5" "$W/$1/xenc_scores3.parquet" text=text_b mode=$2 batch=128
}
downstream() {   # downstream <suffix>: features -> matcher B -> stack -> uncertain pairs, train then test
    stage feat_tr$1  run features --split train --variant c
    stage train_b$1  run train-b
    stage stack_tr$1 run stack-features --split train
    stage train_c$1  run train-c
    stage band_tr$1  run xenc-export --split train
    stage feat_te$1  run features --split test --variant c
    stage stack_te$1 run stack-features --split test
    stage band_te$1  run xenc-export --split test
}

# pass 1
stage prepare      run prepare --split all
stage translit     run translit --split all
stage folds        run folds
stage density      run match-density
stage rank_train   run rank-train
stage rank_tr      run rank --split train
stage rank_te      run rank --split test
downstream ""
# transformers, trained on pass 1's uncertain train pairs (dev folds only)
stage xenc_u2      xenc train "$W/train/xenc_band.parquet" "$M/xenc" text=text_b halves=0,1 batch=128 maxlen=128 epochs=2 lr=5e-5
stage xenc_base    xenc train "$W/train/xenc_band.parquet" "$M/xenc_base" text=text_b halves=0,1 batch=128 maxlen=128 epochs=1 lr=5e-5
stage pseudo_fr    run pseudo-fr
stage xenc2        xenc train "$M/xenc_f5/pairs.parquet" "$M/xenc2" text=text_b halves=0,1 batch=128 maxlen=128 init="$M/xenc_base" lr=2e-5
stage xenc_e5      xenc train "$W/train/xenc_band.parquet" "$M/xenc_e5" text=text_b halves=0,1 batch=32 maxlen=128 epochs=1 lr=3e-5 \
                       model=intfloat/multilingual-e5-base
# pass 2: second-hop candidates from pass 1's scores, then everything downstream again
if [ "$HOP" = 1 ]; then
    stage hop      run hop --split all --src "$W"
    downstream 2
fi
stage xs_tr        score3 train oof
stage xs_te        score3 test mean
stage link_train   run link-train
stage submit       run_shift submit-x --version "$VERSION"
stage submit_ns    run submit-x --version "${VERSION}ns"
if [ -n "${TEAM:-}" ] && [ -n "${DOC:-}" ]; then
    stage package  run package --version "$VERSION" --team "$TEAM" --doc "$DOC"
fi
say "done: $W/submissions/$VERSION (decoy shift) and $W/submissions/${VERSION}ns"
