#!/usr/bin/env bash
# V21 = V20 + the 27 Sep retrieval and decision fixes (experiments/{retrieval2,decision2}): see their REPORT.md.
# Reuses V17 prepared data/ranker/candidates (as V20) and V20 transformer scores (superset) so only new band pairs go
# to the GPU. Submission v21 (no ns variant). Log: work_v21/run_v21.log.
# (V20 header follows)
# V19 = V18 (V17 + e5 as third link transformer) + second-hop candidates: ranker.second_hop adds the targets
# that share an exact address key or a house number + street words key with one of the S1's confident V17
# matches (sealed: 26-27% of V17's retrieval misses each, ~+4 candidates/S1). Reuses V17's prepared data,
# folds, ranker and ranked candidates, reruns from features. Transformer scores are reused from V17/V18 for
# pairs already scored (a score depends only on the pair, the model and the target half); only new band pairs
# go to the GPU. LINK_EXTRA=1 adds the structured stack features to the link (xlink EXTRA).
# Submissions v19 (US/India re-scored + decoy shift x0.5, France stack-only) with its package, and v19ns (no
# decoy shift). Log: work_v19/run_v19.log.
# Runs alongside V17 and run_e5.sh: train stages start at once, test stages once V17's test stages are done, and each
# transformer step waits for the V17/e5 scores it reuses. Below-normal CPU priority (e5 training and V17 are on the
# critical path) and each stage waits for enough free commit memory (peaks measured on V17's run of the same stage).
set -uo pipefail
ROOT="C:/Users/Lenovo/Desktop/Ml challange"
CODE="$(cd "$(dirname "$0")" && pwd -W)"
V15="${V15:-$ROOT/work_v15}"
V17="${V17:-$ROOT/work_v17}"
V18="${V18:-$ROOT/work_v18}"
V19="${V19:-$ROOT/work_v19}"
V20="${V20:-$ROOT/work_v20}"
W="${W:-$ROOT/work_v21}"
VERSION="${VERSION:-v21}"
DATA="${DATA:-$ROOT/6ab10eb3b23ba_student_resource/student_resource/dataset}"
export BER_UPLOAD_DIR="${BER_UPLOAD_DIR-$ROOT/submissions}"
P12=python
P13=python
SITE="$ROOT/baseline_us_india_70_15_15/venv/Lib/site-packages"
CLI="$CODE/business_entity_resolution/src/xenc_cli.py"
export BER_TEXT_RULES=0   # the decoy signals are model features in V20
export PYTHONIOENCODING=utf-8 PYTHONDONTWRITEBYTECODE=1 HF_HUB_OFFLINE=1 HF_HUB_DISABLE_SYMLINKS_WARNING=1 TRANSFORMERS_VERBOSITY=error
[ "${LINK_EXTRA:-0}" = 1 ] && export BER_LINK_EXTRA=1
say() { echo "[v21 $(date +%T)] $*"; }
free_gb() { powershell -NoProfile -Command "[math]::Floor((Get-CimInstance Win32_OperatingSystem).FreeVirtualMemory/1MB)" 2>/dev/null | tr -dc '0-9'; }
mem_gb() {   # free commit (GB) a stage needs before it starts: V17's peak for that stage + ~6 GB margin
    case $1 in train_c) echo 32;; stack_*|band_te) echo 22;; train_b) echo 20;; feat_*) echo 17;; hop_*|link_train|submit*) echo 16;; *) echo 8;; esac; }
wait_mem() { local need=$1 f n=0
    [ "${MEM_GATE:-1}" = 1 ] || return 0   # MEM_GATE=0: no gate (small test data)
    while f=$(free_gb); [ -n "$f" ] && [ "$f" -lt "$need" ]; do
        [ $((n % 10)) = 0 ] && say "waiting for memory: $f GB free, $need GB needed"; n=$((n + 1)); sleep 60; done; }
need() {   # need <file>...: wait until V17 / run_e5.sh wrote them (stop if either failed)
    local f n=0
    for f in "$@"; do until [ -e "$f" ]; do
        grep -qs "FAILED" "$V17/run_v17.log" "$V18/run_e5.log" "$V19/run_v19.log" && { say "upstream FAILED while waiting for $f"; exit 1; }
        [ $((n % 30)) = 0 ] && say "waiting for $f"; n=$((n + 1)); sleep 60; done; done; }
# Audit: a stage that (re)runs makes every later stage's marker stale (it would reuse outputs of the old inputs);
# markers of this script's stages not yet reached in this invocation go to done/stale/ before the stage runs
# (other scripts' markers in the same work dir, e.g. run_e5.sh's e5_*, are left alone).
SEEN=" "; STEPS=" setup_tr hop_tr feat_tr train_b stack_tr train_c band_tr setup_te hop_te feat_te stack_te band_te xs1_train xs2_train xs3_train xs2_test xs1_test xs3_test link_train submit package "
step() { local name=$1 m b; shift; SEEN="$SEEN$name "; [ -e "$W/done/$name" ] && { say "skip $name (done)"; return; }
    for m in "$W"/done/*; do b=${m##*/}
        [ -f "$m" ] && [[ "$STEPS" == *" $b "* ]] && [[ "$SEEN" != *" $b "* ]] || continue
        mkdir -p "$W/done/stale" && mv -f "$m" "$W/done/stale/" && say "stale marker $b moved to done/stale"; done
    wait_mem "$(mem_gb "$name")"; say "=== $name"; "$@" || { say "FAILED: $name"; exit 1; }; touch "$W/done/$name"; }
run() { (cd "$CODE/business_entity_resolution" && PYTHONPATH="$SITE" BER_WORK_DIR="$W" BER_DATA_DIR="$DATA" "$P12" src/run.py "$@"); }
run_shift() { (export BER_DECOY_SHIFT="${DECOY_SHIFT:-0.5}"; run "$@"); }   # xlink.decoy_shift (US/India test decoys)

setup_tr() {
    mkdir -p "$W/train" "$W/test" "$W/models"
    for f in s1 tgt gt folds s1_sk tgt_sk; do cp "$V17/train/$f.parquet" "$W/train/"; done
    cp "$V17/train/density.json" "$W/train/"
    rm -rf "$W/train/cand_b" && cp -r "$V17/train/cand_b" "$W/train/"
    rm -rf "$W/models/ranker" && cp -r "$V17/models/ranker" "$W/models/"
}
setup_te() {
    for f in s1 tgt s1_sk tgt_sk; do cp "$V17/test/$f.parquet" "$W/test/"; done
    rm -rf "$W/test/cand_b" && cp -r "$V17/test/cand_b" "$W/test/"
}

# reuse <split> <k>: transformer k (1 U2, 2 V15's, 3 e5) scores for this run's band; pairs scored before are copied
reuse() {
    local split=$1 k=$2 mode=oof prev model out batch=256
    [ "$split" = test ] && mode=mean
    case $k in
        1) model="$V17/models/xenc"; out=xenc_scores.parquet; prev="$V20/$split/xenc_scores.parquet" ;;
        2) model="$V15/models/xenc"; out=xenc_scores2.parquet; prev="$V20/$split/xenc_scores2.parquet" ;;
        3) model="$V18/models/xenc_e5"; out=xenc_scores3.parquet; prev="$V20/$split/xenc_scores3.parquet"; batch=128 ;;
    esac
    PYTHONPATH="$SITE" "$P12" -c "
import polars as pl
band = pl.read_parquet(r'$W/$split/xenc_band.parquet')
prev = pl.read_parquet(r'$prev', columns=['s1', 't', 'xenc'])
band.join(prev, on=['s1', 't'], how='anti').write_parquet(r'$W/$split/xenc_miss$k.parquet')
band.select('s1', 't').join(prev, on=['s1', 't']).write_parquet(r'$W/$split/xenc_hit$k.parquet')
print(f'$split transformer $k: {band.height:,} band pairs, {band.height - pl.read_parquet(r\"$W/$split/xenc_hit$k.parquet\").height:,} to score')" || return 1
    env -u PYTHONPATH "$P13" "$CLI" score "$W/$split/xenc_miss$k.parquet" "$model" "$W/$split/xenc_new$k.parquet" \
        text=text_b mode=$mode batch=$batch maxlen=128 || return 1
    PYTHONPATH="$SITE" "$P12" -c "
import polars as pl
x = pl.concat([pl.read_parquet(r'$W/$split/xenc_hit$k.parquet'), pl.read_parquet(r'$W/$split/xenc_new$k.parquet')])
assert x.height == pl.read_parquet(r'$W/$split/xenc_band.parquet', columns=['s1']).height, 'band not fully scored'
x.write_parquet(r'$W/$split/$out')"
}

[ -e "$V17/done/train_c" ] || { say "V17's train stack is not finished"; exit 1; }
[ -r /proc/$$/winpid ] && powershell -NoProfile -Command "(Get-Process -Id $(cat /proc/$$/winpid)).PriorityClass='BelowNormal'" >/dev/null 2>&1
need "$V20/done/package"
say "start: V17=$V17 V18=$V18 V20=$V20 work=$W link_extra=${LINK_EXTRA:-0} (below-normal priority, memory-gated)"
mkdir -p "$W/done"
step setup_tr   setup_tr
step hop_tr     run hop --split train --src "$V17"
step feat_tr    run features --split train --variant c
step train_b    run train-b
step stack_tr   run stack-features --split train
step train_c    run train-c
step band_tr    run xenc-export --split train
need "$V17/done/band_te"          # V17's test candidates and test stack scores (the hop seeds)
step setup_te   setup_te
step hop_te     run hop --split test --src "$V17"
step feat_te    run features --split test --variant c
step stack_te   run stack-features --split test
step band_te    run xenc-export --split test
# transformer scores, in the order V17 / run_e5.sh produce the ones they reuse
# transformer scores: V19's (its band's pairs, each transformer) are the cache
need "$V20/done/xs1_train"; step xs1_train reuse train 1
need "$V20/done/xs2_train"; step xs2_train reuse train 2
need "$V20/done/xs3_train"; step xs3_train reuse train 3
need "$V20/done/xs2_test";  step xs2_test  reuse test 2
need "$V20/done/xs1_test";  step xs1_test  reuse test 1
need "$V20/done/xs3_test";  step xs3_test  reuse test 3
step link_train run link-train
step submit     run_shift submit-x --version "$VERSION"
step package    run package --version "$VERSION" --team "${TEAM:-team}" --doc "$CODE/Documentation_template.md"
say "ALL DONE"
