#!/usr/bin/env bash
# V13 train-side pipeline (US + India, sealed fold 4 = local score). Test inference is not run.
# Assumes work_v13/train already holds s1, tgt, gt, folds (copied from V10) and fresh skeletons
# (run.py translit --split train). Log: work_v13/run_v13.log
set -euo pipefail
ROOT="C:/Users/Lenovo/Desktop/Ml challange"
PY=python
export PYTHONPATH="$ROOT/baseline_us_india_70_15_15/venv/Lib/site-packages" PYTHONDONTWRITEBYTECODE=1 PYTHONIOENCODING=utf-8
export BER_WORK_DIR="$ROOT/work_v13" BER_DATA_DIR="$ROOT/6ab10eb3b23ba_student_resource/student_resource/dataset"
cd "$ROOT/code_V13/business_entity_resolution"
run() { echo "=== $*"; "$PY" src/run.py "$@"; }
{
  run rank-train
  run rank --split train
  run features --split train --variant c
  run train-b
  run stack-features --split train
  run train-c
  echo "=== ALL DONE"
} 2>&1 | tee -a "$ROOT/work_v13/run_v13.log"
