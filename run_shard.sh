#!/usr/bin/env bash
# RCR 편 주 실험 — 시나리오 x 진단용량 x 리튬
#   서버1: bash run_shard.sh 0
#   서버2: bash run_shard.sh 1
# 중단해도 안전. 같은 명령을 다시 치면 안 끝난 셀부터 이어서 돈다.
set -euo pipefail
export PYTHONIOENCODING=utf-8
SHARD="${1:?사용법: bash run_shard.sh <0|1>}"
NSHARD="${NSHARD:-2}"
ROOT="${ROOT:-$(pwd)}"
PY="${PY:-python3}"
OUT="${OUT:-$ROOT/results/rcr}"
W="${W:-4,5,6}"; CAP="${CAP:-1,2,3}"; NARR="${NARR:-1,2,3,4}"; LI="${LI:-35.8357}"
WORKERS="${WORKERS:-$(nproc)}"
mkdir -p "$OUT"
echo "shard=$SHARD/$NSHARD W=$W cap=$CAP narr=$NARR li=$LI workers=$WORKERS"
cd "$ROOT"
"$PY" run_rcr.py --out "$OUT" --shard "$SHARD" --nshard "$NSHARD" \
  --W "$W" --cap "$CAP" --narr "$NARR" --li "$LI" --workers "$WORKERS" \
  2>&1 | tee -a "$OUT/shard${SHARD}.log"
