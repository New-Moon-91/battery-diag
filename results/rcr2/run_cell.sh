#!/usr/bin/env bash
# rcr2 셀 러너 — 실측 계측 포함. 사용법: bash run_cell.sh <tag> <W> <CAP> <NARR> [CUDA_VISIBLE_DEVICES]
set -uo pipefail
export PYTHONIOENCODING=utf-8
TAG="${1:?tag}"; W="${2:?W}"; CAP="${3:?cap}"; NARR="${4:?narr}"
[ "${5:-}" ] && export CUDA_VISIBLE_DEVICES="$5"
ROOT=/home/user/battery-diag; OUT=$ROOT/results/rcr2
LOG=$OUT/probe_${TAG}.log; SAMP=$OUT/sample_${TAG}.csv
cd "$ROOT"

# 샘플러: 5초마다 python RSS 합계(KB) 와 GPU 사용량(MiB) 실측
( echo "ts,py_rss_kb_total,py_rss_kb_max,gpu0_mib,gpu1_mib"
  while :; do
    R=$(ps -eo rss=,comm= | awk '$2 ~ /^python/ {s+=$1; if($1>m) m=$1} END {printf "%d,%d", s+0, m+0}')
    G=$(nvidia-smi --query-gpu=memory.used --format=csv,noheader,nounits 2>/dev/null | paste -sd, -)
    echo "$(date +%s),$R,${G:-,}"
    sleep 5
  done ) > "$SAMP" &
SPID=$!
trap 'kill $SPID 2>/dev/null' EXIT

echo "=== $TAG  W=$W cap=$CAP narr=$NARR  시작 $(date -Is) ===" | tee "$LOG"
/usr/bin/time -v python3 run_rcr.py \
  --out "$OUT" --W "$W" --cap "$CAP" --narr "$NARR" --li 35.8357 \
  --shard 0 --nshard 1 --workers "${WORKERS:-16}" 2>&1 | tee -a "$LOG"
RC=${PIPESTATUS[0]}
echo "=== $TAG  종료 $(date -Is)  rc=$RC ===" | tee -a "$LOG"
kill $SPID 2>/dev/null
echo "$RC" > "$OUT/.rc_${TAG}"
