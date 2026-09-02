#!/usr/bin/env bash
# 서버1 — W=4 에서 역전점 구간(부하 1.20~1.35)을 lam 으로 훑는다.
# 서버2가 W=5 를 맡는다. 둘을 대조해 load** 가 창고에 반응하는지 본다.
set -uo pipefail
cd ~/battery-diag
PY="${PY:-python3}"
OUT="$PWD/results/rcr2"
WK="${WORKERS:-6}"
mkdir -p "$OUT"

guard () {  # 가용 5GB 미만이면 커널보다 먼저 멈춘다
  local avail; avail=$(free -g | awk '/^Mem:/{print $7}')
  if [ "$avail" -lt 5 ]; then
    echo "!! 가용 ${avail}GB — 통제 중단"; pkill -f run_rcr.py; return 1
  fi
}

run () { # W CAP NARR LAM 설명
  local W=$1 C=$2 N=$3 L=$4 D=$5
  local LOAD; LOAD=$($PY -c "print(f'{$L*$N/$C:.4f}')")
  echo; echo "════ W=$W CAP=$C NARR=$N lam=$L → 부하 $LOAD  ($D) ════"
  echo "시작 $(date '+%T')  가용 $(free -g | awk '/^Mem:/{print $7}')GB"
  ( while sleep 10; do guard || break; done ) & local G=$!
  /usr/bin/time -f "  ⏱ %e초  단일최대RSS %M KB" \
    "$PY" run_rcr.py --out "$OUT" --W "$W" --cap "$C" --narr "$N" --lam "$L" \
      --li 35.8357 --shard 0 --nshard 1 --workers "$WK" \
    2>&1 | tee -a "$OUT/w4lam.log" | grep -vE '^shard 0/1'
  kill $G 2>/dev/null
  echo "  최대 합산 RSS: $(ps -eo rss,comm | awk '/python/{s+=$1} END{printf "%.1f GB", s/1048576}') (종료 후 잔량)"
}

run 4 3 4 0.90  "앵커 — 기존 값 U(S1)=123.86 U(S2)=122.11 과 일치해야 한다"
run 4 3 4 0.945 "신규"
run 4 3 4 0.975 "신규"
run 4 3 4 1.00  "신규 (분산 0)"
run 4 2 3 0.80  "반대 방향에서 부하 1.20"
run 4 2 3 0.85  "신규"
run 4 2 3 0.90  "앵커 — 기존 부하 1.35, U(S1)=83.55 U(S2)=92.76"

echo; echo "════ 취합 ════"
"$PY" collect_lam.py "$OUT"
