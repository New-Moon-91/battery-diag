# TASK rcr2 — 서버1: 순환 성과 역전점 $\mathrm{load}^{**}$ 의 부하 결정성 검증

2026-09-02. **`TASK_srv2_교차점.md` 는 폐기한다.** 그 파일의 목표 격자(load 1.08 /
1.125 / 1.20)는 이미 답이 나온 구간이다. 아래를 대신 수행하라.

---

## 0. 먼저 읽고, 먼저 지켜라

- **`~/batdiag-cache` (659GB) 는 절대 지우지 마라.** 자산이지 잔해가 아니다.
- 리튬은 **반드시 `--li 35.8357` 로 고정**한다. `cap_sweep.csv` 에 리튬 스윕 행이
  같은 `(W,CAP,NARR)` 키로 섞여 있어서, 고정하지 않으면 표가 오염된다.
- 출력은 **`results/rcr2`** 로 보낸다. 기존 32셀(`results/rcr`)을 건드리지 마라.
- **30분 이상 진전이 없으면 `BLOCKED_rcr2.md` 에 적고 멈춰라.** 원인 진단까지 적어라.
  검증할 수 없는 숫자는 "확인 불가"라고 쓰고 거기서 멈춰라. 추론으로 메우지 마라.
- tmux 안에서 돌려라: `tmux new -As rcr2`

## 1. 무엇을 구하는가

이미 나온 결과에서 $U(S2)-U(S1)$ 의 부호가 바뀐다:

| load | (CAP,NARR) | $\lambda_{\mathrm{eff}}$ | $U(S2)-U(S1)$ |
|---:|---|---:|---:|
| 0.90 | 3,3 | 2.7 | $-11.91$ |
| **1.20** | **3,4** | **3.6** | $\mathbf{-0.22}$ |
| **1.35** | **2,3** | **2.7** | $\mathbf{+10.90}$ |
| 1.80 | 2,4 | 3.6 | $+20.72$ |

논문은 **"부호는 부하만으로 결정된다"** 고 주장한다. 그 주장을 직접 시험한다.
같은 load 를 **다른 $\lambda_{\mathrm{eff}}$** 로 만들어 부호와 크기를 대조하라.

| 셀 | load | $\lambda_{\mathrm{eff}}$ | 대조 대상 |
|---|---:|---:|---|
| **A. W=5, CAP=4, NARR=6** | 1.35 | 5.4 | 기존 1.35 (λ=2.7) |
| **B. W=5, CAP=6, NARR=8** | 1.20 | 7.2 | 기존 1.20 (λ=3.6) |

**부호가 일치하면** 미이행률처럼 닫힌 형태로 쓸 수 있다는 뜻이다.
**다르면 그게 더 중요한 결과다.** 어느 쪽이든 그대로 보고하라.

## 2. ★ 비용을 모른다 — 반드시 직렬 프로브부터

지금까지 돌린 **최대 셀이 CAP=4, NARR=4** 다. A·B 는 둘 다 그보다 크다.
회수된 결과에 셀별 시간·nnz·메모리가 없어 스케일링을 추정할 근거가 없다.
W=6 이 RAM 90GB 로 막힌 전례(`BLOCKED_rcr1.md` B3)가 있다.

**GPU 는 2장이지만 RAM 64GB 는 공유다. `build.py` 는 in-RAM 이다.**
큰 셀 2개를 동시에 띄우면 호스트가 OOM 으로 죽는다.

### 2-1. 셀 A 를 혼자 돌린다

```bash
cd ~/battery-diag
mkdir -p results/rcr2
/usr/bin/time -v python3 run_rcr.py \
  --out $PWD/results/rcr2 \
  --W 5 --cap 4 --narr 6 --li 35.8357 \
  --shard 0 --nshard 1 --workers 16 \
  2>&1 | tee results/rcr2/probe_A_W5_c4_n6.log
```

끝나면 **`Maximum resident set size`(KB)** 와 벽시계 시간을 기록하라.

### 2-2. 분기 판단 — 이 규칙을 지켜라

| 셀 A 최대 RSS | 다음 행동 |
|---|---|
| **< 25 GB** | 셀 B 를 **두 번째 GPU 에 병렬로** 붙여도 된다 (`CUDA_VISIBLE_DEVICES=1`) |
| **25–45 GB** | 셀 B 를 **직렬로** 돌린다. 병렬 금지 |
| **> 45 GB** | 셀 B 는 시도하지 마라. `BLOCKED_rcr2.md` 에 실측치를 적고 멈춰라 |

셀 A 가 **90분을 넘기면** 중단하고 시간·진행 로그를 기록한 뒤 멈춰라.
`DONE.json` 이 있으면 재실행 시 건너뛰므로 중단은 안전하다.

### 2-3. 셀 B

```bash
python3 run_rcr.py \
  --out $PWD/results/rcr2 \
  --W 5 --cap 6 --narr 8 --li 35.8357 \
  --shard 0 --nshard 1 --workers 16 \
  2>&1 | tee results/rcr2/probe_B_W5_c6_n8.log
```

## 3. 여유가 있으면 (선택)

셀 A·B 가 모두 40분 안에 끝나고 RSS 가 25GB 미만이면, **W=4 로 같은 두 셀**을
추가로 돌려라. 교차점 근방에서 창고 효과가 실제로 있는지 재는 것이다 —
기존 격자에서 load 1.20 이 W=4 에서 $-1.75$, W=5 에서 $-0.22$ 로 갈렸다.
**§3.1 이 "저장은 부하 1 바로 위 좁은 구간에서만 흡수한다"고 쓴 그 구간이다.**

```bash
python3 run_rcr.py --out $PWD/results/rcr2 \
  --W 4 --cap 4,6 --narr 6,8 --li 35.8357 --shard 0 --nshard 1 --workers 16
```
(격자 곱이라 4셀이 나온다. 필요 없는 조합은 그냥 두면 된다.)

## 4. 취합

```bash
python3 scripts/report_rcr.py --out results/rcr2
```

`cap_sweep.csv` / `gap_table.csv` 가 `results/rcr2/` 에 생긴다.

## 5. 보고 — `results/rcr2/DIGEST_rcr2.md` 에 쓸 것

1. **실측 비용표**: 셀별 벽시계 시간, 최대 RSS(GB), 최대 GPU 메모리(GiB)
2. **결과표**: 각 셀의 `U_kwh` 를 **S1, S2, S1inf, S2b, INDEX, Bfast0** 전부
3. **핵심 판정 한 줄**: 셀 A 의 $U(S2)-U(S1)$ 부호가 기존 load 1.35 와 같은가?
   셀 B 는 기존 load 1.20 과 같은가? **크기도 같은가, 아니면 $\lambda_{\mathrm{eff}}$
   에 비례하는가?**
4. 막힌 것은 `BLOCKED_rcr2.md` 에 원인 진단까지

마지막에 `rcr2_$(hostname).tar.gz` 로 `results/rcr2` 를 묶어라.

## 6. 하지 말 것

- **W=6 은 시도하지 마라.** `prune=False` 에서 nnz≈7.5e9 / RAM 90GB 다. 실장 64GB.
  `streambuild`+`bigexact` 로 갈아타는 건 이 라운드의 일이 아니다.
- `metrics_w.stationary` 의 수렴 판정을 고치지 마라(`BLOCKED_rcr1.md` B2).
  고치면 기존 셀과 새 셀의 기준이 달라져 **같은 표 안에 두 기준이 섞인다.**
  INDEX 가 느린 건 알려진 문제이고 **결과는 정확하다. 기다려라.**
- 방법론 비교(DCL·PPO)를 늘리지 마라. 이 논문의 기여가 아니다.
