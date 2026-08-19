# battery-diag — 폐배터리 선별센터 진단정책 최적화

평균보상 MDP + 근사 정책반복(DCL). **GPU 정확해**로 벤치마크를 만들고 학습정책을 검증한다.

---

## 0. Claude Code 에게 (서버에서 먼저 할 일)

```bash
conda env create -f environment.yml
conda activate batdiag
# Blackwell(sm_120) 은 cu128 휠이 필요 — conda 의 pytorch 를 쓰지 말 것
pip install torch --index-url https://download.pytorch.org/whl/cu128
python -c "import torch;print(torch.__version__, torch.cuda.get_device_name(0), torch.cuda.device_count())"

# 이식 검증 — 여기부터 통과해야 나머지를 신뢰할 수 있다
pytest -q tests/test_parity.py
```

`test_parity.py` 가 확인하는 것

| 항목 | 기대값 |
|---|---|
| 상태 수 / 행동 수 | 1,485 / 78,000 |
| 경제영역 분류 | 레이=R1, 코나=R2, SM3=R3 |
| $g^\*$ (NARR=1~4) | 기존 CPU 결과와 0.2% 이내 |
| 벤치마크 갭 4종 | 기존과 0.25%p 이내 |
| 배치 디코더 | 항상 MDP 행동집합 안의 행동을 생성 |

**주의.** 기준값은 2026-08-15 CPU 실행분이고 당시 $q_P$ 를 소수 3자리로 반올림해 썼다.
지금 코드는 원자료 정밀값을 쓰므로 $g^\*$ 가 0.15% 정도 다르다 — 정상이다.
`test_dcl_shapes` 가 실패하면 `net.py` 의 마스킹이나 `encode.py` 의 슬롯 정렬을 의심할 것.

---

## 1. 구조

```
battery-diag/
├─ environment.yml
├─ data/                     pool.csv, pass_soh.csv, recycle_post.csv, params.json
├─ src/battery_diag/
│   ├─ data.py               원자료 → 차종 요약통계 FLEET, 결함비중(P_DET)
│   ├─ instance.py           MDP 정의 (상태·행동·전이·경제영역 R1/R2/R3)
│   ├─ build.py              전이 → CSR 배열, 24코어 병렬 + 디스크 캐시
│   ├─ exact.py              ExactSolver(GPU) / NumpySolver(참조) / policy_iteration
│   ├─ policies.py           B1, B_fast, B2, INDEX 벤치마크
│   ├─ encode.py             상태 → 패딩 텐서, 배정 ↔ MDP 행동 변환
│   ├─ net.py                Deep Sets 인코더 + 배터리별 헤드 (**배치판**)
│   ├─ dcl.py                근사 정책반복 + 라운드별 체크포인트
│   ├─ ckpt.py               원자적 체크포인트 저장/재개
│   └─ runner.py             실험 그리드 → 2 GPU 워커풀
├─ scripts/run_one.py        단일 설정 (정확해+벤치마크+DCL)
├─ scripts/run_sweep.py      그리드 실행
├─ configs/*.yaml            부하 / 선택압 / C_p / 항온실 스윕
└─ tests/test_parity.py
```

산출물은 전부 `/home/data/batdiag/` 아래에 둔다 (캐시·결과·체크포인트).

```bash
mkdir -p /home/data/batdiag/{cache,results}
```

---

## 2. 실행

```bash
# 단일 설정
python scripts/run_one.py --config-json '{"NARR":3,"Mcyc":1,"seed":0}' \
       --out /home/data/batdiag/results/probe --job-id probe

# 그리드 (2 GPU × 워커 2 = 4 병렬)
python scripts/run_sweep.py configs/sweep_load.yaml
python scripts/run_sweep.py configs/sweep_selectivity.yaml
python scripts/run_sweep.py configs/sweep_cp.yaml
python scripts/run_sweep.py configs/sweep_hslot.yaml

# 대형 인스턴스 (NMAX=3/SMAX=3) — 스트리밍 빌드 + 워커 메모리 실측. 캐시는 NVMe 로.
python scripts/build_big.py --nmax 3 --smax 3 --workers 16 \
       --cache /home/user/batdiag-cache --solve
```

**중단·재개.** 같은 명령을 다시 실행하면 된다.
- 그리드 수준: `DONE.json` 이 있는 잡은 건너뛴다.
- 잡 내부: DCL 은 라운드마다 `state.pt` 를 원자적으로 저장하고, 재실행하면
  신경망·옵티마이저·현재 정책·RNG 상태를 복원해 **다음 라운드부터** 이어 돈다.
- 전이 CSR 은 설정 해시로 캐시되므로 재빌드하지 않는다.

장시간 작업은 tmux 로:
```bash
tmux new -s batdiag
python scripts/run_sweep.py configs/sweep_selectivity.yaml 2>&1 | tee /home/data/batdiag/results/sel.log
# Ctrl-b d 로 분리, tmux attach -t batdiag 로 복귀
```

---

## 3. GPU 를 어디에 쓰는가 (그리고 어디에 안 쓰는가)

**쓰는 곳 — 정확해.** 상태-행동 전이를 CSR 행렬 $A$ 로 두면 RVI 한 스텝이
$q = r + Ah$ (SpMV) → $h' = \mathrm{segment\_max}(q)$ 두 커널로 끝난다.
파이썬 이중루프 대비 수백 배. 정밀도는 **float64 고정** — 값이 $10^6$ 규모인데
$g$ 를 $10^{-7}$ 상대오차로 봐야 한다. SpMV 는 메모리 바운드라
소비자용 Blackwell 의 낮은 FP64 연산성능이 병목이 되지 않는다.

**쓰는 곳 — 신경망.** 이전 구현은 상태마다 파이썬 루프를 돌아 병목이었다.
`net.py` 는 상태를 배치로 쌓고 **배터리 슬롯 축으로만 루프**($\le 8$회)한다.
1,485개 상태 전체를 한 번에 처리한다.

**안 쓰는 곳 — 분산학습.** 은닉 128짜리 MLP에 DDP 를 붙이면 통신 오버헤드만 늘어난다.
GPU 2장은 **독립 실험을 병렬로 돌리는 데** 쓴다 (`runner.py`).
GPU당 워커 2개가 기본 — 정확해 SpMV 와 학습이 번갈아 돌아 점유율이 채워진다.

---

## 4. 예상 소요시간 (i9-13900K 24코어 + RTX 5080 ×2)

| 작업 | 기존 CPU | 이 서버 |
|---|---:|---:|
| 기본 인스턴스 1점 ($\|S\|$=1,485, 정확해+벤치마크 4종) | ~700초 | **~24초** (실측, CSR 빌드가 지배) |
| 부하 스윕 6점 × 3시드 (DCL 포함) | 3.5시간 | **~18분** (실측) |
| DCL 5라운드 × 3시드 | 3시간 | **~3분** |
| $C_p$ 구간 스윕 (7점) | 1.4시간 | **~30초** |
| 대형 인스턴스 ($\|S\|$≈14,000, 선택압 8.3%) | 불가(메모리) | 빌드 **23분** + 해 **2분** (스트리밍 경로, 4.1 참조) |
| 전체 민감도 그리드 (~200 설정) | 3주 | **30~60분** |
| **1편 전체 실험 스위트** | 3주+ | **하루** |

근거: 측정된 CSR 빌드 8초(2코어, NARR=1) → 24코어에서 ~1초.
numpy CSR RVI 가 이미 파이썬 루프 대비 20배(67회 반복 4초, nnz 7.7M).
GPU SpMV 는 nnz 7.7M × 20 B = 154 MB/반복, 실효 대역 ~900 GB/s 에서 0.2 ms —
반복 전체가 밀리초 단위로 끝난다. 병목은 전적으로 **CSR 빌드(CPU)** 로 옮겨간다.

### 정확해의 실질 한계

GPU 16 GB, float64 CSR(값 8 B + 열 int32 4 B = 12 B/nnz) 기준
안전하게 **nnz ≈ 5억, n_sa ≈ 1,200만**까지.

| 버퍼 | $\|S\|$ | $n_{sa}$ | nnz (실측) | CSR | 선택압 | 정확해 |
|---|---:|---:|---:|---:|---:|---|
| NMAX=2, SMAX=2 | 1,485 | 78,000 | 0.02G | 0.2GB | 12.5% | in-RAM + RVI |
| NMAX=3, SMAX=2 | 3,520 | 436,000 | 0.61G | 7.3GB | 9.1% | 스트리밍 + PI (빌드 133초, 해 467초) |
| NMAX=3, SMAX=3 | 14,080 | 3,108,000 | **4.43G** | **53.2GB** | **8.3%** | 스트리밍 + PI (빌드 1,388초, 해 **120초** / PI 5회) |
| NMAX=4, SMAX=4 | ~89,000 | — | — | — | 6.25% | 메모리 초과 → 학습정책 영역 |

nnz 는 2026-08-19 실측(NARR=4, Mcyc=1). 이전 판의 "빌드 ~20초, nnz 5억" 추정은
분기 수 분포의 꼬리를 무시한 값이었다 — 실제 3/3 은 그 **9배**다.

**NMAX=3/SMAX=3 이 실제 센터의 선택압(창고 220대 / 충방전기 ~20대 ≈ 9%)과 일치한다.**
지금까지 계산 한계로 못 하던 "현실적 선택압에서의 정확해"가 이 서버에서 가능해진다.
이게 이번 하드웨어로 열리는 가장 중요한 능력이다.

### 4.1 대형 인스턴스 경로 (streambuild + bigexact)

CSR 이 53GB 라 **부모 메모리에 모을 수도, GPU 에 통째로 올릴 수도 없다.**
`build.py`(in-RAM) 대신 `streambuild.build_stream` → `bigexact.StreamSolver` 를 쓴다.
`run_one.py` 가 표본으로 CSR 을 추정해 8GB(`BATDIAG_STREAM_GB`)를 넘으면 자동 전환한다.

**빌드가 세 번 죽은 이유와 대책** (2026-08-16 ~ 08-19)

| # | 증상 | 원인 | 대책 |
|---|---|---|---|
| 1 | 48시간 무진행 | 멀티스레드 상태에서 `fork` → 풀리지 않는 락 | start method 를 `spawn` 으로 고정 |
| 2 | 워커 부하 편차 | 상태를 균등 분할 (뒤쪽 상태일수록 행동 수십 배) | 누적 **행동 수** 기준 분할 (`_partition`) |
| 3 | OOM (워커당 4GB × 12) | 청크 결과를 워커가 통째로 보유 | 워커가 임시파일로 **흘려 쓰기** (`flush_nnz`) |

3번이 핵심이다. 행동 수로 균등하게 잘라도 메모리는 균등해지지 않는다 —
상태별 분기 수가 **중앙 116 / 90% 1,856 / 99.9% 32,768 / 최대 131,072** 로
극단적으로 치우쳐 있어, 행동 수가 같은 청크끼리 nnz 가 수십 배 차이 난다.
이제 워커는 버퍼가 `flush_nnz`(기본 8M nnz ≈ 0.1GB)를 넘으면 즉시 파일로 비우고,
부모는 완료된 청크를 순서대로 **스트림 복사**만 한다(8MB 버퍼). 청크 크기와
무관하게 상주분이 상수로 묶인다.

**실측 (ps 표본, `RSSWatch` — 추정치 금지).** 결과는 `meta.json` 의 `rss_peak_gb` 에 남는다.

| 인스턴스 | 워커 | 워커최대 | 워커합계 | 부모 | 전체 피크 | MemAvailable 최저 |
|---|---:|---:|---:|---:|---:|---:|
| NMAX=3, SMAX=2 | 12 | 0.75GB | 8.79GB | 0.70GB | 9.47GB | 56.4GB |
| NMAX=3, SMAX=3 | 16 | **0.33GB** | 4.28GB | 0.61GB | **4.87GB** | 54.4GB |

3/2 쪽 워커가 더 큰 것은 부하가 아니라 **임포트** 탓이다. spawn 워커는 실행
스크립트를 `__mp_main__` 으로 재임포트하므로, 모듈 최상단에서 torch 를 끌면
워커마다 torch 상주분이 복제된다(0.75GB → 0.33GB). 그래서 `run_one.py` 와
`scripts/build_big.py` 는 torch 계열을 `main()` 안에서 임포트한다.

**손잡이**
- `BATDIAG_FLUSH_NNZ` — 워커 버퍼 상한. 메모리가 모자라면 낮춘다.
- `BATDIAG_BUILD_WORKERS` — 워커 수.
- `BATDIAG_SLAB_NNZ` — 개선(greedy) 단계의 GPU 슬랩 상한(기본 96M nnz). GPU 가 모자라면 낮춘다.
- `--cache` 는 **NVMe** 로. 이 서버의 `/home/data` 는 회전 디스크라 53GB CSR 을
  정책반복마다 다시 읽으면 디스크 대역에 묶인다.

**정책평가의 수렴판정.** `StreamSolver.evaluate` 는 절대 tol 만 쓰면 끝나지 않는다.
$h$ 가 $10^6$ 규모라 float64 표현 간격이 ~1e-10 인데 `tol=1e-10` 이라,
실측 `max|Δh|` 가 2.33e-10 에서 바닥을 치고도 판정이 성립하지 않아
매 호출이 itmax(20,000)회를 전부 돌았다 — 같은 인스턴스에서 1e-9 도달은 **50회**면 된다.
`tol + rtol*max|h|` (rtol=1e-13) 로 고쳤다.

**결과 (2026-08-19, NARR=4, Mcyc=1, Cp=50만, Cf=2만).**
NMAX=3/SMAX=2 는 $g^*$=4,233,193.82 (PI 4회), NMAX=3/SMAX=3 은
$g^*$=**4,344,695.30** (PI 5회, 120초). 선택압 8.3% 에서의 정확해를 처음으로 얻었다.

### 4.2 DCL 은 스트리밍 경로에서 그대로 돈다 — 다만 갭 기준을 못 지킨다

`run_dcl` 은 solver 의 `evaluate` / `improve` / `stationary` 만 쓰고, `StreamSolver`
가 셋을 `ExactSolver` 와 같은 시그니처로 제공하므로 **코드 수정 없이 붙는다**
(작은 인스턴스에서 스트리밍 경로를 강제해 확인). NMAX=3/SMAX=3 라운드당 실측
24~69초 — 무거운 것은 전량 스윕인 `improve`(17초)뿐이고, 나머지(정책평가·정상분포·
학습 40에폭·디코딩)는 합쳐 10초 미만이다. 6라운드 총 4분.

| 선택압 | 12.5% | 11.1% | 10.0% | 9.1% | 8.3% |
|---|---:|---:|---:|---:|---:|
| gap_DCL | 0.52% | 0.89% | 1.34% | 2.93% | 1.34% |

**사전등록 기준(갭 ≤ 1%)은 12.5% / 11.1% 두 칸에서만 성립한다.** 선택압이 실제
센터 수준(~9%)으로 내려가면 신경망 정책이 1%를 넘는다. 반면 1단계 개선정책은
같은 인스턴스에서 갭 0.000% 까지 내려가므로(위 로그 r4~r5), 병목은 정확해 쪽이
아니라 **정책을 신경망으로 옮기는 단계**다 — 학습 예산(6라운드×40에폭)과
Deep Sets 인코딩의 표현력을 함께 의심해야 한다. 표는
`results/sweep_selectivity/summary.csv` 에 있다.

---

## 5. 모형 요약 (코드를 고칠 때 알아야 할 것)

**자원 (2026-08 정정판).** 신속검사(EIS 진단기, 10~15분)와 정밀검사(충방전, $T_p$=14h)는
**별도 장비**다 — 포항 인라인 자동평가센터가 MBT-1000 EIS 진단기 + 충방전기 구성이다.
따라서 두 검사는 경합하지 않고, 제약은 정밀검사 용량뿐이다.

$$\mathrm{CAP} = \min\left(M_{\rm cyc},\ \left\lfloor H_{\rm slot}\cdot T_p/T_{\rm hold}\right\rfloor\right),
\qquad \text{설계규칙}\quad H_{\rm slot} \ge M_{\rm cyc}\cdot\frac{T_{\rm hold}}{T_p} = 1.71\,M_{\rm cyc}$$

**보상은 커밋 시점 계상.** 정밀검사를 착수하는 순간 기대수익을 장부에 단다.
평균보상 기준에서 실현시점 계상과 동일한 $g$ 를 주면서 분산과 신용할당 지연을 줄인다.
$C_f, C_p$ 는 **변동비만** — 장비 시간은 용량제약으로 이미 반영되므로 이중계상 금지.

**경제영역.** 차종별로 신호 $b$ 가 처분을 바꾸는지에 따라
R1(전 신호 재활용, 정보가치 0) / R2(전 신호 정밀) / R3(신호가 처분을 뒤집음)로 갈린다.
$C_p$=50만원에서 레이=R1, 코나=R2, SM3=R3.

**지배 가지치기.** R1 유형과 $V^{PS}\le V^S$ 인 선별품은 즉시매각으로 강제한다.
안전하며(보관비 $h>0$, 미래가치 상한이 $V^S$) 행동수를 180만 → 7.8만으로 줄인다.

---

## 5.1 GPU 워치독 주의

`nvidia-smi` 의 Processes 에 `Xorg`/`gnome-shell` 이 붙은 GPU 는 **디스플레이 GPU** 이고,
커널 하나가 수 초를 넘기면 드라이버가 `cudaErrorLaunchTimeout` 으로 죽인다.
큰 인스턴스를 돌릴 때는 디스플레이가 없는 GPU 만 쓰도록 config 의 `gpus:` 를 지정할 것.

v3 에서 `evaluate`/`stationary` 를 정책 부분행렬 방식으로 바꿔 반복당 작업량을
행동수/상태수 배(50~100배) 줄였으므로 워치독에 걸릴 여지는 크게 줄었다.

## 6. 알려진 과제

1. **표현오차 ~1% 벽 (v2 에서 손봄).** 은닉 128·임베딩 64로 키워도 갭이 0.5~1.6%에서
   내려오지 않았다(2026-08-15 sweep_load, 18잡). 용량 문제가 아니라 **순위 정보 부재**로
   판단해 v2 에서 두 가지를 바꿨다.
   - 배터리별 **상대(경쟁) 특징 3개** 추가: 정밀 순가치 `gain`, `n_better/CAP`
     (이 상태에서 나보다 나은 후보 수), 최고 후보와의 격차. 최적정책은 상위 CAP개를
     고르는 순위 규칙이므로 이 신호가 직접 필요하다.
   - 인코더에 **max 풀링을 mean 과 병행**. 평균만으로는 '최고 후보'가 표현되지 않는다.
   특징 차원 6 → 9. 효과가 없으면 다음 후보는 (a) 헤드에 어텐션, (b) 순차 디코딩에
   이미 배정된 대상의 요약을 넘기기, (c) 라벨을 소프트(행동가치 기반)로 바꾸기.
2. **다중 채널의 규모 효과 미해명.** $M$ 을 키우면 갭이 줄었는데, 버퍼를 고정한 탓인지
   진짜 규모 효과인지 분리되지 않았다. `sweep_selectivity.yaml` 이 이걸 겨눈다.
3. **선별버퍼 초과 시 강제매각**은 인위적이다. SMAX 를 키워 발동빈도를 확인할 것.
4. 원자료 중 트위지·CEVO-C·블루온은 PASS 표본이 0~1건이라 $\mu$=0.85, $\sigma$=0.06 대체값.
   $q_P\approx0$ 이라 R1 이므로 결과에 영향은 없다.
