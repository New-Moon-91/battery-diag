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
| 대형 인스턴스 ($\|S\|$≈14,000, 선택압 9%) | 불가(메모리) | 빌드 ~20초 + 해 ~5초 |
| 전체 민감도 그리드 (~200 설정) | 3주 | **30~60분** |
| **1편 전체 실험 스위트** | 3주+ | **하루** |

근거: 측정된 CSR 빌드 8초(2코어, NARR=1) → 24코어에서 ~1초.
numpy CSR RVI 가 이미 파이썬 루프 대비 20배(67회 반복 4초, nnz 7.7M).
GPU SpMV 는 nnz 7.7M × 20 B = 154 MB/반복, 실효 대역 ~900 GB/s 에서 0.2 ms —
반복 전체가 밀리초 단위로 끝난다. 병목은 전적으로 **CSR 빌드(CPU)** 로 옮겨간다.

### 정확해의 실질 한계

GPU 16 GB, float64 CSR(값 8 B + 열 int32 4 B = 12 B/nnz) 기준
안전하게 **nnz ≈ 5억, n_sa ≈ 1,200만**까지.

| 버퍼 | $\|S\|$ | 선택압 CAP/버퍼 | 정확해 |
|---|---:|---:|---|
| NMAX=2, SMAX=2 (현재) | 1,485 | 12.5% | 여유 |
| NMAX=3, SMAX=3 | ~14,000 | **9.1%** | **가능** (빌드 ~20초) |
| NMAX=4, SMAX=4 | ~89,000 | 7.1% | 메모리 초과 → 학습정책 영역 |

**NMAX=3/SMAX=3 이 실제 센터의 선택압(창고 220대 / 충방전기 ~20대 ≈ 9%)과 일치한다.**
지금까지 계산 한계로 못 하던 "현실적 선택압에서의 정확해"가 이 서버에서 가능해진다.
이게 이번 하드웨어로 열리는 가장 중요한 능력이다.

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
