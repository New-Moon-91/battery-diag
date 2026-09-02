# BLOCKED rcr2 — 서버1

2026-09-02. 지시서 `TASK_rcr2_srv1_v2.md`.

---

## B1. `apply_lam_flag.py` 가 적용되지 않는다 — 전제 패치("민감도 패치")가 이 서버에 없다

### 증상

```
$ python3 apply_lam_flag.py
실패: --Cp 인자 줄을 찾지 못했다. 민감도 패치가 먼저 적용돼야 한다.
종료코드=1
```

`run_rcr.py` 는 **변경되지 않았다** (md5 `7892010e836770b5e897d8634acc0a90`,
수정시각 8월 31 17:09 그대로, `.pre-lam.bak` 생성 안 됨). 스크립트가 쓰기 전에
빠져나가므로 부작용은 없다.

### 진단

`apply_lam_flag.py` 는 `run_rcr.py` 에서 세 군데를 앵커로 잡는다. **세 개 모두
이 서버에 없다.** 셋 다 이전 "민감도 패치"가 만들어 놓았어야 할 것들이다.

| # | 필요한 앵커 | 서버1 `run_rcr.py` 의 실제 상태 |
|---|---|---|
| 1 | `ap.add_argument('--Cp' ...)` — `--lam` 을 그 뒤에 끼워 넣을 자리 | **없음.** `--Cp` / `--Cf` 인자 자체가 없다 |
| 2 | `Config(lam=0.9, NARR=NARR, Mcyc=CAP, W=W, prune=False, pu_all=True,` (뒤에 쉼표) | `run_rcr.py:81` 이 `... pu_all=True)` — **쉼표 없이 닫힌다.** 인자가 더 적은 구버전 |
| 3 | `_tag += f'_Cp{a.Cp:g}'` — jid 접미사 블록 | **없음.** `run_rcr.py:107` 이 `jid = f'W{W}_cap{CAP}_narr{NARR}_li{Li:g}'` 한 줄로 끝난다 |

즉 서버1의 `run_rcr.py` 는 rcr1 번들(8월 31일) 세대 그대로이고,
서버2의 `run_rcr.py` 는 그 뒤에 민감도 패치가 얹힌 세대다. `apply_lam_flag.py` 는
후자를 전제로 쓰였다. 이 서버에는 그 민감도 패치 스크립트가 없다
(`~/`, `~/rcrbundle/`, `~/battery-diag/` 검색 — `apply_lam_flag.py`,
`fix_lam_label.py`, `collect_lam.py` 세 개만 오늘 도착해 있다).

### ★ 그냥 밀어붙이면 안 되는 이유 — 앵커 3번이 결과를 조용히 오염시킨다

앵커 3번(jid 접미사)이 없으면 `run_rcr.py:107` 의 jid 에 lam 이 들어가지 않는다.
`RUN_srv1_w4lam.sh` 의 7개 셀은 lam 만 다르고 `(W,CAP,NARR,Li)` 가 겹친다:

| 스크립트의 셀 | lam | 접미사 없을 때의 jid |
|---|---:|---|
| W=4 CAP=3 NARR=4 | 0.90 | `W4_cap3_narr4_li35.8357` |
| W=4 CAP=3 NARR=4 | 0.945 | `W4_cap3_narr4_li35.8357` ← 충돌 |
| W=4 CAP=3 NARR=4 | 0.975 | `W4_cap3_narr4_li35.8357` ← 충돌 |
| W=4 CAP=3 NARR=4 | 1.00 | `W4_cap3_narr4_li35.8357` ← 충돌 |
| W=4 CAP=2 NARR=3 | 0.80 | `W4_cap2_narr3_li35.8357` |
| W=4 CAP=2 NARR=3 | 0.85 | `W4_cap2_narr3_li35.8357` ← 충돌 |
| W=4 CAP=2 NARR=3 | 0.90 | `W4_cap2_narr3_li35.8357` ← 충돌 |

`run_rcr.py:main` 은 `DONE.json` 이 있으면 `skip` 한다. 따라서 **7셀 중 2셀만 돌고
나머지 5셀은 조용히 건너뛴다.** 로그에 `skip` 만 찍히고 오류는 나지 않는다.
2판이 경계한 "리튬 혼입과 같은 종류의 조용한 오염"과 정확히 같은 종류다.

## B2. `fix_lam_label.py` 는 **일부러 돌리지 않았다** — 단독 적용하면 `run_rcr.py` 가 깨진다

`fix_lam_label.py` 의 앵커(`run_rcr.py:118` 의 기록부)는 이 서버에 **있다.**
그래서 단독으로 돌리면 성공한다. 그런데 그 패치가 넣는 코드가 `a.lam` 을 참조한다:

```python
rec = dict(cell=dict(..., lam=a.lam, lam_eff=a.lam*NARR, load=a.lam*NARR/max(I.CAP, 1), ...))
```

`a.lam` 은 `apply_lam_flag.py` 가 만드는 인자다. B1 때문에 그것이 없으므로,
`fix_lam_label.py` 만 적용하면 **모든 셀이 첫 결과 기록 시점에 `AttributeError`
로 죽는다.** 지금 돌아가는 것마저 못 돌게 된다.

두 패치는 순서 의존이고 B1 이 풀리기 전에는 B2 도 적용할 수 없다.
지시서가 "`fix_lam_label.py` 를 건너뛰지 마라"고 한 취지(라벨 오염 방지)에는 동의하며,
B1 이 풀리는 즉시 **두 개를 이 순서대로** 적용한다.

## B3. 결과적으로 막힌 것

- `RUN_srv1_w4lam.sh` 미실행. **W=4 lam 스윕 결과 없음.**
- W=4 의 $\mathrm{load}^{**}$ 정확한 위치: **확인 불가.**
  기존 격자로 1.20 과 1.35 사이라는 것까지만 확인됨 (`DIGEST_rcr2.md` §3).
- $\mathrm{load}^{**}$ 가 W 에 따라 움직이는가(2판의 본 질문): **확인 불가.**
  서버2의 W=5 lam 스윕과 대조할 W=4 쪽 점들이 없다.

막히지 않은 것은 `DIGEST_rcr2.md` 에 적었다 — 셀 A 2회 실측, 서버2 OOM,
NARR/CAP 비용 비대칭, 그리고 앵커 2점 검산(둘 다 일치).

## B4. 풀려면 필요한 것

둘 중 하나면 된다.

1. **민감도 패치 스크립트**(서버2의 `run_rcr.py` 에 `--Cp`/`--Cf` 인자와 `_tag` jid
   접미사 블록을 넣은 그것)를 서버1에도 보내 준다. 그 뒤 `apply_lam_flag.py`
   → `fix_lam_label.py` 순서로 적용하면 된다. **이쪽을 권한다** — 두 서버의
   `run_rcr.py` 세대가 같아져야 jid 규칙과 라벨이 일치하고, 그래야 W=4 와 W=5 를
   같은 표에 놓을 수 있다.
2. 또는 서버2의 `run_rcr.py` 를 그대로 보내 준다(패치 적용 완료 상태).

**손으로 고치지 않았다.** 서버1만 다른 방식으로 기워 놓으면 jid 규칙이나 라벨이
서버2와 어긋날 수 있고, 그 어긋남은 두 서버 결과를 대조하는 이 라운드의
목적 자체를 조용히 무너뜨린다. B1 의 오염과 같은 종류의 위험이다.

## B5. 시스템 상태 (정리 완료)

| 항목 | 상태 |
|---|---|
| 계산 프로세스 | 0개 |
| 메모리 | 가용 59GB / 62GB |
| GPU | 0번 38MiB, 1번 15MiB (유휴) |
| `~/batdiag-cache` (659GB) | 손대지 않음 |
| `run_rcr.py` | **무변경** (md5 `7892010e836770b5e897d8634acc0a90`) |
| `results/rcr` (rcr1 32셀) | 손대지 않음 |
| `results/rcr2` | 완료 셀 없음 (`DONE.json` 없음). 실측 원자료·로그·DIGEST·러너만 있음 |
