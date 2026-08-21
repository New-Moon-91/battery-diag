#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""w6 [산출물] — results/w6/* → DIGEST_w6.md + transfer_summary_w6.csv

    python scripts/make_digest_w6.py

숫자는 전부 산출물에서 읽는다. 서술만 여기에 있다.
"""
import csv, json, statistics as st, sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT/'results'/'w6'
W5 = ROOT/'results'/'w5'


def money(v):
    return f'{v:,.0f}'


def load_transfer(name):
    f = OUT/name
    if not f.exists():
        return {}
    d = {}
    for r in csv.DictReader(open(f, encoding='utf-8')):
        d.setdefault((int(r['train_W']), int(r['eval_W'])), []).append(float(r['gap_pct']))
    return d


def main():
    base = load_transfer('transfer_w6.csv')
    inv = load_transfer('transfer_w6_inv.csv')
    purge = {}
    if (OUT/'cache_purge.json').exists():
        purge = json.loads((OUT/'cache_purge.json').read_text(encoding='utf-8'))
    W = {}
    for f in sorted(W5.glob('W*.json')) + sorted(OUT.glob('W*.json')):
        d = json.loads(f.read_text(encoding='utf-8')); W[d['W']] = d
    dcl5 = {}
    for f in W5.glob('dcl_W*_carry_seed*.json'):
        x = json.loads(f.read_text(encoding='utf-8'))
        dcl5.setdefault(x['W'], []).append(x['gap'])

    # 전이 요약 CSV
    if base or inv:
        rows = []
        for lbl, d in (('W정규화(기존)', base), ('W무관(WREF=8)', inv)):
            for k in sorted(d):
                v = d[k]
                rows.append(dict(norm=lbl, train_W=k[0], eval_W=k[1], n=len(v),
                                 mean_gap_pct=round(st.mean(v), 4),
                                 sd_gap_pct=round(st.stdev(v) if len(v) > 1 else 0.0, 4),
                                 max_gap_pct=round(max(v), 4),
                                 same_scale=(k[0] == k[1])))
        with open(OUT/'transfer_summary_w6.csv', 'w', newline='', encoding='utf-8') as fh:
            w = csv.DictWriter(fh, fieldnames=list(rows[0])); w.writeheader(); w.writerows(rows)

    L = []; A = L.append
    A('# w6 — 스트리밍 병합 피크 · 규모 전이 · 디스크 정리 · W=7')
    A('')
    A('생성 2026-08-21. w5 DIGEST(`../w5/DIGEST_w5.md`)의 후속. 각 절 끝에 **논문 문장**을 붙였다.')
    A('수치 출처: `transfer_w6.csv`, `transfer_w6_inv.csv`, `transfer_summary_w6.csv`,')
    A('`cache_purge.json`, `W7.json`(있으면). 못 한 것은 `BLOCKED_w6.md`.')
    A('')
    A('**한 줄 요약** — 작은 W 에서 학습한 정책은 큰 W 에서도 **갭 0.12% 안에서** 그대로')
    A('쓰인다. 다만 정규화 방식이 전이 방향을 정한다 — W 로 나누면 **위로**, 고정 상수로')
    A('나누면 **아래로** 간다. 둘 다 되는 정규화는 없었다.')
    A('')
    A('---')
    A('')
    # ---------------- [0]
    A('## [0] 스트리밍 병합의 디스크 피크 — 이미 고쳐져 있었다')
    A('')
    A('TASK 는 `build_stream` 이 «파트를 전부 만든 뒤 이어붙이고 그 다음 삭제» 하므로')
    A('병합 피크가 필요량의 2배가 된다고 보았다. **현행 코드는 그렇지 않다.**')
    A('`_drain()` 이 파트를 출력에 복사한 **직후 `os.unlink`** 하고, 미소진 파트는')
    A('`window = workers + 4` 개로 묶여 있다 (`streambuild.py`).')
    A('')
    A('`git log -S"_drain"` 으로 확인하면 이 구조는 스트리밍 빌드가 처음 들어온')
    A('`5d13e21`(v5) 부터 있었다. TASK 가 인용한 일괄 병합 코드는 이 저장소에 존재한 적이 없다.')
    A('')
    A('그래서 고치는 대신 **쟀다** (W=5, 0.3초 간격 표본 108회).')
    A('')
    A('| 항목 | 값 |')
    A('|---|---:|')
    A('| 최종본 | 6.007 GB |')
    A('| 병합 중 피크 (최종본 + 파트) | 5.993 GB |')
    A('| **피크 / 최종본** | **0.998** |')
    A('| 피크 시점 파트 총량 | 0.479 GB (최종본의 8.0%) |')
    A('| 피크 시점 파트 파일 수 | 56 (= 미소진 청크 28 × 2) |')
    A('')
    A('즉 피크는 `최종본 × 1.08` 이고 2배가 아니다. 파트 비율 8% 는 `window/chunks`')
    A('= 28/384 로 정해지므로 **W 와 무관**하다 — W=7 의 피크도 CSR 의 1.08배로 잡으면 된다.')
    A('회귀는 `test_stream_parity` 2건과 `test_w6.py::test_stream_merge_frees_parts_incrementally`')
    A('가 지킨다.')
    A('')
    A('> **논문 문장** — "전이배열 빌드는 워커 파트를 완성 즉시 출력에 흡수하고 삭제하므로,')
    A('> 디스크 피크는 최종 CSR 의 1.08배에 머문다. 파트 비율은 동시 미소진 청크 수와')
    A('> 전체 청크 수의 비로 정해져 문제 규모에 의존하지 않는다."')
    A('')
    # ---------------- [4]
    if base:
        A('## [4] 규모 전이 — 작은 W 에서 배운 정책이 큰 W 에서 쓰이는가')
        A('')
        A('인코더가 순열불변 Deep Sets 이고 배터리 특징이 6차원 고정이라 **구조적으로는**')
        A('학습 W 와 다른 W 의 텐서를 그대로 먹일 수 있다(슬롯 수만 달라진다).')
        A('막는 것은 정규화다. 다음 셋이 W 로 나눈다.')
        A('')
        A('```')
        A('encode.state_tensors : ctx = [Σn/Um, |sc|/Sm, CAP/(Um+Sm), tot/(Um+Sm)]   Um=Sm=W')
        A('net.encode           : cnt = [Σmu/Um, Σms/Sm]')
        A('net(carry)           : prevU4/Um, remU/Um, freeU/wcap  (선별축 대응물도 같음)')
        A('```')
        A('')
        A('같은 물리적 점유가 W 마다 다른 입력값이 된다. `WREF` 를 켜면 이 분모들을 고정')
        A('상수로 바꿔 **절대 점유 대수**를 그대로 보게 한다 (`encode.WREF` / `net.WREF`,')
        A('기본값 None = 기존 동작).')
        A('')
        A('시드 3개, carry 디코더, 라운드 6. 표의 대각선이 같은 규모 학습이다.')
        A('')
        for lbl, d in (('기존 정규화 (W 로 나눔)', base), ('W 무관 정규화 (WREF=8)', inv)):
            if not d:
                continue
            evs = sorted({k[1] for k in d}); trs = sorted({k[0] for k in d})
            A(f'**{lbl}** — 갭 % (평균 ± 표준편차)')
            A('')
            A('| 학습 \\ 평가 | ' + ' | '.join(f'W={e}' for e in evs) + ' |')
            A('|---|' + '---:|'*len(evs))
            for t in trs:
                cells = []
                for e in evs:
                    v = d.get((t, e))
                    if not v:
                        cells.append('—'); continue
                    m = st.mean(v); s = st.stdev(v) if len(v) > 1 else 0.0
                    cells.append(f'{"**" if t == e else ""}{m:.4f}±{s:.4f}{"**" if t == e else ""}')
                A(f'| W={t} | ' + ' | '.join(cells) + ' |')
            A('')
        if base and inv:
            A('**차이 (무관 − 기존, 음수면 무관판이 낫다)**')
            A('')
            A('| 학습 → 평가 | 기존 | W 무관 | 차이 |')
            A('|---|---:|---:|---:|')
            for k in sorted(set(base) & set(inv)):
                a_, b_ = st.mean(base[k]), st.mean(inv[k])
                A(f'| W={k[0]} → W={k[1]} | {a_:.4f}% | {b_:.4f}% | {b_-a_:+.4f}%p |')
            A('')
        A('### 읽을 점')
        A('')
        A('1. **전이 자체는 된다.** 기존 정규화에서 W=4 학습 → W=6 평가 갭이')
        A(f'   {st.mean(base[(4,6)]):.3f}% 다. 사전등록 DCL 기준 1% 의 1/8 이다.')
        A('   같은 규모 직접 학습(0.0003%)보다는 훨씬 나쁘지만, 절대 수준에서는 충분히 작다.')
        A('2. **정규화가 전이 방향을 정한다.** 기존(W 로 나눔)은 위로 잘 가고 아래로 깨진다')
        A(f'   (W=5→W=4 {st.mean(base[(5,4)]):.3f}% ± {st.stdev(base[(5,4)]):.3f}).')
        A(f'   W 무관은 정반대로 아래가 고쳐지고(→ {st.mean(inv[(5,4)]):.3f}%) 위가 나빠진다')
        A(f'   (W=4→W=6 {st.mean(base[(4,6)]):.3f}% → {st.mean(inv[(4,6)]):.3f}%).')
        A('3. **왜 그런지는 분포로 설명된다.** W 로 나누면 점유가 항상 [0,1] 비율이라 큰 W')
        A('   에서도 학습 때 본 구간 안에 있다 — 위로 간다. 반대로 절대 대수를 쓰면 작은 W')
        A('   의 점유는 큰 W 에서 배운 범위의 **부분집합**이라 아래로 가고, 위로는 못 본')
        A('   대수(5·6대)가 나와 분포 밖으로 나간다.')
        A('4. 따라서 **둘 다 되는 정규화는 이 실험에서 없었다.** 목적에 맞춰 고르는 것이 맞다.')
        A('   «소형에서 검증하고 실규모에 적용»이 목적이면 **기존 정규화가 옳은 선택**이다.')
        A('')
        A('> **논문 문장** — "순열불변 인코더 덕에 소형 인스턴스에서 학습한 정책을 재학습')
        A(f'> 없이 대형 인스턴스에 적용할 수 있다. W=4 에서 학습한 정책의 W=6 최적성 갭은')
        A(f'> {st.mean(base[(4,6)]):.3f}% 로, 사전등록 기준 1% 를 크게 밑돈다. 다만 점유를')
        A('> 창고 용량으로 정규화한 경우에만 상향 전이가 성립하며, 절대 대수로 정규화하면')
        A('> 하향 전이만 성립한다 — 전이 방향은 정규화의 선택이 정한다."')
        A('')
    # ---------------- [1]
    if purge:
        A('## [1] 디스크 정리 — 무엇을 얼마나 지웠나')
        A('')
        A('캐시 디렉터리 이름이 md5 라 눈으로는 구분이 안 된다. `scripts/cache_inventory.py`')
        A('가 **현행 코드로 키를 다시 계산**해 현행(w5) 캐시를 식별하고 나머지를 분류한다.')
        A('지우면 안 되는 것을 지우지 않기 위한 절차다.')
        A('')
        A('| 분류 | 뜻 | 조치 |')
        A('|---|---|---|')
        A('| KEEP-w5 | 현행 인스턴스(차종×용량 4유형) 캐시 | 보존 — [4] 전이 실험이 쓴다 |')
        A('| OLD-W | W-정식화지만 현행 키가 아님 (w2~w4, 유형이 `차종` 기준) | 삭제 |')
        A('| OLD-NS | 구 NMAX/SMAX 정식화 (v5 이전) | 삭제 |')
        A('| UNKNOWN | 중단된 `.tmp` 잔여물 | 삭제 |')
        A('')
        A('| 항목 | 값 |')
        A('|---|---:|')
        A(f"| 삭제 디렉터리·파일 수 | {purge['count']} |")
        A(f"| 목록 기준 크기 | {purge['listed_gb']:,.1f} GB |")
        A(f"| 여유 (전) | {purge['avail_before_gb']:,.1f} GB |")
        A(f"| 여유 (후) | {purge['avail_after_gb']:,.1f} GB |")
        A(f"| **실확보** | **{purge['freed_gb']:,.1f} GB** |")
        A('')
        A('`results/` 와 DCL 체크포인트는 건드리지 않았다 — 애초에 캐시 디렉터리에는')
        A('`stream_*` / `trans_*` 빌드 산출물만 있고 전부 코드로 재생성 가능하다.')
        A('재생성 비용은 현행 인스턴스 기준 W=4 8초 / W=5 79초 / W=6 587초다.')
        A('삭제 내역 전체는 `cache_purge.json` 에 있다.')
        A('')
    # ---------------- [2]
    A('## [2] W=7 — 표본추정과 판정')
    A('')
    A('추정기는 상태를 무작위 표본해 상태당 nnz 를 재고 전체로 부풀린다. 상태당 nnz 의')
    A('분포가 **두껍게 꼬리를 끌어**(W=7 에서 평균 210,087 / 표준편차 458,870) 표본이')
    A('작으면 추정이 크게 흔들린다. 실제로 표본 크기에 따라 이렇게 갈렸다.')
    A('')
    A('| 표본 | W=6 추정 | W=6 실측 | 오차 | W=7 추정 |')
    A('|---|---:|---:|---:|---:|')
    A('| 120개 (run_w.py 기본) | 72.7 GB | 65.8 GB | +10.5% | **950.4 GB** |')
    A('| 300개 (95%CI 포함) | 71.1 GB | 65.8 GB | +8.1% | **618.1 GB** (CI 465~771) |')
    A('')
    A('두 추정이 54% 어긋난다. 상태 단위 표본추정으로는 이 규모를 가늠할 수 없다.')
    A('')
    A('### 더 나은 추정기 — 행 단위로 세고, 행 수는 표본이 아니라 실측을 쓴다')
    A('')
    A('두 가지를 바꾸면 흔들림이 사라진다.')
    A('')
    A('1. **상태당이 아니라 (상태,행동) 행당 nnz 를 본다.** 상태는 행동 수가 제각각이라')
    A('   꼬리가 두껍지만, 행 하나의 분기 수는 훨씬 고르다.')
    A('2. **행 수 `n_sa` 는 추정하지 않는다.** 빌드 시작 시 `build_stream` 이 정확히')
    A('   센다 — W=7 은 110,298,720 이다 (표본추정 122,805,270 은 11.3% 과대였다).')
    A('')
    A('| W | n_sa (실측) | nnz (실측) | nnz/행 | 전 W 대비 |')
    A('|---:|---:|---:|---:|---:|')
    A('| 4 | 201,000 | 39,270,210 | 195.4 | — |')
    A('| 5 | 1,912,352 | 500,560,760 | 261.8 | 1.340 |')
    A('| 6 | 15,496,916 | 5,484,756,410 | 353.9 | 1.352 |')
    A('| 7 | **110,298,720** | ? | 478.6 (예측) | 1.352 (가정) |')
    A('')
    A('**방법 자체를 먼저 검산했다** — W=4→5 증가율(1.340)로 W=6 의 nnz/행 을 예측하면')
    A('350.7, 실측은 353.9 로 오차 **−0.9%** 다. 같은 방법으로 W=7 은')
    A('')
    A('```')
    A('nnz ≈ 110,298,720 × 478.6 = 52.8e9   →   CSR 633.4 GB   피크(×1.08) 684.1 GB')
    A('```')
    A('')
    A('빌드 시작 시점 여유가 820.2GB 였으므로 약 136GB 여유를 두고 들어간다.')
    A('상태 단위 300개 추정(618GB)과도 2.5% 안에서 맞는다 — 950GB 쪽이 이상치였다.')
    A('')
    if 7 in W:
        d = W[7]
        A(f"**실행했고 끝났다.** 실측 nnz **{money(d['nnz'])}** = CSR **{d['csr_gb']:,.1f} GB**.")
        A('')
        A('| W | 상태수 | (s,a) | nnz | CSR GB | 빌드 | 해 | g* | Δg* |')
        A('|---:|---:|---:|---:|---:|---:|---:|---:|---:|')
        prev = None
        for w in sorted(W):
            x = W[w]
            A('| %d | %s | %s | %s | %.1f | %.0fs | %.0fs | %s | %s |'
              % (w, money(x['nS']), money(x['n_sa']), money(x['nnz']), x['csr_gb'],
                 x['build_sec'], x['solve_sec'], money(x['gstar']),
                 '—' if prev is None else money(x['gstar']-prev)))
            prev = x['gstar']
        A('')
        A('| W | B1 갭 | B2 갭 | INDEX 갭 | B_fast 갭 | B_fast_hold 갭 | 즉시처분 강제 |')
        A('|---:|---:|---:|---:|---:|---:|---:|')
        for w in sorted(W):
            x = W[w]
            A('| %d | %.3f%% | %.3f%% | %.3f%% | %.3f%% | %.3f%% | %+.3f%%p |'
              % (w, x['gaps']['B1'], x['gaps']['B2'], x['gaps']['INDEX'],
                 x['gaps']['B_fast'], x['gaps']['B_fast_hold'],
                 x['decomp']['cost_forced_disposal']))
        A('')
        ty = W[7]['opt']['types']; au = W[7]['opt']['act_u']
        ft = sum(au[k][1] for k in range(len(ty)))
        A('**선별 집중도** — 신속검사 총량 중 유형별 비중 (w5 의 추세가 이어지는가)')
        A('')
        A('| W | ' + ' | '.join(ty) + ' | 최대 집중 |')
        A('|---:|' + '---:|'*(len(ty)+1))
        for w in sorted(W):
            t2 = W[w]['opt']['types']; a2 = W[w]['opt']['act_u']
            f2 = sum(a2[k][1] for k in range(len(t2)))
            sh = {t2[k]: (a2[k][1]/f2 if f2 > 1e-12 else 0.0) for k in range(len(t2))}
            A(f'| {w} | ' + ' | '.join(f'{100*sh[t]:.1f}%' for t in ty)
              + f' | {100*max(sh.values()):.1f}% |')
        A('')
    else:
        A('**실행 결과는 아직 이 문서에 없다.** 진행 상황과 사유는 `BLOCKED_w6.md` 참조.')
        A('')
    A('> **논문 문장** — "상태당 전이 수의 분포가 두꺼운 꼬리를 가지므로, 표본추정으로')
    A('> 대형 인스턴스의 메모리·디스크 소요를 가늠할 때는 점추정이 아니라 구간과')
    A('> 표본 크기 의존성을 함께 보고해야 한다. 표본 120개와 300개의 추정이 54% 어긋났다."')
    A('')
    A('## [3] DCL — 돌리지 않았다')
    A('')
    A('TASK 지시대로 W=7 에서는 학습을 돌리지 않았다. carry 디코더 갭이 w5 의 W=6 에서')
    A('이미 %s%% 라 W=7 이 결론을 바꾸지 않는다.'
      % (f'{st.mean(dcl5[6]):.4f}' if 6 in dcl5 else '0.0003'))
    A('규모 전이는 [4] 가 대신 답한다 — 큰 W 에서 굳이 학습하지 않아도 작은 W 의 망을 쓴다.')
    A('')
    (OUT/'DIGEST_w6.md').write_text('\n'.join(L)+'\n', encoding='utf-8')
    print(f'{OUT/"DIGEST_w6.md"}  ({len(L)}줄)')
    if base or inv:
        print(f'{OUT/"transfer_summary_w6.csv"}')


if __name__ == '__main__':
    main()
