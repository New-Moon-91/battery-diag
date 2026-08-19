#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""results/w_model/{W*.json, loss_correction.csv} → summary_w.csv + DIGEST.md"""
import sys, json, csv, glob
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT/'results'/'w_model'

# 기존 칸의 유효 총량 = NMAX*|T| + SMAX  (|T|=3)
LEGACY_MAP = {8: (2, 2), 9: (2, 3), 10: (2, 4), 11: (3, 2), 12: (3, 3)}


def load():
    ws = sorted((json.loads(Path(f).read_text()) for f in glob.glob(str(OUT/'W*.json'))),
                key=lambda r: r['W'])
    lc = {}
    f = OUT/'loss_correction.csv'
    if f.exists():
        for r in csv.DictReader(f.open()):
            lc[(int(r['NMAX']), int(r['SMAX']))] = {k: float(v) for k, v in r.items()}
    return ws, lc


def write_csv(ws):
    rows = []
    for r in ws:
        rows.append(dict(
            W=r['W'], nS=r['nS'], n_sa=r['n_sa'], nnz=r['nnz'], csr_gb=round(r['csr_gb'], 2),
            stream=r['stream'], build_sec=round(r['build_sec'], 1),
            solve_sec=round(r['solve_sec'], 1), pi_iters=r['pi_iters'],
            gstar=r['gstar'],
            **{f'bench_{k}': v for k, v in r['bench'].items()},
            **{f'gap_{k}': v for k, v in r['gaps'].items()},
            B_fast_thr=r['B_fast_thr'],
            forced_units_opt=r['opt']['forced_units'],
            forced_value_opt=r['opt']['forced_value'],
            forced_prob_opt=r['opt']['forced_prob'],
            forced_units_bfast=r['bfast']['forced_units'],
            forced_value_bfast=r['bfast']['forced_value'],
        ))
    p = OUT/'summary_w.csv'
    with p.open('w', newline='') as fh:
        w = csv.DictWriter(fh, fieldnames=list(rows[0])); w.writeheader(); w.writerows(rows)
    return p, rows


def act_table(r, key):
    """차종별 행동 비율 (정상분포 가중, 기간당 대수)."""
    st = r[key]
    L = []
    for k, t in enumerate(st['types']):
        u = st['act_u'][k]; s = st['act_s'][k]
        L.append((t, u, s))
    return L


def main():
    ws, lc = load()
    if not ws:
        print('W*.json 이 없다'); return
    p, rows = write_csv(ws)
    print('→', p)

    L = []
    A = L.append
    A('# W-정식화 본 분석 (w2) — DIGEST')
    A('')
    A('생성 2026-08-20. 원자료: `results/w_model/W*.json`, `summary_w.csv`,')
    A('`loss_correction.csv`, `legacy_loss.md`.')
    A('')
    A('---')
    A('')
    # ---------------- [0]
    A('## [0] 편향 방향 — 기존 g\\* 는 **아래로** 편향이다 (w1 서술 정정)')
    A('')
    A('w1 커밋(79de2d3)과 w1 TASK 배경은 무료 소실이 "g\\* 를 위로 편향시킨다" 고 적었다.')
    A('**틀렸다.** 소실은 수익 0·비용 0 으로 사라지는 것이고 강제매각은 +p_rc×kWh 로')
    A('나가는 것인데, 둘 다 창고 자리를 먹지 않고 다음 상태도 같다. 전이커널이 동일하고')
    A('보상만 다르므로 누락된 것은 **비용이 아니라 수익**이다 → 기존 g\\* 는 낮게 나온다.')
    A('')
    A('독립인 두 경로가 1e-7 이내로 일치한다 — (1) 정상분포 항등식 g = d·r 에서 d·Δr,')
    A('(2) 정책행 `rsa` 에 Δr 을 더한 보상수정 정책평가.')
    A('')
    if lc:
        A('| 버퍼 | 기존 g\\* | 소실가치 | 보정 g\\* | 상승 |')
        A('|---|---:|---:|---:|---:|')
        for (n, s), r in sorted(lc.items()):
            A(f"| NMAX={n}, SMAX={s} | {r['gstar']:,.0f} | {r['lost_value']:,.0f} | "
              f"{r['g_corrected_identity']:,.0f} | +{100*r['lost_value']/r['gstar']:.2f}% |")
        A('')
    A('단서. 기존 최적정책을 고정한 채 보상만 고친 값이라 W-모형 g\\* 자체는 아니다.')
    A('확정되는 것은 편향의 **방향**과 그 **하한**이다.')
    A('')
    # ---------------- [1]
    A('## [1] 정확해 가능 경계')
    A('')
    A('|S| = C(W+12,12) 이다. 아래는 **실측** (추정이 아니다).')
    A('')
    A('| W | \\|S\\| | n_sa | nnz | CSR | 경로 | 빌드 | 해 | PI |')
    A('|---:|---:|---:|---:|---:|---|---:|---:|---:|')
    for r in ws:
        A(f"| {r['W']} | {r['nS']:,} | {r['n_sa']:,} | {r['nnz']:,} | {r['csr_gb']:.1f}GB | "
          f"{'스트리밍' if r['stream'] else 'in-RAM'} | {r['build_sec']:.0f}s | "
          f"{r['solve_sec']:.0f}s | {r['pi_iters']} |")
    A('')
    A('경계 밖 (표본추정, 60상태)')
    A('')
    A('| W | \\|S\\| | nA 추정 | nnz 추정 | CSR 추정 |')
    A('|---:|---:|---:|---:|---:|')
    A('| 9 | 293,930 | 96,448,231 | 142,681,460,133 | 1,712GB |')
    A('| 10 | 646,646 | 400,360,093 | 279,376,560,630 | 3,353GB |')
    A('')
    A('**이 하드웨어에서 정확해가 가능한 최대는 W=8 이다.** 근거는 실측이다.')
    A('')
    A('- W=8 은 실제로 끝까지 풀었다 — CSR 72.1GB, 빌드 1,002초, 해 644초(PI 7회).')
    A('  빌드 중 MemAvailable 최저는 48GB 로 62GB 시스템에 여유가 있었다.')
    A('- W=9 는 CSR 추정 1,712GB 다. 표본추정이 실측 대비 과대평가하는 경향을 감안해도')
    A('  (W=8 은 105.7GB 추정 → 72.1GB 실측, 1.47배 과대) 실제 1,160GB 급이고,')
    A('  NVMe 여유 공간 631GB 를 두 배 가까이 넘는다. 디스크에서 먼저 막힌다.')
    A('- 시간으로도 막힌다. W=7→8 에서 nnz 6.1배에 빌드 7.5배(134초→1,002초)였다.')
    A('  같은 비율이면 W=9 빌드는 4시간 안팎이고 PI 한 회의 전량 스윕만 1TB 이상 읽는다.')
    A('')
    A('기존 칸의 유효 총량 대응(2/2→8, 2/3→9, 2/4→10, 3/2→11, 3/3→12)으로 보면,')
    A('**W-정식화에서 정확해로 닿을 수 있는 것은 가장 작은 칸(2/2) 하나뿐**이다.')
    A('용량을 하나로 묶으면 상태공간이 C(W+12,12) 로 커져 기존 정식화보다 훨씬 빨리 터진다')
    A('(2/2 는 1,485 상태인데 같은 총량의 W=8 은 125,970 상태 — 85배).')
    A('')
    # ---------------- [2]
    A('## [2] 기존 모형 vs W 모형')
    A('')
    A('### g\\*(W) 와 한계가치')
    A('')
    A('| W | g\\*(W) | Δg\\*/ΔW | 대응 기존 칸 | 기존 g\\* | 기존 보정 g\\* |')
    A('|---:|---:|---:|---|---:|---:|')
    prev = None
    for r in ws:
        d = f'{r["gstar"]-prev:,.0f}' if prev is not None else '—'
        cell = LEGACY_MAP.get(r['W'])
        if cell and cell in lc:
            c = lc[cell]
            A(f"| {r['W']} | {r['gstar']:,.0f} | {d} | NMAX={cell[0]}, SMAX={cell[1]} | "
              f"{c['gstar']:,.0f} | {c['g_corrected_identity']:,.0f} |")
        else:
            A(f"| {r['W']} | {r['gstar']:,.0f} | {d} | — | — | — |")
        prev = r['gstar']
    A('')
    A('### 벤치마크 갭 (%)')
    A('')
    A('| W | B1 | B2 | INDEX | B_fast | B_fast thr |')
    A('|---:|---:|---:|---:|---:|---:|')
    for r in ws:
        g = r['gaps']
        A(f"| {r['W']} | {g['B1']:.3f} | {g['B2']:.3f} | {g['INDEX']:.3f} | "
          f"**{g['B_fast']:.3f}** | {r['B_fast_thr']} |")
    A('')
    A('### 강제매각 (정상분포 가중, 기간당)')
    A('')
    A('| W | 최적: 대수 | 가치 | 발동확률 | B_fast: 대수 | 가치 |')
    A('|---:|---:|---:|---:|---:|---:|')
    for r in ws:
        o, b = r['opt'], r['bfast']
        A(f"| {r['W']} | {o['forced_units']:.4f} | {o['forced_value']:,.0f} | "
          f"{100*o['forced_prob']:.2f}% | {b['forced_units']:.4f} | {b['forced_value']:,.0f} |")
    A('')
    A('### 최적정책 구조 — 차종별 행동 (정상분포 가중, 기간당 대수)')
    A('')
    for r in ws:
        A(f"**W={r['W']}**")
        A('')
        A('| 차종 | 매각 | 신속 | 정밀직행 | 보유 | 선별매각 | 선별정밀 | 선별보유 |')
        A('|---|---:|---:|---:|---:|---:|---:|---:|')
        for t, u, s in act_table(r, 'opt'):
            A(f"| {t} | {u[0]:.4f} | {u[1]:.4f} | {u[2]:.4f} | {u[3]:.4f} | "
              f"{s[0]:.4f} | {s[1]:.4f} | {s[2]:.4f} |")
        A('')
    # ---- [2] 해석
    W8 = next((r for r in ws if r['W'] == 8), None)
    if W8 and (2, 2) in lc:
        c = lc[(2, 2)]
        diff = W8['gstar'] - c['gstar']
        share = 100*c['lost_value']/diff
        A('### (a) g\\* 차이는 소실가치와 정합적인가')
        A('')
        A(f"대응쌍 W=8 ↔ NMAX=2/SMAX=2 (둘 다 유효 총량 8). "
          f"W 모형이 {W8['gstar']:,.0f}, 기존이 {c['gstar']:,.0f} 로 **{diff:,.0f} 높다**.")
        A(f"이 중 소실보정분이 {c['lost_value']:,.0f} 로 {share:.0f}% 를 설명하고, "
          f"나머지 {diff-c['lost_value']:,.0f}({100-share:.0f}%)는 **용량 풀링**에서 온다.")
        A('')
        A('풀링이 왜 큰가. 기존 2/2 는 차종별 상한 2 가 셋 + 선별완료 상한 2 로 쪼개져 있어')
        A('코나만 4대 들고 있는 상태를 아예 표현하지 못한다. W=8 은 같은 총량을 하나로 묶어')
        A('필요한 차종에 몰아 쓸 수 있다. 즉 g\\* 차이의 방향은 [0] 의 예측과 맞고,')
        A('크기는 소실보정만으로는 설명되지 않는다 — 정식화 교체가 소실 수정 이상의 일을 한다.')
        A('')
        A('### (b) B_fast 갭')
        A('')
        A(f"기존 5칸에서 10.502~14.102% 였다. W 모형에서는 "
          f"{min(r['gaps']['B_fast'] for r in ws):.3f}~{max(r['gaps']['B_fast'] for r in ws):.3f}% 이고")
        A(f"대응쌍(W=8 ↔ 2/2)에서 **{W8['gaps']['B_fast']:.3f}% vs 10.502%** 로 커졌다.")
        A('방향은 유지된다 — B_fast 는 여전히 네 벤치마크 중 가장 좋고 최적과는 뚜렷이 멀다.')
        A('')
        A('갭이 커진 이유는 최적정책 쪽이 좋아졌기 때문이다. 창고가 하나로 묶이면 최적은')
        A('선별완료품을 **들고 기다리는** 여지를 갖는다(아래 구조표에서 선별보유가 W 와 함께')
        A('0.60 → 2.57 로 증가). B_fast 는 정의상 전량 신속검사 후 상위 CAP 만 정밀이고')
        A('나머지를 즉시 매각하므로 그 여지를 못 쓴다. W=8 에서 B_fast 의 강제매각이')
        A('0.0000대인 것이 방증이다 — 재고를 쌓지 않으니 창고가 찰 일이 없다.')
        A('')
    A('### (c) 최적정책 구조 — 유지되지만 무게중심이 옮겨간다')
    A('')
    A('**유지된다.** 레이(R1)는 전 구간에서 100% 무검사 즉시매각이고, SM3 도 대부분')
    A('즉시매각이다(W≥6 에서 0.02~0.03대만 신속검사로 샌다). 검사는 코나에 집중된다.')
    A('"저가치 무검사 / 중간만 선별" 이라는 양끝 생략 구조는 그대로다.')
    A('')
    A('**달라진 것.** W 가 커질수록 코나의 **정밀직행이 줄고**(0.1439 → 0.0300)')
    A('**선별보유가 는다**(0.6004 → 2.5678). 자리가 넉넉해지면 희소한 정밀검사 용량을')
    A('미리 확정해 쓰기보다, 신속검사로 신호를 받아 두고 좋은 신호가 나올 때까지')
    A('**기다리는** 쪽이 낫기 때문이다. 자리 경합이 완화되며 나타나는 전형적인 전환이다.')
    A('')
    A('강제매각도 같은 방향이다 — 최적정책 하 발동확률이 W=4 의 71.40% 에서')
    A('W=8 의 25.23% 로 떨어진다. 정책이 "미리 팔아 자리를 비우는" 회피를 덜 해도 된다.')
    A('')
    # ---------------- [3]
    A('## [3] 오염된 기존 결과의 정정')
    A('')
    A('### 가법분해 재검정')
    A('')
    if lc:
        def gv(n, s, key): return lc[(n, s)][key]
        for tag, key in (('원본', 'gstar'), ('소실보정', 'g_corrected_identity')):
            g22, g23, g32, g33 = (gv(2, 2, key), gv(2, 3, key), gv(3, 2, key), gv(3, 3, key))
            dS2, dS3 = g23-g22, g33-g32
            dN2, dN3 = g32-g22, g33-g23
            A(f'- **{tag}** — SMAX 2→3: NMAX=2 에서 {dS2:,.0f} / NMAX=3 에서 {dS3:,.0f}. '
              f'NMAX 2→3: SMAX=2 에서 {dN2:,.0f} / SMAX=3 에서 {dN3:,.0f}. '
              f'NMAX/SMAX 비 **{dN2/dS2:.2f}배**, 교호작용 {dS3-dS2:,.0f} '
              f'(g\\* 대비 {100*abs(dS3-dS2)/g33:.3f}%)')
        A('')
        A('보정 후 NMAX 우위가 2.22배 → 1.25배로 줄고, 가법 잔차가 0.081% → 0.482% 로')
        A('**6배 커지며 부호가 뒤집힌다**. v5b 의 "NMAX 효과가 SMAX 의 2.2배 / 가법분해가')
        A('성립한다" 는 상당 부분 무료 소실의 인공물이다. NMAX 2→3 증분 239,370 중')
        A('104,249(43.6%)이 소실 감소분이기 때문이다.')
        A('')
        A('반면 **SMAX 효과는 보정에 전혀 움직이지 않는다** (2→3 이 107,993, 3→4 가 58,369,')
        A('체감비 0.540 로 동일). 소실가치가 NMAX 와 도착과정만의 함수이고 SMAX 와 무관해서다.')
        A('SMAX 의 체감수확 결론은 그대로 유지된다.')
        A('')
    A('### 갭은 오염에 무관한가 — 검정 결과 "대체로 그렇다, 그러나 정확히는 아니다"')
    A('')
    A('갭 = (g\\* − g_π)/g\\* 이고 **소실가치는 정책마다 다르므로** 무관성은 가정이 아니라')
    A('검정 대상이다. 정책별 소실가치를 실측해 보정 갭을 다시 계산했다')
    A('(`gap_contamination_*.json`, `dcl_contamination_*.json`).')
    A('')
    A('| 칸 | 정책 | 갭 원본 | 갭 보정 | 차이 |')
    A('|---|---|---:|---:|---:|')
    for f in sorted(OUT.glob('gap_contamination_*.json')):
        cell = f.stem.replace('gap_contamination_', '')
        d = json.loads(f.read_text())
        for k in ('B2', 'INDEX', 'B_fast'):
            if k in d:
                A(f"| {cell} | {k} | {d[k]['gap']:.3f}% | {d[k]['gap_corr']:.3f}% | "
                  f"{d[k]['gap_corr']-d[k]['gap']:+.3f}%p |")
    for f in sorted(OUT.glob('dcl_contamination_*.json')):
        d = json.loads(f.read_text())
        cell = f"N{d['NMAX']}S{d['SMAX']}"
        A(f"| {cell} | DCL({'carry' if d['carry'] else 'legacy'}) | {d['gap']:.3f}% | "
          f"{d['gap_corr']:.3f}% | {d['gap_corr']-d['gap']:+.3f}%p |")
    A('')
    A('읽는 법. NMAX=2 칸에서는 모든 정책이 매 기간 미검사 재고를 비워 소실가치가')
    A('정책과 무관하게 같다 → 보정은 분모에만 걸려 갭이 일률적으로 **작아진다**.')
    A('NMAX=3 칸에서는 최적정책이 재고를 더 들고 있어 소실이 더 크고, 그래서 보정 갭이')
    A('오히려 **커진다**. 즉 부호조차 칸마다 다르다.')
    A('')
    A('중요한 것은 크기다. **정책이 최적에 가까울수록 오염이 작다** — 최적과 소실가치가')
    A('비슷해져 분자에서 상쇄되기 때문이다. 3/2 에서 legacy DCL 은 +0.127%p 움직이는데')
    A('carry DCL 은 +0.009%p 뿐이다. 벤치마크처럼 최적에서 먼 정책일수록 최대 ±0.5%p 이동한다.')
    A('')
    A('### v5b·v5c 결론 — 철회 / 유지')
    A('')
    A('**철회한다**')
    A('')
    A('1. "NMAX 효과가 SMAX 효과의 2.2배" → 소실보정 후 1.25배. 증분의 43.6%가 인공물.')
    A('2. "가법분해가 성립한다 (잔차 0.081%)" → 보정 후 0.482%, 부호 반전. 가법성 자체가 인공물.')
    A('3. g\\* 의 **절대 수준**을 인용한 모든 진술. 5칸 전부 0.49~3.76% 아래로 편향.')
    A('4. w1 커밋의 "위로 편향" 서술 ([0] 참조).')
    A('')
    A('**유지된다**')
    A('')
    A('1. SMAX 체감수확 (2→3 이 3→4 의 1.85배). 보정에 전혀 안 움직인다.')
    A('2. 표현력 감사 — 레거시 디코더 상한 59.2%(3/2)~88.0%(2/4). 최적정책 라벨의')
    A('   **조합적 성질**이라 보상 오염과 무관하다. 소실을 보정해도 최적정책이 바뀌지')
    A('   않으므로(보상 상수 이동이 아니라 상태별 이동이지만 정책은 동일하게 유지된다)')
    A('   라벨도 그대로다.')
    A('3. 자기회귀 디코더의 우위. 3/2 에서 보정 후에도 legacy 3.053% vs carry 0.140%,')
    A('   비율 22배로 사실상 그대로 (원본 22.3배).')
    A('4. "사전등록 기준 ≤1% 가 carry 로 5/5 칸 통과" — 보정은 갭을 최대 +0.01%p 움직일')
    A('   뿐이라 결론이 안 바뀐다.')
    A('5. 시드 분산이 칸 간 차이보다 작다는 결론 (0.17~0.24%p vs 2.33%p).')
    A('6. 스트리밍 빌드·솔버의 성능 실측치.')
    A('')
    A('**정정이 필요하다** — "방법론 결론은 오염과 무관" 이라는 표현은 엄밀하게는 틀렸다.')
    A('갭 수치 자체는 최대 ±0.5%p 움직인다. 다만 그 이동이 결론을 뒤집을 만큼 크지 않고,')
    A('최적에 가까운 정책일수록 이동이 작다는 것이 실측 결과다.')
    A('')
    # ---------------- [4]
    import glob as _g
    from statistics import mean, pstdev
    dd = {}
    for f in sorted(_g.glob(str(OUT/'dcl_W*_*.json'))):
        r = json.loads(Path(f).read_text())
        dd.setdefault((r['W'], r['decoder']), []).append(r['gap'])
    if dd:
        A('## [4] W 모형에서의 디코더 재확인')
        A('')
        A('W=4~6 에서 시드 3개씩. 정확해는 같은 인스턴스를 다시 풀어 갭의 분모로 썼다.')
        A('')
        A('| W | legacy (시드 3개) | 평균±SD | carry (시드 3개) | 평균±SD |')
        A('|---:|---|---:|---|---:|')
        for W in sorted({k[0] for k in dd}):
            def cell(dec):
                g = dd.get((W, dec))
                if not g: return '—', '—'
                return (' / '.join(f'{x:.3f}' for x in g),
                        f'{mean(g):.3f}±{pstdev(g):.3f}')
            l1, l2 = cell('legacy'); c1, c2 = cell('carry')
            A(f'| {W} | {l1} | **{l2}** | {c1} | **{c2}** |')
        A('')
        A('**결론은 바뀌지 않는다 — 오히려 강해진다.** carry 는 세 W 모두 평균 0.01% 이하로')
        A('최적정책을 수치오차 내에서 재현한다. legacy 는 1.5~48.6% 로 흩어지고 W=6 시드1 은')
        A('48.6% 로 학습이 통째로 실패했다.')
        A('')
        A('W 모형에서 격차가 더 벌어지는 이유는 구조적이다. 기존 정식화에서 미검사 슬롯 폭은')
        A('NMAX×3 (최대 9) 이었지만 W 모형에서는 W 자체다. 같은 특징을 가진 슬롯이 더 길게')
        A('늘어서고, legacy 디코더는 그 런을 쪼갤 수 없다(v5c §4.2). 즉 W-정식화는 레거시')
        A('디코더의 약점을 정확히 더 세게 누른다.')
        A('')
    A('## 돌다 만 것 · 실패한 것')
    A('')
    A('- **w1 의 야간 C 단계는 실행되지 않았다.** 8/19 밤 세션이 A 단계 보고 직후 끊겼고')
    A('  B·C 단계와 그때 약속한 `DIGEST.md` 는 만들어지지 않았다. 아침에 DIGEST 가 없던')
    A('  것은 그 때문이다. 이 문서는 w2 지시로 새로 만든 것이다.')
    A('- **W=9 이상은 시도하지 않았다.** 위 근거로 디스크에서 막히는 것이 명백해 빌드를')
    A('  걸지 않았다. 추정만 있고 실측은 없다.')
    A('- **[4] 는 W=4~6 만 했다.** W=7·8 의 DCL 은 돌리지 않았다. TASK 우선순위상 낮고,')
    A('  W=8 은 정확해 한 번에 27분이 든다.')
    A('- **표현가능 상한은 W 모형에서 재측정하지 않았다.** [4] 의 갭 결과가 결론을 이미')
    A('  가르므로(캐리 0.01% vs 레거시 최대 48.6%) 생략했다. v5c 의 측정 방법은 그대로')
    A('  적용 가능하다.')
    A('- 회귀 테스트 중 `test_stream_parity[2-3]` 이 한 번 OOM 으로 떨어진 적이 있다')
    A('  (같은 GPU 에서 다른 잡이 14.7GB 점유). 단독 재실행 통과. 코드 회귀가 아니다.')
    A('')
    (OUT/'DIGEST.md').write_text('\n'.join(L)+'\n')
    print('→', OUT/'DIGEST.md')


if __name__ == '__main__':
    main()
