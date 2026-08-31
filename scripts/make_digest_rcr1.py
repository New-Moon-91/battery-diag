#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""results/rcr/*.csv → DIGEST_rcr1.md 본문 표. 수치를 손으로 옮기지 않는다."""
from __future__ import annotations
import csv, json, math, sys
from pathlib import Path

OUT = Path('results/rcr')
LI_TASK = 35.8357          # TASK_rcr1.md 가 지정한 기준
LI_PARAM = 39.74937728937729   # data/params_rcr_2layer.json 의 Li_ref
LAMBDA, NCENTER, DYEAR = 107500., 60, 245

cap = [r for r in csv.DictReader(open(OUT/'cap_sweep.csv'))]
gap = [r for r in csv.DictReader(open(OUT/'gap_table.csv'))]
F = float; I = int
def num(x):
    try: return float(x)
    except (TypeError, ValueError): return float('nan')
def fm(x, n=3):
    return '—' if (x is None or (isinstance(x, float) and math.isnan(x))) else f'{x:,.{n}f}'

L = []
A = L.append

# ---------------------------------------------------------------- [1]
A('## [1] 벤치마크 갭 재산출 — `prune=False, pu_all=True`, 2층 가격\n')
A('기존 표(`results/w5/DIGEST_w5.md` [4])는 `prune=True`, 가격 `PriceW5`(낙찰가 직접회귀),')
A('`NARR=4, M_cyc=1(CAP=1)` 이다. 그 셀과 **직접 대응**하는 rcr 셀을 나란히 놓는다.\n')
A('| W | | g\\* | B1/S0 | B2/S2b | INDEX | B_fast | 비고 |')
A('|---:|---|---:|---:|---:|---:|---:|---|')
BASE = {4: (8045931, 32.730, 7.734, 13.606, 4.902),
        5: (8189573, 33.909, 9.352, 15.157, 4.010),
        6: (8280433, 34.635, 10.347, 16.125, 3.661)}
for W in (4, 5, 6):
    g0, b1, b2, ix, bf = BASE[W]
    A(f'| {W} | 기존 (prune=True, PriceW5) | {g0:,} | {b1:.3f} | {b2:.3f} | {ix:.3f} | {bf:.3f} | 원고 표 |')
    r = [x for x in gap if I(x['W']) == W and I(x['CAP']) == 1 and I(x['NARR']) == 4
         and abs(F(x['Li']) - LI_TASK) < 1e-6]
    if r:
        r = r[0]
        A(f"| {W} | **rcr (prune=False, PriceRCR)** | {num(r['gstar']):,.0f} | {num(r['gap_S0']):.3f} | "
          f"{num(r['gap_S2b']):.3f} | {num(r['gap_INDEX']):.3f} | {num(r['gap_Bfast']):.3f} | "
          f"S1 갭 {num(r['gap_S1']):.3f} |")
    else:
        A(f'| {W} | **rcr (prune=False, PriceRCR)** | — | — | — | — | — | 미산출 (BLOCKED B3) |')
A('')

# 용량별 순위
A('### 순위는 행동공간이 아니라 **용량**이 바꾼다\n')
A('| W | CAP | NARR | load | S0 | S2b | INDEX | B_fast(최선) | S1 | 갭 최소 정책 |')
A('|---:|---:|---:|---:|---:|---:|---:|---:|---:|---|')
for r in sorted(gap, key=lambda r: (I(r['W']), I(r['NARR']), I(r['CAP']))):
    if abs(F(r['Li']) - LI_TASK) > 1e-6: continue
    cand = {'S0': num(r['gap_S0']), 'S2b': num(r['gap_S2b']), 'INDEX': num(r['gap_INDEX']),
            f"B_fast({r['Bfast_best']})": num(r['gap_Bfast']), 'S1': num(r['gap_S1'])}
    best = min(cand, key=lambda k: cand[k])
    A(f"| {r['W']} | {r['CAP']} | {r['NARR']} | {F(r['load']):.2f} | {num(r['gap_S0']):.3f} | "
      f"{num(r['gap_S2b']):.3f} | {num(r['gap_INDEX']):.3f} | {num(r['gap_Bfast']):.3f} | "
      f"{num(r['gap_S1']):.3f} | **{best}** |")
A('')

# ---------------------------------------------------------------- [2]
A('## [2] 용량 스윕 — load 대 순환 성과\n')
A('| W | CAP | NARR | load | S1 미이행률 | S1 U_kwh | S1 L_total | S1 dE_kg | S1inf U | '
  'ΔU=병목비용 | ΔdE | q_규제 | q_용량 |')
A('|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|')
by = {}
for r in cap:
    if abs(F(r['Li']) - LI_TASK) > 1e-6: continue
    by[(I(r['W']), I(r['CAP']), I(r['NARR']), r['policy'])] = r
for (W, C, N, p), r in sorted(by.items()):
    if p != 'S1': continue
    si = by.get((W, C, N, 'S1inf'))
    dU = (num(si['U_kwh']) - num(r['U_kwh'])) if si else float('nan')
    dE = (num(si['dE_kg']) - num(r['dE_kg'])) if si else float('nan')
    A(f"| {W} | {C} | {N} | {F(r['load']):.2f} | **{num(r['forced_rate']):.4f}** | "
      f"{num(r['U_kwh']):.2f} | {num(r['L_total_kwh']):.2f} | {num(r['dE_kg']):.0f} | "
      f"{fm(num(si['U_kwh']),2) if si else '—'} | {fm(dU,2)} | {fm(dE,0)} | "
      f"{num(r['q_regulatory_per_center_per_day']):.2f} | {num(r['q_cap_per_center_per_day']):.2f} |")
A('')
A('### 강제매각률은 load 만의 함수다\n')
A('| W | CAP | NARR | load | 실측 forced_rate | 1 − CAP/λ_eff | 차 |')
A('|---:|---:|---:|---:|---:|---:|---:|')
for (W, C, N, p), r in sorted(by.items()):
    if p != 'S1': continue
    ld = F(r['load']); fr = num(r['forced_rate']); th = max(0.0, 1 - 1/ld)
    A(f'| {W} | {C} | {N} | {ld:.2f} | {fr:.6f} | {th:.6f} | {fr-th:+.6f} |')
A('')

# CAP*
A('### 규제 미이행률 5% 미만이 되는 최소 용량 CAP\\*\n')
A('| NARR | λ_eff | CAP\\*−1 (미달) | 그 때 미이행률 | **CAP\\*** | 그 때 미이행률 | load | '
  '전국 건/일 | **검사소당 건/일** |')
A('|---:|---:|---:|---:|---:|---:|---:|---:|---:|')
for N in (2, 3, 4):
    lam = 0.9*N
    caps = sorted({C for (W, C, NN, p) in by if NN == N and p == 'S1' and W == 5})
    star = None
    for C in caps:
        fr = num(by[(5, C, N, 'S1')]['forced_rate'])
        if fr < 0.05: star = C; break
    if star is None: continue
    prev = star - 1
    frp = num(by[(5, prev, N, 'S1')]['forced_rate']) if (5, prev, N, 'S1') in by else float('nan')
    frs = num(by[(5, star, N, 'S1')]['forced_rate'])
    q = LAMBDA*(star/lam)/(NCENTER*DYEAR)
    A(f'| {N} | {lam:.1f} | {prev} | {fm(frp,4)} | **{star}** | {frs:.4f} | {lam/star:.2f} | '
      f'{q*NCENTER:,.1f} | **{q:.2f}** |')
A('')

# ---------------------------------------------------------------- [3]
A('## [3] 리튬 스윕\n')
A('### 가격층 — 명제의 전제는 성립한다\n')
sys.path.insert(0, 'src')
from battery_diag.price_rcr import PriceRCR
A('| Li | 행정 예정가 g | 재사용 기대낙찰가 | 재활용가 | **채널비** |')
A('|---:|---:|---:|---:|---:|')
for Li in (8.16, 15, 25, LI_TASK, LI_PARAM, 50, 70.42):
    p = PriceRCR.from_json('data/params_rcr_2layer.json', Li=Li)
    A(f'| {Li:g} | {p.g(0.8,64):,.0f} | {p.Erev(0.8,64):,.0f} | {p.VS(64):,.0f} | '
      f'**{p.Erev(0.8,64)/p.VS(64):.3f}** |')
A('')
A('### 순환 성과층 — 명제는 성립하지 않는다\n')
liall = [r for r in csv.DictReader(open(OUT/'li_sweep.csv')) if I(r['W']) == 5 and I(r['NARR']) == 3]
for C in (1, 2):
    sub = [r for r in liall if I(r['CAP']) == C]
    lis = sorted({F(r['Li']) for r in sub})
    if not lis: continue
    A(f'**W=5, CAP={C}, NARR=3 (load={0.9*3/C:.2f})**\n')
    A('| Li | g\\* | S2 U | S1 U | **ΔU (선별의 가치)** | S2 L_tot | S1 L_tot | S2 dE | S1 dE | '
      '**ΔdE** | S1 갭% | S2b 갭% | INDEX 갭% |')
    A('|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|')
    for Li in lis:
        d = {r['policy']: r for r in sub if F(r['Li']) == Li}
        if 'S2' not in d or 'S1' not in d: continue
        gp = num(d['S2']['gap_pct']); gs = num(d['S2']['g'])/(1-gp/100) if gp else num(d['S2']['g'])
        mark = ' ★' if abs(Li-LI_PARAM) < 1e-6 else (' ☆' if abs(Li-LI_TASK) < 1e-6 else '')
        A(f"| {Li:g}{mark} | {gs:,.0f} | {num(d['S2']['U_kwh']):.2f} | {num(d['S1']['U_kwh']):.2f} | "
          f"**{num(d['S2']['U_kwh'])-num(d['S1']['U_kwh']):+.2f}** | {num(d['S2']['L_total_kwh']):.2f} | "
          f"{num(d['S1']['L_total_kwh']):.2f} | {num(d['S2']['dE_kg']):.0f} | {num(d['S1']['dE_kg']):.0f} | "
          f"**{num(d['S2']['dE_kg'])-num(d['S1']['dE_kg']):+.0f}** | {num(d['S1']['gap_pct']):.3f} | "
          f"{num(d['S2b']['gap_pct']):.3f} | {num(d['INDEX']['gap_pct']):.3f} |")
    A('')
A('☆ = TASK 가 지정한 기준 35.8357 (실은 `params_w5.json` 의 Li_ref).')
A('★ = `data/params_rcr_2layer.json` 의 Li_ref = 39.7494. **RCR 편 반사실 기준은 이쪽이다.**\n')

# ---------------------------------------------------------------- [4]
A('## [4] 아이오닉 — 최적정책에서의 검사율\n')
for C in (1, 2, 3):
    f = OUT/f'policy_types_W5_cap{C}_narr3.json'
    if not f.exists(): continue
    d = json.loads(f.read_text()); c = d['cell']
    A(f"**W=5, CAP={c['CAP']}, NARR=3, Li={c['Li']:g} (load={0.9*3/max(c['CAP'],1):.2f})**\n")
    A('| 유형 | 정책 | 신속검사율 | 정밀검사율 | 미검사매각률 | 창고보유 |')
    A('|---|---|---:|---:|---:|---:|')
    for t in d['rates'].get('S2', {}):
        for pnm in ('S2', 'S1', 'S2b', 'INDEX'):
            r = d['rates'].get(pnm, {}).get(t)
            if not r: continue
            A(f"| {t if pnm=='S2' else ''} | {pnm} | {r['fast_rate']:.4f} | {r['prec_rate']:.4f} | "
              f"{r['sell_rate']:.4f} | {r['hold']:.4f} |")
    A('')
print('\n'.join(L))
