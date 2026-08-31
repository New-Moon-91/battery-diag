#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""rcr1 결과 취합 — results/rcr/*/result.json → 표(csv/markdown).

    python scripts/report_rcr.py --out results/rcr

산출
  gap_table.csv     [1] 갭 재산출 (정책 x 셀)
  cap_sweep.csv     [2] load 대 (U, L_total, forced_rate, q)
  li_sweep.csv      [3] 리튬 대 (U, L_total, dE, S2-S1 격차)

주의 — q 의 두 얼굴
  result.json 의 S1.q 는 rho_required=1.0 으로 계산된 **규제 수요**다
  (전수평가이므로 도착 전량이 정밀검사 대상 = 검사소당 7.31건/일, CAP 무관).
  이 표는 그와 별도로 **용량 환산** q_cap 을 낸다:
      q_cap = Lambda x (CAP/lam_eff) / (n_center x D) = 7.3129 / load
  모형의 기간당 정밀용량 CAP 을 도착률로 정규화해 실제 시간축에 얹은 값이다.
  CAP* 를 검사소 처리능력으로 읽으려면 이쪽을 봐야 한다.
"""
from __future__ import annotations
import argparse, csv, json, math
from pathlib import Path

ap = argparse.ArgumentParser()
ap.add_argument('--out', default='results/rcr')
ap.add_argument('--Lambda', type=float, default=107500.)
ap.add_argument('--ncenter', type=int, default=60)
ap.add_argument('--Dyear', type=int, default=245)
a = ap.parse_args()
OUT = Path(a.out)

CELLS = []
for f in sorted(OUT.glob('*/result.json')):
    r = json.loads(f.read_text())
    c, R = r['cell'], r['results']
    bf = {k: v for k, v in R.items() if k.startswith('Bfast')}
    best_bf = min(bf.items(), key=lambda kv: kv[1].get('gap_pct', math.inf)) if bf else (None, {})
    CELLS.append(dict(jid=f.parent.name, cell=c, gstar=r['gstar'], R=R,
                      summary=r['summary'], sec=r['sec'],
                      bf_name=best_bf[0], bf=best_bf[1]))

def g(d, k, default=float('nan')):
    v = d.get(k, default)
    return default if v is None else v

def qcap(c):
    """용량 환산 검사소당 일처리 건수 = Lambda x (CAP/lam_eff) / (n x D)."""
    lam = c['lam_eff']
    return a.Lambda*(c['CAP']/lam)/(a.ncenter*a.Dyear) if lam else float('nan')

# ---------------------------------------------------------------- [1] 갭 표
POLS = ['S0', 'S2b', 'INDEX', 'S1', 'S1inf']
with (OUT/'gap_table.csv').open('w', newline='') as fh:
    w = csv.writer(fh)
    w.writerow(['jid','W','Mcyc','CAP','NARR','lam_eff','load','Li','nS','gstar','sec',
                'Bfast_best','gap_Bfast','gap_INDEX','gap_S2b','gap_S0','gap_S1','gap_S1inf','gap_S2'])
    for e in CELLS:
        c, R = e['cell'], e['R']
        w.writerow([e['jid'], c['W'], c['Mcyc'], c['CAP'], c['NARR'], c['lam_eff'],
                    round(c['load'],4), c['Li'], e['summary']['nS'],
                    e['gstar'], round(e['sec'],1), e['bf_name'],
                    *[round(g(R.get(p,{}),'gap_pct'),4) if not math.isnan(g(R.get(p,{}),'gap_pct')) else ''
                      for p in ['__bf__']],
                    ] if False else [
                    e['jid'], c['W'], c['Mcyc'], c['CAP'], c['NARR'], c['lam_eff'],
                    round(c['load'],4), c['Li'], e['summary']['nS'],
                    e['gstar'], round(e['sec'],1), e['bf_name'],
                    g(e['bf'],'gap_pct'), g(R.get('INDEX',{}),'gap_pct'),
                    g(R.get('S2b',{}),'gap_pct'), g(R.get('S0',{}),'gap_pct'),
                    g(R.get('S1',{}),'gap_pct'), g(R.get('S1inf',{}),'gap_pct'),
                    g(R.get('S2',{}),'gap_pct')])

# ---------------------------------------------------------------- [2] 용량
with (OUT/'cap_sweep.csv').open('w', newline='') as fh:
    w = csv.writer(fh)
    w.writerow(['jid','W','Mcyc','CAP','NARR','lam_eff','load','Li','policy',
                'U_kwh','L_total_kwh','L_sell_kwh','L_screen_kwh','L_forced_kwh',
                'dE_kg','forced_rate','rho','prec_per_period','fast_per_period',
                'g','gap_pct','q_regulatory_per_center_per_day','q_cap_per_center_per_day'])
    for e in CELLS:
        c = e['cell']
        for nm, m in e['R'].items():
            w.writerow([e['jid'], c['W'], c['Mcyc'], c['CAP'], c['NARR'], c['lam_eff'],
                        round(c['load'],4), c['Li'], nm,
                        g(m,'U_kwh'), g(m,'L_total_kwh'), g(m,'L_sell_kwh'),
                        g(m,'L_screen_kwh'), g(m,'L_forced_kwh'), g(m,'dE_kg'),
                        g(m,'forced_rate'), g(m,'rho'), g(m,'prec_per_period'),
                        g(m,'fast_per_period'), g(m,'g'), g(m,'gap_pct'),
                        m.get('q',{}).get('per_center_per_day'), round(qcap(c),4)])

# ---------------------------------------------------------------- [3] 리튬
with (OUT/'li_sweep.csv').open('w', newline='') as fh:
    w = csv.writer(fh)
    w.writerow(['W','CAP','NARR','load','Li','policy','U_kwh','L_total_kwh','dE_kg',
                'forced_rate','g','gap_pct'])
    for e in sorted(CELLS, key=lambda e:(e['cell']['W'],e['cell']['CAP'],e['cell']['NARR'],e['cell']['Li'])):
        c = e['cell']
        for nm, m in e['R'].items():
            w.writerow([c['W'], c['CAP'], c['NARR'], round(c['load'],4), c['Li'], nm,
                        g(m,'U_kwh'), g(m,'L_total_kwh'), g(m,'dE_kg'),
                        g(m,'forced_rate'), g(m,'g'), g(m,'gap_pct')])

print(f'{len(CELLS)}개 셀 → gap_table.csv, cap_sweep.csv, li_sweep.csv')

# ---------------------------------------------------------------- 콘솔 요약
def fmt(x, n=3):
    return '—' if (x is None or (isinstance(x,float) and math.isnan(x))) else f'{x:,.{n}f}'

print('\n### [1] 갭 재산출 (prune=False, pu_all=True, 2층가격)')
print('| 셀 | nS | g* | S0 | S2b | INDEX | Bfast(최선) | S1 | S1inf |')
print('|---|---:|---:|---:|---:|---:|---:|---:|---:|')
for e in sorted(CELLS, key=lambda e:(e['cell']['Li'],e['cell']['W'],e['cell']['CAP'],e['cell']['NARR'])):
    c, R = e['cell'], e['R']
    print(f"| W{c['W']} CAP{c['CAP']} N{c['NARR']} Li{c['Li']:g} | {e['summary']['nS']:,} | "
          f"{fmt(e['gstar'],0)} | {fmt(g(R.get('S0',{}),'gap_pct'))} | {fmt(g(R.get('S2b',{}),'gap_pct'))} | "
          f"{fmt(g(R.get('INDEX',{}),'gap_pct'))} | {e['bf_name']}={fmt(g(e['bf'],'gap_pct'))} | "
          f"{fmt(g(R.get('S1',{}),'gap_pct'))} | {fmt(g(R.get('S1inf',{}),'gap_pct'))} |")

print('\n### [2] load 대 S1 성과')
print('| W | CAP | NARR | load | S1 미이행률 | S1 U_kwh | S1 L_total | S1 dE_kg | S1inf 미이행 | S1inf U | q_cap(건/일/소) |')
print('|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|')
for e in sorted(CELLS, key=lambda e:(e['cell']['W'],e['cell']['NARR'],e['cell']['CAP'])):
    c, R = e['cell'], e['R']
    if c['Li'] != 35.8357: continue
    s1, si = R.get('S1',{}), R.get('S1inf',{})
    print(f"| {c['W']} | {c['CAP']} | {c['NARR']} | {c['load']:.2f} | {fmt(g(s1,'forced_rate'),4)} | "
          f"{fmt(g(s1,'U_kwh'),1)} | {fmt(g(s1,'L_total_kwh'),1)} | {fmt(g(s1,'dE_kg'),0)} | "
          f"{fmt(g(si,'forced_rate'),4)} | {fmt(g(si,'U_kwh'),1)} | {fmt(qcap(c),2)} |")
