#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""rcr3 — 유찰 보정 가격(_cen) 셀과 무보정 셀(results/rcr, rcr2, rcr3) 대조.

표 1: 기준 인스턴스 (W=4,5 × CAP 1~3, NARR=3) — g*, U(S1), U(S2), ΔU, L_total, 미이행률, 벤치마크 갭.
표 2: 리튬 스윕 (W=5, NARR=3, CAP 1,2) — ΔU, ΔdE, 갭.
표 3: 역전점 — CAP=3·NARR=4 (lam 0.85~1.0), CAP=2·NARR=3 (lam 0.75~0.9), W=4,5: S2−S1 (kWh) 와 선형보간 load**.
"""
import json, glob, re
from pathlib import Path
import pandas as pd, numpy as np

ROOT = Path(__file__).resolve().parents[1]
rows = []
for f in glob.glob(str(ROOT/'results'/'rcr*'/'*'/'result.json')):
    d = json.load(open(f, encoding='utf-8')); c = d['cell']; r = d['results']
    jid = Path(f).parent.name
    base = dict(src=Path(f).parents[1].name, jid=jid, cen=jid.endswith('_cen'), W=c['W'], CAP=c['CAP'], NARR=c['NARR'],
                lam=c.get('lam', 0.9), load=c['load'], Li=c['Li'], gstar=d.get('gstar'))
    for pol in ('S1', 'S2', 'S2b', 'INDEX', 'Bfast0', 'S0', 'S1inf'):
        if pol in r:
            m = r[pol]
            rows.append(dict(base, pol=pol, g=m.get('g'), U=m.get('U_kwh'), L=m.get('L_total_kwh'),
                             forced=m.get('forced_rate'), gap=m.get('gap_pct'), dE=m.get('dE_kg')))
df = pd.DataFrame(rows)
# 같은 셀이 여러 src 에 있으면 최신(rcr3 > rcr2 > rcr) 우선
pri = {'rcr3': 3, 'rcr2': 2, 'rcr': 1, 'rcr3_smoke': 0}
df['pri'] = df.src.map(pri).fillna(0)
df = df.sort_values('pri', ascending=False).drop_duplicates(['cen', 'W', 'CAP', 'NARR', 'lam', 'Li', 'pol'])
pd.set_option('display.width', 240); pd.set_option('display.max_rows', 400)

def wide(sub):
    p = sub.pivot_table(index=['W', 'CAP', 'NARR', 'lam', 'Li', 'cen'], columns='pol', values=['U', 'gap', 'g', 'forced', 'L'])
    out = pd.DataFrame({'g*': p[('g', 'S2')], 'U_S1': p[('U', 'S1')], 'U_S2': p[('U', 'S2')],
                        'dU': p[('U', 'S2')] - p[('U', 'S1')], 'L_S1': p[('L', 'S1')], 'forced_S1': p[('forced', 'S1')],
                        'gap_S1': p[('gap', 'S1')], 'gap_S2b': p[('gap', 'S2b')], 'gap_INDEX': p[('gap', 'INDEX')],
                        'gap_Bfast0': p[('gap', 'Bfast0')]})
    return out

print('=== 표1 기준 인스턴스 (NARR=3, lam=0.9, Li=35.8357) ===')
t1 = wide(df[(df.NARR == 3) & (df.lam == 0.9) & (np.isclose(df.Li, 35.8357))]).reset_index()
print(t1.round(3).to_string(index=False))

print('\n=== 표2 리튬 스윕 (W=5, NARR=3) ===')
t2 = wide(df[(df.W == 5) & (df.NARR == 3) & (df.lam == 0.9)]).reset_index()
t2 = t2.sort_values(['CAP', 'cen', 'Li']); print(t2.round(3).to_string(index=False))

print('\n=== 표3 역전점 — S2−S1 (kWh) ===')
t3 = wide(df[np.isclose(df.Li, 35.8357) & (((df.CAP == 3) & (df.NARR == 4)) | ((df.CAP == 2) & (df.NARR == 3)))]).reset_index()
t3 = t3.sort_values(['CAP', 'W', 'cen', 'lam']); print(t3[['W', 'CAP', 'NARR', 'lam', 'cen', 'U_S1', 'U_S2', 'dU', 'forced_S1', 'gap_S1']].round(3).to_string(index=False))

def cross(sub):
    sub = sub.sort_values('lam'); x = (sub.lam*sub.NARR/sub.CAP).values; y = sub.dU.values
    for i in range(len(x)-1):
        if y[i] <= 0 < y[i+1] or y[i] < 0 <= y[i+1]:
            return x[i] + (0 - y[i])*(x[i+1]-x[i])/(y[i+1]-y[i])
    return None
print('\n역전점 load** (선형보간):')
for (W, CAP, cen), g in t3.groupby(['W', 'CAP', 'cen']):
    print(f'  W={W} CAP={CAP} {"보정" if cen else "무보정"}: load**={cross(g)}  (점: ' + ', '.join(f'{l*n/c:.3f}:{d:+.2f}' for l, n, c, d in zip(g.lam, g.NARR, g.CAP, g.dU)) + ')')
df.to_csv(ROOT/'results'/'rcr3'/'rcr3_long.csv', index=False)
t1.to_csv(ROOT/'results'/'rcr3'/'rcr3_t1_base.csv', index=False); t2.to_csv(ROOT/'results'/'rcr3'/'rcr3_t2_li.csv', index=False); t3.to_csv(ROOT/'results'/'rcr3'/'rcr3_t3_reversal.csv', index=False)
