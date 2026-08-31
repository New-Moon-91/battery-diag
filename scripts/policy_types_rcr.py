#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""[4] 유형별 검사율 — 최적정책(S2)에서 각 차종이 실제로 어떤 처분을 받는가.

    python scripts/policy_types_rcr.py --W 5 --cap 2 --narr 3 --li 35.8357

정상분포 mu 로 가중한 기간당 기대 대수를 유형별로 집계하고, 같은 유형의
기간당 도착 대수(lam_eff x MIX[t])로 나눠 **비율**로 낸다.

  fast_rate  신속검사(EIS 선별)를 받는 비율
  prec_rate  정밀 성능평가를 받는 비율 (미검사분 pu + 선별완료분 ps)
  sell_rate  검사 없이 재활용 매각되는 비율 (sv + 선별 후 매각 ss)
  forced     창고 초과 강제매각은 유형 분해가 정책행동이 아니라 매각순서 규칙이라
             여기서는 잔차(1 - fast/prec/sell 로 설명되지 않는 몫)로만 남는다.
"""
from __future__ import annotations
import argparse, json, sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parent.parent / 'src'))
import numpy as np
from battery_diag.data import fleet_w5, SEL_W5
from battery_diag.instance import Instance, Config
from battery_diag.price_rcr import PriceRCR
from battery_diag.build import build
from battery_diag import policies as pol, policies_rcr as prc, metrics_w as M


def per_type_rates(I, arrays, acts_local):
    P = M.policy_rows(arrays, acts_local)
    mu = M.stationary(P)
    TY = I.TY; nT = len(TY)
    acc = {k: np.zeros(nT) for k in ('fast', 'prec', 'sell', 'hold')}
    for si, st in enumerate(I.ST):
        w = mu[si]
        if w < 1e-15: continue
        n, scr = st
        sv, fv, pu, ss, ps = I.actions(st)[int(acts_local[si])]
        for k in range(nT):
            acc['fast'][k] += w*fv[k]; acc['prec'][k] += w*pu[k]; acc['sell'][k] += w*sv[k]
            acc['hold'][k] += w*(n[k]-sv[k]-fv[k]-pu[k])
        for j in ps:
            acc['prec'][TY.index(scr[j][0])] += w
        for j in ss:
            acc['sell'][TY.index(scr[j][0])] += w
    lam_eff = I.cfg.lam * I.cfg.NARR
    arr = np.array([lam_eff*I.MIX[t] for t in TY])
    return mu, acc, arr


if __name__ == '__main__':
    ap = argparse.ArgumentParser()
    ap.add_argument('--W', type=int, default=5); ap.add_argument('--cap', type=int, default=2)
    ap.add_argument('--narr', type=int, default=3); ap.add_argument('--li', type=float, default=35.8357)
    ap.add_argument('--workers', type=int, default=12)
    ap.add_argument('--out', default='')
    a = ap.parse_args()

    FL = fleet_w5('data', SEL_W5); tot = sum(v[0] for v in FL.values())
    TYPES = {k: (v[1], v[2], v[3], v[4], v[0]/tot) for k, v in FL.items()}
    price = PriceRCR.from_json('data/params_rcr_2layer.json', Li=a.li)
    I = Instance(TYPES, price, Config(lam=0.9, NARR=a.narr, Mcyc=a.cap, W=a.W,
                                      prune=False, pu_all=True))
    arrays = build(I, workers=a.workers)

    POL = {'S1': prc.s1_mandatory_precise(I), 'S2b': pol.b2_no_screening(I),
           'INDEX': pol.index_myopic(I)}
    try:
        import torch
        from battery_diag.exact import ExactSolver
        S = ExactSolver(arrays, device='cuda' if torch.cuda.is_available() else 'cpu')
        g, h, _ = S.solve()
        POL['S2'] = S.greedy(h)
        print(f'g* = {g:,.2f}')
    except Exception as e:
        print('정확해 생략:', e)

    rec = {}
    for nm, acts in POL.items():
        mu, acc, arr = per_type_rates(I, arrays, acts)
        print(f'\n### {nm}  (W={a.W} CAP={I.CAP} NARR={a.narr} Li={a.li:g}, '
              f'load={0.9*a.narr/max(I.CAP,1):.2f})')
        print('| 유형 | 도착/기간 | 신속검사율 | 정밀검사율 | 미검사매각률 | 보유 | 모형등급 | 실측등급 |')
        print('|---|---:|---:|---:|---:|---:|:--:|:--:|')
        rec[nm] = {}
        for k, t in enumerate(I.TY):
            r = dict(arrivals=arr[k], fast=acc['fast'][k], prec=acc['prec'][k],
                     sell=acc['sell'][k], hold=acc['hold'][k],
                     fast_rate=acc['fast'][k]/arr[k], prec_rate=acc['prec'][k]/arr[k],
                     sell_rate=acc['sell'][k]/arr[k])
            rec[nm][t] = r
            print(f"| {t} | {arr[k]:.4f} | {r['fast_rate']:.4f} | {r['prec_rate']:.4f} | "
                  f"{r['sell_rate']:.4f} | {acc['hold'][k]:.4f} | {I.REG[t]} | {I.REG_EMP[t]} |")
    if a.out:
        Path(a.out).write_text(json.dumps(
            dict(cell=dict(W=a.W, Mcyc=a.cap, CAP=I.CAP, NARR=a.narr, Li=a.li), rates=rec),
            ensure_ascii=False, indent=1, default=float))
        print('\n→', a.out)
