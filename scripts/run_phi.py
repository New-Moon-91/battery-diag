#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""w3 [5] — φ 민감도. W=6, |T|=3 에서 φ ∈ {0, .25, .5, .75, 1} 정확해 스윕.

φ = 미공개 결함(FAIL_L, 정체불명 9.3%) 중 **신속검사로 못 잡는** 비율.
P_DET = F_E + (1-φ)·F_U 이므로 φ=0 이면 미공개 결함까지 전부 잡고, φ=1 이면
확정 판별 결함만 잡는다. 현행 기본값은 φ=1 (가장 보수적).

산출: g*, B_fast 갭, 선별량(신속검사 대수).
"""
import sys, json, csv, time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]/'src'))
# w1~w3 스크립트다. 인스턴스가 레이·코나·SM3 로 고정돼 있고 커밋된 결과도
# 그 조합·구 가격으로 냈으므로, 가격도 params_v5.json 으로 고정한다.
# w4 재캘리브레이션된 params.json 을 여기에 물리면 구 결과와 섞인다.
SEL = ['레이', '코나', 'SM3']
CACHE = '/home/user/batdiag-cache'
OUT = Path(__file__).resolve().parents[1]/'results'/'w_model'
PHIS = [0.0, 0.25, 0.5, 0.75, 1.0]
W = 6


def main():
    import numpy as np, torch
    from battery_diag.data import load_cached
    from battery_diag.instance import Instance, PriceParams, Config
    from battery_diag.build import build
    from battery_diag.exact import ExactSolver
    from battery_diag import policies as pol

    root = Path(__file__).resolve().parents[1]
    FLEET, FS = load_cached(str(root/'data'))
    tot = sum(FLEET[t][0] for t in SEL)
    types = {t: (FLEET[t][1], FLEET[t][2], FLEET[t][3], FLEET[t][4], FLEET[t][0]/tot)
             for t in SEL}
    price = PriceParams.from_json(root/'data'/'params_v5.json')
    dev = 'cuda' if torch.cuda.is_available() else 'cpu'
    OUT.mkdir(parents=True, exist_ok=True)

    rows = []
    for phi in PHIS:
        t0 = time.time()
        cfg = Config(Mcyc=1, Cp=500000, Cf=20000, phi=phi, NARR=4, W=W,
                     F_E=FS['F_E'], F_U=FS['F_U'])
        I = Instance(types, price, cfg)
        A = build(I, cache_dir=CACHE, tag=','.join(SEL))
        S = ExactSolver(A, device=dev)
        g, h, _ = S.solve()
        acts = S.greedy(h)
        d = np.asarray(S.stationary(acts)); d = d/d.sum()
        nT = len(I.TY)
        fast = np.zeros(len(I.ST)); prec = np.zeros(len(I.ST)); sold = np.zeros(len(I.ST))
        for si, st in enumerate(I.ST):
            sv, fv, pu, ss, ps = I.actions(st)[acts[si]]
            fast[si] = sum(fv); prec[si] = sum(pu) + len(ps); sold[si] = sum(sv) + len(ss)
        bench = {k: float(S.evaluate(a)[0]) for k, a in
                 (('B1', pol.b1_sell_all(I)), ('B2', pol.b2_no_screening(I)),
                  ('INDEX', pol.index_myopic(I)))}
        bf = {t: float(S.evaluate(pol.b_fast(I, t))[0]) for t in range(cfg.SB+1)}
        bench['B_fast'] = max(bf.values())
        gaps = {k: 100*(g-v)/g for k, v in bench.items()}
        rows.append(dict(phi=phi, P_DET=I.P_DET, gstar=float(g),
                         fast_per_period=float(d @ fast),
                         precise_per_period=float(d @ prec),
                         sold_per_period=float(d @ sold),
                         **{f'gap_{k}': v for k, v in gaps.items()}))
        print(f'φ={phi:<5} P_DET={I.P_DET:.4f}  g*={g:>12,.2f}  '
              f'신속 {d @ fast:.4f}대  정밀 {d @ prec:.4f}대  '
              f'B_fast 갭 {gaps["B_fast"]:6.3f}%  ({time.time()-t0:.0f}s)', flush=True)
        del S, A
        if dev == 'cuda': torch.cuda.empty_cache()

    with (OUT/'phi_sweep.csv').open('w', newline='') as fh:
        w = csv.DictWriter(fh, fieldnames=list(rows[0])); w.writeheader(); w.writerows(rows)
    g = [r['gstar'] for r in rows]
    print(f'\ng* 범위 {min(g):,.0f}~{max(g):,.0f}  변동폭 {100*(max(g)-min(g))/max(g):.3f}%')
    print('→', OUT/'phi_sweep.csv')


if __name__ == '__main__':
    main()
