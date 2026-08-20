#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""w3 [1] — 소실보정 MDP 를 **재최적화**해서 가법분해를 확정한다.

w2 [3] 의 보정 g* 는 기존 최적정책을 고정한 채 보상만 고친 값이라 보정 MDP 의
하한이지 최적값이 아니었다. 여기서는 같은 상태공간·같은 전이커널 위에서 보상만
고친 MDP(`Config.credit_loss=True`)를 정확히 다시 푼다.

산출: results/w_model/reopt.csv, 그리고 콘솔에 가법분해 확정치.
"""
import sys, json, csv, time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]/'src'))

CELLS = [(2, 2), (2, 3), (2, 4), (3, 2), (3, 3)]
# w1~w3 스크립트다. 인스턴스가 레이·코나·SM3 로 고정돼 있고 커밋된 결과도
# 그 조합·구 가격으로 냈으므로, 가격도 params_v5.json 으로 고정한다.
# w4 재캘리브레이션된 params.json 을 여기에 물리면 구 결과와 섞인다.
SEL = ['레이', '코나', 'SM3']
CACHE = '/home/user/batdiag-cache'
OUT = Path(__file__).resolve().parents[1]/'results'/'w_model'


def solve(I, dev, stream, tag):
    from battery_diag.build import build
    from battery_diag.streambuild import build_stream
    from battery_diag.exact import ExactSolver
    from battery_diag.bigexact import StreamSolver
    from battery_diag import policies as pol
    if stream:
        A = build_stream(I, cache_dir=CACHE, tag=tag, log=print)
        S = StreamSolver(A, device=dev)
        g, h, it = S.solve(acts0=pol.index_myopic(I), log=print)
    else:
        A = build(I, cache_dir=CACHE, tag=tag)
        S = ExactSolver(A, device=dev)
        g, h, it = S.solve()
    return S, A, float(g), h, it


def main():
    import numpy as np, torch
    from battery_diag.data import load_cached
    from battery_diag.instance import Instance, PriceParams, Config

    root = Path(__file__).resolve().parents[1]
    FLEET, FS = load_cached(str(root/'data'))
    tot = sum(FLEET[t][0] for t in SEL)
    types = {t: (FLEET[t][1], FLEET[t][2], FLEET[t][3], FLEET[t][4], FLEET[t][0]/tot)
             for t in SEL}
    price = PriceParams.from_json(root/'data'/'params_v5.json')
    dev = 'cuda' if torch.cuda.is_available() else 'cpu'
    OUT.mkdir(parents=True, exist_ok=True)
    tag = ','.join(SEL)

    fixed = {}
    f = OUT/'loss_correction.csv'
    if f.exists():
        for r in csv.DictReader(f.open()):
            fixed[(int(r['NMAX']), int(r['SMAX']))] = float(r['g_corrected_identity'])

    rows = []
    for NMAX, SMAX in CELLS:
        t0 = time.time()
        kw = dict(Mcyc=1, Cp=500000, Cf=20000, phi=1.0, NARR=4,
                  NMAX=NMAX, SMAX=SMAX, F_E=FS['F_E'], F_U=FS['F_U'])
        I0 = Instance(types, price, Config(**kw))
        nS = len(I0.ST)
        stream = nS >= 14000 or (NMAX, SMAX) == (2, 4)
        S0, A0, g0, h0, _ = solve(I0, dev, stream, tag)
        acts0 = S0.greedy(h0)
        del S0, A0, h0
        if dev == 'cuda': torch.cuda.empty_cache()

        I1 = Instance(types, price, Config(credit_loss=True, **kw))
        S1, A1, g1, h1, it1 = solve(I1, dev, stream, tag)
        acts1 = S1.greedy(h1)
        g_fixed_reeval, _ = S1.evaluate(acts0)      # 보정 MDP 에서 기존정책 재평가
        chg = float((acts0 != acts1).mean())
        del S1, A1, h1
        if dev == 'cuda': torch.cuda.empty_cache()

        gf = fixed.get((NMAX, SMAX), float(g_fixed_reeval))
        rows.append(dict(NMAX=NMAX, SMAX=SMAX, nS=nS, g_orig=g0,
                         g_fixed=float(g_fixed_reeval), g_fixed_w2=gf,
                         g_reopt=g1, uplift=g1-float(g_fixed_reeval),
                         policy_change=chg, sec=time.time()-t0))
        print(f'NMAX={NMAX} SMAX={SMAX}  원본 {g0:>12,.2f}  고정보정 {g_fixed_reeval:>12,.2f}  '
              f'재최적 {g1:>12,.2f}  추가이득 {g1-float(g_fixed_reeval):>9,.2f}  '
              f'정책변경 {100*chg:5.2f}%  ({time.time()-t0:.0f}s)', flush=True)

    with (OUT/'reopt.csv').open('w', newline='') as fh:
        w = csv.DictWriter(fh, fieldnames=list(rows[0])); w.writeheader(); w.writerows(rows)
    print('\n→', OUT/'reopt.csv')

    G = {(r['NMAX'], r['SMAX']): r for r in rows}
    print()
    for tag2, key in (('원본', 'g_orig'), ('고정정책 보정', 'g_fixed'), ('재최적 보정', 'g_reopt')):
        g22, g23, g32, g33 = (G[(2, 2)][key], G[(2, 3)][key], G[(3, 2)][key], G[(3, 3)][key])
        dS2, dS3 = g23-g22, g33-g32
        dN2, dN3 = g32-g22, g33-g23
        print(f'[{tag2}] SMAX 2→3: {dS2:>10,.0f} / {dS3:>10,.0f}   '
              f'NMAX 2→3: {dN2:>10,.0f} / {dN3:>10,.0f}   '
              f'비 {dN2/dS2:.2f}배   교호작용 {dS3-dS2:>9,.0f} ({100*abs(dS3-dS2)/g33:.3f}%)')
    print(json.dumps(rows, ensure_ascii=False, default=float))


if __name__ == '__main__':
    main()
