#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""벤치마크 '갭' 이 무료 소실에 오염됐는가 — 정책별 소실가치로 검정.

v5b·v5c 의 방법론 결론(표현력 감사, 자기회귀 디코더, 갭 0.118%)은 "같은 모형을
푸는 두 방법의 비교라 모형 오염과 무관" 이라는 것이 w2 이전의 판단이었다.
그러나 갭 = (g* - g_pi)/g* 이고 **소실가치는 정책마다 다르다**. 재고를 오래 들고
있는 정책일수록 야적장이 더 자주 차서 더 많이 잃는다. 따라서 보정 전후로 갭이
움직일 수 있고, 무관성은 가정이 아니라 검정 대상이다.

각 정책 pi 에 대해 EV(pi) = 정상분포 가중 소실가치를 구해
  갭_보정 = ((g* + EV*) - (g_pi + EV_pi)) / (g* + EV*)
를 원래 갭과 비교한다.

    python scripts/gap_contamination.py [NMAX] [SMAX]
"""
import sys, json
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]/'src'))
sys.path.insert(0, str(Path(__file__).resolve().parent))
from loss_correct import loss_per_state

SEL = ['레이', '코나', 'SM3']
CACHE = '/home/user/batdiag-cache'


def main():
    import numpy as np, torch
    from battery_diag.data import load_cached
    from battery_diag.instance import Instance, PriceParams, Config
    from battery_diag.build import build
    from battery_diag.exact import ExactSolver
    from battery_diag import policies as pol

    NMAX = int(sys.argv[1]) if len(sys.argv) > 1 else 2
    SMAX = int(sys.argv[2]) if len(sys.argv) > 2 else 2
    root = Path(__file__).resolve().parents[1]
    FLEET, FS = load_cached(str(root/'data'))
    tot = sum(FLEET[t][0] for t in SEL)
    types = {t: (FLEET[t][1], FLEET[t][2], FLEET[t][3], FLEET[t][4], FLEET[t][0]/tot)
             for t in SEL}
    price = PriceParams.from_json(root/'data'/'params.json')
    dev = 'cuda' if torch.cuda.is_available() else 'cpu'
    cfg = Config(Mcyc=1, Cp=500000, Cf=20000, phi=1.0, NARR=4,
                 NMAX=NMAX, SMAX=SMAX, F_E=FS['F_E'], F_U=FS['F_U'])
    I = Instance(types, price, cfg)
    A = build(I, cache_dir=CACHE, tag=','.join(SEL))
    S = ExactSolver(A, device=dev)
    gstar, h, _ = S.solve()
    pols = {'OPT': S.greedy(h), 'B1': pol.b1_sell_all(I), 'B2': pol.b2_no_screening(I),
            'INDEX': pol.index_myopic(I)}
    bf = {thr: pol.b_fast(I, thr) for thr in range(cfg.SB)}
    bfv = {thr: S.evaluate(a)[0] for thr, a in bf.items()}
    pols['B_fast'] = bf[max(bfv, key=bfv.get)]

    print(f'NMAX={NMAX} SMAX={SMAX}  g*={gstar:,.2f}')
    print(f"{'정책':>7} {'g':>14} {'소실가치':>11} {'보정 g':>14} "
          f"{'갭 원본':>9} {'갭 보정':>9} {'차이(%p)':>9}")
    out = {}
    gs = {k: (gstar if k == 'OPT' else float(S.evaluate(a)[0])) for k, a in pols.items()}
    evs = {}
    for k, a in pols.items():
        d = np.asarray(S.stationary(a)); d = d/d.sum()
        _, v_lost = loss_per_state(I, a, cfg)
        evs[k] = float(d @ v_lost)
    gc = {k: gs[k]+evs[k] for k in pols}
    for k in ('OPT', 'B1', 'B2', 'INDEX', 'B_fast'):
        gap0 = 100*(gs['OPT']-gs[k])/gs['OPT']
        gap1 = 100*(gc['OPT']-gc[k])/gc['OPT']
        print(f'{k:>7} {gs[k]:>14,.0f} {evs[k]:>11,.0f} {gc[k]:>14,.0f} '
              f'{gap0:>8.3f}% {gap1:>8.3f}% {gap1-gap0:>+8.3f}')
        out[k] = dict(g=gs[k], ev=evs[k], g_corr=gc[k], gap=gap0, gap_corr=gap1)
    p = root/'results'/'w_model'/f'gap_contamination_N{NMAX}S{SMAX}.json'
    p.write_text(json.dumps(out, ensure_ascii=False, indent=1, default=float))
    print('→', p)


if __name__ == '__main__':
    main()
