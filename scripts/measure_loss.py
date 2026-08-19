#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""기존 NMAX 정식화의 '무료 소실' 오염도 측정.

기존 모형은 도착을 `n2[k] = min(n2[k]+1, NMAX)` 로 잘라낸다. 야적장이 차면
도착 배터리가 **비용도 수익도 없이 사라진다**. 현실 대응물이 없고 g* 를 위로
편향시킨다 (팔았으면 받았을 p_rc x kWh 를 안 받았는데 재고비도 안 냈다).

각 칸에서 기존 최적정책의 정상분포를 구해 다음을 잰다.
  (a) 소실 발생확률 — 그 기간에 1대 이상 잃을 확률.
      행동 직후 재고만 보면 0 이 나온다. 최적정책은 재고를 비우고 기간을 시작하지만
      소실은 NARR 개 도착 슬롯이 **순차로** 채워지는 도중에 생기기 때문이다.
  (b) 기간당 평균 소실 대수
  (c) 기간당 소실 가치 (p_rc x kWh 기준) 와 g* 대비 비율

    python scripts/measure_loss.py [출력.md]
"""
import sys, json
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]/'src'))

CELLS = [(2, 2), (2, 3), (2, 4), (3, 2), (3, 3)]
SEL = ['레이', '코나', 'SM3']
CACHE = '/home/user/batdiag-cache'


def main():
    import numpy as np, torch
    from battery_diag.data import load_cached
    from battery_diag.instance import Instance, PriceParams, Config
    from battery_diag.build import build
    from battery_diag.streambuild import build_stream
    from battery_diag.exact import ExactSolver
    from battery_diag.bigexact import StreamSolver
    from battery_diag import policies as pol

    out_md = Path(sys.argv[1]) if len(sys.argv) > 1 else None
    FLEET, FS = load_cached(str(Path(__file__).resolve().parents[1]/'data'))
    tot = sum(FLEET[t][0] for t in SEL)
    types = {t: (FLEET[t][1], FLEET[t][2], FLEET[t][3], FLEET[t][4], FLEET[t][0]/tot)
             for t in SEL}
    price = PriceParams.from_json(Path(__file__).resolve().parents[1]/'data'/'params.json')
    dev = 'cuda' if torch.cuda.is_available() else 'cpu'

    rows = []
    for NMAX, SMAX in CELLS:
        cfg = Config(Mcyc=1, Cp=500000, Cf=20000, phi=1.0, NARR=4,
                     NMAX=NMAX, SMAX=SMAX, F_E=FS['F_E'], F_U=FS['F_U'])
        I = Instance(types, price, cfg); nS = len(I.ST)
        if nS >= 14000 or (NMAX, SMAX) == (2, 4):
            A = build_stream(I, cache_dir=CACHE, tag=','.join(SEL))
            S = StreamSolver(A, device=dev); gstar, h, _ = S.solve(acts0=pol.index_myopic(I))
        else:
            A = build(I, cache_dir=CACHE, tag=','.join(SEL))
            S = ExactSolver(A, device=dev); gstar, h, _ = S.solve()
        acts = S.greedy(h)
        d = np.asarray(S.stationary(acts))
        d = d/d.sum()

        # 상태별 소실 기대치. 행동 뒤 남는 미검사 rem 에서 시작해 도착을 축차 적용,
        # 각 슬롯에서 상한에 걸리는 순간이 곧 소실이다 (arr() 과 같은 논리).
        lam, NARR = cfg.lam, cfg.NARR
        n_lost = np.zeros(nS); v_lost = np.zeros(nS); p_sat = np.zeros(nS)
        for si, st in enumerate(I.ST):
            a = I.actions(st)[acts[si]]
            sv, fv, pu, ss, ps = a
            rem = tuple(st[0][k]-sv[k]-fv[k]-pu[k] for k in range(len(I.TY)))
            # (누적확률, 평균 소실대수, 평균 소실가치, 1대 이상 소실 확률)
            o = {rem: (1.0, 0.0, 0.0, 0.0)}
            for _ in range(NARR):
                nx = {}
                for nn, (p0, cl, cv, ca) in o.items():
                    def add(key, p, dl, dv, da, cl=cl, cv=cv, ca=ca):
                        a0, b0, c0, d0 = nx.get(key, (0., 0., 0., 0.))
                        nx[key] = (a0+p, b0+p*(cl+dl), c0+p*(cv+dv), d0+p*max(ca, da))
                    add(nn, p0*(1-lam), 0., 0., 0.)
                    for k, t in enumerate(I.TY):
                        p = p0*lam*I.MIX[t]
                        if nn[k] >= cfg.NMAX:                 # 상한 → 무료 소실
                            add(nn, p, 1., I.VS[t], 1.)
                        else:
                            n2 = list(nn); n2[k] += 1
                            add(tuple(n2), p, 0., 0., 0.)
                o = {k: (v[0], v[1]/max(v[0], 1e-300), v[2]/max(v[0], 1e-300),
                         v[3]/max(v[0], 1e-300)) for k, v in nx.items()}
            n_lost[si] = sum(p*cl for p, cl, cv, ca in o.values())
            v_lost[si] = sum(p*cv for p, cl, cv, ca in o.values())
            p_sat[si] = sum(p*ca for p, cl, cv, ca in o.values())

        EL = float(d @ n_lost); EV = float(d @ v_lost); PS = float(d @ p_sat)
        rows.append(dict(NMAX=NMAX, SMAX=SMAX, nS=nS, gstar=gstar,
                         p_saturated=PS, lost_per_period=EL, lost_value=EV,
                         lost_share=100*EV/gstar))
        print(f'NMAX={NMAX} SMAX={SMAX}  nS={nS:>6,}  g*={gstar:>12,.0f}  '
              f'소실확률 {100*PS:6.2f}%  소실 {EL:6.3f}대/기간  '
              f'소실가치 {EV:>11,.0f}  g* 대비 {100*EV/gstar:5.2f}%', flush=True)

    if out_md:
        out_md.parent.mkdir(parents=True, exist_ok=True)
        L = ['# 기존 NMAX 정식화의 무료 소실 오염도', '',
             '`n2[k] = min(n2[k]+1, NMAX)` — 야적장이 차면 도착 배터리가 비용도 수익도',
             '없이 사라진다. 아래는 각 칸의 기존 최적정책 정상분포에서 잰 값이다.',
             '소실가치는 재활용 매각가 `p_rc x kWh` 기준 — W-정식화였다면 실제로',
             '받았을 금액이다.', '',
             '| 버퍼 | \\|S\\| | g* | 소실 발생확률/기간 | 소실 대수/기간 | 소실가치/기간 | g* 대비 |',
             '|---|---:|---:|---:|---:|---:|---:|']
        for r in rows:
            L.append(f"| NMAX={r['NMAX']}, SMAX={r['SMAX']} | {r['nS']:,} | {r['gstar']:,.0f} | "
                     f"{100*r['p_saturated']:.2f}% | {r['lost_per_period']:.3f} | "
                     f"{r['lost_value']:,.0f} | {r['lost_share']:.2f}% |")
        L += ['', '소실 발생확률은 "그 기간에 1대 이상 잃을" 확률이다. 최적정책은 재고를 비우고',
              '기간을 시작하므로 행동 직후 포화는 0 이지만, 소실은 NARR 개 도착 슬롯이 순차로',
              '채워지는 도중에 생긴다. 소실 대수·가치는 정상분포 가중 기대값이다.']
        out_md.write_text('\n'.join(L)+'\n')
        print('\n→', out_md)
    print(json.dumps(rows, ensure_ascii=False, default=float))


if __name__ == '__main__':
    main()
