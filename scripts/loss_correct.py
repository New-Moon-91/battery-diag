#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""무료 소실의 편향 **방향**과 크기 — 기존 정식화 g* 의 소실보정.

논지. 기존 모형에서 야적장이 차면 도착 배터리는 수익 0·비용 0 으로 사라진다.
W-정식화에서는 같은 배터리가 재활용 매각으로 나가며 p_rc x kWh 를 번다.
**두 경우 모두 배터리는 창고 자리를 먹지 않고 다음 상태도 같다.** 즉 전이커널은
동일하고 보상만 다르다. 따라서 기존 g* 는 **아래로** 편향돼 있다 (매각수익 누락).

보정값을 서로 독립인 두 경로로 구해 대조한다.
  (1) 정상분포 항등식 — 평균보상 MDP 에서 g = d·r 이므로, 보상에 Δr 을 더하면
      g 는 정확히 d·Δr 만큼 는다. Δr(s) = 그 상태에서의 기대 소실가치.
  (2) 보상수정 정책평가 — 정책행의 rsa 에 Δr 을 더하고 멱반복을 다시 돌린다.
둘이 일치하면 계산 오류가 아니라 구조적 결론이다.

주의. 이것은 **기존 최적정책을 고정한 채** 보상만 고친 값이라 W-모형의 g* 자체는
아니다. W-모형은 용량구조도 다르고(차종별 상한 → 창고 총량) 재최적화 여지도 있다.
여기서 확정되는 것은 편향의 **방향**과 그 하한이다.

    python scripts/loss_correct.py [출력.csv]
"""
import sys, json
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]/'src'))

CELLS = [(2, 2), (2, 3), (2, 4), (3, 2), (3, 3)]
SEL = ['레이', '코나', 'SM3']
CACHE = '/home/user/batdiag-cache'


def loss_per_state(I, acts, cfg):
    """상태별 (기대 소실대수, 기대 소실가치) — 선택된 행동 기준."""
    import numpy as np
    nS = len(I.ST); lam, NARR = cfg.lam, cfg.NARR
    n_lost = np.zeros(nS); v_lost = np.zeros(nS)
    for si, st in enumerate(I.ST):
        sv, fv, pu, ss, ps = I.actions(st)[acts[si]]
        rem = tuple(st[0][k]-sv[k]-fv[k]-pu[k] for k in range(len(I.TY)))
        o = {rem: (1.0, 0.0, 0.0)}
        for _ in range(NARR):
            nx = {}
            for nn, (p0, cl, cv) in o.items():
                def add(key, p, dl, dv, cl=cl, cv=cv):
                    a0, b0, c0 = nx.get(key, (0., 0., 0.))
                    nx[key] = (a0+p, b0+p*(cl+dl), c0+p*(cv+dv))
                add(nn, p0*(1-lam), 0., 0.)
                for k, t in enumerate(I.TY):
                    p = p0*lam*I.MIX[t]
                    if nn[k] >= cfg.NMAX:
                        add(nn, p, 1., I.VS[t])
                    else:
                        n2 = list(nn); n2[k] += 1
                        add(tuple(n2), p, 0., 0.)
            o = {k: (v[0], v[1]/max(v[0], 1e-300), v[2]/max(v[0], 1e-300))
                 for k, v in nx.items()}
        n_lost[si] = sum(p*cl for p, cl, cv in o.values())
        v_lost[si] = sum(p*cv for p, cl, cv in o.values())
    return n_lost, v_lost


def main():
    import numpy as np, torch
    from battery_diag.data import load_cached
    from battery_diag.instance import Instance, PriceParams, Config
    from battery_diag.build import build
    from battery_diag.streambuild import build_stream
    from battery_diag.exact import ExactSolver
    from battery_diag.bigexact import StreamSolver
    from battery_diag import policies as pol

    out_csv = Path(sys.argv[1]) if len(sys.argv) > 1 else None
    root = Path(__file__).resolve().parents[1]
    FLEET, FS = load_cached(str(root/'data'))
    tot = sum(FLEET[t][0] for t in SEL)
    types = {t: (FLEET[t][1], FLEET[t][2], FLEET[t][3], FLEET[t][4], FLEET[t][0]/tot)
             for t in SEL}
    price = PriceParams.from_json(root/'data'/'params.json')
    dev = 'cuda' if torch.cuda.is_available() else 'cpu'

    rows = []
    for NMAX, SMAX in CELLS:
        cfg = Config(Mcyc=1, Cp=500000, Cf=20000, phi=1.0, NARR=4,
                     NMAX=NMAX, SMAX=SMAX, F_E=FS['F_E'], F_U=FS['F_U'])
        I = Instance(types, price, cfg); nS = len(I.ST)
        stream = nS >= 14000 or (NMAX, SMAX) == (2, 4)
        if stream:
            A = build_stream(I, cache_dir=CACHE, tag=','.join(SEL))
            S = StreamSolver(A, device=dev); g0, h, _ = S.solve(acts0=pol.index_myopic(I))
        else:
            A = build(I, cache_dir=CACHE, tag=','.join(SEL))
            S = ExactSolver(A, device=dev); g0, h, _ = S.solve()
        acts = S.greedy(h)
        d = np.asarray(S.stationary(acts)); d = d/d.sum()
        n_lost, v_lost = loss_per_state(I, acts, cfg)
        EL = float(d @ n_lost); EV = float(d @ v_lost)

        # (1) 정상분포 항등식
        g_id = g0 + EV
        # (2) 보상수정 정책평가 — 정책행의 rsa 에만 Δr 을 더하고 다시 푼다.
        #     첫 솔버를 먼저 놓아준다 — 두 개를 동시에 GPU 에 올리면 16GB 를 넘는다.
        aptr = np.asarray(A['aptr']); rsa2 = np.array(A['rsa'], dtype=np.float64, copy=True)
        rsa2[aptr[:-1] + acts] += v_lost
        del S, h
        torch.cuda.empty_cache() if dev == 'cuda' else None
        A2 = dict(A); A2['rsa'] = rsa2
        S2 = (StreamSolver(A2, device=dev) if stream else ExactSolver(A2, device=dev))
        g_ev, _ = S2.evaluate(acts)

        rows.append(dict(NMAX=NMAX, SMAX=SMAX, nS=nS, gstar=float(g0),
                         lost_per_period=EL, lost_value=EV,
                         g_corrected_identity=float(g_id),
                         g_corrected_eval=float(g_ev),
                         mismatch=abs(g_id-g_ev)))
        print(f'NMAX={NMAX} SMAX={SMAX}  g*={g0:>12,.2f}  소실가치={EV:>10,.2f}  '
              f'보정 g*: 항등식 {g_id:>12,.2f} / 재평가 {g_ev:>12,.2f}  '
              f'차이 {abs(g_id-g_ev):.3e}', flush=True)
        del S2, A, A2
        torch.cuda.empty_cache() if dev == 'cuda' else None

    print()
    G = {(r['NMAX'], r['SMAX']): r for r in rows}
    for tag, key in (('원본', 'gstar'), ('소실보정', 'g_corrected_identity')):
        g22, g23, g32, g33 = (G[(2, 2)][key], G[(2, 3)][key], G[(3, 2)][key], G[(3, 3)][key])
        dS_n2 = g23-g22; dS_n3 = g33-g32; dN_s2 = g32-g22; dN_s3 = g33-g23
        inter = dS_n3 - dS_n2
        print(f'[{tag}] SMAX 2→3 효과: NMAX=2 에서 {dS_n2:>10,.0f} / NMAX=3 에서 {dS_n3:>10,.0f}')
        print(f'         NMAX 2→3 효과: SMAX=2 에서 {dN_s2:>10,.0f} / SMAX=3 에서 {dN_s3:>10,.0f}')
        print(f'         NMAX/SMAX 비 {dN_s2/dS_n2:.2f}배   교호작용 {inter:>10,.0f} '
              f'(g* 대비 {100*abs(inter)/g33:.3f}%)')
    if out_csv:
        import csv
        out_csv.parent.mkdir(parents=True, exist_ok=True)
        with open(out_csv, 'w', newline='') as fh:
            w = csv.DictWriter(fh, fieldnames=list(rows[0])); w.writeheader(); w.writerows(rows)
        print('\n→', out_csv)
    print(json.dumps(rows, ensure_ascii=False, default=float))


if __name__ == '__main__':
    main()
