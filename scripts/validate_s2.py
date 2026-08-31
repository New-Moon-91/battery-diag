#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""S2 검증 — 롤아웃 h 가 정확 h 를 대신할 수 있는가.

    python scripts/validate_s2.py [--W 4] [--grid]

보고 지표 (중요도 순)
  1. **개선 단계 행동 일치율** — argmax_a[r + ΣP·ĥ] 가 정확 h 로 고른 것과 같은 비율.
     h 의 절대오차보다 이것이 관건이다. 개선이 같은 행동을 고르면 DCL 은 같은 길을 간다.
  2. 상관계수 (중심화 후) — 순서를 얼마나 보존하는가
  3. 상대오차 분포 — |ĥ−h| / (h 의 범위)
  4. ĝ 오차
"""
import argparse, csv, json, sys, time
from pathlib import Path
import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT/'src')); sys.path.insert(0, str(ROOT/'scripts'))
from battery_diag.simulate import Simulator
from battery_diag.rollout import estimate_g, estimate_h, center
from battery_diag import policies as pol
from validate_s1 import build

OUT = ROOT/'results'/'w7'


def exact(I, dev='cuda'):
    import torch
    from battery_diag.build import build as build_ram
    from battery_diag.exact import ExactSolver, NumpySolver
    A = build_ram(I, cache_dir='/home/user/batdiag-cache', tag=','.join(I.TY))
    S = (ExactSolver(A, device=dev) if torch.cuda.is_available() else NumpySolver(A))
    return S


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--W', type=int, default=4)
    ap.add_argument('--H', nargs='+', type=int, default=[60])
    ap.add_argument('--M', nargs='+', type=int, default=[20])
    ap.add_argument('--seed', type=int, default=0)
    ap.add_argument('--gT', type=int, default=200_000)
    ap.add_argument('--policy', default='index', choices=('index', 'opt'))
    ap.add_argument('--no-crn', action='store_true', help='공통난수를 끈다 (대조군)')
    a = ap.parse_args()
    OUT.mkdir(parents=True, exist_ok=True)

    I = build(a.W)
    S = exact(I)
    sim = Simulator(I, arrivals='binomial')
    nS = len(I.ST)

    acts = pol.index_myopic(I)
    if a.policy == 'opt':
        g0, h0, _ = S.solve()
        acts = S.greedy(h0)
    g_ex, h_ex = S.evaluate(acts)
    h_ex = np.asarray(h_ex, float)
    base_best = S.greedy(h_ex)
    g_base = float(S.evaluate(base_best)[0])
    # 기준선 — h 를 아예 안 쓰면(근시안) 얼마나 맞나. 일치율의 «바닥» 이다.
    zero_best = S.greedy(np.zeros(nS))
    agree_zero = float((zero_best == base_best).mean())
    g_zero = float(S.evaluate(zero_best)[0])
    print(f'W={a.W}  nS={nS:,}  정책={a.policy}  정확 g={g_ex:,.2f}', flush=True)
    print(f'  개선 1회(정확 h) → g={g_base:,.2f}', flush=True)
    print(f'  기준선: 근시안(h=0) 행동일치 {100*agree_zero:.2f}%, '
          f'g={g_zero:,.2f} (개선분의 {100*(g_zero-g_ex)/(g_base-g_ex):.1f}%)', flush=True)

    t0 = time.time()
    g_hat = estimate_g(sim, I, acts, T=a.gT, seed=a.seed)
    tg = time.time()-t0
    print(f'  ĝ = {g_hat:,.2f}   오차 {100*(g_hat/g_ex-1):+.3f}%   '
          f'({a.gT:,} 스텝, {tg:.0f}s)', flush=True)

    rows = []
    for H in a.H:
        for M in a.M:
            t0 = time.time()
            h_hat = estimate_h(sim, I, acts, g_hat, H=H, M=M, seed=a.seed,
                                   crn=not a.no_crn)
            sec = time.time()-t0
            ce, ch = center(h_ex), center(h_hat)
            corr = float(np.corrcoef(ce, ch)[0, 1])
            rng_h = float(ce.max()-ce.min())
            relerr = np.abs(ch-ce)/rng_h
            best = S.greedy(h_hat)
            agree = float((best == base_best).mean())
            # 행동 일치율보다 중요한 것 — 그 행동으로 만든 정책의 **실제 가치**.
            # 근사 h 로 개선한 정책을 정확히 평가해 정확 h 로 개선한 것과 견준다.
            g_hat_pol = float(S.evaluate(best)[0])
            recov = (g_hat_pol-g_ex)/(g_base-g_ex) if g_base > g_ex else float('nan')
            slope = float(np.dot(ch, ce)/np.dot(ce, ce))
            rows.append(dict(W=a.W, H=H, M=M, crn=not a.no_crn, slope=slope,
                             steps=nS*H*M, sec=round(sec, 1),
                             g_hat=g_hat, g_err_pct=100*(g_hat/g_ex-1),
                             corr=corr, relerr_med=float(np.median(relerr)),
                             relerr_p90=float(np.percentile(relerr, 90)),
                             relerr_max=float(relerr.max()), act_agree=agree,
                             g_policy=g_hat_pol, improve_recovered=recov,
                             g_policy_gap_vs_exact_improve=100*(g_base-g_hat_pol)/g_base))
            print(f'  H={H:3d} M={M:3d}  스텝 {nS*H*M:>11,}  {sec:6.0f}s  '
                  f'상관 {corr:.4f}  기울기 {slope:.3f}  '
                  f'행동일치 {100*agree:5.2f}%  **개선회수 {100*recov:6.2f}%**  '
                  f'(g={g_hat_pol:,.0f})', flush=True)

    with open(OUT/f's2_accuracy_W{a.W}{"_nocrn" if a.no_crn else ""}.csv', 'w', newline='', encoding='utf-8') as fh:
        w = csv.DictWriter(fh, fieldnames=list(rows[0])); w.writeheader(); w.writerows(rows)
    json.dump(dict(W=a.W, policy=a.policy, g_exact=float(g_ex), g_hat=float(g_hat),
                   gT=a.gT, rows=rows),
              open(OUT/f's2_summary_W{a.W}{"_nocrn" if a.no_crn else ""}.json', 'w', encoding='utf-8'),
              ensure_ascii=False, indent=1)
    print(f'\n→ {OUT}/s2_accuracy_W{a.W}.csv')


if __name__ == '__main__':
    main()
