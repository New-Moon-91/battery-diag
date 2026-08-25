#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""S1 검증 — 시뮬레이터의 전이 빈도가 정확 전이행렬과 일치하는가.

    python scripts/validate_s1.py [--W 4 5 6] [--pairs 40] [--draws 20000]

각 (상태, 행동) 표본마다
  · 총변동거리 TVD = ½ Σ|p̂ − p|
  · 카이제곱 적합도 (기대도수 5 미만 칸은 하나로 묶음) 와 p-값
  · 기대보상 일치 — (r̄ − E[r]) / SE 의 z 값
을 낸다. p-값이 균등분포에 가깝고 |z| 가 표준정규처럼 흩어지면 통과다.
**여기서 어긋나면 S2·S3 는 무의미하다.**
"""
import argparse, csv, json, math, sys
from pathlib import Path
import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT/'src'))
from battery_diag.data import SEL_W5 as SEL, fleet_w5, load_cached
from battery_diag.instance import Instance, PriceW5, Config
from battery_diag.simulate import Simulator

OUT = ROOT/'results'/'w7'


def build(W, types_sel=None):
    sel = list(types_sel or SEL)
    _, FS = load_cached(str(ROOT/'data'))
    F = fleet_w5(ROOT/'data', sel=sel)
    tot = sum(F[t][0] for t in sel)
    types = {t: (F[t][1], F[t][2], F[t][3], F[t][4], F[t][0]/tot) for t in sel}
    cfg = Config(Mcyc=1, Cp=500000, Cf=20000, phi=1.0, NARR=4, W=W,
                 F_E=FS['F_E'], F_U=FS['F_U'])
    return Instance(types, PriceW5.from_json(ROOT/'data'/'params_w5.json'), cfg)


def chi2_pool(obs, exp, n):
    """기대도수 5 미만 칸을 꼬리로 묶어 카이제곱. (chi2, df) 반환."""
    idx = np.argsort(-exp)
    obs, exp = obs[idx], exp[idx]
    keep = exp*n >= 5
    if keep.sum() < 1:
        return float('nan'), 0
    o = list(obs[keep]*n); e = list(exp[keep]*n)
    to, te = float(obs[~keep].sum()*n), float(exp[~keep].sum()*n)
    if te > 0:
        o.append(to); e.append(te)
    o, e = np.array(o), np.array(e)
    return float(((o-e)**2/e).sum()), len(o)-1


def chi2_sf(x, df):
    """카이제곱 생존함수 — scipy 없이 정규근사(Wilson–Hilferty)."""
    if df <= 0 or not np.isfinite(x):
        return float('nan')
    z = ((x/df)**(1/3) - (1 - 2/(9*df))) / np.sqrt(2/(9*df))
    return float(0.5*math.erfc(z/math.sqrt(2)))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--W', nargs='+', type=int, default=[4, 5, 6])
    ap.add_argument('--pairs', type=int, default=40, help='W 마다 검사할 (상태,행동) 수')
    ap.add_argument('--draws', type=int, default=20000, help='(상태,행동) 마다 표본수')
    ap.add_argument('--seed', type=int, default=0)
    a = ap.parse_args()
    OUT.mkdir(parents=True, exist_ok=True)

    rows = []
    for W in a.W:
        I = build(W)
        sim = Simulator(I, arrivals='binomial')
        rng = np.random.default_rng(a.seed + W)
        nS = len(I.ST)
        # 점유가 0 인 상태만 뽑히면 검정이 시시해진다 — 점유가 있는 상태를 고른다.
        cand = [i for i in rng.choice(nS, size=min(nS, 4000), replace=False)
                if sum(I.ST[i][0]) + len(I.ST[i][1]) > 0]
        picks = cand[:a.pairs] if len(cand) >= a.pairs else cand
        print(f'=== W={W}  nS={nS:,}  검사 (상태,행동) {len(picks)}개 × {a.draws:,} 표본 ===',
              flush=True)
        for si in picks:
            st = I.ST[si]
            acts = I.actions(st)
            ai = int(rng.integers(len(acts)))
            act = acts[ai]
            # 정확 분포
            ex = {}
            er = 0.0
            for p, j, r in I.step_dist(st, act):
                ex[j] = ex.get(j, 0.0) + p
                er += p*r
            # 보상의 정확 2차모멘트 → 표준편차
            er2 = sum(p*r*r for p, _j, r in I.step_dist(st, act))
            sd = np.sqrt(max(er2 - er*er, 0.0))
            keys = sorted(ex)
            kidx = {k: i for i, k in enumerate(keys)}
            exp = np.array([ex[k] for k in keys])
            cnt = np.zeros(len(keys))
            rs = np.empty(a.draws)
            bad = 0
            for d in range(a.draws):
                r, ns = sim.step(st, act, rng)
                rs[d] = r
                j = I.SI.get(ns)
                if j is None or j not in kidx:
                    bad += 1
                else:
                    cnt[kidx[j]] += 1
            obs = cnt/a.draws
            tvd = 0.5*np.abs(obs-exp).sum()
            chi2, df = chi2_pool(obs, exp, a.draws)
            p = chi2_sf(chi2, df)
            se = sd/np.sqrt(a.draws)
            z = (rs.mean()-er)/se if se > 0 else 0.0
            rows.append(dict(W=W, state=si, action=ai, n_out=len(keys), draws=a.draws,
                             tvd=tvd, chi2=chi2, df=df, pval=p, z_reward=z,
                             unseen=bad, E_r_exact=er, r_bar=float(rs.mean())))
            if bad:
                print(f'  [경고] 상태 {si}: 정확분포에 없는 다음상태 {bad}건', flush=True)
        sub = [r for r in rows if r['W'] == W]
        print(f'  TVD 중앙 {np.median([r["tvd"] for r in sub]):.5f} '
              f'최대 {max(r["tvd"] for r in sub):.5f}')
        print(f'  p-값 중앙 {np.median([r["pval"] for r in sub]):.3f}  '
              f'p<0.01 인 표본 {sum(1 for r in sub if r["pval"] < 0.01)}/{len(sub)}')
        print(f'  |z| 중앙 {np.median([abs(r["z_reward"]) for r in sub]):.3f}  '
              f'최대 {max(abs(r["z_reward"]) for r in sub):.3f}')
        print(f'  정확분포 밖 전이 총 {sum(r["unseen"] for r in sub)}건', flush=True)

    with open(OUT/'s1_transition_check.csv', 'w', newline='', encoding='utf-8') as fh:
        w = csv.DictWriter(fh, fieldnames=list(rows[0])); w.writeheader(); w.writerows(rows)
    summ = {}
    for W in a.W:
        sub = [r for r in rows if r['W'] == W]
        summ[str(W)] = dict(
            n=len(sub), draws=a.draws,
            tvd_med=float(np.median([r['tvd'] for r in sub])),
            tvd_max=float(max(r['tvd'] for r in sub)),
            pval_med=float(np.median([r['pval'] for r in sub])),
            n_p_lt_01=int(sum(1 for r in sub if r['pval'] < 0.01)),
            absz_med=float(np.median([abs(r['z_reward']) for r in sub])),
            absz_max=float(max(abs(r['z_reward']) for r in sub)),
            unseen=int(sum(r['unseen'] for r in sub)))
    json.dump(summ, open(OUT/'s1_summary.json', 'w', encoding='utf-8'),
              ensure_ascii=False, indent=1)
    print(f'\n→ {OUT}/s1_transition_check.csv, s1_summary.json')


if __name__ == '__main__':
    main()
