#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""scale1 결과 표 — 상계 대비 갭(원값·통제변량 보정), 시드 표준오차, 운영통계.

    python scripts/make_digest_scale1.py results/scale/*.json

통제변량: B1(전량 즉시매각)의 기대값은 닫힌식 E[B1] = λ Σ_k mix_k V^S_k 이다.
같은 시드는 모든 정책이 같은 도착열을 보므로, 시드 s 의 B1 편차 (B1_s − E[B1]) 는
«그 시드의 도착이 얼마나 풍부했나» 의 측도다. 정책 g_s 에서 이를 빼면 도착 잡음이 대부분
제거된다 (gap_cv). 두 갭이 비슷하면 보정이 결론을 바꾸지 않는다는 뜻이다.
"""
import sys, json
from pathlib import Path
import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT/'src'))
from battery_diag.instance import Instance, PriceW5, Config, load_price


def load(path):
    d = json.load(open(path, encoding='utf-8'))
    types = {t: tuple(v) for t, v in d['types'].items()}
    a = d['args']; lam = d['lam_eff']
    price = load_price(a.get('params', ROOT/'data'/'params_w5.json'))
    if a.get('Li') is not None: price.Li_ref = float(a['Li'])   # 리튬 덮어쓰기 셀의 통제변량 기준
    rows = []
    for c in d['cells']:
        cfg = Config(lam=1.0, NARR=int(round(lam)), Cp=a['Cp'], Cf=20_000, phi=1.0,
                     Mcyc=c['M'], W=a['W'], F_E=d['F_E'], F_U=d['F_U'])
        I = Instance.lite(types, price, cfg)
        EB1 = lam*sum(I.MIX[t]*I.VS[t] for t in I.TY)
        b1 = np.array([r['g'] for r in c['policies']['B1']['runs']])
        dev = b1 - EB1
        for name, p in c['policies'].items():
            gs = np.array([r['g'] for r in p['runs']]); cv = gs - dev
            n = len(gs)
            rows.append(dict(file=Path(path).name, lam=lam, W=a['W'], M=c['M'], CAP=c['CAP'],
                             bound=c['bound'], mu=c['mu'], prec_lp=c['prec_use_lp'], pol=name,
                             g=gs.mean(), se=gs.std(ddof=1)/np.sqrt(n) if n > 1 else 0.0,
                             gap=100*(1-gs.mean()/c['bound']),
                             gap_cv=100*(1-cv.mean()/c['bound']),
                             se_cv=100*(cv.std(ddof=1)/np.sqrt(n))/c['bound'] if n > 1 else 0.0,
                             occ=p['occ'], occ_p99=np.mean([r.get('occ_p99', np.nan) for r in p['runs']]),
                             prec=p['prec_per'], fast=p['fast_per'], forced=p['forced_per'],
                             EB1=EB1, b1dev_pct=100*dev.mean()/EB1))
    return rows


def main():
    rows = []
    for f in sys.argv[1:]:
        rows += load(f)
    import pandas as pd
    df = pd.DataFrame(rows)
    pd.set_option('display.width', 250); pd.set_option('display.max_rows', 500)
    cols = ['lam', 'W', 'M', 'CAP', 'pol', 'bound', 'mu', 'prec_lp', 'g', 'gap', 'gap_cv', 'se_cv',
            'occ', 'occ_p99', 'prec', 'fast', 'forced', 'b1dev_pct']
    out = df[cols].copy()
    for k in ('bound', 'g', 'mu'): out[k] = (out[k]/1e3).round(0)
    for k in ('gap', 'gap_cv', 'se_cv', 'b1dev_pct'): out[k] = out[k].round(3)
    for k in ('occ', 'occ_p99', 'prec', 'fast', 'forced', 'prec_lp'): out[k] = out[k].round(2)
    print(out.to_string(index=False))
    o = Path(sys.argv[1]).parent/'table.csv'
    df.to_csv(o, index=False); print('->', o)


if __name__ == '__main__':
    main()
