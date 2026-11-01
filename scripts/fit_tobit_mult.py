#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""유찰을 포함한 낙찰배수 Tobit — Omega 원고(Moon & Kim, 입찰규칙)의 발견을 가격모형에 반영한다.

문제. 우리의 재사용 채널 기대수익은 낙찰 273건만으로 회귀했다 (PriceW5.Erev, PriceRCR 의 M).
그것은 E[W | W ≥ R] 이다. 그러나 재사용급 로트의 35~65% 는 유찰되고 예정가 ≈ R 에 수의계약된다
(Omega 원고 §3.2~3.3). 판매자(선별센터)가 받는 기대수익은
    E[max(R, W)] = R · E[max(1, M~)],   M~ = W/R  (잠재 배수, 유찰이면 M~ < 1)
이다. 낙찰 표본만 쓰면 E[M | M ≥ 1] 을 E[max(1, M~)] 대신 써서 10~45% 과대평가한다.

자료. Omega 원고의 추정 패널 `~/research/slb2/panel_tobit.csv` (재사용급 423로트, 6차종,
2022-03~2025-03, 유찰 205). 여기에 우리 입찰자료의 월별 리튬 시세를 붙인다.

모형 (우리 2층 구조와 같은 공변량).
    ln M~ = m0 + m_cap ln cap + m_s ln s + m_li ln Li + ε,  ε ~ N(0, σ²),  관측: 낙찰이면 ln(낙찰가/R), 유찰이면 ln M~ < 0 (좌측 검열)
연식·지역은 넣지 않는다 — 우리 유형 정의에 없다. (Omega Tobit 은 차종 고정효과+연식+지역+추세.)

산출. data/params_tobit_mult.json — 계수, σ, 그리고 유형별 G(1)=P(M~<1), E[max(1,M~)] 를
Li_ref 에서 계산한 표. 가격 클래스 PriceCensored(instance.py) 가 이 json 을 읽는다.
"""
import json, sys
from pathlib import Path
import numpy as np, pandas as pd
from scipy.optimize import minimize
from scipy.stats import norm

ROOT = Path(__file__).resolve().parents[1]
PANEL = ROOT/'data'/'panel_tobit_omega.csv'   # 사본. 원본은 ~/research/slb2/panel_tobit.csv (Omega 원고 추정 패널; CSV 는 gitignore)

# ── 리튬 월별 시세: 우리 입찰자료에서 ──
li = []
for f in ('입찰_재사용__0819.xlsx', '입찰_재활용__0819.xlsx'):
    x = pd.read_excel(ROOT/'data'/f)
    li.append(pd.DataFrame({'ym': pd.to_datetime(x['공고시작일시']).dt.to_period('M'), 'Li': x['리튬']}))
li = pd.concat(li).groupby('ym')['Li'].median()
li = li.reindex(pd.period_range(li.index.min(), li.index.max(), freq='M')).interpolate()

p = pd.read_csv(PANEL)
p['ym'] = pd.to_datetime(p['date']).dt.to_period('M')
p['Li'] = p['ym'].map(li)
miss = p['Li'].isna().sum()
if miss:
    p['Li'] = p['Li'].fillna(li.iloc[-1] if p['ym'].max() > li.index.max() else li.iloc[0])
p['s'] = p['soh']/100.0
p['lnM'] = np.log(p['won']/p['reserve'])          # 낙찰이면 관측
cens = p['failed'].astype(bool).values
X = np.column_stack([np.ones(len(p)), np.log(p['cap_kwh']), np.log(p['s']), np.log(p['Li'])])
y = p['lnM'].values

def nll(th):
    b, ls = th[:-1], th[-1]; s = np.exp(ls)
    mu = X @ b
    ll_obs = norm.logpdf((y[~cens]-mu[~cens])/s) - ls
    ll_cen = norm.logcdf((0.0 - mu[cens])/s)
    return -(ll_obs.sum() + ll_cen.sum())

th0 = np.array([0.7, 0.0, 0.0, 0.0, np.log(0.5)])
res = minimize(nll, th0, method='BFGS')
b, sig = res.x[:-1], float(np.exp(res.x[-1]))
H = res.hess_inv; se = np.sqrt(np.diag(H))[:-1]
names = ['m0', 'm_cap', 'm_s', 'm_li']
print('Tobit ln M~ (n=%d, censored=%d)  -LL=%.1f' % (len(p), cens.sum(), res.fun))
for n_, v, e in zip(names, b, se): print(f'  {n_:6s} {v:8.4f} (se {e:.4f})')
print(f'  sigma  {sig:8.4f}')

def mu_of(cap, s, Li): return b[0] + b[1]*np.log(cap) + b[2]*np.log(s) + b[3]*np.log(Li)
def G_of(cap, s, Li): return float(norm.cdf(-mu_of(cap, s, Li)/sig))
def Emax_of(cap, s, Li):
    m = mu_of(cap, s, Li)
    return float(norm.cdf(-m/sig) + np.exp(m + 0.5*sig**2)*norm.cdf((m + sig**2)/sig))
def Econd_of(cap, s, Li):   # E[M | M>=1]
    m = mu_of(cap, s, Li)
    return float(np.exp(m + 0.5*sig**2)*norm.cdf((m + sig**2)/sig)/max(norm.cdf(m/sig), 1e-12))

# 적합도: 차종별 유찰률 예측 vs 실측 (전기간, 2022-23)
p['G_hat'] = [G_of(c, s_, l) for c, s_, l in zip(p['cap_kwh'], p['s'], p['Li'])]
fit = p.groupby('model').agg(n=('failed', 'size'), fail_obs=('failed', 'mean'), fail_hat=('G_hat', 'mean'),
                             mult_obs=('mult', 'mean'))
fit['E_cond_hat'] = [np.mean([Econd_of(c, s_, l) for c, s_, l in zip(g['cap_kwh'], g['s'], g['Li'])])
                     for _, g in p.groupby('model')]
print(fit.round(3))
f23 = p[p['period'] == '2022-23'].groupby('model').agg(fail_obs=('failed', 'mean'), fail_hat=('G_hat', 'mean'))
print('2022-23 only:'); print(f23.round(3))

# 유형표 (types_w5.json 전 유형 + pool16) at Li_ref of 2-layer params and of w5 params
P2 = json.load(open(ROOT/'data'/'params_rcr_2layer.json', encoding='utf-8'))
P5 = json.load(open(ROOT/'data'/'params_w5.json', encoding='utf-8'))
sys.path.insert(0, str(ROOT/'src'))
from battery_diag.data import load_cached, POOL_MIN
F, _ = load_cached(ROOT/'data', by_capacity=True)
rows = {}
for t, (n, qP, cap, mu_s, sd) in F.items():
    if n < POOL_MIN: continue
    rows[t] = {}
    for tag, Li in (('Li_w5', P5['Li_ref']), ('Li_rcr', P2['Li_ref'])):
        rows[t][tag] = dict(G=G_of(cap, mu_s, Li), Emax=Emax_of(cap, mu_s, Li), Econd=Econd_of(cap, mu_s, Li),
                            ratio=Emax_of(cap, mu_s, Li)/Econd_of(cap, mu_s, Li))
print('\n유형별 (Li=%.2f):  G(1)  E[max(1,M)]  E[M|M>=1]  보정비' % P5['Li_ref'])
for t, r in sorted(rows.items(), key=lambda kv: -kv[1]['Li_w5']['ratio']):
    r = r['Li_w5']; print(f"  {t:14s} {r['G']:.3f} {r['Emax']:8.3f} {r['Econd']:8.3f} {r['ratio']:.3f}")

out = dict(_schema='tobit_mult_v1',
           _note='ln(W/R) Tobit, 좌측검열 0, Omega 패널 423로트 + 우리 월별 리튬. 계수는 ln cap, ln s, ln Li.',
           n=int(len(p)), n_censored=int(cens.sum()), nll=float(res.fun),
           coef=dict(zip(names, map(float, b))), se=dict(zip(names, map(float, se))), sigma=sig,
           fit_by_model=json.loads(fit.round(4).to_json(orient='index')),
           fit_2223=json.loads(f23.round(4).to_json(orient='index')),
           types=rows, Li_ref_w5=P5['Li_ref'], Li_ref_rcr=P2['Li_ref'],
           M_awarded_only=dict(median=P2['M_median'], lognormal_mean=P2['M_lognormal_mean']))
json.dump(out, open(ROOT/'data'/'params_tobit_mult.json', 'w', encoding='utf-8'), ensure_ascii=False, indent=1)
print('-> data/params_tobit_mult.json')
