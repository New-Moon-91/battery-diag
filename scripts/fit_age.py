#!/usr/bin/env python
# -*- coding: utf-8 -*-
r"""연식(차량 연령) 효과 — Omega 원고(연식 1년당 경쟁입찰 −16%, 예정가 0.1^{a/9})를 우리 가격모형에 넣을 수 있는가.

1. 유형별 연령 분포: 재사용 낙찰 381건(`data/재사용거래_381.xlsx`, 차량 연령 일수) 과
   공고결과 재사용급 473건(`data/공고결과_전체_2503_20250831.xlsx`, 차량 연식 YY.MM + 공고일)에서
   유형(차종×용량)별 경매 시점 연령(년)의 평균·표준편차.
2. 재사용 낙찰가 직접회귀에 연령을 추가: ln P = c0 + a ln cap + b ln s + c ln Li + d·(age − āge).
   (273 입찰자료, 차량연령 일수). 연령 중심화로 절편이 기존 w5 와 비교 가능.
3. Tobit(ln M~)에 연령 추가 (Omega 패널 age_yr).
4. 유형별 예측 vs 실측 평균 낙찰가 — 연령 포함/미포함 비교 (어느 쪽이 유형 수준을 더 잘 맞추나).
산출: data/params_age.json (계수, 유형별 연령, 유형별 보정배수).
"""
import json, sys
from pathlib import Path
import numpy as np, pandas as pd
from scipy.optimize import minimize
from scipy.stats import norm

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT/'src'))
from battery_diag.data import type_key, normalize_cap, load_cached, POOL_MIN

# ── 1. 유형별 연령 ──
b = pd.read_excel(ROOT/'data'/'입찰_재사용__0819.xlsx')
b['age'] = pd.to_numeric(b['차량연령'], errors='coerce')/365.25
b['type'] = [type_key(m, k) for m, k in zip(b['차종'], b['용량'])]
# 레이 16.0→16.4 병합 규칙과 같게
b.loc[(b['차종'] == '레이') & (b['용량'] == 16.0), 'type'] = type_key('레이', 16.4)
age_bid = b.groupby('type')['age'].agg(['mean', 'std', 'size']).rename(columns={'size': 'n'})

a = pd.read_excel(ROOT/'data'/'공고결과_전체_2503_20250831.xlsx', header=3); a.columns = [str(c).strip() for c in a.columns]
a = a[~a['물품명'].astype(str).str.startswith('폐배터리')].copy()
a['date'] = pd.to_datetime(a['공고시작일시'], errors='coerce')
def parse_yymm(x):
    try:
        y, m = str(x).split('.'); y = int(y); m = int(m[:2]) if m else 6
        return pd.Timestamp(year=2000+y, month=max(1, min(12, m)), day=1)
    except Exception:
        return pd.NaT
a['born'] = a['차량 연식'].apply(parse_yymm)
a['age'] = (a['date'] - a['born']).dt.days/365.25
a['cap'] = pd.to_numeric(a['배터리 용량 (kWh)'], errors='coerce')
a['model'] = a['차종'].astype(str).str.replace('Ⅱ', '2').str.replace(' ', '').str.replace('(26kWh)', '').str.replace('(26kwh)', '')
a['type'] = [type_key(m, k) if pd.notna(k) else None for m, k in zip(a['model'], a['cap'])]
age_ann = a.dropna(subset=['age', 'type']).groupby('type')['age'].agg(['mean', 'std', 'size']).rename(columns={'size': 'n'})

print('유형별 연령(년) — 낙찰 273 / 공고 473'); print(age_bid.join(age_ann, lsuffix='_bid', rsuffix='_ann', how='outer').round(2))

# ── 2. 직접회귀 + 연령 ──
d = b.dropna(subset=['age']).copy()
d['lnP'] = np.log(d['낙찰금액']/d['묶음개수'].replace(0, 1)) if '묶음개수' in d else np.log(d['낙찰금액'])
d['lnP'] = np.log(pd.to_numeric(d['낙찰금액'], errors='coerce')/pd.to_numeric(d['묶음개수'], errors='coerce').fillna(1).replace(0, 1))
d['s'] = pd.to_numeric(d['잔존가치'], errors='coerce')/100
d = d.dropna(subset=['lnP', 's', '리튬'])
age_bar = float(d['age'].mean())
X0 = np.column_stack([np.ones(len(d)), np.log(d['용량']), np.log(d['s']), np.log(d['리튬'])])
X1 = np.column_stack([X0, d['age'] - age_bar])
def ols(X, y):
    beta, *_ = np.linalg.lstsq(X, y, rcond=None); res = y - X@beta
    sig = np.sqrt((res**2).sum()/(len(y)-X.shape[1])); r2 = 1 - (res**2).sum()/((y-y.mean())**2).sum()
    cov = sig**2*np.linalg.inv(X.T@X); return beta, np.sqrt(np.diag(cov)), sig, r2
b0, se0, s0, r20 = ols(X0, d['lnP'].values); b1, se1, s1, r21 = ols(X1, d['lnP'].values)
print(f'\n직접회귀 (n={len(d)}) 연령 없음: c0 {b0[0]:.4f} cap {b0[1]:.4f} s {b0[2]:.4f} Li {b0[3]:.4f}  σ {s0:.4f} R² {r20:.3f}')
print(f'직접회귀 연령 포함:       c0 {b1[0]:.4f} cap {b1[1]:.4f} s {b1[2]:.4f} Li {b1[3]:.4f} age {b1[4]:.4f} (se {se1[4]:.4f}) σ {s1:.4f} R² {r21:.3f}  āge={age_bar:.2f}')

# ── 3. Tobit + 연령 (Omega 패널) ──
p = pd.read_csv(ROOT/'data'/'panel_tobit_omega.csv'); p['ym'] = pd.to_datetime(p['date']).dt.to_period('M')
li = pd.concat([pd.DataFrame({'ym': pd.to_datetime(x['공고시작일시']).dt.to_period('M'), 'Li': x['리튬']})
                for x in (pd.read_excel(ROOT/'data'/'입찰_재사용__0819.xlsx'), pd.read_excel(ROOT/'data'/'입찰_재활용__0819.xlsx'))]).groupby('ym')['Li'].median()
li = li.reindex(pd.period_range(li.index.min(), li.index.max(), freq='M')).interpolate()
p['Li'] = p['ym'].map(li).fillna(li.iloc[-1]); p['s'] = p['soh']/100
cens = p['failed'].astype(bool).values; y = np.log(p['won']/p['reserve']).values
tage_bar = float(p['age_yr'].mean())
def tobit(X):
    def nll(th):
        bb, ls = th[:-1], th[-1]; sg = np.exp(ls); mu = X@bb
        return -(np.sum(norm.logpdf((y[~cens]-mu[~cens])/sg) - ls) + np.sum(norm.logcdf(-mu[cens]/sg)))
    r = minimize(nll, np.r_[np.zeros(X.shape[1]), np.log(0.6)], method='BFGS')
    return r.x[:-1], float(np.exp(r.x[-1])), float(r.fun), np.sqrt(np.diag(r.hess_inv))[:-1]
XT0 = np.column_stack([np.ones(len(p)), np.log(p['cap_kwh']), np.log(p['s']), np.log(p['Li'])])
XT1 = np.column_stack([XT0, p['age_yr'] - tage_bar])
t0, ts0, nll0, _ = tobit(XT0); t1, ts1, nll1, tse1 = tobit(XT1)
print(f'\nTobit 연령 없음: {np.round(t0,4)} σ {ts0:.4f} -LL {nll0:.1f}')
print(f'Tobit 연령 포함: {np.round(t1,4)} (age se {tse1[4]:.4f}) σ {ts1:.4f} -LL {nll1:.1f}  āge={tage_bar:.2f}')

# ── 4. 유형 수준 예측력 ──
d['type'] = b.loc[d.index, 'type']
d['pred0'] = np.exp(X0@b0 + 0.5*s0**2); d['pred1'] = np.exp(X1@b1 + 0.5*s1**2); d['P'] = np.exp(d['lnP'])
g = d.groupby('type').agg(n=('P', 'size'), P_obs=('P', 'mean'), P0=('pred0', 'mean'), P1=('pred1', 'mean'), age=('age', 'mean'))
g = g[g.n >= 5]; g['err0'] = g.P0/g.P_obs - 1; g['err1'] = g.P1/g.P_obs - 1
print('\n유형별 평균 낙찰가 예측오차 (연령 없음 err0 / 포함 err1):'); print(g.round(3))
print('MAE err0 %.3f  err1 %.3f' % (g.err0.abs().mean(), g.err1.abs().mean()))

# ── 유형별 보정배수 (연령 포함 회귀 기준, 유형 평균연령) ──
F, _ = load_cached(ROOT/'data', by_capacity=True)
ages = {}
for t in F:
    if F[t][0] < POOL_MIN: continue
    src = None
    if t in age_bid.index and age_bid.loc[t, 'n'] >= 3: src = ('bid', float(age_bid.loc[t, 'mean']))
    elif t in age_ann.index and age_ann.loc[t, 'n'] >= 3: src = ('ann', float(age_ann.loc[t, 'mean']))
    ages[t] = dict(age=(src[1] if src else None), src=(src[0] if src else None),
                   mult_price=(float(np.exp(b1[4]*(src[1]-age_bar))) if src else 1.0),
                   shift_tobit=(float(t1[4]*(src[1]-tage_bar)) if src else 0.0))
print('\n유형별 연령·가격배수·Tobit 이동:')
for t, v in sorted(ages.items(), key=lambda kv: (kv[1]['age'] is None, kv[1]['age'] or 0)): print(f"  {t:14s} age={v['age']}  ×{v['mult_price']:.3f}  Δμ={v['shift_tobit']:+.3f}  ({v['src']})")
out = dict(_schema='age_v1', age_bar_price=age_bar, age_bar_tobit=tage_bar,
           price_coef=dict(c0=float(b1[0]), cap=float(b1[1]), s=float(b1[2]), li=float(b1[3]), age=float(b1[4]), sd=float(s1), r2=float(r21), n=int(len(d))),
           price_coef_noage=dict(c0=float(b0[0]), cap=float(b0[1]), s=float(b0[2]), li=float(b0[3]), sd=float(s0), r2=float(r20)),
           tobit_coef=dict(m0=float(t1[0]), m_cap=float(t1[1]), m_s=float(t1[2]), m_li=float(t1[3]), m_age=float(t1[4]), sigma=ts1, nll=nll1),
           tobit_coef_noage=dict(m0=float(t0[0]), m_cap=float(t0[1]), m_s=float(t0[2]), m_li=float(t0[3]), sigma=ts0, nll=nll0),
           type_fit=json.loads(g.round(5).to_json(orient='index')), types=ages)
json.dump(out, open(ROOT/'data'/'params_age.json', 'w', encoding='utf-8'), ensure_ascii=False, indent=1)
print('-> data/params_age.json')
