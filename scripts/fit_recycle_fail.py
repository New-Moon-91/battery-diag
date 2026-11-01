#!/usr/bin/env python
# -*- coding: utf-8 -*-
r"""재활용 채널의 유찰 보정 — 공고결과 전체 파일(838건)과 전자수의 파일로 추정한다.

재사용급은 예정가 R 이 공개라 Tobit(fit_tobit_mult.py)이 가능하지만, 재활용급은 예정가가
**비공개**라 같은 방법을 쓸 수 없다. 대신 관측 가능한 것 셋으로 보정비를 만든다.
    G_rc(연도)  = 재활용급 로트의 유찰률 (공고결과 파일, 낙찰금액 없음 = 유찰)
    P_auc(연도) = 낙찰 재활용 로트의 kWh 당 낙찰가 중앙값
    P_neg(연도) = 유찰 후 전자수의(수의계약)로 팔린 재활용 묶음의 kWh 당 가격 중앙값
판매자가 받는 기대수익 / 낙찰가 = 1 − G_rc·(1 − P_neg/P_auc) =: r_rc(연도).
연도별 r_rc 를 로트 수로 가중평균한 값을 V^S 에 곱한다 (PriceCensored.rc_ratio).

자료 출처 (E:\SNU\PHD Moon\배터리 재사용 논문\폐배터리 가격 추정\, 2025-08-31 판)
  공고결과_낙찰,부분낙찰_25년3월까지_20250831.xlsx → data/공고결과_전체_2503_20250831.xlsx
  전자수의_수의공고결과_25년3월까지.xlsx           → data/전자수의_2503.xlsx
"""
import json
from pathlib import Path
import numpy as np, pandas as pd

ROOT = Path(__file__).resolve().parents[1]
d = pd.read_excel(ROOT/'data'/'공고결과_전체_2503_20250831.xlsx', header=3)
d.columns = [str(c).strip() for c in d.columns]
d['grade'] = np.where(d['물품명'].astype(str).str.startswith('폐배터리'), 'recycle', 'reuse')
d['won'] = pd.to_numeric(d['낙찰금액'], errors='coerce')
d['kwh'] = pd.to_numeric(d['배터리 용량 (kWh)'], errors='coerce'); d['q'] = pd.to_numeric(d['수량 (대)'], errors='coerce')
d['date'] = pd.to_datetime(d['공고시작일시'], errors='coerce'); d['yr'] = d.date.dt.year
d['failed'] = d.won.isna()
r = d[d.grade == 'recycle'].copy(); r['per_kwh'] = r.won/(r.kwh*r.q)
G = r.groupby('yr').failed.agg(['size', 'mean']).rename(columns={'size': 'n', 'mean': 'G'})
Pauc = r[r.won.notna()].groupby('yr').per_kwh.median().rename('P_auc')

n = pd.read_excel(ROOT/'data'/'전자수의_2503.xlsx', header=3)
n.columns = [str(c).strip() for c in n.columns]; n = n[n['물품명'].notna()].copy()
n['grade'] = np.where(n['물품명'].astype(str).str.startswith('폐배터리'), 'recycle', 'reuse')
n['kwh'] = pd.to_numeric(n['배터리 용량 (kWh)'], errors='coerce'); n['q'] = pd.to_numeric(n['수량 (대)'], errors='coerce')
n['won'] = pd.to_numeric(n['낙찰금액'], errors='coerce'); n['Rsum'] = pd.to_numeric(n['예정가격합계'], errors='coerce')
n['date'] = pd.to_datetime(n['공고시작일시'].astype(str).str[:10], errors='coerce'); n['yr'] = n.date.dt.year
rc = n[n.grade == 'recycle'].copy(); rc['kwhq'] = rc.kwh*rc.q
g = rc.groupby('공고번호').agg(kwhq=('kwhq', 'sum'), won=('won', 'max'), Rsum=('Rsum', 'max'), yr=('yr', 'max'))
g = g[g.won.notna() & (g.kwhq > 0)]; g['per_kwh'] = g.won/g.kwhq
Pneg = g.groupby('yr').per_kwh.median().rename('P_neg'); Nneg = g.groupby('yr').size().rename('n_neg')

T = pd.concat([G, Pauc, Pneg, Nneg], axis=1)
T['rho'] = (T.P_neg/T.P_auc)
# 수의계약 관측이 없는 연도(2022: 유찰 0건)는 rho 가 필요 없다 (G=0). 결측 rho 는 인접연도 중앙값으로 채운다.
T['rho'] = T.rho.fillna(T.rho.median())
T['r_rc'] = 1 - T.G*(1 - T.rho.clip(upper=1.0))
w = T.n/T.n.sum()
r_rc = float((T.r_rc*w).sum())
# 대안: 전기간 풀링 (연도 매칭 없이)
pool_G = float(r[r.yr >= 2023].failed.mean())
pool_rho = float(g.per_kwh.median()/r[(r.won.notna()) & (r.yr >= 2023)].per_kwh.median())
r_rc_pooled = 1 - pool_G*(1 - min(pool_rho, 1.0))
# 재사용급 (참고·대조용): 차종별 유찰률
u = d[d.grade == 'reuse']
reuse_fail = u.groupby('차종').failed.agg(['size', 'mean']).sort_values('size', ascending=False).head(12)

pd.set_option('display.width', 200)
print(T.round(3)); print(f'r_rc (연도별 로트 가중) = {r_rc:.3f}   풀링 대안 = {r_rc_pooled:.3f}  (G={pool_G:.3f}, rho={pool_rho:.3f})')
print('재사용급 차종별 유찰률 (참고):'); print(reuse_fail.round(3))
out = dict(_schema='recycle_fail_v1',
           _note='재활용급 유찰 보정비. r_rc = 1 - G_rc(1 - P_neg/P_auc), 연도별 로트수 가중. 자료: 공고결과 전체(838건)·전자수의(2503까지).',
           by_year=json.loads(T.round(5).to_json(orient='index')),
           r_rc=r_rc, r_rc_pooled=r_rc_pooled, pool_G=pool_G, pool_rho=pool_rho,
           n_recycle_lots=int(len(r)), n_negotiated_bundles=int(len(g)),
           reuse_fail_by_model=json.loads(reuse_fail.round(4).to_json(orient='index')))
json.dump(out, open(ROOT/'data'/'params_recycle_fail.json', 'w', encoding='utf-8'), ensure_ascii=False, indent=1)
print('-> data/params_recycle_fail.json')
