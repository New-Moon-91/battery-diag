# -*- coding: utf-8 -*-
"""실증 원자료 → 차종별 요약통계(FLEET).

원자료 2종 (순환자원정보센터 계약결과 공개, 2022.3~2025.3)
  · 재사용 배터리 거래 데이터_2503까지_20250831.xlsx  (SOH 포함, 381건)
  · 배터리_1대단위_데이터_폐배터리.xlsx               (재활용 652건, 사유 9종)
외관/화재/침수(291건)를 제외한 361건이 '외관검사 통과 후 재활용'이며
재사용 381건과 합쳐 모집단 742건이 된다.
"""
from __future__ import annotations
import numpy as np, pandas as pd
from pathlib import Path

MODEL_MAP = {'BLUEON':'블루온','BOLT':'볼트','BONGO':'봉고','BONGO3':'봉고3','CEVO-C':'CEVO-C',
             'DANIGO':'다니고','IONIQ':'아이오닉','KONA':'코나','MODEL 3':'모델3','NIRO':'니로',
             'PEUGEOT E-2008':'푸조e2008','PORTER':'포터','PORTER2':'포터2','RAY':'레이',
             'SM3':'SM3','SOUL':'쏘울','TWIZY':'트위지'}
MU_FALLBACK, SD_FALLBACK, MIN_PASS = 0.85, 0.06, 5


def build_pool(reuse_xlsx: str | Path, recycle_xlsx: str | Path):
    """원자료 → (pool, pass_soh, recycle_post) 데이터프레임"""
    r = pd.ExcelFile(reuse_xlsx).parse('최종본')
    r['차종'] = r['차종'].astype(str).str.strip()
    r = r.rename(columns={'차종':'model','배터리 용량 (kWh)':'kwh','잔존 가치 (SOH,%)':'soh'})
    r['s'] = r['soh'] / 100.0
    c = pd.read_excel(recycle_xlsx)
    mc = [x for x in c.columns if x.startswith('Model_')]
    c['model'] = c[mc].idxmax(axis=1).str.replace('Model_','',regex=False).map(MODEL_MAP)
    c['kwh'] = c['Battert Capacity (kWh)']
    pre = (c['Appearance Issue'] + c['Fire History'] + c['Flood Damage']) > 0
    post = c[~pre].copy()
    pool = pd.concat([pd.DataFrame({'model':r['model'],'kwh':r['kwh'],'Z':1}),
                      pd.DataFrame({'model':post['model'],'kwh':post['kwh'],'Z':0})],
                     ignore_index=True)
    assert len(pool) == 742, f'모집단 742건이어야 함 (현재 {len(pool)})'
    return pool, r[['model','kwh','s']], post


def fleet_from_pool(pool: pd.DataFrame, pass_soh: pd.DataFrame) -> dict:
    """{차종: (n, q_P, cap_kWh, mu_S, sd_S)}"""
    g = pool.groupby('model').agg(n=('Z','size'), qP=('Z','mean'), cap=('kwh','median'))
    s = pass_soh.groupby('model')['s'].agg(['mean','std','count'])
    out = {}
    for m, x in g.sort_values('qP').iterrows():
        ok = m in s.index and s.loc[m,'count'] >= MIN_PASS and not np.isnan(s.loc[m,'std'])
        mu = float(s.loc[m,'mean']) if ok else MU_FALLBACK
        sd = float(s.loc[m,'std'])  if ok else SD_FALLBACK
        out[m] = (int(x.n), float(x.qP), float(x.cap), mu, sd)
    return out


def failure_shares(recycle_post: pd.DataFrame) -> dict:
    """신속검사 판별 가능 결함 비중 — P_DET 산출용.

    리콜 대상(7건)은 차종·연식으로 사전 식별 가능하므로 **무료 관측정보 w 에 편입**되고
    결함 분류 모집단에서 제외한다(문제정의 확정판 §2.2). 따라서 분모는 361-7=354.
    """
    rec = int(recycle_post['Recall Target'].sum())
    n = len(recycle_post) - rec
    fe = int(recycle_post[['Electrical Fault','Cell Imbalance','Unable to Measure SOH']].sum().sum())
    fu = int(recycle_post['Undisclosed'].sum())
    fs = int(recycle_post['Low SOH'].sum())
    return dict(n_fail=n, n_recall=rec, F_E=fe/n, F_U=fu/n, F_S=fs/n,
                counts=dict(FE=fe, FU=fu, FS=fs, RECALL=rec))


def load_cached(data_dir: str | Path):
    d = Path(data_dir)
    pool = pd.read_csv(d/'pool.csv'); ps = pd.read_csv(d/'pass_soh.csv')
    post = pd.read_csv(d/'recycle_post.csv')
    return fleet_from_pool(pool, ps), failure_shares(post)
