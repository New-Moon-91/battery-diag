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


# ---------------------------------------------------------------------------
# w4 — 실측 손익배수와 차종 선택
# ---------------------------------------------------------------------------
# 정밀검사 손익분기: 정밀검사는 통과확률 q_P 로만 재사용 매출을 얻으므로
#   q_P * (E[재사용 낙찰가] - V^S) > C_p   ⇔   (E[재사용] - V^S) / (C_p/q_P) > 1
# 이 좌변을 **손익배수** 라 부른다. 분모 C_p/q_P = 500,000/0.51348 = 973,753원.
#
# 아래 RATIO_EMP 는 그 손익배수를 **모형을 전혀 쓰지 않고** 실거래만으로 계산한 값이다.
#   E[재사용] = 2024-01 이후 차종별 재사용 낙찰가 중앙값 (환경공단 거점수거센터 입찰)
#   V^S       = p_rc * kWh,  p_rc = 9,491원/kWh (같은 창의 재활용 낙찰 중앙값)
# 즉 최적정책의 검사강도 서열과 대조할 때 **독립 증거**가 된다 (results/w4/validation.md §3).
RATIO_EMP = {'SM3': 0.05, '쏘울': 0.65, '볼트': 1.99, '코나': 2.89}

# 실측 SOH 중앙값 (2024-01 이후 재사용 표본). pass_soh.csv 의 전기간 평균과는 다르다 —
# 가격검증(§1)에서 예측을 만들 때 쓰는 값이고, 모형 인스턴스의 MU 는 전기간 평균을 쓴다.
SOH_MED_EMP = {'SM3': 0.670, '쏘울': 0.892, '볼트': 0.969, '코나': 0.861}

# 손익배수 → 경제영역. 임계 0.8 / 1.5.
#   R1 (<0.8)      정밀검사가 어떤 신호에서도 손해 — 즉시 매각이 지배
#   R3 (0.8~1.5)   손익분기를 걸치는 경계 차종 — 신호에 따라 갈린다
#   R2 (>=1.5)     신호와 무관하게 정밀검사가 이득
# 임계를 1.0 에 붙이지 않고 0.8/1.5 로 벌린 근거: 손익배수는 낙찰배수 잡음
# (M_sd=0.4201, 로그정규 → 대략 ±34% 산포) 위에서 잰 중앙값이라 1.0 근방
# ±20~50% 는 신호 하나로 부호가 바뀔 수 있는 구간이다. 그 구간을 R3 로 따로 둔다.
REG_LO, REG_HI = 0.8, 1.5


def reg_from_ratio(r: float) -> str:
    return 'R1' if r < REG_LO else ('R2' if r >= REG_HI else 'R3')


REG_EMP = {m: reg_from_ratio(r) for m, r in RATIO_EMP.items()}

# w4 [2] 확정 인스턴스. 손익분기 1.0 을 사이에 두고 양쪽에 둘씩 놓이는 사다리이고
# 넷 다 승용이라 시장이 균질하다. (포터2가 유일한 R3 후보였으나 상용차이고
# 2024년 이후 재사용 4건뿐이라 제외.)  기존 w1~w3 인스턴스는 SEL_V5 로 남긴다.
SEL_W4 = ['SM3', '쏘울', '볼트', '코나']
SEL_V5 = ['레이', '코나', 'SM3']
