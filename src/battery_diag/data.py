# -*- coding: utf-8 -*-
"""실증 원자료 → 차종별 요약통계(FLEET).

원자료 2종 (순환자원정보센터 계약결과 공개, 2022.3~2025.3)
  · 재사용 배터리 거래 데이터_2503까지_20250831.xlsx  (SOH 포함, 381건)
  · 배터리_1대단위_데이터_폐배터리.xlsx               (재활용 652건, 사유 9종)
외관/화재/침수(291건)를 제외한 361건이 '외관검사 통과 후 재활용'이며
재사용 381건과 합쳐 모집단 742건이 된다.
"""
from __future__ import annotations
import json
import numpy as np, pandas as pd
from pathlib import Path

MODEL_MAP = {'BLUEON':'블루온','BOLT':'볼트','BONGO':'봉고','BONGO3':'봉고3','CEVO-C':'CEVO-C',
             'DANIGO':'다니고','IONIQ':'아이오닉','KONA':'코나','MODEL 3':'모델3','NIRO':'니로',
             'PEUGEOT E-2008':'푸조e2008','PORTER':'포터','PORTER2':'포터2','RAY':'레이',
             'SM3':'SM3','SOUL':'쏘울','TWIZY':'트위지'}
MU_FALLBACK, SD_FALLBACK, MIN_PASS = 0.85, 0.06, 5

# ---------------------------------------------------------------------------
# w5 — 유형은 `차종` 이 아니라 `차종 × 용량` 이다
# ---------------------------------------------------------------------------
# 같은 차종이라도 배터리 용량이 다르면 다른 배터리다. w4 까지는 차종으로만 묶어
# 서로 다른 팩을 하나로 취급했다. 복수 용량이 실제로 존재하는 차종:
#   쏘울(27/30/64) · SM3(26.6/35.9) · 아이오닉(28/38.3) · 코나(39.2/64/66)
#   볼트(60/66) · 레이(16/16.4/35.2) · 모델3(50/72)
# 이들은 웹 검증상 **실제 사양 차이**다 (볼트 초기형 60 / 2022년형 이후 66,
# 코나 1세대 스탠다드 39.2 / 롱레인지 64, 쏘울 2세대 27·30 / 부스터 64,
# SM3 Z.E. 초기형 26.6 / 개선형 35.9, 아이오닉 2016년형 28 / 2019년형 38.3).
TYPE_SEP = '_'


def type_key(model, kwh) -> str:
    return f'{model}{TYPE_SEP}{float(kwh):.1f}'


# 용량 표기 정규화 — 출처별 표기 차이로 한 사양이 둘로 갈린 것을 되붙인다.
#
# 근거: `pool.csv` 에서 레이가 16.0/16.4 두 유형으로 갈리는데 통과확률이
#   레이_16.0  n=13  q_P=1.00 (전부 재사용)
#   레이_16.4  n=47  q_P=0.00 (전부 재활용)
# 로 **완벽히 0/1 로 분리**된다. 확률이 0과 1로 갈리는 것은 통계가 아니라
# 재사용 파일은 16.0, 재활용 파일은 16.4 로 적었다는 표기 규칙 차이다.
# 웹 검증상 레이 EV 초기형은 16.4kWh 단일 사양이므로 16.4 로 통일한다.
# 병합 후 n=60, 통과 13건, q_P=0.217 로 정상값이 된다.
#
# 규칙: 같은 차종의 두 용량 표기가 CAP_TOL 이내로 근접하면서 q_P 가 각각
# 정확히 0 과 1 이고 양쪽 표본이 MERGE_MIN_N 이상이면 병합한다.
# 세 조건을 모두 걸어야 실제 사양 차이를 잘못 뭉개지 않는다.
#
# CAP_TOL 을 3% 로 둔 이유: 레이 16.0↔16.4 의 상대차가 2.44% 라 정합성 메모의
# 서술("2% 이내")로는 정작 목표인 레이가 걸리지 않는다. 3% 로 올려도 pool.csv
# 안에서 추가로 걸리는 쌍은 없다 — 다음으로 가까운 동일차종 쌍이 볼트 60↔66
# (10.0%) 이라 여유가 크다.
CAP_TOL, MERGE_MIN_N = 0.03, 5

# 유형 채택 하한. pool 표본이 POOL_MIN 이상이고 신규 입찰자료의 재사용·재활용이
# 각각 SIDE_MIN 건 이상이어야 6개 특징을 전부 실측으로 채울 수 있다.
POOL_MIN, SIDE_MIN = 10, 3
CP = 500_000.0                      # 정밀검사 변동비 (Config.Cp 와 같아야 한다)

# 병합하지 **않는** 것: 볼트_66.0 · 코나_39.2 · 아이오닉_38.3 (각 n=1, q_P=0).
# 정합성 메모 §4.1 은 이 셋을 레이와 "같은 패턴" 으로 묶었으나, §4 의 웹 검증이
# 확인했듯 셋 다 실재하는 별도 사양이고 용량 차이도 10~129% 로 표기 오류로 보기
# 어렵다. 66kWh 팩을 60kWh 유형에 합치면 용량 자체가 오염된다. 이들은 병합이
# 아니라 표본수 하한(POOL_MIN)에서 걸러지며, 그 결과는 메모 §5 표와 같다.


def normalize_cap(pool):
    """§4.1 표기 정규화. (정규화된 pool, 적용된 병합 목록) 을 돌려준다."""
    pool = pool.copy()
    merges = []
    for m, grp in pool.groupby('model'):
        caps = sorted(grp['kwh'].unique())
        for i in range(len(caps)):
            for j in range(i+1, len(caps)):
                a, b = caps[i], caps[j]
                if abs(b-a)/max(a, b) > CAP_TOL:
                    continue
                za, zb = grp[grp.kwh == a]['Z'], grp[grp.kwh == b]['Z']
                if len(za) < MERGE_MIN_N or len(zb) < MERGE_MIN_N:
                    continue
                if {za.mean(), zb.mean()} != {0.0, 1.0}:
                    continue
                tgt = b if len(zb) >= len(za) else a          # 표본이 많은 표기로
                src = a if tgt == b else b
                pool.loc[(pool.model == m) & (pool.kwh == src), 'kwh'] = tgt
                merges.append(dict(model=m, src=src, dst=tgt,
                                   n_src=int(len(za if src == a else zb)),
                                   n_dst=int(len(zb if tgt == b else za))))
    return pool, merges


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
    pool, merges = normalize_cap(pool)                       # w5 §4.1 표기 정규화
    ps = r[['model','kwh','s']].copy()
    for mg in merges:                                        # SOH 표본도 같은 표기로
        ps.loc[(ps.model == mg['model']) & (ps.kwh == mg['src']), 'kwh'] = mg['dst']
    return pool, ps, post


def fleet_from_pool(pool: pd.DataFrame, pass_soh: pd.DataFrame,
                    by_capacity: bool = True) -> dict:
    """{유형: (n, q_P, cap_kWh, mu_S, sd_S)}

    by_capacity=True (w5 기본) 면 유형 키가 `차종_용량` 이고, False 면 w4 까지의
    `차종` 이다. 구 결과 재현 경로를 남겨 두기 위해 스위치로 둔다.
    """
    pool = pool.copy(); pass_soh = pass_soh.copy()
    if by_capacity:
        pool['key'] = [type_key(m, k) for m, k in zip(pool.model, pool.kwh)]
        pass_soh['key'] = [type_key(m, k) for m, k in zip(pass_soh.model, pass_soh.kwh)]
    else:
        pool['key'] = pool['model']; pass_soh['key'] = pass_soh['model']
    g = pool.groupby('key').agg(n=('Z','size'), qP=('Z','mean'), cap=('kwh','median'))
    s = pass_soh.groupby('key')['s'].agg(['mean','std','count'])
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


def load_cached(data_dir: str | Path, by_capacity: bool = False):
    """캐시된 CSV → (FLEET, failure_shares).

    `pool.csv` 는 정규화 이전에 떨어진 파일이므로 여기서 normalize_cap 을 한 번 더
    태운다 (멱등이다 — 이미 정규화된 입력에는 병합할 쌍이 없다).

    by_capacity 기본값은 **False(차종 단위)** 다. w1~w4 의 호출부를 전부 그대로
    살리기 위해서다 — w5 경로는 load_cached 가 아니라 fleet_w5() 로 유형을 얻는다.
    """
    d = Path(data_dir)
    pool = pd.read_csv(d/'pool.csv'); ps = pd.read_csv(d/'pass_soh.csv')
    post = pd.read_csv(d/'recycle_post.csv')
    pool, merges = normalize_cap(pool)
    for mg in merges:
        ps.loc[(ps.model == mg['model']) & (ps.kwh == mg['src']), 'kwh'] = mg['dst']
    return fleet_from_pool(pool, ps, by_capacity), failure_shares(post)


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


# ---------------------------------------------------------------------------
# w5 — 유형표(차종×용량)와 실측 손익배수
# ---------------------------------------------------------------------------
# `data/types_w5.json` 은 scripts/build_types_w5.py 가 원자료에서 만든다.
# 원자료(입찰 xlsx 2종)는 공동연구자 미발표분이라 gitignore 대상이므로, 산출물인
# 이 json 만 추적한다 — 원자료 없이도 모형이 그대로 돌아가게 하기 위해서다.
_TYPES_W5 = None


def types_w5(data_dir: str | Path | None = None) -> dict:
    """{유형: 행dict} — 실측 6개 특징 + 실측 손익배수 + 등급."""
    global _TYPES_W5
    if _TYPES_W5 is None:
        d = Path(data_dir) if data_dir else Path(__file__).resolve().parents[2]/'data'
        raw = json.loads((d/'types_w5.json').read_text(encoding='utf-8'))
        _TYPES_W5 = {r['type']: r for r in raw['types']}
        _TYPES_W5['_meta'] = {k: v for k, v in raw.items() if k != 'types'}
    return _TYPES_W5


def fleet_w5(data_dir: str | Path | None = None, sel=None) -> dict:
    """{유형: (n, q_P, cap, mu, sd)} — types_w5.json 에서. sel 이 있으면 그 순서로."""
    T = types_w5(data_dir)
    ks = sel if sel is not None else [k for k in T if k != '_meta']
    return {k: (T[k]['n'], T[k]['qP'], T[k]['cap'], T[k]['mu'], T[k]['sd']) for k in ks}


def ratio_emp_w5(data_dir=None) -> dict:
    T = types_w5(data_dir)
    return {k: v['ratio_emp'] for k, v in T.items()
            if k != '_meta' and v.get('ratio_emp') is not None}


# w5 확정 인스턴스 (정합성 메모 §5.1).
#   아이오닉_28.0 (0.63, R1) · 코나_64.0 (1.42, R3) · 볼트_60.0 (3.53, R2)
#   · 포터2_58.8 (5.91, R2)
# 손익분기 1.0 을 사이에 두고 아래 하나·경계 하나·위 둘이 놓이는 촘촘한 사다리다.
# pool 표본이 68/130/86/21 이라 6개 특징을 전부 실측으로 채운다(기본값 대체 없음).
#
# w4 인스턴스(SM3·쏘울·볼트·코나)와 다른 이유: 유형을 차종×용량으로 쪼개고
# 손익배수에 유형별 q_P 를 반영하자(§4.2) 서열이 바뀌었다. 특히 SM3 는 26.6(0.02,R1)
# 과 35.9(1.27,R3) 로 갈라져 하나의 "차종 SM3" 로는 어느 쪽도 대표하지 못한다.
SEL_W5 = ['아이오닉_28.0', '코나_64.0', '볼트_60.0', '포터2_58.8']
