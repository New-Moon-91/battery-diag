# -*- coding: utf-8 -*-
"""w5 회귀 — 차종×용량 유형, 직접회귀 가격함수, 확정 인스턴스 4유형.

원자료(입찰 xlsx 2종)가 있어야만 도는 검사는 skipif 로 표시한다. 산출물
`data/types_w5.json` 은 추적되므로 원자료 없이도 대부분은 돈다.
"""
import json, sys
from pathlib import Path
import numpy as np
import pytest, torch

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT/'src'))
from battery_diag.data import (load_cached, normalize_cap, fleet_w5, types_w5,
                               ratio_emp_w5, SEL_W5, reg_from_ratio, CP,
                               CAP_TOL, POOL_MIN, SIDE_MIN)
from battery_diag.instance import Instance, PriceParams, PriceW5, Config
from battery_diag.build import build, _price_key
from battery_diag.exact import ExactSolver, NumpySolver
from battery_diag import policies as pol
import pandas as pd

DATA = ROOT/'data'
XLSX = (DATA/'입찰_재사용__0819.xlsx', DATA/'입찰_재활용__0819.xlsx')
need_raw = pytest.mark.skipif(not all(p.exists() for p in XLSX),
                              reason='원자료 미배치 (gitignore 대상)')


# ---------------------------------------------------------------- 유형 정의
def test_normalize_cap_merges_only_ray():
    """§4.1 — 레이 16.0/16.4 만 병합된다. 실제 사양 차이는 건드리지 않는다."""
    pool = pd.read_csv(DATA/'pool.csv')
    p2, merges = normalize_cap(pool)
    assert len(merges) == 1
    m = merges[0]
    assert (m['model'], m['src'], m['dst']) == ('레이', 16.0, 16.4)
    assert len(p2) == len(pool) == 742, '병합은 행을 지우지 않는다'
    # 볼트 60/66 · 코나 39.2/64 · 아이오닉 28/38.3 은 남아 있어야 한다
    keys = {(r.model, r.kwh) for r in p2.itertuples()}
    for k in (('볼트', 66.0), ('코나', 39.2), ('아이오닉', 38.3)):
        assert k in keys, f'{k} 는 실재 사양이므로 병합하면 안 된다'


def test_normalize_cap_idempotent():
    pool = pd.read_csv(DATA/'pool.csv')
    p2, _ = normalize_cap(pool)
    p3, m2 = normalize_cap(p2)
    assert m2 == [], '이미 정규화된 입력에는 병합할 쌍이 없어야 한다'
    assert p3.equals(p2)


def test_ray_qp_becomes_sane():
    """병합 전 q_P 가 0/1 로 갈리던 것이 0.217 로 정상화된다."""
    F = fleet_w5(DATA)
    n, qP, cap, mu, sd = F['레이_16.4']
    assert n == 60 and cap == pytest.approx(16.4)
    assert qP == pytest.approx(13/60, abs=1e-6)


def test_legacy_keying_preserved():
    """load_cached 기본값은 차종 단위 — w1~w4 호출부가 그대로 살아야 한다."""
    F, _ = load_cached(DATA)
    assert '코나' in F and '코나_64.0' not in F
    F2, _ = load_cached(DATA, by_capacity=True)
    assert '코나_64.0' in F2 and '코나' not in F2


# ---------------------------------------------------------------- 실측 손익배수
def test_ratio_emp_definition():
    """§4.2 — 손익배수는 유형별 q_P 를 분자에 쓴다: q_P (E - V^S) / C_p."""
    T = types_w5(DATA)
    for t, x in T.items():
        if t == '_meta' or not x['eligible']:
            continue
        assert x['ratio_emp'] == pytest.approx(
            x['qP']*(x['med_re']-x['med_rc'])/CP, rel=1e-9)
        assert x['reg_emp'] == reg_from_ratio(x['ratio_emp'])


def test_ratio_emp_matches_memo():
    """정합성 메모 §5 표의 실측 손익배수를 그대로 재현한다.

    레이_16.4 만 0.00 → 0.14 로 다르다. 메모 §5 표가 §4.1 의 용량 병합 **이전**
    값(q_P=0)이라 그렇고, 병합 후 q_P=0.217 이 맞다.
    """
    R = ratio_emp_w5(DATA)
    memo = {'SM3_26.6': 0.02, '쏘울_27.0': 0.03, '아이오닉_28.0': 0.63,
            'SM3_35.9': 1.27, '코나_64.0': 1.42, '볼트_60.0': 3.53,
            '니로_64.0': 3.54, '쏘울_64.0': 5.49, '포터2_58.8': 5.91,
            '봉고3_58.8': 11.03}
    for t, v in memo.items():
        assert R[t] == pytest.approx(v, abs=0.005), t
    assert R['레이_16.4'] == pytest.approx(0.14, abs=0.005)


def test_selection_is_a_ladder():
    """§5.1 — 손익분기 1.0 을 사이에 두고 실측 등급이 갈린다."""
    R = ratio_emp_w5(DATA)
    r = [R[t] for t in SEL_W5]
    assert r == sorted(r), '사다리는 오름차순이어야 한다'
    assert r[0] < 1.0 < r[1], '손익분기가 첫 둘 사이에 놓여야 한다'
    assert [reg_from_ratio(x) for x in r] == ['R1', 'R3', 'R2', 'R2']


# ---------------------------------------------------------------- 가격함수
def test_pricew5_matches_memo_effective_intercepts():
    """smear=False 유효절편이 정합성 메모의 *_c0_eff 와 같아야 한다."""
    d = json.load(open(DATA/'params_w5.json', encoding='utf-8'))
    P = PriceW5.from_json(DATA/'params_w5.json'); P.smear = False
    assert P._re0 == pytest.approx(d['reuse_c0_eff'], abs=1e-9)
    assert P._rc0 == pytest.approx(d['recyc_c0_eff'], abs=1e-9)


def test_smear_is_the_expectation():
    """로그정규 보정이 들어가야 E[P] 다. 보정분은 정확히 exp(σ²/2)."""
    P = PriceW5.from_json(DATA/'params_w5.json')
    Pm = PriceW5.from_json(DATA/'params_w5.json'); Pm.smear = False
    assert P.Erev(0.9, 60)/Pm.Erev(0.9, 60) == pytest.approx(np.exp(0.5*P.reuse_sd**2))
    assert P.VS(60)/Pm.VS(60) == pytest.approx(np.exp(0.5*P.recyc_sd**2))


def test_recycle_is_not_linear_in_capacity():
    """§2.3 — VS 의 용량 지수는 1.0 이 아니다. p_rc×cap 판과 갈라진다."""
    P = PriceW5.from_json(DATA/'params_w5.json')
    assert P.recyc_cap == pytest.approx(1.2675, abs=1e-4)
    assert P.p_rc_at(64.0) > P.p_rc_at(16.4), '대용량일수록 kWh당 회수가치가 높다'


@need_raw
def test_smear_matches_sample_mean():
    """표본평균 검증 — 보정을 넣은 예측 평균이 실측 평균을 맞춘다."""
    sys.path.insert(0, str(ROOT/'scripts'))
    from build_types_w5 import load_bids
    R, C = load_bids(ROOT)
    P = PriceW5.from_json(DATA/'params_w5.json')
    ln = P.reuse_c0 + P.reuse_cap*np.log(R.kwh) + P.reuse_s*np.log(R.s) \
        + P.reuse_li*np.log(R['리튬'])
    assert float(np.exp(ln+0.5*P.reuse_sd**2).mean())/R['pack'].mean() == \
        pytest.approx(1.0, abs=0.05)
    assert float(np.exp(ln).mean())/R['pack'].mean() < 0.95, '보정 없으면 과소'


def test_price_cache_keys_are_distinct():
    """세 가격구조가 캐시 해시에서 서로 섞이면 안 된다."""
    k5 = _price_key(PriceParams.from_json(DATA/'params_v5.json'))
    k4 = _price_key(PriceParams.from_json(DATA/'params.json'))
    kw5 = _price_key(PriceW5.from_json(DATA/'params_w5.json'))
    assert k5 == {}, '구 파라미터는 기존 캐시를 살리기 위해 빈 키'
    assert 'price' in k4 and 'price_w5' in kw5
    assert len({json.dumps(x, sort_keys=True) for x in (k5, k4, kw5)}) == 3


# ---------------------------------------------------------------- 인스턴스
def _inst(W=4):
    F = fleet_w5(DATA, sel=SEL_W5)
    _, FS = load_cached(DATA)
    tot = sum(F[t][0] for t in SEL_W5)
    types = {t: (F[t][1], F[t][2], F[t][3], F[t][4], F[t][0]/tot) for t in SEL_W5}
    cfg = Config(Mcyc=1, Cp=500_000, Cf=20_000, phi=1.0, NARR=4, W=W,
                 F_E=FS['F_E'], F_U=FS['F_U'])
    return Instance(types, PriceW5.from_json(DATA/'params_w5.json'), cfg)


def test_instance_uses_w5_ratio_table():
    I = _inst(4)
    assert set(I.RATIO_EMP) == set(SEL_W5)
    assert I.RATIO_EMP['코나_64.0'] == pytest.approx(1.42, abs=0.005)


def test_no_type_is_pruned():
    """w4 는 SM3·쏘울을 R1 로 가지쳤다. w5 인스턴스는 넷 다 살아 있다 —
    행동공간이 그만큼 커지고, 이것이 W=7 이 안 되는 직접 원인이다."""
    I = _inst(4)
    assert I.DUMP == set()
    assert I.REG['아이오닉_28.0'] == 'R3'


def test_gstar_W4():
    I = _inst(4)
    assert len(I.ST) == 4_845
    A = build(I, workers=8)
    S = (ExactSolver(A, device='cuda') if torch.cuda.is_available() else NumpySolver(A))
    g, h, _ = S.solve()
    assert g == pytest.approx(8_045_931.40, rel=2e-6)
    gap = lambda v: 100*(g-float(S.evaluate(v)[0]))/g
    assert gap(pol.b1_sell_all(I)) == pytest.approx(32.730, abs=0.01)
    assert gap(pol.b2_no_screening(I)) == pytest.approx(7.734, abs=0.01)


def test_bfast_hold_is_not_a_relaxation():
    """B_fast_hold 는 B_fast 의 완화가 아니라 **다른 고정정책**이다.

    w3·w4 에서는 gap(B_fast) >= gap(B_fast_hold) 라 「즉시처분 강제의 비용」이
    양수로 읽혔지만, 그건 정리가 아니라 그 인스턴스의 성질이었다. w5 에서는
    부호가 뒤집힌다 — 보유가 창고를 먹어 강제매각을 늘리기 때문이다.
    """
    I = _inst(4)
    A = build(I, workers=8)
    S = (ExactSolver(A, device='cuda') if torch.cuda.is_available() else NumpySolver(A))
    g, _, _ = S.solve()
    bf = max(float(S.evaluate(pol.b_fast(I, t))[0]) for t in range(I.cfg.SB+1))
    bfh = max(float(S.evaluate(pol.b_fast_hold(I, t))[0]) for t in range(I.cfg.SB+1))
    assert bfh < bf, 'w5 에서는 보유판이 오히려 나쁘다'
