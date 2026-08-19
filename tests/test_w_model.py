# -*- coding: utf-8 -*-
"""W-정식화 (단일 창고 제약) 검증.

새 정식화 사양
  · 상태공간 = {(n, sc) : Σ_t n_t + |sc| ≤ W}. NMAX·SMAX 는 쓰지 않는다.
  · 도착이 W 를 넘치면 초과분은 미검사 상태로 즉시 재활용 매각 (수익 p_rc×kWh).
    무료 소실 금지. 초과분은 매각가치가 낮은 차종부터.
  · 신속검사는 미검사→선별완료 이동이라 총 점유가 불변 — W 와 무관.

    pytest -q tests/test_w_model.py
"""
import sys
from math import comb
from pathlib import Path
import numpy as np
import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / 'src'))
from battery_diag.data import load_cached
from battery_diag.instance import Instance, PriceParams, Config

SEL = ['레이', '코나', 'SM3']


def _types():
    FLEET, FS = load_cached('data')
    tot = sum(FLEET[t][0] for t in SEL)
    return ({t: (FLEET[t][1], FLEET[t][2], FLEET[t][3], FLEET[t][4], FLEET[t][0]/tot)
             for t in SEL}, FS)


def _inst(sel=None, **kw):
    sel = sel or SEL
    FLEET, FS = load_cached('data')
    tot = sum(FLEET[t][0] for t in sel)
    types = {t: (FLEET[t][1], FLEET[t][2], FLEET[t][3], FLEET[t][4], FLEET[t][0]/tot)
             for t in sel}
    price = PriceParams.from_json(Path('data')/'params.json')
    return Instance(types, price, Config(Mcyc=1, Cp=500000, Cf=20000, phi=1.0,
                                         F_E=FS['F_E'], F_U=FS['F_U'], **kw))


# ---------------------------------------------------------------- 상태공간
def test_state_count_closed_form():
    """|S| = C(W + K, K),  K = |T| + |T|*SB = 3 + 9 = 12.

    (n, sc) 는 12종(미검사 3차종 + 선별완료 9종)에서 크기 ≤ W 인 멀티셋과 일대일이다.
    """
    for W in (1, 2, 3, 4, 5):
        I = _inst(NARR=4, W=W)
        assert len(I.ST) == comb(W + 12, 12), f'W={W}'
        assert len(I.SI) == len(I.ST), '상태 중복'


def test_state_count_hand_check_W1():
    """W=1 손계산: 빈 상태 1 + 미검사 1대 3가지 + 선별완료 1대 9가지 = 13."""
    I = _inst(NARR=4, W=1)
    assert len(I.ST) == 13
    assert sum(1 for n, sc in I.ST if sum(n) + len(sc) == 0) == 1
    assert sum(1 for n, sc in I.ST if sum(n) == 1) == 3
    assert sum(1 for n, sc in I.ST if len(sc) == 1) == 9


def test_capacity_never_exceeded():
    """어떤 (상태, 행동) 에서도 Σn + |sc| > W 인 상태로 가지 않는다. 확률합도 1."""
    I = _inst(NARR=4, W=3)
    occ = np.array([sum(n) + len(sc) for n, sc in I.ST])
    assert occ.max() <= 3
    for si, st in enumerate(I.ST):
        for a in I.actions(st):
            o = I.step_dist(st, a)
            assert abs(sum(p for p, _, _ in o) - 1.0) < 1e-9, f'확률합 {st} {a}'
            assert all(occ[j] <= 3 for _, j, _ in o), f'W 초과 전이 {st} {a}'


# ---------------------------------------------------------------- 강제매각
def test_forced_sale_reward_exact():
    """포화 상태에서 전량 보유하면 도착분이 전부 강제매각된다.

    기대 즉시보상 = -h·W + NARR·λ·Σ_t MIX_t·(p_rc×kWh_t).
    무료 소실이면 두 번째 항이 0 이 되므로 이 검사가 소실 금지를 고정한다.
    """
    W = 3
    I = _inst(NARR=4, W=W)
    st = ((0, W, 0), ())                       # 코나 W대, 선별완료 없음
    assert sum(st[0]) + len(st[1]) == W and st in I.SI
    hold = next(a for a in I.actions(st)
                if not any(sum(x) for x in a[:3]) and not a[3] and not a[4])
    Er = sum(p*r for p, _, r in I.step_dist(st, hold))
    c = I.cfg
    want = -c.h*W + c.NARR*c.lam*sum(I.MIX[t]*I.VS[t] for t in I.TY)
    assert abs(Er - want) < 1e-6*max(1.0, abs(want)), f'{Er} vs {want}'


def test_forced_sale_order_cheapest_first():
    """초과분은 매각가치 p_rc×kWh 가 낮은 차종부터 판다."""
    I = _inst(NARR=4, W=2)
    order = [I.TY[k] for k in I._sell_order]
    assert order == sorted(I.TY, key=lambda t: I.VS[t]), order
    # slack=0 이면 도착 전량이 매각되고 수용 개수벡터는 0 이다
    for p, a, rev, ns in I._arr_slack(0):
        assert sum(a) == 0
    # slack 이 NARR 이상이면 강제매각이 절대 없다
    for p, a, rev, ns in I._arr_slack(I.cfg.NARR):
        assert rev == 0.0 and ns == 0
    # 대수와 수익이 정합적이다 — 판 대수가 0 이면 수익도 0, 아니면 양수
    for slack in range(I.cfg.NARR+2):
        for p, a, rev, ns in I._arr_slack(slack):
            assert (ns == 0) == (rev == 0.0)


def test_no_free_loss_probability_conserved():
    """도착 확률질량이 어디서도 사라지지 않는다 (소실은 매각으로 흡수된다)."""
    I = _inst(NARR=4, W=2)
    for slack in range(0, I.cfg.NARR+2):
        assert abs(sum(p for p, _, _, _ in I._arr_slack(slack)) - 1.0) < 1e-12


# ---------------------------------------------------------------- 기존 모형과의 관계
def test_legacy_path_untouched():
    """W=None 이면 기존 상태공간이 그대로 나온다 (v5b 재현성)."""
    I = _inst(NARR=4, NMAX=3, SMAX=3)
    assert I.W is None
    assert len(I.ST) == 14080
    assert I.summary()['selectivity'] == pytest.approx(1/12)


def test_large_W_matches_legacy():
    """용량이 어디에도 안 걸릴 만큼 크면 두 정식화의 g* 가 일치한다.

    한 차종·NARR=1 로 줄여 두 모형을 모두 정확해로 푼다. W 를 키우면 강제매각
    확률이 0 으로 가고, 같은 크기로 NMAX=SMAX 를 키운 기존 모형과 g* 가 만난다.
    (기존 모형의 소실도 이 극한에서 0 이므로 두 모형이 같은 문제가 된다.)

    W 를 더 키워도 결과는 같지만 기존 모형의 행동 수가 터진다 — W=5 에서 이미
    nA=222,285 / nnz=17M 이다. 검증에는 W=4 로 충분하다 (양쪽 합쳐 2초).
    """
    from battery_diag.build import build
    from battery_diag.exact import ExactSolver
    W = 4
    Iw = _inst(sel=['코나'], NARR=1, W=W)
    Il = _inst(sel=['코나'], NARR=1, NMAX=W, SMAX=W)
    Sw = ExactSolver(build(Iw, workers=8), device='cpu'); gw, hw, _ = Sw.solve()
    Sl = ExactSolver(build(Il, workers=8), device='cpu'); gl, _, _ = Sl.solve()
    d = np.asarray(Sw.stationary(Sw.greedy(hw)))
    occ = np.array([sum(n) + len(sc) for n, sc in Iw.ST])
    assert float(d[occ >= W].sum()) < 1e-6, '이 W 에서는 아직 창고가 찬다 — 극한이 아니다'
    assert abs(gw - gl) <= 1e-6*abs(gl), f'g* 불일치 {gw:,.4f} vs {gl:,.4f}'
