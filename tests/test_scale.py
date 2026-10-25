# -*- coding: utf-8 -*-
"""scale1 회귀 — 유체 LP 상계·Lite 인스턴스·규모 확장 정책이 소형 정확해와 맞는가."""
import sys, json
from pathlib import Path
import numpy as np
import pytest

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT/'src'))
from battery_diag.data import SEL_W5, fleet_w5
from battery_diag.instance import Instance, PriceW5, Config
from battery_diag.simulate import Simulator
from battery_diag import scale as S

DATA = ROOT/'data'


def _types():
    F = fleet_w5(DATA, sel=SEL_W5); tot = sum(F[t][0] for t in SEL_W5)
    return {t: (F[t][1], F[t][2], F[t][3], F[t][4], F[t][0]/tot) for t in SEL_W5}


def _cfg(W):
    return Config(Mcyc=1, Cp=500_000, Cf=20_000, phi=1.0, NARR=4, W=W)


def test_lite_matches_full_economics():
    """Instance.lite 의 경제량은 정규 인스턴스와 비트 단위로 같다."""
    types = _types(); p = PriceW5.from_json(DATA/'params_w5.json')
    I = Instance(types, p, _cfg(4)); L = Instance.lite(types, p, _cfg(4))
    assert I.VS == L.VS and I.VPU == L.VPU and I.VPS == L.VPS
    assert I.REG == L.REG and I.DUMP == L.DUMP and I.PUOK == L.PUOK and I.CAP == L.CAP
    assert I._sell_order == L._sell_order
    assert L.ST is None


@pytest.mark.parametrize('W', [4, 5, 6])
def test_fluid_bound_dominates_exact(W):
    """LP 상계 ≥ g*(정확해), 강쌍대(LP = min_μ L(μ)), 갭은 W 가 클수록 줄어든다."""
    f = ROOT/'results'/'w5'/f'W{W}.json'
    if not f.exists():
        pytest.skip('w5 결과 없음')
    g = json.load(open(f, encoding='utf-8'))['gstar']
    L = Instance.lite(_types(), PriceW5.from_json(DATA/'params_w5.json'), _cfg(W))
    fb = S.fluid_bound(L, 0.9*4, L.CAP)
    assert fb['bound'] >= g
    assert abs(fb['bound'] - fb['lagr_min'])/fb['bound'] < 1e-6
    assert abs(fb['bound'] - fb['lagr'])/fb['bound'] < 1e-6
    # 상계는 W 에 무관 — 갭은 g*(W) 가 커지며 줄어든다 (w5: 4.95 → 3.11 → 1.98%)
    assert fb['bound']/g - 1 < 0.06


def test_simulated_benchmarks_match_exact_w4():
    """시뮬레이션 평균보상이 정확 정책평가(W4.json)와 통계적으로 일치한다 (|z|<4)."""
    f = ROOT/'results'/'w5'/'W4.json'
    if not f.exists():
        pytest.skip('w5 결과 없음')
    d = json.load(open(f, encoding='utf-8'))
    L = Instance.lite(_types(), PriceW5.from_json(DATA/'params_w5.json'), _cfg(4))
    sim = Simulator(L)
    for name, act, ref in [('B1', S.pol_b1(L), d['bench']['B1']),
                           ('B_fast0', S.pol_b_fast(L, 0), d['B_fast_all']['0']),
                           ('INDEX', S.pol_index(L), d['bench']['INDEX'])]:
        r = S.evaluate(sim, act, T=30000, seed=7, burn=500)
        assert abs(r['g'] - ref)/max(r['se'], 1.0) < 4.0, (name, r['g'], ref, r['se'])
        assert abs(r['g'] - ref)/ref < 0.01, name


def test_lagr_policy_is_feasible_and_bounded():
    """라그랑주/대기열 정책은 CAP 을 넘지 않고, 평균보상은 LP 상계 아래다."""
    L = Instance.lite(_types(), PriceW5.from_json(DATA/'params_w5.json'), _cfg(6))
    fb = S.fluid_bound(L, 3.6, L.CAP); sim = Simulator(L)
    for kp in (None, 1.0):
        act = S.pol_lagr(L, fb['mu'], 3.6, kappa=kp)
        rng = np.random.default_rng(3); st = (tuple([0]*len(L.TY)), ())
        for _ in range(2000):
            sv, fv, pu, ss, ps = act(st)
            n, scr = st
            assert sum(pu) + len(ps) <= L.CAP
            assert all(sv[k]+fv[k]+pu[k] <= n[k] for k in range(len(n)))
            assert not (ss & ps) and max(list(ss)+list(ps)+[-1]) < len(scr)
            _, st = sim.step(st, (sv, fv, pu, ss, ps), rng)
        r = S.evaluate(sim, act, T=20000, seed=5, burn=500)
        assert r['g'] < fb['bound']
