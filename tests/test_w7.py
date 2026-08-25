# -*- coding: utf-8 -*-
"""w7 회귀 — 시뮬레이터(S1)가 정확 전이구조와 같은 분포를 내는가.

전체 검증은 `scripts/validate_s1.py` 가 하고(W=4~6 × 40쌍 × 2만 표본), 여기서는
빠르게 도는 핵심 불변량만 못 박는다.
"""
import sys
from pathlib import Path
import numpy as np
import pytest

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT/'src'))
from battery_diag.data import SEL_W5, fleet_w5, load_cached
from battery_diag.instance import Instance, PriceW5, Config
from battery_diag.simulate import Simulator

DATA = ROOT/'data'


def _inst(W=4, sel=None):
    sel = list(sel or SEL_W5)
    F = fleet_w5(DATA, sel=sel); _, FS = load_cached(DATA)
    tot = sum(F[t][0] for t in sel)
    types = {t: (F[t][1], F[t][2], F[t][3], F[t][4], F[t][0]/tot) for t in sel}
    return Instance(types, PriceW5.from_json(DATA/'params_w5.json'),
                    Config(Mcyc=1, Cp=500_000, Cf=20_000, phi=1.0, NARR=4, W=W,
                           F_E=FS['F_E'], F_U=FS['F_U']))


def test_simulator_stays_in_support():
    """표본추출된 다음상태는 반드시 정확 전이의 지지집합 안에 있어야 한다."""
    I = _inst(4); sim = Simulator(I); rng = np.random.default_rng(0)
    for si in (100, 900, 2500, 4000):
        st = I.ST[si]
        act = I.actions(st)[0]
        sup = {j for _p, j, _r in I.step_dist(st, act)}
        for _ in range(300):
            _, ns = sim.step(st, act, rng)
            assert I.SI[ns] in sup, (si, ns)


def test_simulator_matches_exact_distribution():
    """총변동거리가 표본오차 수준인가 (2만 표본이면 대략 0.1 이내)."""
    I = _inst(4); sim = Simulator(I); rng = np.random.default_rng(1)
    st = I.ST[1500]
    act = I.actions(st)[len(I.actions(st))//2]
    ex = {}
    for p, j, _r in I.step_dist(st, act):
        ex[j] = ex.get(j, 0.0) + p
    N = 20000
    cnt = {}
    for _ in range(N):
        _, ns = sim.step(st, act, rng)
        j = I.SI[ns]; cnt[j] = cnt.get(j, 0) + 1
    tvd = 0.5*sum(abs(cnt.get(j, 0)/N - p) for j, p in ex.items())
    assert tvd < 0.05, tvd


def test_simulator_mean_reward_matches():
    """기대보상이 정확값과 표본오차 안에서 같아야 한다."""
    I = _inst(4); sim = Simulator(I); rng = np.random.default_rng(2)
    st = I.ST[2000]
    act = I.actions(st)[0]
    d = I.step_dist(st, act)
    er = sum(p*r for p, _j, r in d)
    er2 = sum(p*r*r for p, _j, r in d)
    sd = np.sqrt(max(er2-er*er, 0.0))
    N = 20000
    rs = np.array([sim.step(st, act, rng)[0] for _ in range(N)])
    z = (rs.mean()-er)/(sd/np.sqrt(N)) if sd > 0 else 0.0
    assert abs(z) < 4.0, (z, er, rs.mean())


def test_arrival_model_switch():
    """이항/포아송은 평균이 같고 분산이 다르다 — 정확해 대조에는 이항만 쓴다."""
    I = _inst(4)
    rng = np.random.default_rng(3)
    lam_eff = I.cfg.lam*I.cfg.NARR
    for mode, var in (('binomial', I.cfg.NARR*I.cfg.lam*(1-I.cfg.lam)), ('poisson', lam_eff)):
        sim = Simulator(I, arrivals=mode)
        a = np.array([sim._arrivals(rng).sum() for _ in range(20000)], float)
        assert abs(a.mean()-lam_eff) < 0.05, (mode, a.mean())
        assert abs(a.var()-var) < 0.12*max(var, 1e-9) + 0.05, (mode, a.var(), var)


def test_holding_cost_charged_on_pre_arrival_occupancy():
    """보관비는 도착 **전** 점유에 걸린다 (step_dist 와 같은 시점).

    도착이 0 인 사건은 확률 (1-lam)^NARR = 1e-4 라 표본으로 기다릴 수 없다.
    도착 표본추출만 0 으로 고정해 보상 분해가 정확히 맞는지 본다.
    """
    I = _inst(4); sim = Simulator(I, arrivals='binomial')
    rng = np.random.default_rng(4)
    sim._arrivals = lambda _rng: np.zeros(len(I.TY), np.int64)
    st = I.ST[3000]
    n, scr = st
    act = next(a for a in I.actions(st) if sum(a[1]) == 0)   # 신속검사 없음 → 결정적
    sv, fv, pu, ss, ps = act
    rem = sum(n[k]-sv[k]-fv[k]-pu[k] for k in range(len(I.TY)))
    occ = rem + (len(scr) - len(ss) - len(ps))
    base = (sum(sv[k]*I.VS[t] for k, t in enumerate(I.TY))
            + sum(pu[k]*I.VPU[t] for k, t in enumerate(I.TY))
            + sum(I.VS[scr[j][0]] for j in ss) + sum(I.VPS[scr[j]] for j in ps))
    r, ns = sim.step(st, act, rng)
    assert abs(r - (base - I.cfg.h*occ)) < 1e-6, (r, base, occ)
    assert sum(ns[0]) == rem


def test_forced_sale_sells_cheapest_first():
    """창고가 넘치면 재활용 매각가치가 낮은 유형부터 판다 (instance 규칙과 동일)."""
    I = _inst(4); sim = Simulator(I, arrivals='binomial')
    rng = np.random.default_rng(5)
    # 창고를 꽉 채운 상태 = 도착분이 전부 강제매각된다
    full = next(st for st in I.ST if sum(st[0]) + len(st[1]) == I.W)
    act = I.actions(full)[0]          # 꽉 찬 상태에는 «전부 보유» 행동이 없다(가지치기)
    order = [I.TY[k] for k in sim.sell_order]
    vs = [I.VS[t] for t in order]
    assert vs == sorted(vs), '매각 순서가 V^S 오름차순이어야 한다'
    for _ in range(50):
        _, ns = sim.step(full, act, rng)
        assert sum(ns[0]) + len(ns[1]) <= I.W, '창고 상한을 넘으면 안 된다'
