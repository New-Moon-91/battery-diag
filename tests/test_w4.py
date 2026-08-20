# -*- coding: utf-8 -*-
"""w4 회귀 — 재캘리브레이션된 params.json 과 확정 인스턴스(SM3·쏘울·볼트·코나).

`test_parity.py` 는 구 파라미터(params_v5.json)로 CPU↔GPU 이식만 검증한다.
새 파라미터의 값 자체는 여기서 못 박는다.
"""
import json, sys
from pathlib import Path
import pytest, torch

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT/'src'))
from battery_diag.data import (load_cached, SEL_W4, RATIO_EMP, REG_EMP,
                               reg_from_ratio, REG_LO, REG_HI)
from battery_diag.instance import Instance, PriceParams, Config
from battery_diag.build import build, _price_key
from battery_diag.exact import ExactSolver, NumpySolver
from battery_diag import policies as pol

DATA = ROOT/'data'


def _inst(W=4):
    FLEET, FS = load_cached(DATA)
    tot = sum(FLEET[t][0] for t in SEL_W4)
    types = {t: (FLEET[t][1], FLEET[t][2], FLEET[t][3], FLEET[t][4], FLEET[t][0]/tot)
             for t in SEL_W4}
    cfg = Config(Mcyc=1, Cp=500_000, Cf=20_000, phi=1.0, NARR=4, W=W,
                 F_E=FS['F_E'], F_U=FS['F_U'])
    return Instance(types, PriceParams.from_json(DATA/'params.json'), cfg)


def test_params_recalibrated():
    d = json.load(open(DATA/'params.json'))
    assert d['M_s'] == 0.0, '낙찰배수는 SOH 와 무관해야 한다 (w4 [1])'
    assert d['p_rc'] == pytest.approx(9490.96, abs=0.01)
    assert d['g_cap'] == pytest.approx(1.99359, abs=1e-4)
    assert d['_calib']['n_reuse'] == 94
    v5 = json.load(open(DATA/'params_v5.json'))
    assert v5['M_s'] == pytest.approx(-0.9472, abs=1e-4), '구 파라미터가 보존돼야 한다'


def test_price_in_cache_key():
    """가격이 캐시 해시에 들어가되, 구 파라미터일 때만 빠진다."""
    assert _price_key(PriceParams.from_json(DATA/'params_v5.json')) == {}
    assert 'price' in _price_key(PriceParams.from_json(DATA/'params.json'))


def test_reg_thresholds():
    assert (REG_LO, REG_HI) == (0.8, 1.5)
    assert [reg_from_ratio(r) for r in (0.05, 0.79, 0.8, 1.49, 1.5, 2.9)] == \
        ['R1', 'R1', 'R3', 'R3', 'R2', 'R2']
    assert REG_EMP == {'SM3': 'R1', '쏘울': 'R1', '볼트': 'R2', '코나': 'R2'}


def test_reg_agreement():
    """모형 내부 경제영역과 실측 손익배수 분류가 네 차종 모두에서 일치한다 (w4 [2])."""
    I = _inst(4)
    assert I.REG == REG_EMP
    assert all(I.REG_AGREE.values()), I.REG_AGREE
    assert I.DUMP == {'SM3', '쏘울'}
    assert I.PUOK == {'볼트', '코나'}


def test_ratio_emp_matches_breakeven():
    """실측 손익배수의 정의가 (E-V^S)/(C_p/q_P) 임을 되풀기로 검산.

    구 파라미터로 코나(64kWh, SOH 0.87)의 모형 함의 배율을 내면 TASK [0] 의 8.14 가
    나와야 한다 — 정의를 옳게 복원했다는 확인이다.
    """
    p5 = PriceParams.from_json(DATA/'params_v5.json')
    assert float(p5.Erev(0.87, 64.0))/(p5.p_rc*64.0) == pytest.approx(8.14, abs=0.01)
    assert set(RATIO_EMP) == set(SEL_W4)


def test_gstar_W4():
    I = _inst(4)
    assert len(I.ST) == 4_845
    A = build(I, workers=8)
    S = (ExactSolver(A, device='cuda') if torch.cuda.is_available() else NumpySolver(A))
    g, h, _ = S.solve()
    assert g == pytest.approx(3_854_563.56, rel=2e-6)
    gap = lambda v: 100*(g-float(S.evaluate(v)[0]))/g
    assert gap(pol.b1_sell_all(I)) == pytest.approx(56.367, abs=0.01)
    assert gap(pol.b_fast(I, 0)) == pytest.approx(7.125, abs=0.01)
    assert gap(pol.b_fast_hold(I, 0)) == pytest.approx(3.259, abs=0.01)
