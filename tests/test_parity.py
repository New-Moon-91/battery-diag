# -*- coding: utf-8 -*-
"""이식 검증 — GPU 재작성본이 기존 CPU 결과를 재현하는가.
서버에서 가장 먼저 실행할 것:  pytest -q tests/test_parity.py
기준값은 2026-08-15 CPU 실행분(레이·코나·SM3, Mcyc=1, NMAX=SMAX=2, Cp=50만, Cf=2만).

**가격 파라미터는 params_v5.json 을 쓴다.** 이 파일의 기준값은 구 파라미터로 낸
CPU 실행분이므로, w4 재캘리브레이션된 params.json 을 쓰면 당연히 어긋난다.
여기서 검증하는 것은 "가격이 얼마인가" 가 아니라 "같은 가격에서 GPU 경로가 CPU 경로를
재현하는가" 이므로 파라미터를 고정하는 것이 옳다. 새 파라미터의 회귀는 test_w4.py 가 맡는다."""
import json, sys
from pathlib import Path
import numpy as np, pytest, torch
sys.path.insert(0, str(Path(__file__).resolve().parents[1]/'src'))
from battery_diag.data import load_cached
from battery_diag.instance import Instance, PriceParams, Config
from battery_diag.build import build
from battery_diag.exact import ExactSolver, NumpySolver
from battery_diag import policies as pol

# 기준값은 2026-08-15 CPU 실행분. 당시 q_P 를 소수 3자리로 반올림해 썼으므로
# 원자료 정밀값을 쓰는 현재 코드와 g* 가 0.15% 이내에서 다르다 (정상).
REF = {   # NARR: (g*, B1, B_fast, B2, INDEX)  — 갭(%)
    1: (1_288_181, 74.63, 0.00, 5.68, 3.75),
    2: (2_540_933, 74.28, 9.64, 18.87, 9.23),
    3: (3_434_569, 72.82, 11.46, 22.98, 17.11),
    4: (3_994_357, 71.03, 10.50, 23.73, 20.28),
}
GTOL, GAPTOL = 2e-3, 0.25
DATA = Path(__file__).resolve().parents[1]/'data'


def _inst(NARR):
    FLEET, FS = load_cached(DATA)
    sel = ['레이','코나','SM3']; tot = sum(FLEET[t][0] for t in sel)
    types = {t: (FLEET[t][1], FLEET[t][2], FLEET[t][3], FLEET[t][4], FLEET[t][0]/tot) for t in sel}
    cfg = Config(NARR=NARR, F_E=FS['F_E'], F_U=FS['F_U'])
    return Instance(types, PriceParams.from_json(DATA/'params_v5.json'), cfg)


def test_action_count():
    I = _inst(2)
    assert len(I.ST) == 1485
    assert I.n_actions() == 78_000
    assert I.REG == {'레이':'R1', '코나':'R2', 'SM3':'R3'}


@pytest.mark.parametrize('NARR', [1, 2, 3, 4])
def test_gstar_and_benchmarks(NARR):
    I = _inst(NARR)
    A = build(I, workers=8)
    S = (ExactSolver(A, device='cuda') if torch.cuda.is_available() else NumpySolver(A))
    g, h, _ = S.solve()
    ref_g, b1, bf, b2, ix = REF[NARR]
    assert abs(g - ref_g)/ref_g < GTOL, f'g*={g:,.0f} vs 기준 {ref_g:,.0f}'
    bench = pol.all_benchmarks(I, S)
    gap = lambda v: 100*(g-v)/g
    for name, ref in [('B1', b1), ('B_fast', bf), ('B2', b2), ('INDEX', ix)]:
        assert abs(gap(bench[name]) - ref) < GAPTOL, f'{name} 갭 {gap(bench[name]):.2f}% vs 기준 {ref}%'


def test_dcl_shapes():
    """배치 인코더/디코더가 MDP 행동집합 안의 행동만 내는지"""
    from battery_diag.encode import state_tensors, actions_from_assign
    from battery_diag.net import PolicyNet
    dev = 'cuda' if torch.cuda.is_available() else 'cpu'
    I = _inst(2); tens = state_tensors(I, device=dev)
    net = PolicyNet().to(dev)
    b = {k: tens[k] for k in ('U','mu','S','ms','ctx','allow_u','allow_s')}
    au, as_ = net.decode(b, I.CAP)
    acts = actions_from_assign(I, tens, au, as_)
    assert acts.shape == (len(I.ST),)
    assert all(0 <= acts[i] < len(I.actions(st)) for i, st in enumerate(I.ST))
