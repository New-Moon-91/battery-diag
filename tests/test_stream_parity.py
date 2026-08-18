# -*- coding: utf-8 -*-
"""스트리밍 경로 검증 — build_stream + StreamSolver 가 기존 build + ExactSolver 와
같은 답을 내는지 확인한다.

작은 인스턴스에서만 돌린다. 두 경로가 모두 가능한 구간이어야 대조가 되기 때문.
슬랩은 일부러 잘게 강제해(slab_nnz 작게) 스트리밍 분기가 실제로 타도록 한다.

    pytest -q tests/test_stream_parity.py
"""
import sys, tempfile
from pathlib import Path
import numpy as np
import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / 'src'))
from battery_diag.data import load_cached
from battery_diag.instance import Instance, PriceParams, Config
from battery_diag.build import build
from battery_diag.streambuild import build_stream
from battery_diag.exact import ExactSolver
from battery_diag.bigexact import StreamSolver

try:
    import torch
    DEV = 'cuda' if torch.cuda.is_available() else 'cpu'
except ImportError:
    DEV = 'cpu'

SEL = ['레이', '코나', 'SM3']


def _inst(NMAX, SMAX, NARR=4):
    FLEET, FS = load_cached('data')
    tot = sum(FLEET[t][0] for t in SEL)
    types = {t: (FLEET[t][1], FLEET[t][2], FLEET[t][3], FLEET[t][4], FLEET[t][0] / tot) for t in SEL}
    price = PriceParams.from_json(Path('data') / 'params.json')
    cfg = Config(Mcyc=1, Cp=500000, Cf=20000, phi=1.0, NARR=NARR,
                 NMAX=NMAX, SMAX=SMAX, F_E=FS['F_E'], F_U=FS['F_U'])
    return Instance(types, price, cfg)


@pytest.mark.parametrize('NMAX,SMAX', [(2, 2), (2, 3)])
def test_stream_matches_exact(NMAX, SMAX):
    I = _inst(NMAX, SMAX)
    A = build(I, workers=8)
    with tempfile.TemporaryDirectory() as td:
        B = build_stream(I, cache_dir=td, workers=8)

        # 1) 빌드 산출물이 배열 단위로 동일한가
        for k in ('indptr', 'rsa', 'aptr'):
            assert np.array_equal(A[k], B[k]), f'{k} 불일치'
        assert np.array_equal(A['indices'], np.asarray(B['indices'])), 'indices 불일치'
        assert np.array_equal(A['probs'], np.asarray(B['probs'])), 'probs 불일치'

        E = ExactSolver(A, device=DEV)
        # 슬랩을 잘게 잘라 스트리밍 분기를 강제한다
        S = StreamSolver(B, device=DEV, slab_nnz=max(A['indptr'][-1] // 20, 1000))
        assert len(S.slabs) > 3, '슬랩이 갈리지 않아 스트리밍 경로를 검증하지 못함'

        rng = np.random.default_rng(0)
        na = np.diff(A['aptr'])

        # 2) 정책평가 — 무작위 정책 3개
        for _ in range(3):
            acts = np.array([rng.integers(0, n) for n in na], np.int64)
            g1, h1 = E.evaluate(acts)
            g2, h2 = S.evaluate(acts)
            assert abs(g1 - g2) <= 1e-6 * max(1.0, abs(g1)), f'g 불일치 {g1} vs {g2}'
            assert np.max(np.abs(h1 - h2)) <= 1e-4 * max(1.0, np.max(np.abs(h1)))

        # 3) 1단계 개선 — 배열 완전일치가 아니라 **정책 성능**으로 본다.
        #
        # q 는 1e6 규모이고 두 경로의 덧셈 순서가 달라 반올림 잡음이 ~1e-9 생긴다.
        # 참으로 값이 같은 행동들 사이에서 어느 쪽이 뽑히는지는 그 잡음이 정하므로
        # 비트 단위 일치는 원리적으로 불가능하다. 의미 있는 기준은 성능이다.
        for k in range(3):
            h = rng.normal(0, 1e5, E.nS)
            aE, aS = E.improve(h), S.improve(h)
            nd = int((aE != aS).sum())
            gE, _ = E.evaluate(aE)
            gS, _ = E.evaluate(aS)          # 동일 솔버로 평가해야 공정
            rel = abs(gE - gS) / max(1.0, abs(gE))
            print(f'  improve {k}: 선택차 {nd}/{E.nS}  g {gE:,.4f} vs {gS:,.4f}  상대차 {rel:.2e}')
            assert rel <= 1e-9, f'개선 정책 성능 불일치 (상대차 {rel:.3e})'

        # 4) 최적해 — RVI(기준) vs 스트리밍 정책반복
        g_r, h_r, _ = E.solve()
        g_s, h_s, _ = S.solve()
        print(f'  g*: RVI {g_r:,.4f} vs PI {g_s:,.4f}  상대차 {abs(g_r-g_s)/abs(g_r):.2e}')
        assert abs(g_r - g_s) <= 1e-7 * abs(g_r), f'g* 불일치 {g_r:,.4f} vs {g_s:,.4f}'

        # 5) 정상상태 분포
        acts_star = E.greedy(h_r)
        d1 = E.stationary(acts_star)
        d2 = S.stationary(acts_star)
        assert np.max(np.abs(d1 - d2)) <= 1e-9
