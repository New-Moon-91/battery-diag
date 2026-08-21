# -*- coding: utf-8 -*-
"""w6 회귀 — 스트리밍 병합의 디스크 피크, 그리고 W 무관 정규화(WREF).

WREF 는 기본값 None 이면 **기존 동작 그대로**여야 한다. w1~w5 결과가 전부 그
경로로 나왔기 때문이다. 아래 첫 두 검사가 그것을 못 박는다.
"""
import os, sys, shutil, tempfile
from pathlib import Path
import numpy as np
import pytest, torch

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT/'src'))
from battery_diag import encode as enc, net as netmod
from battery_diag.data import SEL_W5, fleet_w5, load_cached
from battery_diag.instance import Instance, PriceW5, Config
from battery_diag.encode import state_tensors
from battery_diag.net import PolicyNet

DEV = 'cuda' if torch.cuda.is_available() else 'cpu'
DATA = ROOT/'data'


@pytest.fixture(autouse=True)
def _reset_wref():
    """검사마다 전역 WREF 를 되돌린다 — 새는 순간 다른 검사가 조용히 깨진다."""
    a, b = enc.WREF, netmod.WREF
    yield
    enc.WREF, netmod.WREF = a, b


def _inst(W):
    F = fleet_w5(DATA, sel=SEL_W5); _, FS = load_cached(DATA)
    tot = sum(F[t][0] for t in SEL_W5)
    types = {t: (F[t][1], F[t][2], F[t][3], F[t][4], F[t][0]/tot) for t in SEL_W5}
    return Instance(types, PriceW5.from_json(DATA/'params_w5.json'),
                    Config(Mcyc=1, Cp=500_000, Cf=20_000, phi=1.0, NARR=4, W=W,
                           F_E=FS['F_E'], F_U=FS['F_U']))


# ------------------------------------------------------------------ 기본값 보존
def test_wref_default_is_none():
    assert enc.WREF is None and netmod.WREF is None, \
        '기본값이 켜져 있으면 w1~w5 재현이 조용히 깨진다'


def test_ctx_default_matches_legacy_formula():
    """WREF=None 이면 ctx 는 예전 식 그대로 — Um·Sm·(Um+Sm) 으로 나눈다."""
    I = _inst(4)
    t = state_tensors(I, device='cpu')
    Um = Sm = I.W
    ctx = t['ctx'].numpy()
    for si, (n, scr) in enumerate(I.ST[:200]):
        tot = sum(n) + len(scr)
        want = [sum(n)/Um, len(scr)/Sm, I.CAP/(Um+Sm), tot/(Um+Sm)]
        assert np.allclose(ctx[si], want, atol=1e-6), (si, ctx[si], want)


# ------------------------------------------------------------------ W 무관성
def test_wref_makes_ctx_w_invariant():
    """같은 물리적 점유는 W 가 달라도 같은 ctx 여야 한다 (WREF 켠 경우)."""
    enc.WREF = netmod.WREF = 8.0
    t4 = state_tensors(_inst(4), device='cpu')
    t6 = state_tensors(_inst(6), device='cpu')
    I4, I6 = _inst(4), _inst(6)
    # 같은 (n, scr) 상태를 양쪽에서 찾아 ctx 를 비교한다.
    idx6 = {st: i for i, st in enumerate(I6.ST)}
    n_cmp = 0
    for i4, st in enumerate(I4.ST):
        j = idx6.get(st)
        if j is None:
            continue
        assert np.allclose(t4['ctx'][i4].numpy(), t6['ctx'][j].numpy(), atol=1e-6), st
        n_cmp += 1
    assert n_cmp > 100, f'비교한 공통 상태가 너무 적다 ({n_cmp})'


def test_ctx_is_w_dependent_without_wref():
    """대조군 — 끄면 실제로 W 에 의존한다. 이게 전이가 깨지는 원인이다."""
    I4, I6 = _inst(4), _inst(6)
    t4 = state_tensors(I4, device='cpu'); t6 = state_tensors(I6, device='cpu')
    idx6 = {st: i for i, st in enumerate(I6.ST)}
    diff = 0
    for i4, st in enumerate(I4.ST):
        j = idx6.get(st)
        if j is None:
            continue
        if not np.allclose(t4['ctx'][i4].numpy(), t6['ctx'][j].numpy(), atol=1e-6):
            diff += 1
    assert diff > 0, 'W 의존이 없다면 [4] 의 전제가 틀린 것이다'


# ------------------------------------------------------------------ carry 일관성
@pytest.mark.parametrize('W', [3, 4])
def test_wref_carry_decode_matches_teacher_forcing(W):
    """WREF 를 켜도 decode 의 축차 carry == carry_from_labels 의 누적합이어야 한다.

    두 경로가 갈리면 학습(교사강요)과 추론이 다른 함수가 된다. WREF 는 분모만
    바꾸므로 이 항등식은 유지돼야 하고, 유지되지 않으면 패치가 틀린 것이다.
    """
    enc.WREF = netmod.WREF = 8.0
    I = _inst(W)
    t = state_tensors(I, device=DEV)
    sel = slice(0, min(256, len(I.ST)))
    b = {k: t[k][sel] for k in ('U', 'mu', 'S', 'ms', 'ctx', 'allow_u', 'allow_s', 'wcap')}
    torch.manual_seed(0)
    net = PolicyNet(carry=True).to(DEV)
    au, as_, cU_dec, cS_dec = net.decode(b, I.CAP, return_carry=True)
    lu = torch.as_tensor(au, device=DEV); ls = torch.as_tensor(as_, device=DEV)
    cU_lab, cS_lab, _, _ = net.carry_from_labels(b, lu, ls, I.CAP)
    assert torch.allclose(cU_dec, cU_lab, atol=1e-6), '미검사축 carry 불일치'
    assert torch.allclose(cS_dec, cS_lab, atol=1e-6), '선별축 carry 불일치'


def test_net_accepts_foreign_W_tensors():
    """W=4 에서 만든 망이 W=6 텐서를 그대로 먹는다 — 전이의 구조적 전제."""
    I4, I6 = _inst(4), _inst(6)
    t4 = state_tensors(I4, device=DEV); t6 = state_tensors(I6, device=DEV)
    torch.manual_seed(0); net = PolicyNet(carry=True).to(DEV)
    keys = ('U', 'mu', 'S', 'ms', 'ctx', 'allow_u', 'allow_s', 'wcap')
    a4 = net.decode({k: t4[k][:64] for k in keys}, I4.CAP)[0]
    a6 = net.decode({k: t6[k][:64] for k in keys}, I6.CAP)[0]
    assert a4.shape[1] == 4 and a6.shape[1] == 6, '슬롯 수가 W 를 따라야 한다'


# ------------------------------------------------------------------ 병합 디스크 피크
def test_stream_merge_frees_parts_incrementally():
    """w6 [0] — 파트를 즉시 unlink 하므로 병합 피크가 2배가 되지 않는다.

    빌드가 끝난 시점에 parts/ 가 비어 있고 최종본만 남는지로 확인한다.
    (실측 피크는 scripts 밖에서 따로 쟀다 — W=5 에서 피크/최종 = 0.998.)
    """
    from battery_diag.streambuild import build_stream
    I = _inst(3)
    with tempfile.TemporaryDirectory() as td:
        A = build_stream(I, cache_dir=td, tag='t', workers=2, log=lambda *a: None)
        nnz = int(A['indptr'][-1])
        d = next(Path(td).glob('stream_*'))
        assert not (d/'parts').exists(), 'parts 디렉터리가 남아 있으면 안 된다'
        got = (d/'indices.i32').stat().st_size + (d/'probs.f64').stat().st_size
        assert got == nnz*12, (got, nnz*12)
        total = sum(f.stat().st_size for f in d.rglob('*') if f.is_file())
        assert total < nnz*12*1.5, f'디렉터리 총량이 최종본의 1.5배 미만이어야 한다 ({total})'
