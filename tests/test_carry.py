# -*- coding: utf-8 -*-
"""자기회귀 디코더 검증.

1) 레거시 동치 — carry=False 의 ce_loss 가 v5b 구현과 수치까지 같은가.
   (carry_from_labels 로 갈아끼우면서 기존 결과가 재현 불가능해지면 안 된다)
2) carry 정합 — decode 가 슬롯마다 축차로 만드는 carry 와 ce_loss 가 라벨에서
   누적합으로 만드는 carry 가 같은가. 어긋나면 학습과 추론이 다른 함수가 된다.
3) 체크포인트 — v5b 가 남긴 net_best.pt 가 carry=False 로 그대로 로드되는가.

    pytest -q tests/test_carry.py
"""
import sys
from pathlib import Path
import numpy as np
import pytest
import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / 'src'))
from battery_diag.data import load_cached
from battery_diag.instance import Instance, PriceParams, Config
from battery_diag.encode import state_tensors, labels_from_actions
from battery_diag.net import PolicyNet, NEG
from battery_diag import policies as pol

DEV = 'cuda' if torch.cuda.is_available() else 'cpu'
SEL = ['레이', '코나', 'SM3']


def _inst(NMAX=2, SMAX=2, NARR=4, W=None):
    FLEET, FS = load_cached('data')
    tot = sum(FLEET[t][0] for t in SEL)
    types = {t: (FLEET[t][1], FLEET[t][2], FLEET[t][3], FLEET[t][4], FLEET[t][0] / tot) for t in SEL}
    price = PriceParams.from_json(Path('data') / 'params.json')
    cfg = Config(Mcyc=1, Cp=500000, Cf=20000, phi=1.0, NARR=NARR,
                 NMAX=NMAX, SMAX=SMAX, W=W, F_E=FS['F_E'], F_U=FS['F_U'])
    return Instance(types, price, cfg)


def _batch(I, n=512, seed=0):
    tens = state_tensors(I, device=DEV)
    rng = np.random.default_rng(seed)
    sel = rng.choice(len(I.ST), size=min(n, len(I.ST)), replace=False)
    sel = np.sort(sel)
    keys = ['U', 'mu', 'S', 'ms', 'ctx', 'allow_u', 'allow_s']
    if 'wcap' in tens: keys.append('wcap')
    b = {k: tens[k][sel] for k in keys}
    return tens, sel, b


def _legacy_ce(net, batch, lab_u, lab_s, cap):
    """v5b 의 ce_loss 를 그대로 옮긴 참조 구현."""
    U, mu, S, ms, ctx = batch['U'], batch['mu'], batch['S'], batch['ms'], batch['ctx']
    allow_u, allow_s = batch['allow_u'], batch['allow_s']
    B, Um = mu.shape; Sm = ms.shape[1]
    z = net.encode(U, mu, S, ms, ctx)
    useU = ((lab_u == 2) & (mu > 0)).to(U.dtype)
    useS = ((lab_s == 1) & (ms > 0)).to(U.dtype)
    prevU = torch.cumsum(useU, 1) - useU
    prevS = useU.sum(1, keepdim=True) + torch.cumsum(useS, 1) - useS
    budU = (cap - prevU) / max(cap, 1)
    budS = (cap - prevS) / max(cap, 1)
    zz = z.unsqueeze(1).expand(B, Um, z.shape[-1])
    lgU = net.headU(torch.cat([zz, U, budU.unsqueeze(-1)], -1))
    mU = allow_u.clone(); mU[..., 2] &= ((cap - prevU) >= 1)
    lpU = torch.log_softmax(lgU.masked_fill(~mU, NEG), -1)
    lossU = -(lpU.gather(-1, lab_u.unsqueeze(-1)).squeeze(-1) * mu).sum()
    zs = z.unsqueeze(1).expand(B, Sm, z.shape[-1])
    lgS = net.headS(torch.cat([zs, S, budS.unsqueeze(-1)], -1))
    mS = allow_s.clone(); mS[..., 1] &= ((cap - prevS) >= 1)
    lpS = torch.log_softmax(lgS.masked_fill(~mS, NEG), -1)
    lossS = -(lpS.gather(-1, lab_s.unsqueeze(-1)).squeeze(-1) * ms).sum()
    return (lossU + lossS) / B


@pytest.mark.parametrize('NMAX,SMAX', [(2, 2), (3, 2)])
def test_legacy_ce_unchanged(NMAX, SMAX):
    """carry=False 는 v5b 와 같은 손실을 낸다 — 기존 결과 재현성 보장."""
    I = _inst(NMAX=NMAX, SMAX=SMAX)
    tens, sel, b = _batch(I)
    LU, LS = labels_from_actions(I, tens, pol.index_myopic(I))   # 전 상태 → 부분집합
    lu = torch.as_tensor(LU[sel], device=DEV); ls = torch.as_tensor(LS[sel], device=DEV)
    torch.manual_seed(0); net = PolicyNet(carry=False).to(DEV)
    new = net.ce_loss(b, lu, ls, I.CAP)
    ref = _legacy_ce(net, b, lu, ls, I.CAP)
    assert torch.allclose(new, ref, rtol=0, atol=0), f'레거시 손실 불일치 {new} vs {ref}'


@pytest.mark.parametrize('kw', [dict(NMAX=2, SMAX=2), dict(NMAX=3, SMAX=2),
                                dict(NMAX=3, SMAX=3), dict(W=3), dict(W=4)],
                         ids=['N2S2', 'N3S2', 'N3S3', 'W3', 'W4'])
def test_decode_ce_carry_match(kw):
    """decode 의 축차 carry == carry_from_labels 의 누적합 carry.

    W-정식화(잔여용량이 W 기준, 보유 미검사분도 자리를 먹음)에서도 걸어야 한다 —
    두 경로의 정의가 갈리면 학습과 추론이 다른 함수가 된다.
    """
    I = _inst(**kw)
    tens, sel, b = _batch(I)
    torch.manual_seed(0); net = PolicyNet(carry=True).to(DEV)
    au, as_, cU_dec, cS_dec = net.decode(b, I.CAP, return_carry=True)
    lu = torch.as_tensor(au, device=DEV); ls = torch.as_tensor(as_, device=DEV)
    cU_lab, cS_lab, _, _ = net.carry_from_labels(b, lu, ls, I.CAP)
    assert cU_dec.shape == cU_lab.shape and cS_dec.shape == cS_lab.shape
    assert torch.allclose(cU_dec, cU_lab, atol=1e-6), '미검사축 carry 불일치'
    assert torch.allclose(cS_dec, cS_lab, atol=1e-6), '선별축 carry 불일치'


@pytest.mark.parametrize('kw', [dict(NMAX=2, SMAX=2), dict(NMAX=3, SMAX=2), dict(W=4)],
                         ids=['N2S2', 'N3S2', 'W4'])
def test_teacher_forcing_reproduces_decode(kw):
    """교사강요 로짓의 argmax(마스크 적용)가 decode 의 선택과 같다.

    carry 정의가 같으므로, 자기 출력을 라벨로 넣으면 학습 경로가 보는 로짓은
    추론 경로가 본 로짓과 같아야 한다. 레거시는 decode 가 초기 예산만 쓰기 때문에
    이 성질이 성립하지 않는다 — 그래서 carry=True 에서만 건다.
    """
    I = _inst(**kw)
    tens, sel, b = _batch(I)
    torch.manual_seed(0); net = PolicyNet(carry=True).to(DEV)
    au, as_ = net.decode(b, I.CAP)
    lu = torch.as_tensor(au, device=DEV); ls = torch.as_tensor(as_, device=DEV)
    with torch.no_grad():
        z = net.encode(b['U'], b['mu'], b['S'], b['ms'], b['ctx'])
        cU, cS, mU, mS = net.carry_from_labels(b, lu, ls, I.CAP)
        B, Um = b['mu'].shape; Sm = b['ms'].shape[1]
        zz = z.unsqueeze(1).expand(B, Um, z.shape[-1])
        aU = net.headU(torch.cat([zz, b['U'], cU], -1)).masked_fill(~mU, NEG).argmax(-1)
        zs = z.unsqueeze(1).expand(B, Sm, z.shape[-1])
        aS = net.headS(torch.cat([zs, b['S'], cS], -1)).masked_fill(~mS, NEG).argmax(-1)
    aU = torch.where(b['mu'] > 0, aU, torch.zeros_like(aU))
    aS = torch.where(b['ms'] > 0, aS, torch.zeros_like(aS))
    assert torch.equal(aU, lu), '미검사축 교사강요 argmax != decode'
    assert torch.equal(aS, ls), '선별축 교사강요 argmax != decode'


def test_v5b_checkpoint_loads():
    """v5b 가 남긴 net_best.pt 가 carry=False 로 그대로 로드된다."""
    root = Path('/home/data/batdiag/results/sweep_selectivity')
    pts = sorted(root.glob('*/net_best.pt'))
    if not pts:
        pytest.skip('v5b 체크포인트가 없다')
    sd = torch.load(pts[0], map_location='cpu')
    PolicyNet(carry=False).load_state_dict(sd)          # 엄격 모드로 통과해야 한다
    with pytest.raises(RuntimeError):                    # 새 디코더와는 차원이 다르다
        PolicyNet(carry=True).load_state_dict(sd)
