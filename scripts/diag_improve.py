# -*- coding: utf-8 -*-
"""improve 불일치 진단 — 원인이 부동소수 오차인지, 로직 오류인지 가른다.

    python scripts/diag_improve.py
"""
import sys
from pathlib import Path
import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / 'src'))
from battery_diag.data import load_cached
from battery_diag.instance import Instance, PriceParams, Config
from battery_diag.build import build
from battery_diag.exact import ExactSolver
from battery_diag.bigexact import StreamSolver

import torch
DEV = 'cuda' if torch.cuda.is_available() else 'cpu'
SEL = ['레이', '코나', 'SM3']


def main():
    FLEET, FS = load_cached('data')
    tot = sum(FLEET[t][0] for t in SEL)
    types = {t: (FLEET[t][1], FLEET[t][2], FLEET[t][3], FLEET[t][4], FLEET[t][0]/tot) for t in SEL}
    price = PriceParams.from_json(Path('data') / 'params.json')
    cfg = Config(Mcyc=1, Cp=500000, Cf=20000, phi=1.0, NARR=4,
                 NMAX=2, SMAX=2, F_E=FS['F_E'], F_U=FS['F_U'])
    I = Instance(types, price, cfg)
    A = build(I, workers=8)

    E = ExactSolver(A, device=DEV)
    S = StreamSolver(A, device=DEV, slab_nnz=max(int(A['indptr'][-1]) // 20, 1000))
    print(f'슬랩 {len(S.slabs)}개, nS={E.nS}, n_sa={E.n_sa}')

    rng = np.random.default_rng(0)
    h = rng.normal(0, 1e5, E.nS)

    aE = E.improve(h)
    aS = S.improve(h)
    bad = np.nonzero(aE != aS)[0]
    print(f'\n불일치 상태 {len(bad)}/{E.nS} ({100*len(bad)/E.nS:.2f}%)')

    # 두 경로의 q 를 같은 방식으로 꺼내 비교
    qE = E._q(torch.as_tensor(h, device=E.dev, dtype=E.dt)).cpu().numpy()
    ip, aptr, rsa = A['indptr'], A['aptr'], A['rsa']
    ixa, pra = A['indices'], A['probs']
    qS = np.empty_like(qE)
    for r in range(len(rsa)):
        lo, hi = ip[r], ip[r+1]
        qS[r] = rsa[r] + float(pra[lo:hi] @ h[ixa[lo:hi]])

    print(f'q 최대절대차 = {np.max(np.abs(qE-qS)):.3e}   |q| 중앙값 = {np.median(np.abs(qE)):.3e}')

    if len(bad):
        print('\n불일치 상태 표본 (상위 5개):')
        for s in bad[:5]:
            lo, hi = aptr[s], aptr[s+1]
            seg = qE[lo:hi]
            qa, qb = qE[lo+aE[s]], qE[lo+aS[s]]
            print(f'  s={s:5d}  |A(s)|={hi-lo:4d}  ExactSolver 선택 {aE[s]:4d} / Stream 선택 {aS[s]:4d}'
                  f'  q차={abs(qa-qb):.3e}  상대={abs(qa-qb)/max(1.0,abs(qa)):.3e}'
                  f'  최대와의차={seg.max()-min(qa,qb):.3e}')

    # 결정적 질문: 두 정책의 성능이 같은가
    gE, _ = E.evaluate(aE)
    gS, _ = E.evaluate(aS)          # 같은 솔버로 평가해야 공정
    print(f'\n정책 성능 (동일 솔버로 평가)')
    print(f'  g(ExactSolver 정책) = {gE:,.6f}')
    print(f'  g(Stream 정책)      = {gS:,.6f}')
    print(f'  상대차 = {abs(gE-gS)/max(1.0,abs(gE)):.3e}')

    print('\n판정: q차가 1e-12 를 넘고 정책 성능 상대차가 1e-9 이하면'
          ' → 부동소수 동점 문제(무해). 그 외면 로직 오류.')


if __name__ == '__main__':
    main()
