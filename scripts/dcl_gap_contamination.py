#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""v5b/v5c 의 DCL 갭이 무료 소실에 오염됐는가 — 저장된 신경망 정책으로 직접 검정.

benchmark 정책과 달리 DCL 신경망 정책은 최적정책에 가깝다. 최적정책과 소실가치가
비슷하면 보정은 분모 효과만 남아 갭이 거의 안 움직인다. 그 가정을 실측한다.

    python scripts/dcl_gap_contamination.py <NMAX> <SMAX> <net_best.pt> [carry]
"""
import sys, json
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]/'src'))
sys.path.insert(0, str(Path(__file__).resolve().parent))
from loss_correct import loss_per_state

# w1~w3 스크립트다. 인스턴스가 레이·코나·SM3 로 고정돼 있고 커밋된 결과도
# 그 조합·구 가격으로 냈으므로, 가격도 params_v5.json 으로 고정한다.
# w4 재캘리브레이션된 params.json 을 여기에 물리면 구 결과와 섞인다.
SEL = ['레이', '코나', 'SM3']
CACHE = '/home/user/batdiag-cache'


def main():
    import numpy as np, torch
    from battery_diag.data import load_cached
    from battery_diag.instance import Instance, PriceParams, Config
    from battery_diag.build import build
    from battery_diag.streambuild import build_stream
    from battery_diag.exact import ExactSolver
    from battery_diag.bigexact import StreamSolver
    from battery_diag.encode import state_tensors
    from battery_diag.net import PolicyNet
    from battery_diag.dcl import greedy_policy
    from battery_diag import policies as pol

    NMAX, SMAX = int(sys.argv[1]), int(sys.argv[2])
    pt = Path(sys.argv[3])
    carry = len(sys.argv) > 4 and sys.argv[4] == 'carry'
    root = Path(__file__).resolve().parents[1]
    FLEET, FS = load_cached(str(root/'data'))
    tot = sum(FLEET[t][0] for t in SEL)
    types = {t: (FLEET[t][1], FLEET[t][2], FLEET[t][3], FLEET[t][4], FLEET[t][0]/tot)
             for t in SEL}
    price = PriceParams.from_json(root/'data'/'params_v5.json')
    dev = 'cuda' if torch.cuda.is_available() else 'cpu'
    cfg = Config(Mcyc=1, Cp=500000, Cf=20000, phi=1.0, NARR=4,
                 NMAX=NMAX, SMAX=SMAX, F_E=FS['F_E'], F_U=FS['F_U'])
    I = Instance(types, price, cfg); nS = len(I.ST)
    stream = nS >= 14000 or (NMAX, SMAX) == (2, 4)
    if stream:
        A = build_stream(I, cache_dir=CACHE, tag=','.join(SEL))
        S = StreamSolver(A, device=dev); gstar, h, _ = S.solve(acts0=pol.index_myopic(I))
    else:
        A = build(I, cache_dir=CACHE, tag=','.join(SEL))
        S = ExactSolver(A, device=dev); gstar, h, _ = S.solve()
    acts_opt = S.greedy(h)

    net = PolicyNet(carry=carry).to(dev)
    net.load_state_dict(torch.load(pt, map_location=dev))
    tens = state_tensors(I, device=dev)
    acts_net = greedy_policy(net, I, tens)
    g_net = float(S.evaluate(acts_net)[0])

    res = {}
    for name, a, g in (('OPT', acts_opt, float(gstar)), ('DCL', acts_net, g_net)):
        d = np.asarray(S.stationary(a)); d = d/d.sum()
        n_l, v_l = loss_per_state(I, a, cfg)
        res[name] = dict(g=g, ev=float(d @ v_l), units=float(d @ n_l))
    gap0 = 100*(res['OPT']['g']-res['DCL']['g'])/res['OPT']['g']
    gc_o = res['OPT']['g']+res['OPT']['ev']; gc_d = res['DCL']['g']+res['DCL']['ev']
    gap1 = 100*(gc_o-gc_d)/gc_o
    print(f'NMAX={NMAX} SMAX={SMAX} decoder={"carry" if carry else "legacy"}')
    print(f"  OPT g={res['OPT']['g']:,.2f}  소실 {res['OPT']['units']:.4f}대 "
          f"{res['OPT']['ev']:,.0f}원")
    print(f"  DCL g={res['DCL']['g']:,.2f}  소실 {res['DCL']['units']:.4f}대 "
          f"{res['DCL']['ev']:,.0f}원")
    print(f'  갭 원본 {gap0:.3f}%  →  보정 {gap1:.3f}%   차이 {gap1-gap0:+.3f}%p')
    out = dict(NMAX=NMAX, SMAX=SMAX, carry=carry, gap=gap0, gap_corr=gap1, **res)
    p = root/'results'/'w_model'/f'dcl_contamination_N{NMAX}S{SMAX}_{"carry" if carry else "legacy"}.json'
    p.write_text(json.dumps(out, ensure_ascii=False, indent=1, default=float))
    print('→', p)


if __name__ == '__main__':
    main()
