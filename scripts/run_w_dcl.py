#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""W-정식화에서 DCL 갭 + 표현가능 상한 (w2 [4]).

    python scripts/run_w_dcl.py <W> <decoder:legacy|carry> <seed...>

정확해는 매번 다시 푼다 (캐시 적중이면 빌드는 공짜). 결과는
results/w_model/dcl_W{W}_{decoder}_seed{s}.json.
"""
import sys, json, time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]/'src'))
# w5 확정 인스턴스. 근거는 battery_diag.data.SEL_W5 주석 참조.
from battery_diag.data import SEL_W5 as SEL
import os
CACHE = os.environ.get('BATDIAG_CACHE', '/home/user/batdiag-cache')
OUT = Path(os.environ.get('BATDIAG_OUT',
           Path(__file__).resolve().parents[1]/'results'/'w5'))


def main():
    import numpy as np, torch
    from battery_diag.data import load_cached, fleet_w5
    from battery_diag.instance import Instance, PriceW5, Config
    from battery_diag.build import build
    from battery_diag.streambuild import build_stream
    from battery_diag.exact import ExactSolver
    from battery_diag.bigexact import StreamSolver
    from battery_diag.dcl import run_dcl
    from battery_diag.ckpt import Checkpoint
    from battery_diag import policies as pol

    W = int(sys.argv[1]); dec = sys.argv[2]
    seeds = [int(x) for x in sys.argv[3:]] or [0]
    root = Path(__file__).resolve().parents[1]
    _, FS = load_cached(str(root/'data'))
    FLEET = fleet_w5(root/'data', sel=SEL)
    tot = sum(FLEET[t][0] for t in SEL)
    types = {t: (FLEET[t][1], FLEET[t][2], FLEET[t][3], FLEET[t][4], FLEET[t][0]/tot)
             for t in SEL}
    price = PriceW5.from_json(root/'data'/'params_w5.json')
    dev = 'cuda' if torch.cuda.is_available() else 'cpu'
    cfg = Config(Mcyc=1, Cp=500000, Cf=20000, phi=1.0, NARR=4, W=W,
                 F_E=FS['F_E'], F_U=FS['F_U'])
    I = Instance(types, price, cfg); nS = len(I.ST)
    OUT.mkdir(parents=True, exist_ok=True)
    big = nS >= 40000
    if big:
        A = build_stream(I, cache_dir=CACHE, tag=','.join(SEL))
        S = StreamSolver(A, device=dev); gstar, h, _ = S.solve(acts0=pol.index_myopic(I))
    else:
        A = build(I, cache_dir=CACHE, tag=','.join(SEL))
        S = ExactSolver(A, device=dev); gstar, h, _ = S.solve()
    print(f'W={W} nS={nS:,}  g*={gstar:,.2f}', flush=True)
    acts0 = pol.index_myopic(I)
    for seed in seeds:
        f = OUT/f'dcl_W{W}_{dec}_seed{seed}.json'
        if f.exists():
            print(f'[skip] {f.name}', flush=True); continue
        import tempfile
        t0 = time.time()
        with tempfile.TemporaryDirectory() as td:
            net, gbest, hist = run_dcl(I, S, acts0, Checkpoint(Path(td), 'dcl'),
                                       rounds=6, epochs=40, seed=seed, device=dev,
                                       log=print, gstar=gstar, carry=(dec == 'carry'))
        gap = 100*(gstar-gbest)/gstar
        f.write_text(json.dumps(dict(W=W, decoder=dec, seed=seed, gstar=float(gstar),
                                     best=float(gbest), gap=float(gap), hist=hist,
                                     sec=time.time()-t0),
                                ensure_ascii=False, indent=1, default=float))
        print(f'== W={W} {dec} seed{seed}: gap {gap:.3f}%  ({time.time()-t0:.0f}s)', flush=True)


if __name__ == '__main__':
    main()
