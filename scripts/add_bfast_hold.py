#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""w3 [2] — 기존 W{W}.json 에 B_fast_hold 와 임계 정의역 감사 결과를 덧붙인다.

    python scripts/add_bfast_hold.py <W> [<W> ...]

정확해는 캐시 적중이라 빌드가 공짜다. 새로 계산하는 것은 정책평가 몇 번뿐이다.
"""
import sys, json, time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]/'src'))
# w1~w3 스크립트다. 인스턴스가 레이·코나·SM3 로 고정돼 있고 커밋된 결과도
# 그 조합·구 가격으로 냈으므로, 가격도 params_v5.json 으로 고정한다.
# w4 재캘리브레이션된 params.json 을 여기에 물리면 구 결과와 섞인다.
SEL = ['레이', '코나', 'SM3']
CACHE = '/home/user/batdiag-cache'
OUT = Path(__file__).resolve().parents[1]/'results'/'w_model'


def main():
    import numpy as np, torch
    from battery_diag.data import load_cached
    from battery_diag.instance import Instance, PriceParams, Config
    from battery_diag.build import build
    from battery_diag.streambuild import build_stream
    from battery_diag.exact import ExactSolver
    from battery_diag.bigexact import StreamSolver
    from battery_diag import policies as pol

    root = Path(__file__).resolve().parents[1]
    FLEET, FS = load_cached(str(root/'data'))
    tot = sum(FLEET[t][0] for t in SEL)
    types = {t: (FLEET[t][1], FLEET[t][2], FLEET[t][3], FLEET[t][4], FLEET[t][0]/tot)
             for t in SEL}
    price = PriceParams.from_json(root/'data'/'params_v5.json')
    dev = 'cuda' if torch.cuda.is_available() else 'cpu'

    for W in [int(x) for x in sys.argv[1:]]:
        f = OUT/f'W{W}.json'
        res = json.loads(f.read_text())
        if 'B_fast_hold' in res:
            print(f'[skip] W={W}', flush=True); continue
        t0 = time.time()
        cfg = Config(Mcyc=1, Cp=500000, Cf=20000, phi=1.0, NARR=4, W=W,
                     F_E=FS['F_E'], F_U=FS['F_U'])
        I = Instance(types, price, cfg)
        stream = res['stream']
        if stream:
            A = build_stream(I, cache_dir=CACHE, tag=','.join(SEL))
            S = StreamSolver(A, device=dev)
            gstar, h, _ = S.solve(acts0=pol.index_myopic(I))
        else:
            A = build(I, cache_dir=CACHE, tag=','.join(SEL))
            S = ExactSolver(A, device=dev); gstar, h, _ = S.solve()
        assert abs(gstar - res['gstar']) <= 1e-6*abs(gstar), 'g* 재현 실패'

        # (a) 임계 정의역 감사 — SB 개 구간이면 후보는 0..SB (SB+1 개) 가 전부다.
        #     thr=SB 는 "아무도 자격 없음" = 정밀검사를 아예 안 쓰는 퇴화 케이스.
        bfa = {}
        for thr in range(cfg.SB + 1):
            bfa[thr] = float(S.evaluate(pol.b_fast(I, thr))[0])
        # (b) B_fast_hold
        bfh = {thr: float(S.evaluate(pol.b_fast_hold(I, thr))[0])
               for thr in range(cfg.SB + 1)}
        thr_f = max(bfa, key=bfa.get); thr_h = max(bfh, key=bfh.get)
        gap = lambda v: 100*(gstar-v)/gstar
        res['B_fast_all_full'] = {str(k): v for k, v in bfa.items()}
        res['B_fast_hold_all'] = {str(k): v for k, v in bfh.items()}
        res['B_fast_hold'] = bfh[thr_h]
        res['B_fast_hold_thr'] = int(thr_h)
        res['gaps']['B_fast_hold'] = gap(bfh[thr_h])
        res['bench']['B_fast_hold'] = bfh[thr_h]
        res['decomp'] = dict(
            gap_B_fast=gap(bfa[thr_f]),
            gap_B_fast_hold=gap(bfh[thr_h]),
            cost_full_inspection=gap(bfh[thr_h]),
            cost_forced_disposal=gap(bfa[thr_f]) - gap(bfh[thr_h]))
        f.write_text(json.dumps(res, ensure_ascii=False, indent=1, default=float))
        print(f"W={W}: B_fast 갭 {res['decomp']['gap_B_fast']:.3f}% = "
              f"전량검사강제 {res['decomp']['cost_full_inspection']:.3f}%p + "
              f"즉시처분강제 {res['decomp']['cost_forced_disposal']:.3f}%p   "
              f"(thr: B_fast {thr_f}, hold {thr_h})  {time.time()-t0:.0f}s", flush=True)
        print(f"   B_fast thr 전역 {bfa}", flush=True)
        del S, A
        if dev == 'cuda': torch.cuda.empty_cache()


if __name__ == '__main__':
    main()
