#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""대형 인스턴스 빌드·풀이 + 워커 메모리 실측.

  python scripts/build_big.py --nmax 3 --smax 2 --workers 12          # 측정만
  python scripts/build_big.py --nmax 3 --smax 3 --workers 12 --solve  # 빌드 후 PI

빌드 중 RSSWatch 가 ps 로 spawn 워커의 RSS 를 표본해 최대치를 찍는다.
메모리가 모자라면 --flush(=BATDIAG_FLUSH_NNZ) 를 낮추거나 --workers 를 줄인다.
GPU 가 모자라면 BATDIAG_SLAB_NNZ 로 슬랩을 줄인다.

--cache 는 **NVMe 쪽**을 지정할 것. 기본값 /home/data 는 회전 디스크라
수십 GB CSR 을 정책반복마다 다시 읽으면 디스크 대역에 묶인다.
"""
import argparse, json, sys, time
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parents[1] / 'src'))
import numpy as np
from battery_diag.data import load_cached
from battery_diag.instance import Instance, PriceParams, Config
from battery_diag.streambuild import build_stream

# 무거운 임포트(torch)는 main() 안에서 한다.
# spawn 워커는 이 스크립트를 __mp_main__ 으로 **재임포트**하므로, 여기서 torch 를
# 끌어오면 워커마다 torch 상주분이 그대로 붙는다.


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--nmax', type=int, required=True)
    ap.add_argument('--smax', type=int, required=True)
    ap.add_argument('--narr', type=int, default=4)
    ap.add_argument('--mcyc', type=int, default=1)
    ap.add_argument('--workers', type=int, default=12)
    ap.add_argument('--flush', type=int, default=None, help='워커 버퍼 상한 (nnz)')
    ap.add_argument('--chunks-per-worker', type=int, default=16)
    ap.add_argument('--cache', default='/home/data/batdiag/cache')
    ap.add_argument('--data', default='data')
    ap.add_argument('--types', default='레이,코나,SM3')
    ap.add_argument('--solve', action='store_true')
    a = ap.parse_args()

    def log(*x): print(*x, flush=True)

    FLEET, FS = load_cached(a.data)
    sel = a.types.split(','); tot = sum(FLEET[t][0] for t in sel)
    types = {t: (FLEET[t][1], FLEET[t][2], FLEET[t][3], FLEET[t][4], FLEET[t][0] / tot) for t in sel}
    price = PriceParams.from_json(Path(a.data) / 'params.json')
    cfg = Config(Mcyc=a.mcyc, Cp=500000, Cf=20000, phi=1.0, NARR=a.narr,
                 NMAX=a.nmax, SMAX=a.smax, F_E=FS['F_E'], F_U=FS['F_U'])
    I = Instance(types, price, cfg)
    log(f'NMAX={a.nmax} SMAX={a.smax} NARR={a.narr}  nS={len(I.ST):,}')

    t0 = time.time()
    A = build_stream(I, cache_dir=a.cache, tag=a.types, workers=a.workers, log=log,
                     chunks_per_worker=a.chunks_per_worker, flush_nnz=a.flush)
    log(f'빌드 완료 {time.time()-t0:.0f}s  n_sa={A["meta"]["n_sa"]:,} nnz={A["meta"]["nnz"]:,} '
        f'({A["meta"]["nnz"]*12/1e9:.1f}GB)')
    log('메타: ' + json.dumps({k: v for k, v in A['meta'].items() if k != 'bytes'},
                             ensure_ascii=False))

    if a.solve:
        import torch
        from battery_diag.bigexact import StreamSolver
        from battery_diag import policies as pol
        dev = 'cuda' if torch.cuda.is_available() else 'cpu'
        S = StreamSolver(A, device=dev)
        log(f'슬랩 {len(S.slabs)}개 / device={dev}')
        acts0 = pol.index_myopic(I)
        t0 = time.time()
        g, h, it = S.solve(acts0=acts0, log=log)
        log(f'g* = {g:,.2f}  (PI {it}회, {time.time()-t0:.0f}s)')
        np.save(Path(a.cache) / f'h_N{a.nmax}S{a.smax}.npy', h)


if __name__ == '__main__':
    main()
