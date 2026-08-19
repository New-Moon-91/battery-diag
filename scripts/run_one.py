#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""단일 설정 실행 — 정확해 + 벤치마크 + DCL. runner 가 이 스크립트를 호출한다.
직접 실행도 가능:
  python scripts/run_one.py --config-json '{"NARR":3,"Mcyc":1,"seed":0}' --out results/tmp --job-id tmp
"""
import argparse, json, sys, time
from pathlib import Path
import numpy as np
sys.path.insert(0, str(Path(__file__).resolve().parents[1]/'src'))
from battery_diag.data import load_cached
from battery_diag.instance import Instance, PriceParams, Config
from battery_diag.build import build
from battery_diag.streambuild import build_stream

# torch 계열 임포트는 main() 안에서 한다 — build_stream 의 spawn 워커는 이 파일을
# __mp_main__ 으로 재임포트하므로, 모듈 최상단에서 torch 를 끌면 워커 수만큼
# torch 상주분이 복제된다. 62GB 시스템에서 그만큼이 그대로 빌드 여유를 깎는다.


def main():
    import torch
    from battery_diag.bigexact import StreamSolver
    from battery_diag.exact import ExactSolver
    from battery_diag import policies as pol
    from battery_diag.dcl import run_dcl
    from battery_diag.ckpt import Checkpoint

    ap = argparse.ArgumentParser()
    ap.add_argument('--config-json', required=True)
    ap.add_argument('--out', required=True)
    ap.add_argument('--job-id', required=True)
    ap.add_argument('--data', default='data')
    ap.add_argument('--cache', default='/home/data/batdiag/cache')
    ap.add_argument('--types', default='레이,코나,SM3')
    ap.add_argument('--rounds', type=int, default=6)
    ap.add_argument('--epochs', type=int, default=40)
    ap.add_argument('--no-dcl', action='store_true')
    ap.add_argument('--decoder', choices=['legacy', 'carry'], default='carry',
                    help='carry=자기회귀 디코더(기본, v5c). legacy=v5b 디코더 — 옛 결과 재현용')
    a = ap.parse_args()
    C = json.loads(a.config_json)
    out = Path(a.out); out.mkdir(parents=True, exist_ok=True)
    def log(*x): print(*x, flush=True)

    FLEET, FS = load_cached(a.data)
    sel = a.types.split(','); tot = sum(FLEET[t][0] for t in sel)
    types = {t: (FLEET[t][1], FLEET[t][2], FLEET[t][3], FLEET[t][4], FLEET[t][0]/tot) for t in sel}
    price = PriceParams.from_json(Path(a.data)/'params.json')
    cfg = Config(**{k: v for k, v in C.items() if k in Config.__dataclass_fields__},
                 F_E=FS['F_E'], F_U=FS['F_U'])
    I = Instance(types, price, cfg)
    log('instance', json.dumps(I.summary(), ensure_ascii=False))

    # nnz 를 표본으로 추정해 in-RAM 경로가 감당 가능한지 먼저 판단한다.
    import os as _os, random as _random
    rng = _random.Random(0)
    smp = rng.sample(range(len(I.ST)), min(150, len(I.ST)))
    _rows = _nnz = 0
    for _si in smp:
        _st = I.ST[_si]
        for _act in I.actions(_st):
            _rows += 1; _nnz += len(I.step_dist(_st, _act))
    est_gb = (_nnz / max(_rows, 1) * I.n_actions()) * 12 / 1e9
    thr_gb = float(_os.environ.get('BATDIAG_STREAM_GB', 8))
    dev = 'cuda' if torch.cuda.is_available() else 'cpu'
    if est_gb > thr_gb:
        log(f'추정 CSR {est_gb:.1f}GB > {thr_gb:.0f}GB → 스트리밍 경로 (build_stream + PI)')
        t0 = time.time(); arrays = build_stream(I, cache_dir=a.cache, tag=a.types, log=log)
        log(f'build_stream {time.time()-t0:.1f}s  n_sa={len(arrays["rsa"]):,} nnz={arrays["meta"]["nnz"]:,}')
        S = StreamSolver(arrays, device=dev)
        acts0 = pol.index_myopic(I)          # 근시안 지표에서 출발 → PI 조기 수렴
        t0 = time.time(); gstar, h, it = S.solve(acts0=acts0, log=log)
        log(f'g* = {gstar:,.2f}  (PI {it}회, {time.time()-t0:.1f}s)')
    else:
        t0 = time.time(); arrays = build(I, cache_dir=a.cache, tag=a.types, log=log)
        log(f'build {time.time()-t0:.1f}s  n_sa={len(arrays["rsa"]):,} nnz={len(arrays["probs"]):,}')
        S = ExactSolver(arrays, device=dev)
        t0 = time.time(); gstar, h, it = S.solve(); log(f'g* = {gstar:,.2f}  (RVI {it}회, {time.time()-t0:.1f}s)')
    bench = pol.all_benchmarks(I, S)
    for k, v in bench.items():
        if k != 'B_fast_thr': log(f'  {k:8s} {v:12,.0f}  갭 {100*(gstar-v)/gstar:6.3f}%')
    res = dict(cfg=C, summary=I.summary(), gstar=gstar,
               bench={k: float(v) for k, v in bench.items()},
               gaps={k: float(100*(gstar-v)/gstar) for k, v in bench.items() if k != 'B_fast_thr'})
    ck = Checkpoint(out, 'dcl')
    if not a.no_dcl:
        acts0 = pol.index_myopic(I)
        net, gbest, hist = run_dcl(I, S, acts0, ck, rounds=a.rounds, epochs=a.epochs,
                                   seed=int(C.get('seed', 0)), device=dev, log=log, gstar=gstar,
                                   carry=(a.decoder == 'carry'))
        res['decoder'] = a.decoder
        res['dcl'] = dict(best=float(gbest), gap=float(100*(gstar-gbest)/gstar), hist=hist)
        torch.save(net.state_dict(), out/'net_best.pt')
    (out/'result.json').write_text(json.dumps(res, ensure_ascii=False, indent=1, default=float))
    Checkpoint(out, '.').mark_done(res)
    log('DONE', a.job_id)


if __name__ == '__main__':
    main()
