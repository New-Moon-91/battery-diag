#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""단일 설정 실행 — 정확해 + 벤치마크 + DCL. runner 가 이 스크립트를 호출한다.
직접 실행도 가능:
  python scripts/run_one.py --config-json '{"NARR":3,"Mcyc":1,"seed":0}' --out results/tmp --job-id tmp
"""
import argparse, json, sys, time
from pathlib import Path
import numpy as np, torch
sys.path.insert(0, str(Path(__file__).resolve().parents[1]/'src'))
from battery_diag.data import load_cached
from battery_diag.instance import Instance, PriceParams, Config
from battery_diag.build import build
from battery_diag.exact import ExactSolver, policy_iteration
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

t0 = time.time(); arrays = build(I, cache_dir=a.cache, tag=a.types)
log(f'build {time.time()-t0:.1f}s  n_sa={len(arrays["rsa"]):,} nnz={len(arrays["probs"]):,}')
dev = 'cuda' if torch.cuda.is_available() else 'cpu'
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
                               seed=int(C.get('seed', 0)), device=dev, log=log, gstar=gstar)
    res['dcl'] = dict(best=float(gbest), gap=float(100*(gstar-gbest)/gstar), hist=hist)
    torch.save(net.state_dict(), out/'net_best.pt')
(out/'result.json').write_text(json.dumps(res, ensure_ascii=False, indent=1, default=float))
Checkpoint(out, '.').mark_done(res)
log('DONE', a.job_id)
