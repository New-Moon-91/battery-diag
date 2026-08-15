# -*- coding: utf-8 -*-
"""실험 그리드 → 2 GPU 워커풀.

이 문제의 신경망은 작아서(은닉 128) 분산학습(DDP)은 오버헤드만 늘린다.
GPU 2장은 **독립 실험을 병렬로 돌리는 데** 쓰는 것이 맞다.
GPU당 워커 2개(총 4)가 기본값 — 정확해 SpMV 와 학습이 번갈아 돌아 점유율이 채워진다.

각 워커는 자기 job_id 로 체크포인트를 남기고, 이미 DONE 인 잡은 건너뛴다.
따라서 중단 후 같은 명령을 다시 실행하면 남은 잡부터 이어서 돈다.
"""
from __future__ import annotations
import itertools, json, os, subprocess, sys, time
from pathlib import Path
from concurrent.futures import ProcessPoolExecutor


def expand_grid(base: dict, grid: dict):
    keys = list(grid)
    for vals in itertools.product(*[grid[k] for k in keys]):
        cfg = dict(base); cfg.update(dict(zip(keys, vals)))
        yield cfg


def job_id(cfg: dict) -> str:
    return '_'.join(f'{k}{v}' for k, v in sorted(cfg.items())
                    if k in ('NARR','Mcyc','Hslot','Cp','Cf','phi','NMAX','SMAX','seed','lam'))


def _worker(args):
    cfg, script, out_root, gpu = args
    env = dict(os.environ)
    env['CUDA_VISIBLE_DEVICES'] = str(gpu)
    env['OMP_NUM_THREADS'] = '4'
    jid = job_id(cfg)
    cmd = [sys.executable, script, '--config-json', json.dumps(cfg),
           '--out', str(Path(out_root)/jid), '--job-id', jid]
    t0 = time.time()
    p = subprocess.run(cmd, env=env, capture_output=True, text=True)
    return dict(job=jid, gpu=gpu, rc=p.returncode, sec=time.time()-t0,
                tail=(p.stdout or '')[-2000:], err=(p.stderr or '')[-2000:])


def run_grid(cfgs, script, out_root, gpus=(0, 1), per_gpu=2, log=print):
    out_root = Path(out_root); out_root.mkdir(parents=True, exist_ok=True)
    todo = [c for c in cfgs if not (out_root/job_id(c)/'DONE.json').exists()]
    log(f'전체 {len(cfgs)}개 중 미완료 {len(todo)}개 실행 '
        f'(GPU {list(gpus)} x {per_gpu} = 워커 {len(gpus)*per_gpu}개)')
    args = [(c, script, str(out_root), gpus[i % len(gpus)]) for i, c in enumerate(todo)]
    res = []
    with ProcessPoolExecutor(max_workers=len(gpus)*per_gpu) as ex:
        for r in ex.map(_worker, args):
            res.append(r)
            log(f"  [{'OK ' if r['rc']==0 else 'FAIL'}] {r['job']} gpu{r['gpu']} {r['sec']:.0f}s")
            if r['rc'] != 0: log(r['err'][-800:])
    (out_root/'runlog.json').write_text(json.dumps(res, ensure_ascii=False, indent=1))
    return res
