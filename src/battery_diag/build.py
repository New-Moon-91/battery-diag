# -*- coding: utf-8 -*-
"""전이구조 → CSR 배열. 24코어 병렬 + 디스크 캐시.

산출물 (모두 numpy):
  indptr  : (n_sa+1,)  CSR 행 포인터        — 행 = (상태,행동) 쌍
  indices : (nnz,)     다음 상태 인덱스
  probs   : (nnz,)     전이확률
  rsa     : (n_sa,)    기대 즉시보상 E[r | s,a]
  seg     : (n_sa,)    각 행이 속한 상태 인덱스
  aptr    : (nS+1,)    상태별 행동 시작 위치 (argmax 복원용)
"""
from __future__ import annotations
import hashlib, json, os
import numpy as np
from pathlib import Path
from concurrent.futures import ProcessPoolExecutor

_INST = None


def _init(inst):
    global _INST
    _INST = inst


def _chunk(args):
    lo, hi = args
    I = _INST
    idx_l, prb_l, cnt_l, rsa_l = [], [], [], []
    for si in range(lo, hi):
        st = I.ST[si]
        for a in I.actions(st):
            o = I.step_dist(st, a)
            p = np.fromiter((x[0] for x in o), float, len(o))
            i = np.fromiter((x[1] for x in o), np.int32, len(o))
            r = np.fromiter((x[2] for x in o), float, len(o))
            idx_l.append(i); prb_l.append(p); cnt_l.append(len(o)); rsa_l.append(float(p @ r))
    return (np.concatenate(idx_l), np.concatenate(prb_l),
            np.asarray(cnt_l, np.int64), np.asarray(rsa_l, float))


def build(inst, workers: int | None = None, cache_dir: str | Path | None = None, tag: str = ''):
    key = None
    if cache_dir is not None:
        h = hashlib.md5(json.dumps({
            'ty': {k: list(v) for k, v in inst.types.items()},
            'cfg': inst.cfg.__dict__, 'tag': tag}, sort_keys=True, default=str).encode()).hexdigest()[:16]
        key = Path(cache_dir) / f'trans_{h}.npz'
        if key.exists():
            z = np.load(key)
            return {k: z[k] for k in z.files}
    workers = workers or min(os.cpu_count() or 4, 24)
    nS = len(inst.ST)
    bounds = np.linspace(0, nS, workers + 1).astype(int)
    jobs = [(int(bounds[i]), int(bounds[i+1])) for i in range(workers) if bounds[i+1] > bounds[i]]
    with ProcessPoolExecutor(max_workers=workers, initializer=_init, initargs=(inst,)) as ex:
        parts = list(ex.map(_chunk, jobs))
    indices = np.concatenate([p[0] for p in parts])
    probs   = np.concatenate([p[1] for p in parts])
    counts  = np.concatenate([p[2] for p in parts])
    rsa     = np.concatenate([p[3] for p in parts])
    indptr  = np.zeros(len(counts) + 1, np.int64); np.cumsum(counts, out=indptr[1:])
    na = np.asarray([len(inst.actions(st)) for st in inst.ST], np.int64)
    aptr = np.zeros(nS + 1, np.int64); np.cumsum(na, out=aptr[1:])
    seg = np.repeat(np.arange(nS, dtype=np.int64), na)
    out = dict(indptr=indptr, indices=indices, probs=probs, rsa=rsa, seg=seg, aptr=aptr)
    if key is not None:
        key.parent.mkdir(parents=True, exist_ok=True)
        np.savez_compressed(key, **out)
    return out
