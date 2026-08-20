# -*- coding: utf-8 -*-
"""전이구조 → CSR 배열. 병렬 + 디스크 캐시.

산출물 (모두 numpy):
  indptr  : (n_sa+1,)  CSR 행 포인터        — 행 = (상태,행동) 쌍
  indices : (nnz,)     다음 상태 인덱스
  probs   : (nnz,)     전이확률
  rsa     : (n_sa,)    기대 즉시보상 E[r | s,a]
  seg     : (n_sa,)    각 행이 속한 상태 인덱스
  aptr    : (nS+1,)    상태별 행동 시작 위치 (argmax 복원용)

병렬화 주의 — 2026-08 수정:
  이전 판은 ProcessPoolExecutor 를 기본 start method(fork) 로 사용했다.
  부모가 torch/numpy 를 임포트해 이미 멀티스레드인 상태에서 fork 하면
  자식이 부모의 락 상태만 물려받고 락을 쥔 스레드는 복제되지 않아
  풀리지 않는 락이 남는다(교착). NMAX=3,SMAX=3 에서 48시간 무진행으로 발현.
  → start method 를 'spawn' 으로 고정한다.
"""
from __future__ import annotations
import hashlib, json, os, sys, time
import multiprocessing as mp
import numpy as np
from pathlib import Path
from concurrent.futures import ProcessPoolExecutor, as_completed

_INST = None



# 캐시 해시에서 **기본값일 때만** 제외하는 필드.
# 해시가 cfg.__dict__ 전체를 쓰기 때문에 Config 에 필드를 하나 추가하면 기존 캐시가
# 통째로 무효가 된다 (credit_loss 를 넣자 W=8 의 72GB 캐시까지 날아갔다).
# 나중에 추가된 필드는 기본값이면 빼서 "그 필드가 없던 시절" 의 사전을 복원한다.
# 기본값이 아니면 전이배열이 실제로 달라지므로 반드시 남긴다.
# **새 필드를 Config 에 넣을 때마다 여기에도 추가할 것.**
_LATE_FIELDS = {'W': None, 'credit_loss': False}


def _cfg_key(cfg):
    d = dict(cfg.__dict__)
    for k, dflt in _LATE_FIELDS.items():
        if k in d and d[k] == dflt:
            d.pop(k)
    return d

def _init(inst):
    # spawn 워커는 부모 환경을 물려받지 않으므로 여기서 스레드 수를 고정한다.
    for v in ('OMP_NUM_THREADS', 'MKL_NUM_THREADS', 'OPENBLAS_NUM_THREADS'):
        os.environ.setdefault(v, '1')
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


def _strip(inst):
    """워커로 보낼 인스턴스의 행동 캐시를 비운다.

    spawn 은 워커마다 인스턴스를 직렬화해 보내므로, 부모가 summary() 등으로
    채워 둔 _acts 를 그대로 보내면 전송량이 워커 수만큼 곱해진다.
    워커는 자기 담당 구간의 행동만 다시 만들면 된다.
    """
    saved = getattr(inst, '_acts', None)
    if saved:
        inst._acts = {}
    return saved


def _partition(na, nchunk):
    """행동 수 누적이 고르도록 상태 구간을 나눈다.

    상태를 균등 분할하면 뒤쪽 상태(대기열·버퍼가 찬 상태)일수록 행동이
    수십 배 많아 워커별 부하가 크게 어긋난다. 누적 행동 수 기준으로 자른다.
    """
    tot = int(na.sum())
    if tot == 0 or nchunk <= 1:
        return [(0, len(na))]
    step = tot / nchunk
    cuts, acc, lo, target = [], 0, 0, step
    for i, v in enumerate(na):
        acc += int(v)
        if acc >= target and i + 1 < len(na):
            cuts.append((lo, i + 1)); lo = i + 1; target += step
    cuts.append((lo, len(na)))
    return [c for c in cuts if c[1] > c[0]]


def build(inst, workers: int | None = None, cache_dir: str | Path | None = None,
          tag: str = '', log=None, chunks_per_worker: int = 8):
    say = log or (lambda *a: None)
    key = None
    if cache_dir is not None:
        h = hashlib.md5(json.dumps({
            'ty': {k: list(v) for k, v in inst.types.items()},
            'cfg': _cfg_key(inst.cfg), 'tag': tag}, sort_keys=True, default=str).encode()).hexdigest()[:16]
        key = Path(cache_dir) / f'trans_{h}.npz'
        if key.exists():
            z = np.load(key)
            say(f'build: 캐시 적중 {key.name}')
            return {k: z[k] for k in z.files}

    env_w = os.environ.get('BATDIAG_BUILD_WORKERS')
    workers = workers or (int(env_w) if env_w else None) or min(os.cpu_count() or 4, 24)
    nS = len(inst.ST)
    workers = max(1, min(workers, nS))

    # 상태별 행동 수 — 부하 분할과 aptr/seg 에 모두 쓰이므로 여기서 한 번만 만든다.
    t0 = time.time()
    na = np.asarray([len(inst.actions(st)) for st in inst.ST], np.int64)
    say(f'build: nS={nS:,} n_sa={int(na.sum()):,} 행동수집 {time.time()-t0:.0f}s')

    jobs = _partition(na, max(workers * chunks_per_worker, workers))
    say(f'build: 워커 {workers}개 / 청크 {len(jobs)}개 (spawn, 행동수 균등)')

    t0 = time.time()
    if workers == 1:                      # 디버그용 순차 경로
        _init(inst)
        parts = [_chunk(j) for j in jobs]
    else:
        saved = _strip(inst)
        try:
            ctx = mp.get_context('spawn')
            with ProcessPoolExecutor(max_workers=workers, mp_context=ctx,
                                     initializer=_init, initargs=(inst,)) as ex:
                fut = {ex.submit(_chunk, j): k for k, j in enumerate(jobs)}
                parts = [None] * len(jobs)
                done = 0
                for f in as_completed(fut):
                    parts[fut[f]] = f.result()
                    done += 1
                    if done % 5 == 0 or done == len(jobs):
                        say(f'  build {done}/{len(jobs)} 청크  {time.time()-t0:.0f}s')
        finally:
            if saved:
                inst._acts = saved

    indices = np.concatenate([p[0] for p in parts])
    probs   = np.concatenate([p[1] for p in parts])
    counts  = np.concatenate([p[2] for p in parts])
    rsa     = np.concatenate([p[3] for p in parts])
    indptr  = np.zeros(len(counts) + 1, np.int64); np.cumsum(counts, out=indptr[1:])
    aptr = np.zeros(nS + 1, np.int64); np.cumsum(na, out=aptr[1:])
    seg = np.repeat(np.arange(nS, dtype=np.int64), na)
    out = dict(indptr=indptr, indices=indices, probs=probs, rsa=rsa, seg=seg, aptr=aptr)
    if key is not None:
        key.parent.mkdir(parents=True, exist_ok=True)
        np.savez_compressed(key, **out)
        say(f'build: 캐시 저장 {key.name}')
    return out
