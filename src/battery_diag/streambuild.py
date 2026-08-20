# -*- coding: utf-8 -*-
"""대형 인스턴스용 스트리밍 빌드 — 청크를 RAM 에 쌓지 않고 디스크에 이어 쓴다.

배경: build.py 는 모든 청크를 부모 메모리에 모은 뒤 np.concatenate 한다.
NMAX=3, SMAX=3 (추정 nnz 3.27G, CSR 39GB) 에서는 원본+사본이 겹치는 순간
78GB 가 필요해 62GB 시스템에서 OOM 으로 죽는다 (2026-08-18 확인).

메모리 설계 (2026-08-19 수정) — 세 번째 실패의 원인과 대책
  청크를 '행동 수' 로 균등 분할해도 메모리는 균등해지지 않는다. 상태별 분기
  수가 중앙 116 / 99.9% 32,768 / 최대 131,072 로 극단적으로 치우쳐 있어,
  행동 수가 같은 청크끼리도 nnz 가 수십 배 차이 난다. 워커가 청크 결과를
  통째로 들고 있으면 운 나쁜 청크 하나가 수 GB 가 되고, 워커 12개가 동시에
  그러면 시스템이 죽는다.
  → 워커는 결과를 자기 임시파일로 **흘려 쓴다**. 버퍼가 flush_nnz 를 넘으면
    즉시 비우므로 청크 크기와 무관하게 워커 상주분이 상수로 묶인다.
  → 부모는 완료된 청크를 순서대로 **스트림 복사**만 한다. 큰 배열이 부모
    주소공간에 올라오는 일이 없다 (예전에는 window 개만큼 들고 있었다).
  → 실측: RSSWatch 가 ps 로 spawn 워커의 RSS 를 표본해 최대치를 남긴다.

산출물: <cache_dir>/stream_<hash>/
  meta.json                 dtype·개수·검증용 합계 (+ 빌드 시 실측 RSS)
  indptr.i64  (n_sa+1)      CSR 행 포인터
  indices.i32 (nnz)         다음 상태
  probs.f64   (nnz)         전이확률
  rsa.f64     (n_sa)        기대 즉시보상
  aptr.i64    (nS+1)        상태별 행동 시작

load_stream() 은 indices/probs 를 np.memmap 으로, 나머지는 RAM 으로 돌려준다.
"""
from __future__ import annotations
import hashlib, json, os, shutil, subprocess, threading, time
import multiprocessing as mp
import numpy as np
from pathlib import Path
from collections import deque
from concurrent.futures import ProcessPoolExecutor

from .build import _init, _strip, _partition

FLUSH_NNZ = 8_000_000          # 워커 버퍼 상한 (nnz) — 약 100MB 상당



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

# 가격 파라미터도 해시에 넣어야 한다 — rsa(기대 즉시보상)가 캐시에 들어가므로
# params.json 을 바꾸고 캐시를 그대로 쓰면 **옛 보상으로 푼 답이 조용히 나온다.**
# (w4 재캘리브레이션에서 실제로 걸릴 뻔했다.)
# 다만 _LATE_FIELDS 와 같은 요령으로, 구 파라미터(v5)와 정확히 같을 때는 키에서 빼서
# "가격이 해시에 없던 시절" 의 사전을 복원한다. 그래야 379GB 짜리 기존 캐시가
# data/params_v5.json 재현용으로 그대로 살아 있다.
# 키 계산은 가격 클래스에 위임한다 (build.py 와 같은 이유 — w5 에서 필드 집합이
# 통째로 바뀌었으므로 빌더가 필드 이름을 알고 있으면 안 된다).


def _price_key(pp):
    return pp.key()


def _key(inst, tag):
    h = hashlib.md5(json.dumps({
        'ty': {k: list(v) for k, v in inst.types.items()},
        'cfg': _cfg_key(inst.cfg), 'tag': tag, **_price_key(inst.pp)},
        sort_keys=True, default=str).encode()).hexdigest()[:16]
    return f'stream_{h}'


def load_stream(d: Path):
    d = Path(d)
    meta = json.loads((d / 'meta.json').read_text())
    out = dict(
        indptr=np.fromfile(d / 'indptr.i64', np.int64),
        rsa=np.fromfile(d / 'rsa.f64', np.float64),
        aptr=np.fromfile(d / 'aptr.i64', np.int64),
        indices=np.memmap(d / 'indices.i32', np.int32, 'r'),
        probs=np.memmap(d / 'probs.f64', np.float64, 'r'),
    )
    out['meta'] = meta
    assert len(out['indptr']) == meta['n_sa'] + 1
    assert int(out['indptr'][-1]) == meta['nnz'] == len(out['indices']) == len(out['probs'])
    return out


# ---------- 워커 메모리 실측 (ps 표본) ----------
class RSSWatch(threading.Thread):
    """spawn 워커들의 RSS 를 ps 로 표본한다.

    추정치는 믿지 않는다 — 이 인스턴스에서 세 번 OOM 으로 죽은 원인이 전부
    '이 정도면 되겠지' 였다. 실제 프로세스의 RSS 를 주기적으로 읽어
    워커 최대·합계와 시스템 여유(MemAvailable)를 남긴다.
    """

    def __init__(self, interval: float = 2.0, log=None, ppid: int | None = None,
                 warn_gb: float = 6.0):
        super().__init__(daemon=True)
        self.interval, self.say = interval, (log or (lambda *a: None))
        self.ppid = ppid or os.getpid()
        self.warn_gb = warn_gb
        self._stop = threading.Event()
        self.peak = dict(worker_max_gb=0.0, worker_sum_gb=0.0, parent_gb=0.0,
                         rss_total_gb=0.0, avail_min_gb=float('inf'), n=0, samples=0)
        self.low_mem = False

    def _sample(self):
        try:
            out = subprocess.run(['ps', '-o', 'pid=,rss=,args=', '--ppid', str(self.ppid)],
                                 capture_output=True, text=True, timeout=10).stdout
        except Exception:
            return None
        rss = [int(f[1]) / 1e6 for f in (ln.split(None, 2) for ln in out.splitlines())
               if len(f) >= 3 and 'spawn_main' in f[2]]          # KB → GB
        try:
            parent = int(next(l for l in open(f'/proc/{self.ppid}/status')
                              if l.startswith('VmRSS')).split()[1]) / 1e6
        except Exception:
            parent = 0.0
        try:
            avail = int(next(l for l in open('/proc/meminfo')
                             if l.startswith('MemAvailable')).split()[1]) / 1e6
        except Exception:
            avail = float('inf')
        return rss, parent, avail

    def run(self):
        while not self._stop.wait(self.interval):
            s = self._sample()
            if not s:
                continue
            rss, parent, avail = s
            p = self.peak
            p['samples'] += 1
            if rss:
                p['n'] = max(p['n'], len(rss))
                p['worker_max_gb'] = max(p['worker_max_gb'], max(rss))
                p['worker_sum_gb'] = max(p['worker_sum_gb'], sum(rss))
            p['parent_gb'] = max(p['parent_gb'], parent)
            p['rss_total_gb'] = max(p['rss_total_gb'], sum(rss) + parent)
            p['avail_min_gb'] = min(p['avail_min_gb'], avail)
            if avail < self.warn_gb and not self.low_mem:
                self.low_mem = True
                self.say(f'  [RSS] 경고: MemAvailable {avail:.1f}GB — '
                         f'워커 {len(rss)}개 합계 {sum(rss):.1f}GB. '
                         f'BATDIAG_FLUSH_NNZ 를 낮추거나 워커를 줄일 것.')

    def stop(self):
        self._stop.set()
        self.join(timeout=self.interval * 2 + 5)
        return self.peak

    def report(self):
        p = dict(self.peak)
        if not p['samples']:
            return '  [RSS] 표본 없음'
        av = p.pop('avail_min_gb')
        return ('  [RSS] 표본 {samples}회 / 워커 {n}개 — 워커최대 {worker_max_gb:.2f}GB, '
                '워커합계 {worker_sum_gb:.2f}GB, 부모 {parent_gb:.2f}GB, '
                '전체 {rss_total_gb:.2f}GB, MemAvailable 최저 '.format(**p)
                + (f'{av:.1f}GB' if av != float('inf') else 'n/a'))


# ---------- 워커: 청크를 임시파일로 흘려 쓴다 ----------
def _chunk_to_file(args):
    """(lo,hi) 구간을 계산해 파일에 쓰고, 작은 메타만 부모에 돌려준다.

    반환 (p_ix, p_pr, counts, rsa, nnz) — counts/rsa 는 행당 8바이트뿐이라
    청크당 수백 KB 수준이다. 큰 배열은 파이프를 타지 않는다.
    """
    lo, hi, seq, outdir, flush_nnz = args
    from . import build as _b
    I = _b._INST                      # _init 이 워커 프로세스에 심어 둔 인스턴스
    p_ix = os.path.join(outdir, f'p{seq:06d}.ix')
    p_pr = os.path.join(outdir, f'p{seq:06d}.pr')
    cnt_l, rsa_l = [], []
    buf_ix, buf_pr = [], []
    buf_n = nnz = 0
    with open(p_ix, 'wb') as f_ix, open(p_pr, 'wb') as f_pr:
        for si in range(lo, hi):
            st = I.ST[si]
            for a in I.actions(st):
                o = I.step_dist(st, a)
                p = np.fromiter((x[0] for x in o), float, len(o))
                i = np.fromiter((x[1] for x in o), np.int32, len(o))
                r = np.fromiter((x[2] for x in o), float, len(o))
                del o
                buf_ix.append(i); buf_pr.append(p)
                cnt_l.append(len(i)); rsa_l.append(float(p @ r))
                buf_n += len(i); nnz += len(i)
                if buf_n >= flush_nnz:
                    np.concatenate(buf_ix).tofile(f_ix)
                    np.concatenate(buf_pr).tofile(f_pr)
                    buf_ix.clear(); buf_pr.clear(); buf_n = 0
            # 행동 캐시는 워커에 계속 쌓인다 — 끝난 상태는 즉시 버린다.
            I._acts.pop(st, None)
        if buf_ix:
            np.concatenate(buf_ix).tofile(f_ix)
            np.concatenate(buf_pr).tofile(f_pr)
        del buf_ix, buf_pr
    assert os.path.getsize(p_ix) == 4 * nnz and os.path.getsize(p_pr) == 8 * nnz
    return p_ix, p_pr, np.asarray(cnt_l, np.int64), np.asarray(rsa_l, float), nnz


def build_stream(inst, cache_dir, tag='', workers: int | None = None, log=None,
                 chunks_per_worker: int = 16, window_extra: int = 4,
                 flush_nnz: int | None = None, rss_watch: bool = True):
    say = log or (lambda *a: None)
    d = Path(cache_dir) / _key(inst, tag)
    if (d / 'meta.json').exists():
        say(f'build_stream: 캐시 적중 {d.name}')
        return load_stream(d)

    env_w = os.environ.get('BATDIAG_BUILD_WORKERS')
    workers = workers or (int(env_w) if env_w else None) or min(os.cpu_count() or 4, 24)
    nS = len(inst.ST)
    workers = max(1, min(workers, nS))
    flush_nnz = flush_nnz or int(os.environ.get('BATDIAG_FLUSH_NNZ', FLUSH_NNZ))

    t0 = time.time()
    na = np.asarray([len(inst.actions(st)) for st in inst.ST], np.int64)
    n_sa = int(na.sum())
    jobs = _partition(na, max(workers * chunks_per_worker, workers))
    say(f'build_stream: nS={nS:,} n_sa={n_sa:,} 워커 {workers}개 / 청크 {len(jobs)}개 '
        f'/ flush {flush_nnz:,} nnz (≈{flush_nnz*12/1e9:.2f}GB 버퍼 상한)')

    tmp = d.with_suffix('.tmp')
    if tmp.exists():
        shutil.rmtree(tmp)
    parts = tmp / 'parts'
    parts.mkdir(parents=True, exist_ok=True)
    f_ix = open(tmp / 'indices.i32', 'wb')
    f_pr = open(tmp / 'probs.f64', 'wb')
    counts_l, rsa_l = [], []
    nnz = 0

    def _drain(res):
        """워커 파일을 순서대로 이어붙이고 즉시 지운다 — 부모는 8MB 버퍼만 쓴다."""
        nonlocal nnz
        p_ix, p_pr, cnt, rsa, k = res
        for src, dst in ((p_ix, f_ix), (p_pr, f_pr)):
            with open(src, 'rb') as fs:
                shutil.copyfileobj(fs, dst, 1 << 23)
            os.unlink(src)
        counts_l.append(cnt); rsa_l.append(rsa)
        nnz += k

    watch = RSSWatch(log=say) if rss_watch else None
    if watch:
        watch.start()
    t0 = time.time()
    try:
        if workers == 1:
            _init(inst)
            for k, j in enumerate(jobs, 1):
                _drain(_chunk_to_file((j[0], j[1], k, str(parts), flush_nnz)))
                say(f'  stream {k}/{len(jobs)}  nnz={nnz:,}  {time.time()-t0:.0f}s')
        else:
            saved = _strip(inst)
            try:
                ctx = mp.get_context('spawn')
                window = workers + window_extra      # 동시에 미소진 상태로 둘 청크 수
                with ProcessPoolExecutor(max_workers=workers, mp_context=ctx,
                                         initializer=_init, initargs=(inst,)) as ex:
                    pend = deque()
                    it = iter(enumerate(jobs))
                    for k, j in it:
                        pend.append(ex.submit(_chunk_to_file, (j[0], j[1], k, str(parts), flush_nnz)))
                        if len(pend) >= window:
                            break
                    done = 0
                    while pend:
                        _drain(pend.popleft().result())      # 순서 보존
                        done += 1
                        nx = next(it, None)
                        if nx is not None:
                            k, j = nx
                            pend.append(ex.submit(_chunk_to_file,
                                                  (j[0], j[1], k, str(parts), flush_nnz)))
                        if done % 5 == 0 or done == len(jobs):
                            say(f'  stream {done}/{len(jobs)}  nnz={nnz:,}  {time.time()-t0:.0f}s')
                            if watch:
                                say(watch.report())
            finally:
                if saved:
                    inst._acts = saved
    finally:
        f_ix.close(); f_pr.close()
        peak = watch.stop() if watch else {}
        if watch:
            say(watch.report())

    counts = np.concatenate(counts_l); del counts_l
    rsa = np.concatenate(rsa_l); del rsa_l
    assert len(counts) == n_sa, (len(counts), n_sa)
    indptr = np.zeros(n_sa + 1, np.int64); np.cumsum(counts, out=indptr[1:])
    del counts
    aptr = np.zeros(nS + 1, np.int64); np.cumsum(na, out=aptr[1:])
    assert int(indptr[-1]) == nnz, (int(indptr[-1]), nnz)
    indptr.tofile(tmp / 'indptr.i64')
    rsa.tofile(tmp / 'rsa.f64')
    aptr.tofile(tmp / 'aptr.i64')
    shutil.rmtree(parts, ignore_errors=True)
    (tmp / 'meta.json').write_text(json.dumps(dict(
        nS=nS, n_sa=n_sa, nnz=nnz, built=time.time(),
        workers=workers, chunks=len(jobs), flush_nnz=flush_nnz,
        secs=round(time.time() - t0, 1),
        rss_peak_gb={k: (None if v == float('inf') else v) for k, v in peak.items()},
        bytes=dict(indices=nnz * 4, probs=nnz * 8))))
    if d.exists():
        shutil.rmtree(d)
    tmp.rename(d)
    say(f'build_stream: 저장 {d.name}  nnz={nnz:,} ({(nnz*12)/1e9:.1f}GB)  {time.time()-t0:.0f}s')
    return load_stream(d)
