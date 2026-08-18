# -*- coding: utf-8 -*-
"""스트리밍 정확해 — CSR 이 GPU(16GB)·RAM 을 넘는 인스턴스용.

설계
  · RVI 는 매 반복 전체 (s,a) 행을 훑는다. 39GB 를 수천 번 읽을 수는 없다.
    → **정책반복(PI)** 으로 바꾼다. 전체 스윕은 개선 단계에서만 필요하고
      PI 는 십수 회면 수렴한다.
  · 평가·정상분포는 정책 고정 시 상태당 한 행 — nS×평균분기 (~15M nnz) 로 작아
    통째로 GPU(또는 RAM) 에 올린다. 알고리즘은 ExactSolver 와 동일(멱반복).
  · 개선(greedy) 은 indices/probs memmap 을 상태 경계에 맞춘 슬랩으로 잘라
    순차로 장치에 올렸다 내린다. 슬랩당 기본 96M nnz ≈ 1.6GB 전송.

인터페이스는 ExactSolver 와 동일: solve / evaluate / improve / greedy / stationary.
solve() 는 (g, h, 반복수) 를 돌려주므로 run_one.py 를 그대로 쓴다.

백엔드: torch 가 있으면 torch(cuda/cpu), 없으면 numpy — 결과는 동일하며
numpy 경로는 GPU 없는 환경에서의 이식 검증용이다.
"""
from __future__ import annotations
import os
import numpy as np

try:
    import torch
except ImportError:
    torch = None

NEG = -1e30


class StreamSolver:
    def __init__(self, arrays: dict, device: str = 'cuda', slab_nnz: int | None = None,
                 tie_rel: float = 1e-9):
        self.ip = np.asarray(arrays['indptr'])          # RAM (n_sa+1)
        self.rsa = np.asarray(arrays['rsa'])            # RAM (n_sa)
        self.aptr = np.asarray(arrays['aptr'])          # RAM (nS+1)
        self.ix = arrays['indices']                     # memmap 또는 ndarray
        self.pr = arrays['probs']
        self.nS = len(self.aptr) - 1
        self.n_sa = len(self.rsa)
        self.na = np.diff(self.aptr)
        self.use_torch = torch is not None
        self.dev = torch.device(device) if self.use_torch else None
        self.dt = torch.float64 if self.use_torch else np.float64
        slab_nnz = slab_nnz or int(os.environ.get('BATDIAG_SLAB_NNZ', 96_000_000))
        self.tie_rel = tie_rel
        self.slabs = self._make_slabs(slab_nnz)         # [(s0,s1,r0,r1,p0,p1), ...]
        self._P = None                                  # 정책행 캐시 (acts 해시 기준)
        self._Pkey = None

    # ---------- 슬랩 경계: 상태 경계에 맞춰 nnz 상한으로 자른다 ----------
    def _make_slabs(self, slab_nnz):
        state_nnz = self.ip[self.aptr[1:]] - self.ip[self.aptr[:-1]]
        slabs, s0, acc = [], 0, 0
        for s in range(self.nS):
            v = int(state_nnz[s])
            if acc and acc + v > slab_nnz:
                slabs.append((s0, s)); s0, acc = s, 0
            acc += v
        slabs.append((s0, self.nS))
        out = []
        for a, b in slabs:
            r0, r1 = int(self.aptr[a]), int(self.aptr[b])
            out.append((a, b, r0, r1, int(self.ip[r0]), int(self.ip[r1])))
        return out

    # ---------- 장치 헬퍼 ----------
    def _t(self, x, dtype=None):
        if self.use_torch:
            t = torch.as_tensor(np.ascontiguousarray(x), device=self.dev)
            return t.to(dtype) if dtype is not None else t
        return np.ascontiguousarray(x)

    # ---------- 정책행 추출: 상태당 한 행 → 작은 배열 ----------
    def _policy_rows(self, acts_local):
        acts = np.asarray(acts_local, np.int64)
        key = acts.tobytes()
        if self._Pkey == key:
            return self._P
        rows = self.aptr[:-1] + acts
        lo, hi = self.ip[rows], self.ip[rows + 1]
        cnt = hi - lo
        # memmap 에서 정책행만 모은다 (fancy-index 는 조각별로)
        cols = np.empty(int(cnt.sum()), np.int64)
        vals = np.empty(int(cnt.sum()), np.float64)
        o = 0
        for l, h_, c in zip(lo, hi, cnt):
            c = int(c)
            cols[o:o+c] = self.ix[l:h_]
            vals[o:o+c] = self.pr[l:h_]
            o += c
        src = np.repeat(np.arange(self.nS, dtype=np.int64), cnt)
        P = dict(cols=self._t(cols), vals=self._t(vals, self.dt if self.use_torch else None),
                 src=self._t(src), r=self._t(self.rsa[rows], self.dt if self.use_torch else None))
        self._P, self._Pkey = P, key
        return P

    def _pi_mv(self, P, h):
        if self.use_torch:
            out = torch.zeros(self.nS, device=self.dev, dtype=self.dt)
            out.scatter_add_(0, P['src'], P['vals'] * h[P['cols']])
            return out
        out = np.zeros(self.nS)
        np.add.at(out, P['src'], P['vals'] * h[P['cols']])
        return out

    # ---------- 정책평가 (ExactSolver.evaluate 와 동일 알고리즘) ----------
    def evaluate(self, acts_local, tol=1e-10, itmax=20000, check_every=25, h0=None,
                 rtol=1e-13):
        """정책 고정 시의 (g, h). 수렴판정은 **절대+상대** 기준이다.

        절대 tol 만 쓰면 이 문제에서는 영원히 성립하지 않는다. h 가 1e6 규모라
        float64 표현 간격이 ~1e-10 이고, 실측한 max|Δh| 는 2.33e-10 에서 바닥을
        친다(2026-08-19, NMAX=3/SMAX=2). 즉 이미 수치적으로 도달한 고정점인데
        판정이 성립하지 않아 매 호출이 itmax(20,000)회를 전부 돌았다.
        같은 인스턴스에서 max|Δh| < 1e-9 는 50회면 닿는다 — 400배 낭비였다.
        → 임계값을 tol + rtol*max|h| (h~1e6 이면 ~1e-7) 로 둔다.
        """
        P = self._policy_rows(acts_local)
        if self.use_torch:
            h = torch.zeros(self.nS, device=self.dev, dtype=self.dt) if h0 is None \
                else torch.as_tensor(h0, device=self.dev, dtype=self.dt).clone()
        else:
            h = np.zeros(self.nS) if h0 is None else np.asarray(h0, np.float64).copy()
        g = 0.0
        for it in range(itmax):
            hn = P['r'] + self._pi_mv(P, h)
            g = hn[0].clone() if self.use_torch else hn[0]
            hn = hn - g
            if (it + 1) % check_every == 0 or it + 1 == itmax:
                if self.use_torch:
                    d = torch.max(torch.abs(hn - h)).item()
                    sc = torch.max(torch.abs(hn)).item()
                else:
                    d = float(np.max(np.abs(hn - h))); sc = float(np.max(np.abs(hn)))
                if d < tol + rtol * sc:
                    h = hn; break
            h = hn
        hf = h.cpu().numpy() if self.use_torch else h
        return float(g), hf

    # ---------- 개선: 슬랩 스트리밍 greedy ----------
    def greedy(self, h):
        h_np = np.asarray(h, np.float64)
        hd = self._t(h_np, self.dt if self.use_torch else None)
        best = np.empty(self.nS, np.int64)
        for (s0, s1, r0, r1, p0, p1) in self.slabs:
            ix = np.array(self.ix[p0:p1])      # memmap -> 쓰기가능 복사본
            pr = np.array(self.pr[p0:p1])
            ipl = self.ip[r0:r1 + 1] - p0                 # 슬랩-로컬 행 포인터
            cnt = np.diff(ipl)
            nr = r1 - r0
            if self.use_torch:
                ixd = torch.as_tensor(ix, device=self.dev).to(torch.long)
                prd = torch.as_tensor(pr, device=self.dev, dtype=self.dt)
                rid = torch.repeat_interleave(
                    torch.arange(nr, device=self.dev),
                    torch.as_tensor(cnt, device=self.dev))
                q = torch.zeros(nr, device=self.dev, dtype=self.dt)
                q.scatter_add_(0, rid, prd * hd[ixd])
                q += torch.as_tensor(self.rsa[r0:r1], device=self.dev, dtype=self.dt)
                q_np = q.cpu().numpy()
                del ixd, prd, rid, q
            else:
                contrib = pr * h_np[ix]
                q_np = np.add.reduceat(contrib, ipl[:-1]) if nr else np.empty(0)
                q_np[cnt == 0] = 0.0
                q_np = q_np + self.rsa[r0:r1]
            # 상태별 argmax — 동점은 **상대** 허용치로 묶고 가장 앞 행동을 고른다.
            #
            # ExactSolver 는 절대 1e-12 를 쓰는데, q 가 1e6 규모면 float64 의
            # 표현 간격(~1e-10)보다 작아 참 동점조차 동점으로 인식되지 않는다.
            # 그러면 선택이 덧셈 순서 잡음에 좌우되고, 정책반복 중 h 가 미세하게
            # 흔들릴 때마다 선택이 튀어 '변경 0' 이 성립하지 않는다(수렴 실패).
            # 상대 1e-9 (q~1e6 이면 절대 1e-3) 로 묶으면 안정되며,
            # 값 손실은 상대 1e-9 이하로 무시할 수준이다.
            for s in range(s0, s1):
                a, b = self.aptr[s] - r0, self.aptr[s + 1] - r0
                seg = q_np[a:b]
                m = seg.max()
                best[s] = int(np.argmax(seg >= m - self.tie_rel * max(1.0, abs(m))))
        return best

    def improve(self, h):
        return self.greedy(h)

    # ---------- 최적해: 정책반복 ----------
    def solve(self, tol=1e-9, itmax=200, h0=None, acts0=None, log=None):
        say = log or (lambda *a: None)
        acts = np.zeros(self.nS, np.int64) if acts0 is None else np.asarray(acts0, np.int64).copy()
        g, h = self.evaluate(acts, h0=h0)
        stall = 0
        for it in range(itmax):
            acts2 = self.greedy(h)
            ch = int((acts2 != acts).sum())
            if ch == 0:
                say(f'  PI {it}: 수렴 g={g:,.2f}')
                return float(g), h, it
            acts = acts2
            g2, h = self.evaluate(acts, h0=h)             # 웜스타트
            say(f'  PI {it}: 변경 {ch}개  g {g:,.2f} → {g2:,.2f}')
            # 진동 방지 — 변경은 남았는데 g 가 더 이상 오르지 않으면 동점 튐이다.
            if g2 <= g + tol * max(1.0, abs(g2)):
                stall += 1
                if stall >= 3:
                    say(f'  PI {it}: g 정체 {stall}회 → 종료 (잔여 변경 {ch}개는 동점)')
                    return float(max(g2, g)), h, it
            else:
                stall = 0
            g = g2
        return float(g), h, itmax

    # ---------- 정상상태 분포 ----------
    def stationary(self, acts_local, itmax=20000, tol=1e-13, check_every=25):
        P = self._policy_rows(acts_local)
        if self.use_torch:
            d = torch.full((self.nS,), 1.0 / self.nS, device=self.dev, dtype=self.dt)
            for it in range(itmax):
                nd = torch.zeros(self.nS, device=self.dev, dtype=self.dt)
                nd.scatter_add_(0, P['cols'], P['vals'] * d[P['src']])
                nd /= nd.sum()
                if (it + 1) % check_every == 0 or it + 1 == itmax:
                    if torch.max(torch.abs(nd - d)).item() < tol:
                        return nd.cpu().numpy()
                d = nd
            return d.cpu().numpy()
        d = np.full(self.nS, 1.0 / self.nS)
        for it in range(itmax):
            nd = np.zeros(self.nS)
            np.add.at(nd, P['cols'], P['vals'] * d[P['src']])
            nd /= nd.sum()
            if (it + 1) % check_every == 0 or it + 1 == itmax:
                if np.max(np.abs(nd - d)) < tol:
                    return nd
            d = nd
        return d
