# -*- coding: utf-8 -*-
"""GPU 정확해 — 상대가치반복(RVI) · 정책평가 · 1단계 개선.

핵심: 상태-행동 전이를 하나의 희소행렬 A (n_sa x nS) 로 두면
      q = r_sa + A h            (SpMV)
      h' = segment_max(q, seg)  (scatter_reduce amax)
두 커널로 RVI 한 스텝이 끝난다. CPU 파이썬 루프 대비 수백 배.

정밀도: 값 크기가 1e6 규모이고 g 를 1e-7 상대오차로 봐야 하므로 **float64 고정**.
        SpMV 는 메모리 바운드라 RTX 5080 의 낮은 FP64 연산성능이 병목이 아니다.
"""
from __future__ import annotations
import numpy as np
try:
    import torch
except ImportError:      # numpy 참조 구현만 쓸 때는 torch 없이도 import 가능
    torch = None

NEG = -1e30


class ExactSolver:
    def __init__(self, arrays: dict, device: str = 'cuda', dtype=None):
        dtype = dtype or torch.float64
        self.dev, self.dt = torch.device(device), dtype
        ip = torch.from_numpy(arrays['indptr']).to(self.dev)
        ix = torch.from_numpy(arrays['indices'].astype(np.int64)).to(self.dev)
        pr = torch.from_numpy(arrays['probs']).to(self.dev, dtype)
        self.n_sa = ip.numel() - 1
        self.nS = int(arrays['aptr'].shape[0] - 1)
        self.A = torch.sparse_csr_tensor(ip, ix, pr, size=(self.n_sa, self.nS), dtype=dtype)
        self.rsa = torch.from_numpy(arrays['rsa']).to(self.dev, dtype)
        self.seg = torch.from_numpy(arrays['seg']).to(self.dev)
        self.aptr = torch.from_numpy(arrays['aptr']).to(self.dev)
        self._ar = torch.arange(self.n_sa, device=self.dev)

    def _q(self, h):
        return self.rsa + torch.mv(self.A, h)

    def _policy_rows(self, acts_local):
        """정책 고정 시 상태당 한 행만 필요 → 그 행들만 평탄화해 보관.

        cuSPARSE 디스크립터(sparse_csr_tensor) 대신 gather + scatter_add 로 곱한다.
        커널 하나로 끝나고, 부분행렬을 매번 새로 만들 때 생기던
        `cusparseCreateCsr: invalid value` 를 원천적으로 피한다.
        """
        rows = self.aptr[:-1] + torch.as_tensor(acts_local, device=self.dev, dtype=torch.long)
        ip, ix, vl = self.A.crow_indices(), self.A.col_indices(), self.A.values()
        starts = ip[rows]; cnt = ip[rows + 1] - starts
        total = int(cnt.sum().item())
        offs = torch.cumsum(cnt, 0) - cnt
        pos = torch.repeat_interleave(starts - offs, cnt) + torch.arange(total, device=self.dev)
        cols = ix[pos].to(torch.long)
        vals = vl[pos]
        src = torch.repeat_interleave(torch.arange(self.nS, device=self.dev), cnt)
        return dict(cols=cols, vals=vals, src=src, r=self.rsa[rows])

    def _pi_mv(self, P, h):
        """(P^pi h)(s) = sum_j p_sj h_j   — 행 방향 집계"""
        out = torch.zeros(self.nS, device=self.dev, dtype=self.dt)
        out.scatter_add_(0, P['src'], P['vals'] * h[P['cols']])
        return out

    def _seg_max(self, q):
        out = torch.full((self.nS,), NEG, device=self.dev, dtype=self.dt)
        return out.scatter_reduce_(0, self.seg, q, reduce='amax', include_self=False)

    def _seg_argmax(self, q):
        m = self._seg_max(q)
        is_max = q >= m[self.seg] - 1e-12
        idx = torch.full((self.nS,), self.n_sa, device=self.dev, dtype=torch.long)
        idx.scatter_reduce_(0, self.seg[is_max], self._ar[is_max], reduce='amin', include_self=False)
        return m, idx - self.aptr[:-1]          # 상태별 로컬 행동 인덱스

    # ---------- 최적해 ----------
    def solve(self, tol=1e-7, itmax=20000, h0=None):
        # 수렴 확인은 10회마다 (GPU-CPU 동기화 비용 절감)
        h = torch.zeros(self.nS, device=self.dev, dtype=self.dt) if h0 is None \
            else torch.as_tensor(h0, device=self.dev, dtype=self.dt).clone()
        g = torch.zeros((), device=self.dev, dtype=self.dt)
        for it in range(itmax):
            hn = self._seg_max(self._q(h))
            g = hn[0].clone(); hn -= g
            if (it + 1) % 10 == 0 or it + 1 == itmax:
                if torch.max(torch.abs(hn - h)).item() < tol:
                    h = hn; break
            h = hn
        return float(g), h.cpu().numpy(), it

    def greedy(self, h):
        h = torch.as_tensor(h, device=self.dev, dtype=self.dt)
        _, loc = self._seg_argmax(self._q(h))
        return loc.cpu().numpy()

    # ---------- 정책평가 ----------
    def evaluate(self, acts_local, tol=1e-10, itmax=20000, check_every=25):
        """acts_local: 상태별 로컬 행동 인덱스 → (장기평균보상 g, 상대가치 h)

        수렴 확인은 check_every 회마다만 — 매 반복 .item() 은 GPU-CPU 동기화라 비싸다."""
        P = self._policy_rows(acts_local)
        h = torch.zeros(self.nS, device=self.dev, dtype=self.dt)
        g = torch.zeros((), device=self.dev, dtype=self.dt)
        for it in range(itmax):
            hn = P['r'] + self._pi_mv(P, h)
            g = hn[0].clone(); hn = hn - g
            if (it + 1) % check_every == 0 or it + 1 == itmax:
                if torch.max(torch.abs(hn - h)).item() < tol:
                    h = hn; break
            h = hn
        return float(g), h.cpu().numpy()

    def improve(self, h):
        return self.greedy(h)

    # ---------- 정상상태 분포 ----------
    def stationary(self, acts_local, itmax=20000, tol=1e-13, check_every=25):
        P = self._policy_rows(acts_local)
        d = torch.full((self.nS,), 1.0/self.nS, device=self.dev, dtype=self.dt)
        for it in range(itmax):
            nd = torch.zeros(self.nS, device=self.dev, dtype=self.dt)
            nd.scatter_add_(0, P['cols'], P['vals'] * d[P['src']])
            nd /= nd.sum()
            if (it + 1) % check_every == 0 or it + 1 == itmax:
                if torch.max(torch.abs(nd - d)).item() < tol:
                    return nd.cpu().numpy()
            d = nd
        return d.cpu().numpy()


def policy_iteration(solver: ExactSolver, acts0, rounds=50, tol=1e-9):
    """정확 정책반복 — DCL 의 상한(oracle) 및 개선 라벨 생성용"""
    acts = np.asarray(acts0).copy(); hist = []
    for r in range(rounds):
        g, h = solver.evaluate(acts)
        acts2 = solver.improve(h)
        hist.append(dict(round=r, g=g, changed=int((acts2 != acts).sum())))
        if (acts2 == acts).all(): break
        acts = acts2
    g, h = solver.evaluate(acts)
    return g, h, acts, hist


# ============================================================
# numpy 참조 구현 — GPU 없이 이식 검증용 (느리지만 동일 결과)
# ============================================================
class NumpySolver:
    def __init__(self, arrays: dict):
        self.ip = arrays['indptr']; self.ix = arrays['indices']; self.pr = arrays['probs']
        self.rsa = arrays['rsa']; self.seg = arrays['seg']; self.aptr = arrays['aptr']
        self.nS = len(self.aptr) - 1; self.n_sa = len(self.rsa)

    def _q(self, h):
        seg_sum = np.add.reduceat(self.pr * h[self.ix], self.ip[:-1])
        empty = np.diff(self.ip) == 0
        seg_sum[empty] = 0.0
        return self.rsa + seg_sum

    def _seg_max(self, q):
        out = np.full(self.nS, -np.inf)
        np.maximum.at(out, self.seg, q)
        return out

    def solve(self, tol=1e-7, itmax=20000):
        h = np.zeros(self.nS)
        for it in range(itmax):
            hn = self._seg_max(self._q(h)); g = hn[0]; hn = hn - g
            if np.max(np.abs(hn - h)) < tol: return float(g), hn, it
            h = hn
        return float(g), h, itmax

    def evaluate(self, acts_local, tol=1e-10, itmax=20000):
        rows = self.aptr[:-1] + np.asarray(acts_local, np.int64)
        h = np.zeros(self.nS)
        for it in range(itmax):
            hn = self._q(h)[rows]; g = hn[0]; hn = hn - g
            if np.max(np.abs(hn - h)) < tol: return float(g), hn
            h = hn
        return float(g), h

    def improve(self, h):
        q = self._q(h); m = self._seg_max(q)
        best = np.zeros(self.nS, np.int64)
        for s in range(self.nS):
            lo, hi = self.aptr[s], self.aptr[s+1]
            best[s] = int(np.argmax(q[lo:hi]))
        return best

    def stationary(self, acts_local, itmax=20000, tol=1e-13):
        rows = self.aptr[:-1] + np.asarray(acts_local, np.int64)
        d = np.full(self.nS, 1.0/self.nS)
        for _ in range(itmax):
            nd = np.zeros(self.nS)
            for s in range(self.nS):
                lo, hi = self.ip[rows[s]], self.ip[rows[s]+1]
                np.add.at(nd, self.ix[lo:hi], self.pr[lo:hi]*d[s])
            nd /= nd.sum()
            if np.max(np.abs(nd - d)) < tol: return nd
            d = nd
        return d
