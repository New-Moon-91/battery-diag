# -*- coding: utf-8 -*-
"""순열불변(Deep Sets) 인코더 + 배터리별 공유 헤드 — **배치 처리판**.

이전 CPU 구현은 상태마다 파이썬 루프를 돌아 병목이었다. 여기서는
  · 상태를 배치로 쌓고 (U_max = NMAX*nT, S_max = SMAX 로 패딩; 둘 다 작은 상수)
  · 순차 배정은 '배터리 슬롯' 축으로만 루프(<=8회), 상태는 전부 병렬
로 바꿔 GPU 한 장에서 수천 상태를 한 번에 처리한다.
"""
from __future__ import annotations
import numpy as np, torch, torch.nn as nn

A_UNT = ['SELL', 'FAST', 'PRECISE', 'HOLD']
A_SCR = ['SELL', 'PRECISE', 'HOLD']
NEG = -1e9


def mlp(i, h, o, act=nn.Tanh):
    return nn.Sequential(nn.Linear(i, h), act(), nn.Linear(h, h), act(), nn.Linear(h, o))


class PolicyNet(nn.Module):
    def __init__(self, fdim=6, ctxdim=4, hid=128, emb=64, nbin=12):
        super().__init__()
        self.nbin = nbin
        self.phiU = mlp(fdim, hid, emb); self.phiS = mlp(fdim, hid, emb)
        self.enc  = mlp(2*emb + ctxdim + 2*nbin + 2, hid, hid)
        self.headU = mlp(hid + fdim + 1, hid, 4)
        self.headS = mlp(hid + fdim + 1, hid, 3)
        self.V = mlp(hid, hid, 1)

    def _hist(self, X, m, col):
        v = torch.clamp(X[..., col], 0., 1.)
        idx = torch.clamp((v*self.nbin).long(), 0, self.nbin-1)
        oh = torch.zeros(*v.shape, self.nbin, device=X.device, dtype=X.dtype)
        oh.scatter_(-1, idx.unsqueeze(-1), 1.0)
        oh = oh * m.unsqueeze(-1)
        return oh.sum(1) / m.sum(1, keepdim=True).clamp(min=1)

    def encode(self, U, mu, S, ms, ctx):
        """U:(B,Um,F) mu:(B,Um) S:(B,Sm,F) ms:(B,Sm) ctx:(B,C) → z:(B,H)"""
        eu = (self.phiU(U) * mu.unsqueeze(-1)).sum(1) / mu.sum(1, keepdim=True).clamp(min=1)
        es = (self.phiS(S) * ms.unsqueeze(-1)).sum(1) / ms.sum(1, keepdim=True).clamp(min=1)
        hu = self._hist(U, mu, 1); hs = self._hist(S, ms, 1)
        cnt = torch.stack([mu.sum(1)/max(U.shape[1],1), ms.sum(1)/max(S.shape[1],1)], -1)
        return self.enc(torch.cat([eu, es, ctx, hu, hs, cnt], -1))

    def value(self, z):
        return self.V(z).squeeze(-1)

    def _logits(self, head, z, feat, bud):
        B, K, F = feat.shape
        zz = z.unsqueeze(1).expand(B, K, z.shape[-1])
        bb = bud.unsqueeze(-1).unsqueeze(1).expand(B, K, 1)
        return head(torch.cat([zz, feat, bb], -1))

    # ---------- 탐욕 디코딩 (슬롯 축 순차, 상태 축 병렬) ----------
    @torch.no_grad()
    def decode(self, batch, cap: int, sample=False, gen=None):
        U, mu, S, ms, ctx = batch['U'], batch['mu'], batch['S'], batch['ms'], batch['ctx']
        allow_u, allow_s = batch['allow_u'], batch['allow_s']
        B, Um = mu.shape; Sm = ms.shape[1]
        z = self.encode(U, mu, S, ms, ctx)
        bud = torch.full((B,), float(cap), device=U.device, dtype=U.dtype)
        au = torch.zeros(B, Um, dtype=torch.long, device=U.device)
        as_ = torch.zeros(B, Sm, dtype=torch.long, device=U.device)
        lgU = self._logits(self.headU, z, U, bud/max(cap,1))
        for i in range(Um):
            m = allow_u[:, i].clone()
            m[:, 2] &= (bud >= 1)
            lg = lgU[:, i].masked_fill(~m, NEG)
            a = (torch.distributions.Categorical(logits=lg).sample() if sample else lg.argmax(-1))
            a = torch.where(mu[:, i] > 0, a, torch.zeros_like(a))
            au[:, i] = a
            bud = bud - ((a == 2) & (mu[:, i] > 0)).to(U.dtype)
        lgS = self._logits(self.headS, z, S, bud/max(cap,1))
        for j in range(Sm):
            m = allow_s[:, j].clone()
            m[:, 1] &= (bud >= 1)
            lg = lgS[:, j].masked_fill(~m, NEG)
            a = (torch.distributions.Categorical(logits=lg).sample() if sample else lg.argmax(-1))
            a = torch.where(ms[:, j] > 0, a, torch.zeros_like(a))
            as_[:, j] = a
            bud = bud - ((a == 1) & (ms[:, j] > 0)).to(U.dtype)
        return au.cpu().numpy(), as_.cpu().numpy()

    # ---------- 교사강요 교차엔트로피 (라벨로 예산 프리픽스 계산) ----------
    def ce_loss(self, batch, lab_u, lab_s, cap: int):
        U, mu, S, ms, ctx = batch['U'], batch['mu'], batch['S'], batch['ms'], batch['ctx']
        allow_u, allow_s = batch['allow_u'], batch['allow_s']
        B, Um = mu.shape; Sm = ms.shape[1]
        z = self.encode(U, mu, S, ms, ctx)
        useU = ((lab_u == 2) & (mu > 0)).to(U.dtype)
        useS = ((lab_s == 1) & (ms > 0)).to(U.dtype)
        # 슬롯 i 진입 시점의 잔여 예산 = cap - (이전 슬롯들의 정밀 배정 수)
        prevU = torch.cumsum(useU, 1) - useU
        prevS = useU.sum(1, keepdim=True) + torch.cumsum(useS, 1) - useS
        budU = (cap - prevU) / max(cap, 1)
        budS = (cap - prevS) / max(cap, 1)
        zz = z.unsqueeze(1).expand(B, Um, z.shape[-1])
        lgU = self.headU(torch.cat([zz, U, budU.unsqueeze(-1)], -1))
        mU = allow_u.clone(); mU[..., 2] &= ((cap - prevU) >= 1)
        lpU = torch.log_softmax(lgU.masked_fill(~mU, NEG), -1)
        lossU = -(lpU.gather(-1, lab_u.unsqueeze(-1)).squeeze(-1) * mu).sum()
        zs = z.unsqueeze(1).expand(B, Sm, z.shape[-1])
        lgS = self.headS(torch.cat([zs, S, budS.unsqueeze(-1)], -1))
        mS = allow_s.clone(); mS[..., 1] &= ((cap - prevS) >= 1)
        lpS = torch.log_softmax(lgS.masked_fill(~mS, NEG), -1)
        lossS = -(lpS.gather(-1, lab_s.unsqueeze(-1)).squeeze(-1) * ms).sum()
        return (lossU + lossS) / B
