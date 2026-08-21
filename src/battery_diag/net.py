# -*- coding: utf-8 -*-
"""순열불변(Deep Sets) 인코더 + 배터리별 공유 헤드 — **배치 처리판**.

이전 CPU 구현은 상태마다 파이썬 루프를 돌아 병목이었다. 여기서는
  · 상태를 배치로 쌓고 (U_max = NMAX*nT, S_max = SMAX 로 패딩; 둘 다 작은 상수)
  · 순차 배정은 '배터리 슬롯' 축으로만 루프(<=8회), 상태는 전부 병렬
로 바꿔 GPU 한 장에서 수천 상태를 한 번에 처리한다.
"""
from __future__ import annotations
import numpy as np, torch, torch.nn as nn, torch.nn.functional as F

A_UNT = ['SELL', 'FAST', 'PRECISE', 'HOLD']
A_SCR = ['SELL', 'PRECISE', 'HOLD']
NEG = -1e9


def mlp(i, h, o, act=nn.Tanh):
    return nn.Sequential(nn.Linear(i, h), act(), nn.Linear(h, h), act(), nn.Linear(h, o))


# W 무관 정규화 (w6 [4]). encode.WREF 와 같은 값으로 함께 설정한다.
# None 이면 기존 그대로 — 슬롯 수(Um·Sm=W)와 wcap 으로 나눈다.
WREF = None


def _ref(default):
    return float(WREF) if WREF else float(default)


# carry 차원 — 미검사축 7, 선별축 6 (아래 carry_from_labels 의 열 순서와 같다)
CARRY_U, CARRY_S = 7, 6


class PolicyNet(nn.Module):
    """carry=False 는 v5b 까지의 디코더 그대로. carry=True 는 자기회귀 디코더.

    레거시 디코더는 슬롯 로짓을 루프 **밖에서 한 번만** 계산한다. 그래서 로짓이
    (z, 슬롯특징, 진입시점 예산) 이 아니라 사실상 (z, 슬롯특징) 에만 의존하고
    — 예산조차 초기값 하나로 고정 — 슬롯 사이를 가르는 것은 마스크뿐이다.
    특징이 같은 슬롯은 마스크가 바뀌기 전까지 반드시 같은 행동을 받으므로
    "같은 유형 3대를 신속 2 + 매각 1 로 쪼개기" 를 표현할 수 없다 (v5b §4.2).

    덧붙여 레거시는 학습·추론이 어긋나 있었다. ce_loss 는 슬롯별 예산 budU 를
    헤드에 넣는데 decode 는 초기 예산만 넣는다. carry=True 는 양쪽 모두
    carry_from_labels 와 같은 정의를 쓰므로 이 불일치도 사라진다.
    """

    def __init__(self, fdim=6, ctxdim=4, hid=128, emb=64, nbin=12, carry=False):
        super().__init__()
        self.nbin = nbin
        self.carry = bool(carry)
        cu, cs = (CARRY_U, CARRY_S) if self.carry else (1, 1)
        self.cu, self.cs = cu, cs
        self.phiU = mlp(fdim, hid, emb); self.phiS = mlp(fdim, hid, emb)
        self.enc  = mlp(2*emb + ctxdim + 2*nbin + 2, hid, hid)
        self.headU = mlp(hid + fdim + cu, hid, 4)
        self.headS = mlp(hid + fdim + cs, hid, 3)
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
        cnt = torch.stack([mu.sum(1)/_ref(max(U.shape[1],1)),
                           ms.sum(1)/_ref(max(S.shape[1],1))], -1)
        return self.enc(torch.cat([eu, es, ctx, hu, hs, cnt], -1))

    def value(self, z):
        return self.V(z).squeeze(-1)

    def _logits(self, head, z, feat, bud):
        B, K, F = feat.shape
        zz = z.unsqueeze(1).expand(B, K, z.shape[-1])
        bb = bud.unsqueeze(-1).unsqueeze(1).expand(B, K, 1)
        return head(torch.cat([zz, feat, bb], -1))

    # ---------- carry: 지금까지의 배정을 요약한 벡터 ----------
    #
    # 열 순서 (carry=True). decode 의 축차 갱신과 carry_from_labels 의 누적합이
    # **같은 정의**여야 학습(교사강요)과 추론이 일치한다 — tests/test_carry.py 가 고정한다.
    #   미검사축 (CARRY_U=7)
    #     0..3  앞선 유효 슬롯에 배정한 SELL/FAST/PRECISE/HOLD 개수 / Um
    #     4     잔여 정밀예산 (cap - 앞선 PRECISE 수) / cap
    #     5     선별버퍼 잔여용량 (SMAX - |scr| - 앞선 FAST 수) / SMAX
    #     6     남은 유효 슬롯 수(자기 포함) / Um
    #   선별축 (CARRY_S=6)
    #     0..2  앞선 유효 슬롯에 배정한 SELL/PRECISE/HOLD 개수 / Sm
    #     3     잔여 정밀예산 (cap - 미검사축 PRECISE 총수 - 앞선 PRECISE 수) / cap
    #     4     선별버퍼 잔여용량 (SMAX - (|scr| - 앞선 SELL - 앞선 PRECISE) - FAST 총수) / SMAX
    #     5     남은 유효 슬롯 수(자기 포함) / Sm
    #
    # 5번(미검사축)이 이번 변경의 핵심이다. FAST 로 보낸 배터리는 결함이 안 잡히면
    # 다음 기 선별버퍼로 들어가고, 버퍼가 차 있으면 step_dist 가 즉시매각으로 흘린다
    # (강제매각). 즉 신속검사 수는 선별버퍼 잔여용량에 맞춰 조절해야 하는데
    # 레거시 디코더에는 그 신호가 아예 없었다. |scr| 는 전량 보유를 가정한 보수적
    # 근사다 — 미검사축이 선별축보다 먼저 디코딩되어 보유 수를 아직 모르기 때문이다.
    # (선별축을 먼저 디코딩하면 근사를 없앨 수 있다. 후속 과제로 남긴다.)

    def _dtype_dev(self, U):
        return U.dtype, U.device

    def carry_from_labels(self, batch, lab_u, lab_s, cap: int):
        """라벨 → (cU, cS, mU, mS). 교사강요용 — 누적합으로 한 번에 계산한다."""
        U, mu, S, ms = batch['U'], batch['mu'], batch['S'], batch['ms']
        allow_u, allow_s = batch['allow_u'], batch['allow_s']
        dt = U.dtype
        Um = mu.shape[1]; Sm = ms.shape[1]
        oneU = F.one_hot(lab_u, 4).to(dt) * mu.unsqueeze(-1)
        oneS = F.one_hot(lab_s, 3).to(dt) * ms.unsqueeze(-1)
        prevU4 = torch.cumsum(oneU, 1) - oneU              # (B,Um,4) 자기 앞까지
        prevS3 = torch.cumsum(oneS, 1) - oneS              # (B,Sm,3)
        prevPrecU, prevFastU = prevU4[..., 2], prevU4[..., 1]
        totPrecU = oneU[..., 2].sum(1, keepdim=True)
        totFastU = oneU[..., 1].sum(1, keepdim=True)
        prevSellS, prevPrecS = prevS3[..., 0], prevS3[..., 1]
        budU = cap - prevPrecU
        budS = cap - totPrecU - prevPrecS
        mU = allow_u.clone(); mU[..., 2] &= (budU >= 1)
        mS = allow_s.clone(); mS[..., 1] &= (budS >= 1)
        if not self.carry:                                  # 레거시: 예산 하나만
            return (budU/max(cap,1)).unsqueeze(-1), (budS/max(cap,1)).unsqueeze(-1), mU, mS
        nscr = ms.sum(1, keepdim=True)
        remU = mu.flip(1).cumsum(1).flip(1)                 # 자기 포함 남은 유효 슬롯
        remS = ms.flip(1).cumsum(1).flip(1)
        # 잔여 저장용량. 기존 정식화는 선별버퍼 SMAX(=Sm) 가 제약이고,
        # W-정식화는 창고 하나(W)를 미검사·선별완료가 나눠 쓴다. 후자에서는
        # 보유(HOLD)한 미검사분도 자리를 먹으므로 함께 센다.
        if 'wcap' in batch:
            W = batch['wcap'].unsqueeze(-1)
            prevHoldU = prevU4[..., 3]
            totHoldU = oneU[..., 3].sum(1, keepdim=True)
            cap_ref = W
            freeU = W - nscr - prevFastU - prevHoldU
            freeS = W - (nscr - prevSellS - prevPrecS) - totFastU - totHoldU
        else:
            cap_ref = float(max(Sm, 1))
            freeU = Sm - nscr - prevFastU
            freeS = Sm - (nscr - prevSellS - prevPrecS) - totFastU
        if WREF:                       # W 무관 정규화 — 분모를 고정 상수로
            cap_ref = _ref(1.0)
        cU = torch.cat([prevU4/_ref(max(Um,1)), (budU/max(cap,1)).unsqueeze(-1),
                        (freeU/cap_ref).unsqueeze(-1),
                        (remU/_ref(max(Um,1))).unsqueeze(-1)], -1)
        cS = torch.cat([prevS3/_ref(max(Sm,1)), (budS/max(cap,1)).unsqueeze(-1),
                        (freeS/cap_ref).unsqueeze(-1),
                        (remS/_ref(max(Sm,1))).unsqueeze(-1)], -1)
        return cU, cS, mU, mS

    def _logits(self, head, z, feat, bud):
        B, K, F_ = feat.shape
        zz = z.unsqueeze(1).expand(B, K, z.shape[-1])
        bb = bud.unsqueeze(-1).unsqueeze(1).expand(B, K, 1)
        return head(torch.cat([zz, feat, bb], -1))

    # ---------- 탐욕 디코딩 (슬롯 축 순차, 상태 축 병렬) ----------
    @torch.no_grad()
    def decode(self, batch, cap: int, sample=False, gen=None, return_carry=False):
        U, mu, S, ms, ctx = batch['U'], batch['mu'], batch['S'], batch['ms'], batch['ctx']
        allow_u, allow_s = batch['allow_u'], batch['allow_s']
        B, Um = mu.shape; Sm = ms.shape[1]
        dt = U.dtype
        z = self.encode(U, mu, S, ms, ctx)
        au = torch.zeros(B, Um, dtype=torch.long, device=U.device)
        as_ = torch.zeros(B, Sm, dtype=torch.long, device=U.device)
        pick = (lambda lg: torch.distributions.Categorical(logits=lg).sample()) if sample \
               else (lambda lg: lg.argmax(-1))

        if not self.carry:                                  # ---- 레거시 (v5b 그대로)
            bud = torch.full((B,), float(cap), device=U.device, dtype=dt)
            lgU = self._logits(self.headU, z, U, bud/max(cap,1))
            for i in range(Um):
                m = allow_u[:, i].clone()
                m[:, 2] &= (bud >= 1)
                a = pick(lgU[:, i].masked_fill(~m, NEG))
                a = torch.where(mu[:, i] > 0, a, torch.zeros_like(a))
                au[:, i] = a
                bud = bud - ((a == 2) & (mu[:, i] > 0)).to(dt)
            lgS = self._logits(self.headS, z, S, bud/max(cap,1))
            for j in range(Sm):
                m = allow_s[:, j].clone()
                m[:, 1] &= (bud >= 1)
                a = pick(lgS[:, j].masked_fill(~m, NEG))
                a = torch.where(ms[:, j] > 0, a, torch.zeros_like(a))
                as_[:, j] = a
                bud = bud - ((a == 1) & (ms[:, j] > 0)).to(dt)
            if return_carry:
                return au.cpu().numpy(), as_.cpu().numpy(), None, None
            return au.cpu().numpy(), as_.cpu().numpy()

        # ---- 자기회귀: 슬롯마다 carry 를 갱신하고 로짓을 그 자리에서 계산
        nscr = ms.sum(1)
        remU = mu.flip(1).cumsum(1).flip(1)
        remS = ms.flip(1).cumsum(1).flip(1)
        wmode = 'wcap' in batch
        Wt = batch['wcap'] if wmode else None
        cap_ref = Wt if wmode else float(max(Sm, 1))
        if WREF:                                # W 무관 정규화 (carry_from_labels 와 동일)
            cap_ref = _ref(1.0)
        cntU = torch.zeros(B, 4, device=U.device, dtype=dt)
        cntS = torch.zeros(B, 3, device=U.device, dtype=dt)
        prevPrecU = torch.zeros(B, device=U.device, dtype=dt)
        prevFastU = torch.zeros(B, device=U.device, dtype=dt)
        prevHoldU = torch.zeros(B, device=U.device, dtype=dt)
        cUs, cSs = [], []
        for i in range(Um):
            budU = cap - prevPrecU
            freeU = (Wt - nscr - prevFastU - prevHoldU) if wmode else (Sm - nscr - prevFastU)
            c = torch.cat([cntU/_ref(max(Um,1)), (budU/max(cap,1)).unsqueeze(-1),
                           (freeU/cap_ref).unsqueeze(-1),
                           (remU[:, i]/_ref(max(Um,1))).unsqueeze(-1)], -1)
            cUs.append(c)
            lg = self.headU(torch.cat([z, U[:, i], c], -1))
            m = allow_u[:, i].clone(); m[:, 2] &= (budU >= 1)
            a = pick(lg.masked_fill(~m, NEG))
            a = torch.where(mu[:, i] > 0, a, torch.zeros_like(a))
            au[:, i] = a
            v = (mu[:, i] > 0).to(dt)
            cntU = cntU + F.one_hot(a, 4).to(dt) * v.unsqueeze(-1)
            prevPrecU = prevPrecU + (a == 2).to(dt) * v
            prevFastU = prevFastU + (a == 1).to(dt) * v
            prevHoldU = prevHoldU + (a == 3).to(dt) * v
        totPrecU, totFastU, totHoldU = prevPrecU, prevFastU, prevHoldU
        prevSellS = torch.zeros(B, device=U.device, dtype=dt)
        prevPrecS = torch.zeros(B, device=U.device, dtype=dt)
        for j in range(Sm):
            budS = cap - totPrecU - prevPrecS
            freeS = ((Wt - (nscr - prevSellS - prevPrecS) - totFastU - totHoldU) if wmode
                     else (Sm - (nscr - prevSellS - prevPrecS) - totFastU))
            c = torch.cat([cntS/_ref(max(Sm,1)), (budS/max(cap,1)).unsqueeze(-1),
                           (freeS/cap_ref).unsqueeze(-1),
                           (remS[:, j]/_ref(max(Sm,1))).unsqueeze(-1)], -1)
            cSs.append(c)
            lg = self.headS(torch.cat([z, S[:, j], c], -1))
            m = allow_s[:, j].clone(); m[:, 1] &= (budS >= 1)
            a = pick(lg.masked_fill(~m, NEG))
            a = torch.where(ms[:, j] > 0, a, torch.zeros_like(a))
            as_[:, j] = a
            v = (ms[:, j] > 0).to(dt)
            cntS = cntS + F.one_hot(a, 3).to(dt) * v.unsqueeze(-1)
            prevSellS = prevSellS + (a == 0).to(dt) * v
            prevPrecS = prevPrecS + (a == 1).to(dt) * v
        if return_carry:
            cU = torch.stack(cUs, 1) if cUs else torch.zeros(B, 0, self.cu, device=U.device, dtype=dt)
            cS = torch.stack(cSs, 1) if cSs else torch.zeros(B, 0, self.cs, device=U.device, dtype=dt)
            return au.cpu().numpy(), as_.cpu().numpy(), cU, cS
        return au.cpu().numpy(), as_.cpu().numpy()

    # ---------- 교사강요 교차엔트로피 (라벨로 carry 를 누적합으로 계산) ----------
    def ce_loss(self, batch, lab_u, lab_s, cap: int):
        U, mu, S, ms, ctx = batch['U'], batch['mu'], batch['S'], batch['ms'], batch['ctx']
        B, Um = mu.shape; Sm = ms.shape[1]
        z = self.encode(U, mu, S, ms, ctx)
        cU, cS, mU, mS = self.carry_from_labels(batch, lab_u, lab_s, cap)
        zz = z.unsqueeze(1).expand(B, Um, z.shape[-1])
        lgU = self.headU(torch.cat([zz, U, cU], -1))
        lpU = torch.log_softmax(lgU.masked_fill(~mU, NEG), -1)
        lossU = -(lpU.gather(-1, lab_u.unsqueeze(-1)).squeeze(-1) * mu).sum()
        zs = z.unsqueeze(1).expand(B, Sm, z.shape[-1])
        lgS = self.headS(torch.cat([zs, S, cS], -1))
        lpS = torch.log_softmax(lgS.masked_fill(~mS, NEG), -1)
        lossS = -(lpS.gather(-1, lab_s.unsqueeze(-1)).squeeze(-1) * ms).sum()
        return (lossU + lossS) / B
