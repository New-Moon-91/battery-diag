# -*- coding: utf-8 -*-
"""상태 → 패딩 텐서 변환, 그리고 배터리별 배정 ↔ MDP 행동 튜플 변환."""
from __future__ import annotations
import numpy as np, torch

SCALE = 1e6


def state_tensors(I, device='cuda', dtype=torch.float32):
    """전 상태를 한 번에 패딩 텐서로. U_max = NMAX*nT, S_max = SMAX (둘 다 작은 상수)"""
    nT = len(I.TY); Um = I.cfg.NMAX*nT; Sm = I.cfg.SMAX; F = 6
    nS = len(I.ST)
    U = np.zeros((nS, Um, F), np.float32); mu = np.zeros((nS, Um), np.float32)
    S = np.zeros((nS, Sm, F), np.float32); ms = np.zeros((nS, Sm), np.float32)
    ctx = np.zeros((nS, 4), np.float32)
    au = np.zeros((nS, Um, 4), bool); as_ = np.zeros((nS, Sm, 3), bool)
    slot_t = np.full((nS, Um), -1, np.int64)          # 슬롯 → 유형 인덱스
    slot_j = np.full((nS, Sm), -1, np.int64)          # 슬롯 → 선별품 원본 위치
    fu, fs = {}, {}
    for t in I.TY:
        pb, msn = I.BP[t]
        fu[t] = np.array([I.QP[t], I.MU[t], I.SD[t], I.KWH[t]/64.,
                          I.VS[t]/SCALE, I.VPU[t]/SCALE], np.float32)
        psd = 1/np.sqrt(1/max(I.SD[t],1e-6)**2 + 1/I.cfg.sig_f**2)
        for b in range(I.cfg.SB):
            fs[(t,b)] = np.array([I.QPOST[t], msn[b], psd, I.KWH[t]/64.,
                                  I.VS[t]/SCALE, I.VPS[(t,b)]/SCALE], np.float32)
    order_t = sorted(range(nT), key=lambda k: -I.QP[I.TY[k]])
    for si, (n, scr) in enumerate(I.ST):
        p = 0
        for k in order_t:
            t = I.TY[k]
            for _ in range(n[k]):
                U[si,p] = fu[t]; mu[si,p] = 1; slot_t[si,p] = k
                if t in I.DUMP: au[si,p] = [True, False, False, False]
                else:
                    au[si,p] = [True, True, I.VPU[t] > I.VS[t], True]
                p += 1
        js = sorted(range(len(scr)), key=lambda j: -I.BP[scr[j][0]][1][scr[j][1]])
        for q, j in enumerate(js):
            x = scr[j]
            S[si,q] = fs[x]; ms[si,q] = 1; slot_j[si,q] = j
            if x in I.SDUMP: as_[si,q] = [True, False, False]
            else:            as_[si,q] = [True, I.VPS[x] > I.VS[x[0]], True]
        tot = sum(n) + len(scr)
        ctx[si] = [sum(n)/max(Um,1), len(scr)/max(Sm,1), I.CAP/max(Um+Sm,1), tot/max(Um+Sm,1)]
    T = lambda a, d=dtype: torch.as_tensor(a, device=device, dtype=d)
    return dict(U=T(U), mu=T(mu), S=T(S), ms=T(ms), ctx=T(ctx),
                allow_u=torch.as_tensor(au, device=device),
                allow_s=torch.as_tensor(as_, device=device),
                slot_t=slot_t, slot_j=slot_j)


def assign_to_action(I, st, slot_t_row, slot_j_row, au_row, as_row):
    """배터리별 배정 → MDP 행동 튜플"""
    nT = len(I.TY); n, scr = st
    sv = [0]*nT; fv = [0]*nT; pu = [0]*nT; ss = set(); ps = set()
    for p, k in enumerate(slot_t_row):
        if k < 0: continue
        a = au_row[p]
        if a == 0: sv[k] += 1
        elif a == 1: fv[k] += 1
        elif a == 2: pu[k] += 1
    for q, j in enumerate(slot_j_row):
        if j < 0: continue
        a = as_row[q]
        if a == 0: ss.add(j)
        elif a == 1: ps.add(j)
    return (tuple(sv), tuple(fv), tuple(pu), frozenset(ss), frozenset(ps))


def actions_from_assign(I, tens, AU, AS):
    """(nS,Um),(nS,Sm) 배정 → 상태별 로컬 행동 인덱스"""
    out = np.zeros(len(I.ST), np.int64)
    for si, st in enumerate(I.ST):
        a = assign_to_action(I, st, tens['slot_t'][si], tens['slot_j'][si], AU[si], AS[si])
        A = I.actions(st)
        try: out[si] = A.index(a)
        except ValueError:
            n, scr = st; z = tuple([0]*len(I.TY))
            out[si] = A.index((tuple(n), z, z, frozenset(range(len(scr))), frozenset()))
    return out


def labels_from_actions(I, tens, acts_local):
    """상태별 MDP 행동 → 배터리별 라벨 (교사강요용)"""
    nS = len(I.ST); Um = tens['slot_t'].shape[1]; Sm = tens['slot_j'].shape[1]
    LU = np.zeros((nS, Um), np.int64); LS = np.zeros((nS, Sm), np.int64)
    for si, st in enumerate(I.ST):
        sv, fv, pu, ss, ps = I.actions(st)[acts_local[si]]
        need = {k: [sv[k], fv[k], pu[k]] for k in range(len(I.TY))}
        for p, k in enumerate(tens['slot_t'][si]):
            if k < 0: LU[si,p] = 0; continue
            if   need[k][0] > 0: LU[si,p] = 0; need[k][0] -= 1
            elif need[k][1] > 0: LU[si,p] = 1; need[k][1] -= 1
            elif need[k][2] > 0: LU[si,p] = 2; need[k][2] -= 1
            else:                LU[si,p] = 3
        for q, j in enumerate(tens['slot_j'][si]):
            if j < 0: LS[si,q] = 0; continue
            LS[si,q] = 0 if j in ss else (1 if j in ps else 2)
    return LU, LS
