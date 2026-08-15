# -*- coding: utf-8 -*-
"""벤치마크 정책 — 모두 상태별 '로컬 행동 인덱스' 배열을 반환."""
from __future__ import annotations
import numpy as np


def _idx(I, st, a):
    A = I.actions(st)
    try: return A.index(a)
    except ValueError:                      # 가지치기로 제거된 행동 → 전량 즉시매각으로 대체
        n, scr = st; z = tuple([0]*len(I.TY))
        return A.index((tuple(n), z, z, frozenset(range(len(scr))), frozenset()))


def b1_sell_all(I):
    z = tuple([0]*len(I.TY))
    return np.array([_idx(I, st, (tuple(st[0]), z, z, frozenset(range(len(st[1]))), frozenset()))
                     for st in I.ST], dtype=np.int64)


def b2_no_screening(I):
    """선별 없이 순가치 상위 CAP건을 직행 정밀검사, 나머지 즉시매각"""
    z = tuple([0]*len(I.TY)); out = []
    for st in I.ST:
        n, scr = st; cand = []
        for k, t in enumerate(I.TY):
            if t in I.PUOK:
                cand += [(I.VPU[t]-I.VS[t], k)] * n[k]
        cand.sort(key=lambda x: -x[0]); pu = [0]*len(I.TY)
        for v, k in cand[:I.CAP]:
            if v > 0: pu[k] += 1
        sv = tuple(n[k]-pu[k] for k in range(len(I.TY)))
        out.append(_idx(I, st, (sv, z, tuple(pu), frozenset(range(len(scr))), frozenset())))
    return np.array(out, dtype=np.int64)


def b_fast(I, thr):
    """원고 구조 — 전량 신속검사 후 신호 thr 이상 상위 CAP건만 정밀, 나머지 매각"""
    z = tuple([0]*len(I.TY)); out = []
    for st in I.ST:
        n, scr = st
        sv = tuple(n[k] if I.TY[k] in I.DUMP else 0 for k in range(len(I.TY)))
        fv = tuple(0 if I.TY[k] in I.DUMP else n[k] for k in range(len(I.TY)))
        cand = sorted([j for j in range(len(scr))
                       if scr[j][1] >= thr and I.VPS[scr[j]] > I.VS[scr[j][0]]],
                      key=lambda j: -I.VPS[scr[j]])[:I.CAP]
        ps = frozenset(cand); ss = frozenset(j for j in range(len(scr)) if j not in ps)
        out.append(_idx(I, st, (sv, fv, z, ss, ps)))
    return np.array(out, dtype=np.int64)


def evsi(I, t):
    q = I.QP[t]; pb, ms = I.BP[t]; pdet = (1-q)*I.P_DET
    v0 = max(I.VS[t], I.VPU[t])
    v = pdet*I.VS[t] + (1-pdet)*sum(pb[b]*max(I.VS[t], I.VPS[(t,b)]) for b in range(I.cfg.SB))
    return v - v0


def index_myopic(I):
    """근시안 EVSI 지표 — 자원이 분리되어 신속·정밀 결정이 독립"""
    out = []
    for st in I.ST:
        n, scr = st; cand = []
        for k, t in enumerate(I.TY):
            if t in I.PUOK: cand += [(I.VPU[t]-I.VS[t], 'U', k)] * n[k]
        for j in range(len(scr)):
            if I.VPS[scr[j]] > I.VS[scr[j][0]]:
                cand.append((I.VPS[scr[j]]-I.VS[scr[j][0]], 'S', j))
        cand.sort(key=lambda x: -x[0])
        pu = [0]*len(I.TY); ps = set()
        for v, kind, arg in cand[:I.CAP]:
            if v <= 0: break
            if kind == 'U': pu[arg] += 1
            else: ps.add(arg)
        sv = [0]*len(I.TY); fv = [0]*len(I.TY)
        for k, t in enumerate(I.TY):
            c = n[k] - pu[k]
            if t in I.DUMP: sv[k] = c
            elif evsi(I, t) - I.cfg.Cf > 0: fv[k] = c
            else: sv[k] = c
        ss = {j for j in range(len(scr))
              if j not in ps and I.VPS[scr[j]] <= I.VS[scr[j][0]]}
        out.append(_idx(I, st, (tuple(sv), tuple(fv), tuple(pu), frozenset(ss), frozenset(ps))))
    return np.array(out, dtype=np.int64)


def all_benchmarks(I, solver):
    res = {}
    for name, acts in [('B1', b1_sell_all(I)), ('B2', b2_no_screening(I)),
                       ('INDEX', index_myopic(I))]:
        res[name] = solver.evaluate(acts)[0]
    bf = {thr: solver.evaluate(b_fast(I, thr))[0] for thr in range(I.cfg.SB)}
    res['B_fast'] = max(bf.values()); res['B_fast_thr'] = int(max(bf, key=bf.get))
    return res
