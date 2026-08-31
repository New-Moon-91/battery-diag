# -*- coding: utf-8 -*-
"""RCR 편 규제 시나리오 정책 — **W 정식화** 전용.

S0  무진단        전량 즉시 재활용 매각. 제도 도입 이전의 사실상 관행
S1  전수 성능평가  **미검사 매각 금지.** 모든 배터리가 정밀 성능평가를 거쳐야 한다.
                  용량 CAP 안에서 배정하고 나머지는 보유한다.
                  창고가 넘치면 초과분은 규칙에 따라 **평가 없이 강제 재활용 매각**된다
                  — 이것이 곧 **규제 미이행**이며, 그 규모가 본 논문의 결과다.
S2b 사전정보 선별  신속검사 없이 w 만으로 정밀 대상 선택 (기존 b2_no_screening)
S2  계층적 진단    정확해 최적정책 (exact.greedy)

**S1 은 Config(prune=False) 에서만 표현 가능하다.** prune=True 면 R1 유형이 DUMP 로
즉시매각 강제되어 '전수 평가'가 성립하지 않는다.
"""
from __future__ import annotations
import numpy as np


def _idx(I, st, a):
    try:
        return I.actions(st).index(a)
    except ValueError:
        return None


def s1_mandatory_precise(I, order: str = 'value') -> np.ndarray:
    """전수 성능평가. sv=0(미검사 매각 금지), CAP 안에서 정밀 배정, 나머지 보유.

    order  용량이 부족할 때의 배정 우선순위
           'value' 기대 순가치(V^PU − V^S) 높은 순 — 규제 하에서도 센터가 합리적으로 행동
           'mix'   도착 비중 순 — 규제의 무차별성을 그대로 반영 (강건성 확인용)
    """
    if I.cfg.prune:
        raise ValueError('S1 은 prune=False 인스턴스에서만 표현 가능하다')
    z = tuple([0]*len(I.TY))
    if order == 'value':
        keys = sorted(range(len(I.TY)), key=lambda k: -(I.VPU[I.TY[k]] - I.VS[I.TY[k]]))
    else:
        keys = sorted(range(len(I.TY)), key=lambda k: -I.MIX[I.TY[k]])
    out = []
    for st in I.ST:
        n, scr = st
        cap_left = I.CAP
        # 선별완료 재고(S1 에서는 발생하지 않지만 도달 가능)를 먼저 정밀로
        ps = set()
        for j in sorted(range(len(scr)), key=lambda j: -I.VPS[scr[j]]):
            if cap_left <= 0: break
            ps.add(j); cap_left -= 1
        pu = [0]*len(I.TY)
        for k in keys:
            if cap_left <= 0: break
            take = min(n[k], cap_left); pu[k] = take; cap_left -= take
        a = (z, z, tuple(pu), frozenset(), frozenset(ps))
        i = _idx(I, st, a)
        if i is None:                                   # 방어: 전량 보유
            i = _idx(I, st, (z, z, z, frozenset(), frozenset()))
        out.append(i)
    return np.array(out, dtype=np.int64)


def s0_no_diagnosis(I) -> np.ndarray:
    """무진단 — 전량 즉시 재활용 매각 (policies.b1_sell_all 과 동일, 이름만 시나리오용)."""
    from battery_diag import policies as pol
    return pol.b1_sell_all(I)


SCENARIOS = {
    'S0':  ('무진단',            s0_no_diagnosis),
    'S1':  ('전수 성능평가',      s1_mandatory_precise),
    'S2b': ('사전정보 선별',      None),   # policies.b2_no_screening
    'S2':  ('계층적 진단(최적해)', None),   # exact.greedy(h)
}
