# -*- coding: utf-8 -*-
"""S1 — 시뮬레이터.  전이행렬을 열거하지 않고 (보상, 다음상태) 를 표본추출한다.

실규모(W=1,097 · 16유형)에서는 `Instance.step_dist` 처럼 전이를 열거할 수 없다.
그래서 같은 확률구조를 **표본추출**로 다시 구현한다. 핵심 요구는 하나다 —
소형에서 이 시뮬레이터의 전이 빈도가 정확 전이행렬 `P(s'|s,a)` 와 통계적으로
일치해야 한다. 어긋나면 이후 단계(S2 롤아웃 h, S3 롤아웃 DCL)가 전부 무의미하다.

`step_dist` 와 한 줄씩 대응시킨 구조 (그래야 검정이 의미를 갖는다):

    보상 = r + rr − hc + rev
      r   행동 즉시보상 — 매각 V^S, 직행정밀 V^PU, 선별품 매각/정밀, 신속검사비 C_f
          (정밀검사 보상은 **커밋 시점 기대값**이라 표본추출하지 않는다)
      rr  신속검사에서 결함이 잡힌 대수 × V^S
      hc  보관비 h × 점유(도착 전) = 남은 미검사 + 선별완료
      rev 도착이 창고를 넘칠 때 강제매각 수익
    다음상태 = (남은 미검사 + 수용된 도착, 선별완료)

**확률적인 것은 둘뿐이다.**
  1. 신속검사 결과 — 확률 pdet=(1−q_P)·P_DET 로 결함 검출(→ 즉시 재활용),
     아니면 신호구간 b ~ BP[t] 로 선별완료가 된다.
  2. 도착 — 개수와 유형 구성, 그리고 초과분의 강제매각.

도착 모형 (`arrivals`)
  'binomial' : NARR 슬롯 각각이 확률 lam 로 도착. **`instance.py` 와 정확히 같다.**
               S1~S3 의 정확해 대조는 반드시 이쪽을 써야 한다.
  'poisson'  : 개수 ~ Poisson(lam·NARR). SPEC §3 의 실규모 도착 모형이다.
               두 모형은 평균이 같고 분산이 다르다(이항 lam·NARR·(1−lam) < 포아송
               lam·NARR). 실규모 전환 시에만 쓰고, 정확해 대조에는 쓰지 않는다.
"""
from __future__ import annotations
import numpy as np


class Simulator:
    def __init__(self, inst, arrivals: str = 'binomial'):
        if arrivals not in ('binomial', 'poisson'):
            raise ValueError(arrivals)
        self.I = inst
        self.arrivals = arrivals
        c = inst.cfg
        self.nT = len(inst.TY)
        self.lam, self.NARR, self.SB = c.lam, c.NARR, c.SB
        self.h, self.Cf = c.h, c.Cf
        self.W = inst.W
        self.mix = np.array([inst.MIX[t] for t in inst.TY], float)
        self.mix = self.mix/self.mix.sum()
        self.VS = np.array([inst.VS[t] for t in inst.TY], float)
        self.VPU = np.array([inst.VPU[t] for t in inst.TY], float)
        self.QP = np.array([inst.QP[t] for t in inst.TY], float)
        self.pdet = (1.0 - self.QP) * inst.P_DET
        self.pb = np.array([inst.BP[t][0] for t in inst.TY], float)   # (nT, SB)
        self.sell_order = list(inst._sell_order) if inst.W is not None else \
            sorted(range(self.nT), key=lambda k: inst.VS[inst.TY[k]])

    # ---------------- 도착 ----------------
    def _arrivals(self, rng):
        """유형별 도착 개수벡터. instance._arr_dist 와 같은 분포여야 한다."""
        if self.arrivals == 'binomial':
            A = int(rng.binomial(self.NARR, self.lam))
        else:
            A = int(rng.poisson(self.lam * self.NARR))
        if A == 0:
            return np.zeros(self.nT, np.int64)
        return rng.multinomial(A, self.mix)

    # ---------------- 한 걸음 ----------------
    def step(self, st, a, rng):
        """(보상, 다음상태) 하나를 표본추출. st=(n, scr), a=(sv,fv,pu,ss,ps)."""
        I = self.I
        n, scr = st
        sv, fv, pu, ss, ps = a

        # r — 행동 즉시보상 (step_dist 첫 항과 같다)
        r = 0.0
        for k in range(self.nT):
            r += sv[k]*self.VS[k] + pu[k]*self.VPU[k]
        for j in ss:
            r += I.VS[scr[j][0]]
        for j in ps:
            r += I.VPS[scr[j]]
        r -= self.Cf * sum(fv)

        rem = [n[k]-sv[k]-fv[k]-pu[k] for k in range(self.nT)]
        sc = [scr[j] for j in range(len(scr)) if j not in ss and j not in ps]

        # 1) 신속검사 결과 — 결함이면 즉시 재활용(자리 안 씀), 아니면 선별완료가 된다
        rr = 0.0
        for k in range(self.nT):
            for _ in range(fv[k]):
                if rng.random() < self.pdet[k]:
                    rr += self.VS[k]
                else:
                    b = int(rng.choice(self.SB, p=self.pb[k]))
                    sc.append((I.TY[k], b))
        sc = tuple(sorted(sc))

        # 2) 보관비 — 도착 **전** 점유에 부과 (step_dist 와 같은 시점)
        occ = sum(rem) + len(sc)
        hc = self.h * occ

        # 3) 도착과 강제매각
        aa = self._arrivals(rng)
        rev = 0.0
        if self.W is not None:
            excess = int(aa.sum()) - (self.W - occ)
            if excess > 0:
                aa = aa.copy()
                for k in self.sell_order:
                    if excess <= 0:
                        break
                    take = min(int(aa[k]), excess)
                    aa[k] -= take; excess -= take; rev += take*self.VS[k]
        else:                                   # 기존 NMAX 정식화 — 상한에서 잘린다
            aa = np.minimum(np.array(rem) + aa, I.cfg.NMAX) - np.array(rem)
        nn = tuple(int(rem[k] + aa[k]) for k in range(self.nT))
        return float(r + rr - hc + rev), (nn, sc)

    # ---------------- 궤적 ----------------
    def rollout(self, st, act_of, H, rng, gbar=0.0):
        """π 를 H 기간 따라가며 Σ(r − gbar) 를 돌려준다. act_of(state) → 행동튜플."""
        tot = 0.0
        for _ in range(H):
            r, st = self.step(st, act_of(st), rng)
            tot += r - gbar
        return tot, st

    def avg_reward(self, st, act_of, T, rng, burn=0):
        """긴 단일 궤적의 평균보상 — g 추정."""
        for _ in range(burn):
            _, st = self.step(st, act_of(st), rng)
        tot = 0.0
        for _ in range(T):
            r, st = self.step(st, act_of(st), rng)
            tot += r
        return tot/T, st
