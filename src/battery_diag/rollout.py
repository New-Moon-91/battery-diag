# -*- coding: utf-8 -*-
"""S2 — 롤아웃 기반 상대가치 h 추정.

평균보상 MDP 에서 정책 π 의 상대가치는

    h(s) = lim_{H→∞} E[ Σ_{t<H} (r_t − g)  |  s_0 = s ]

이므로, 유한 H 로 잘라 표본평균을 내면 추정치가 된다. g 도 추정해야 하는데
긴 단일 궤적의 평균보상을 쓴다.

**절대값이 아니라 순서가 중요하다.** 개선 단계는 argmax_a [ r(s,a) + Σ P h(s') ]
이므로 h 에 상수를 더해도 결과가 같다. 유한 H 절단이 만드는 편의는 상태에 거의
공통이라(체인이 섞이면 사라진다) 순서에는 크게 영향하지 않는다 — 그래서 S2 의
검증 지표는 상관계수보다 **개선 단계 행동 일치율**이 우선이다.

**분산감소 — 공통난수(CRN).** 롤아웃 m 번째에서 모든 시작상태가 **같은 난수열**을
쓴다. h 자체가 아니라 h 의 «차이» 가 결정에 쓰이므로, 상태 간 공통 잡음을 상쇄시키면
같은 표본수로 훨씬 안정된 순서를 얻는다.
"""
from __future__ import annotations
import numpy as np


def estimate_g(sim, I, acts, T=200_000, burn=5_000, seed=0, start=None):
    """긴 단일 궤적의 평균보상으로 g 를 추정한다."""
    rng = np.random.default_rng(seed)
    st = I.ST[start if start is not None else 0]
    for _ in range(burn):
        _, st = sim.step(st, I.actions(st)[acts[I.SI[st]]], rng)
    tot = 0.0
    for _ in range(T):
        r, st = sim.step(st, I.actions(st)[acts[I.SI[st]]], rng)
        tot += r
    return tot/T


def estimate_h(sim, I, acts, g, H=60, M=20, seed=0, crn=True, states=None):
    """정책 acts 의 h 를 롤아웃으로 추정. states=None 이면 전 상태.

    CRN: 롤아웃 m 에서 모든 시작상태가 같은 씨앗을 쓴다.
    """
    idx = range(len(I.ST)) if states is None else list(states)
    out = np.zeros(len(idx))
    for m in range(M):
        for pos, si in enumerate(idx):
            rng = (np.random.default_rng([seed, m]) if crn
                   else np.random.default_rng([seed, m, si]))
            st = I.ST[si]
            tot = 0.0
            for _ in range(H):
                r, st = sim.step(st, I.actions(st)[acts[I.SI[st]]], rng, crn=crn)
                tot += r - g
            out[pos] += tot
    return out/M


def center(x):
    """h 는 상수만큼 자유롭다 — 비교 전에 평균을 뺀다."""
    x = np.asarray(x, float)
    return x - x.mean()
