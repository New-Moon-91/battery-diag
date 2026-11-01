# -*- coding: utf-8 -*-
"""실규모 관문 실험 (scale1) — 유체 LP 상계 · 규모 확장 정책 · 시뮬레이션 평가.

질문은 하나다. **실규모(λ≈50/일, 16유형, W≈1,100)에서 학습정책(DCL)이 필요한가?**
소형(W≤6)에서 INDEX 갭 13~16%, B_fast 갭 4~5% 였던 것은 «작은 수의 효과»일 수 있다.
규모가 커지면 매 기간 50대 중 상위 CAP 대를 고르는 문제의 확률적 변동이 작아져
정적 지수정책이 유체 극한(fluid limit)에 붙을 가능성이 크다. 그 여부를 재려면
(i) 임의 규모에서 계산되는 **상계**와 (ii) 그 상계에 닿는지를 보는 **실행 가능한
정책**이 함께 필요하다. 이 모듈은 둘을 제공하고, 소형에서 정확해로 검산한다.

── 유체 LP 상계 (fluid_bound) ──────────────────────────────────────────────
배터리 한 대의 «여정»은 유형 k 에 대해 다음 중 하나다.
    SELL            보상 V^S_k
    PU (직행정밀)   보상 V^PU_k, 정밀용량 1 소비
    FAST → 구간 b   보상 −C_f + pdet·V^S + (1−pdet)Σ_b p_kb·[V^S + y_kb (V^PS_kb − V^S)],
                    정밀용량 (1−pdet)Σ_b p_kb y_kb 소비,  y_kb∈[0,1]
보유(HOLD)는 시점만 늦추고 여정의 기대보상을 바꾸지 않는다(평균보상·할인 없음).
어떤 정상정책도 장기적으로는 유형별 도착률 a_k = λ_eff·mix_k 를 여정들에 나누는
것이고, 기간당 정밀 CAP 건이라는 **경성 제약은 장기평균 제약을 함의**한다. 보관비
≥ 0 과 강제매각(= V^S 매각) 을 떨어내면 다음 LP 가 g* 의 상계다:

    max  Σ_k a_k [ s_k V^S_k + u_k V^PU_k + f_k(−C_f + V^S_k) + Σ_b z_kb (V^PS_kb − V^S_k) ]
    s.t. s_k + u_k + f_k = 1,  0 ≤ z_kb ≤ f_k (1−pdet_k) p_kb,
         Σ_k a_k ( u_k + Σ_b z_kb ) ≤ CAP.

(여기서 f_k 의 pdet·V^S + (1−pdet)·V^S = V^S 로 묶였다.) 정밀용량 제약의 쌍대가격
μ 가 **용량 한 건의 그림자가격**이고, 라그랑주 쌍대 L(μ) = Σ_k a_k max{V^S,
V^PU−μ, v_fast(μ)} + μ·CAP 의 최소값이 LP 값과 같아야 한다(강쌍대, 자가검산).

── 정책 (모두 act_of(state) 형태, Simulator 와 맞물린다) ─────────────────
    B1       전량 즉시매각
    B_fast   전량 신속검사, 신호 ≥ thr 상위 CAP 정밀, 나머지 매각  (정확해 정의와 같다)
    INDEX    근시안 EVSI 지표                                     (정확해 정의와 같다)
    LAGR     라그랑주 지수정책: 그림자가격 μ 로 유형별 경로(SELL/PU/FAST)를 정하고
             후보를 한계가치순으로 CAP 건 정밀, **미처리는 즉시매각** (대기 없음)
    QUEUE    LAGR + 대기열: 미처리 후보 중 한계가치 m 이 기대 대기비용 κ·h·(순위/CAP) 을
             넘으면 보유, 아니면 매각. 보유 총량은 W − λ_eff 를 넘지 않게 한다.

B_fast·INDEX 는 `policies.py` 의 정의를 상태열거 없이 다시 쓴 것이라, 소형에서
정확 평가값과 시뮬레이션 평균이 통계적으로 일치해야 한다 (tests/test_scale.py).
"""
from __future__ import annotations
import numpy as np


# ───────────────────────── 유체 LP 상계 ─────────────────────────
def _vfast(I, k, mu):
    t = I.TY[k]; q = I.QP[t]; pdet = (1-q)*I.P_DET; pb, _ = I.BP[t]
    vs = I.VS[t]
    return -I.cfg.Cf + pdet*vs + (1-pdet)*sum(pb[b]*max(vs, I.VPS[(t, b)]-mu)
                                               for b in range(I.cfg.SB))


def lagrangian(I, lam_eff, CAP, mu):
    """L(μ) 와 유형별 경로. 경로는 argmax{SELL, PU, FAST} (PU 는 PUOK 유형만)."""
    tot = 0.0; route = {}
    for k, t in enumerate(I.TY):
        a = lam_eff*I.MIX[t]
        cands = {'SELL': I.VS[t], 'FAST': _vfast(I, k, mu)}
        if t in I.PUOK:
            cands['PU'] = I.VPU[t] - mu
        if t in I.DUMP:
            cands = {'SELL': I.VS[t]}
        best = max(cands, key=cands.get)
        route[t] = best; tot += a*cands[best]
    return tot + mu*CAP, route


def fluid_bound(I, lam_eff: float, CAP: int):
    """유체 LP 상계. dict(bound, mu, lagr, route, flows, prec_use)."""
    from scipy.optimize import linprog
    TY = I.TY; nT = len(TY); SB = I.cfg.SB
    # 변수 배치: [s_k, u_k, f_k, z_k0..z_k(SB-1)] × nT
    nv = nT*(3+SB)
    c = np.zeros(nv); A_ub = []; b_ub = []; A_eq = []; b_eq = []
    bounds = []
    cap_row = np.zeros(nv)
    for k, t in enumerate(TY):
        a = lam_eff*I.MIX[t]; o = k*(3+SB)
        q = I.QP[t]; pdet = (1-q)*I.P_DET; pb, _ = I.BP[t]
        c[o+0] = -a*I.VS[t]
        c[o+1] = -a*I.VPU[t]
        c[o+2] = -a*(-I.cfg.Cf + I.VS[t])
        for b in range(SB):
            c[o+3+b] = -a*(I.VPS[(t, b)] - I.VS[t])
            row = np.zeros(nv); row[o+3+b] = 1.0; row[o+2] = -(1-pdet)*pb[b]
            A_ub.append(row); b_ub.append(0.0)
        row = np.zeros(nv); row[o:o+3] = 1.0
        A_eq.append(row); b_eq.append(1.0)
        cap_row[o+1] = a
        for b in range(SB): cap_row[o+3+b] = a
        pu_ok = (t in I.PUOK) and (t not in I.DUMP)
        bounds += [(0, 1), (0, 1 if pu_ok else 0), (0, 0 if t in I.DUMP else 1)] + [(0, None)]*SB
    A_ub.append(cap_row); b_ub.append(float(CAP))
    res = linprog(c, A_ub=np.array(A_ub), b_ub=np.array(b_ub),
                  A_eq=np.array(A_eq), b_eq=np.array(b_eq), bounds=bounds, method='highs')
    if not res.success:
        raise RuntimeError(res.message)
    bound = -res.fun
    mu = float(-res.ineqlin.marginals[-1])          # CAP 제약의 쌍대가격 (≥0)
    # 강쌍대 자가검산 — μ 근방 1차원 최소화
    from scipy.optimize import minimize_scalar
    hi = max(max(I.VPU[t]-I.VS[t] for t in TY),
             max(I.VPS[x]-I.VS[x[0]] for x in I.VPS), 1.0)
    ms = minimize_scalar(lambda m: lagrangian(I, lam_eff, CAP, m)[0],
                         bounds=(0.0, hi*1.05), method='bounded',
                         options={'xatol': 1e-6})
    lagr, route = lagrangian(I, lam_eff, CAP, mu)
    x = res.x; flows = {}
    use = 0.0
    for k, t in enumerate(TY):
        o = k*(3+SB); a = lam_eff*I.MIX[t]
        flows[t] = dict(sell=float(x[o]), pu=float(x[o+1]), fast=float(x[o+2]),
                        z=[float(v) for v in x[o+3:o+3+SB]])
        use += a*(x[o+1] + sum(x[o+3:o+3+SB]))
    return dict(bound=float(bound), mu=mu, lagr=float(lagr), lagr_min=float(ms.fun),
                mu_min=float(ms.x), route=route, flows=flows, prec_use=float(use),
                lam_eff=float(lam_eff), CAP=int(CAP))


# ───────────────────────── 정책 (act_of) ─────────────────────────
def _z(I): return tuple([0]*len(I.TY))


def pol_b1(I):
    z = _z(I)
    def act(st):
        n, scr = st
        return (tuple(n), z, z, frozenset(range(len(scr))), frozenset())
    return act


def pol_b_fast(I, thr=0):
    z = _z(I); TY = I.TY
    def act(st):
        n, scr = st
        sv = tuple(n[k] if TY[k] in I.DUMP else 0 for k in range(len(TY)))
        fv = tuple(0 if TY[k] in I.DUMP else n[k] for k in range(len(TY)))
        cand = sorted([j for j in range(len(scr))
                       if scr[j][1] >= thr and I.VPS[scr[j]] > I.VS[scr[j][0]]],
                      key=lambda j: -I.VPS[scr[j]])[:I.CAP]
        ps = frozenset(cand); ss = frozenset(j for j in range(len(scr)) if j not in ps)
        return (sv, fv, z, ss, ps)
    return act


def pol_index(I, hold_beta=None):
    """policies.index_myopic 과 같은 정의 (상태열거 없음).

    hold_beta: None 이면 원 정의 — 선별품 중 V^PS>V^S 는 무기한 보유. 이 정의는 큰 W 에서 창고를
    채워 강제매각을 유발한다(scale1 §2-4). 숫자면 한계가치 순위 ≤ hold_beta·CAP 인 선별품만 보유하고
    나머지는 판다 (대규모용 재정의, INDEX_h). 신속검사 결정(EVSI−C_f>0)과 용량 배정은 원 정의 그대로.
    """
    from .policies import evsi
    TY = I.TY; nT = len(TY)
    fast_ok = {t: (evsi(I, t) - I.cfg.Cf > 0) for t in TY}
    CAP = max(I.CAP, 1)
    def act(st):
        n, scr = st; cand = []
        for k, t in enumerate(TY):
            if t in I.PUOK and n[k]:
                cand += [(I.VPU[t]-I.VS[t], 'U', k)]*n[k]
        for j in range(len(scr)):
            m = I.VPS[scr[j]] - I.VS[scr[j][0]]
            if m > 0: cand.append((m, 'S', j))
        cand.sort(key=lambda x: -x[0])
        pu = [0]*nT; ps = set()
        for v, kind, arg in cand[:I.CAP]:
            if v <= 0: break
            if kind == 'U': pu[arg] += 1
            else: ps.add(arg)
        sv = [0]*nT; fv = [0]*nT
        for k, t in enumerate(TY):
            c = n[k] - pu[k]
            if t in I.DUMP: sv[k] = c
            elif fast_ok[t]: fv[k] = c
            else: sv[k] = c
        ss = {j for j in range(len(scr)) if j not in ps and I.VPS[scr[j]] <= I.VS[scr[j][0]]}
        if hold_beta is not None:
            rest = [(m, j) for m, kind, j in cand[I.CAP:] if kind == 'S' and j not in ss]
            for r, (m, j) in enumerate(rest, start=1):
                if r > hold_beta*CAP: ss.add(j)
        return (tuple(sv), tuple(fv), tuple(pu), frozenset(ss), frozenset(ps))
    return act


def pol_lagr(I, mu, lam_eff=None, kappa=None, reserve=None, mu_route=None, beta=None):
    """라그랑주 지수정책. kappa=None 이면 대기 없음(LAGR), 숫자면 대기열(QUEUE).

    경로: 유형별 argmax{V^S, V^PU−μ, v_fast(μ)} (DUMP 유형은 SELL).
    용량 배정: 후보(PU 경로 미검사 + 한계가치>0 선별품)를 한계가치순으로 CAP 건.
    미처리 후보: LAGR 은 매각. QUEUE 는 순위 r(1..) 에서 기대 대기 r/CAP 기간,
      한계가치 m ≥ kappa·h·r/CAP 이면 보유, 아니면 매각. 보유 총량 ≤ W − reserve.
    mu_route: 경로 결정에만 쓰는 그림자가격 (기본 = mu). LP 해가 경계 유형을 분할하면
      argmax 경로는 그 유형을 통째로 SELL 로 보내 용량이 논다(M=16 셀, prec 15.4/16).
      mu_route < mu 로 두면 경계 유형이 FAST 로 들어와 노는 용량을 채우되, 보유 판정은
      참 μ 로 하므로 그 유형의 선별품은 자리가 남을 때만 정밀로 가고 아니면 팔린다.
    """
    TY = I.TY; nT = len(TY)
    if mu_route is None: mu_route = mu
    _, route = lagrangian(I, lam_eff if lam_eff is not None else 1.0, 0, mu_route)
    h = I.cfg.h; CAP = max(I.CAP, 1)
    W = I.W if I.W is not None else 10**9
    if reserve is None:
        reserve = lam_eff if lam_eff is not None else 0.0
    hold_max = max(int(W - reserve), 0)

    def act(st):
        n, scr = st
        sv = [0]*nT; fv = [0]*nT; pu = [0]*nT
        cand = []                                # (margin, kind, arg)
        for k, t in enumerate(TY):
            if not n[k]: continue
            r = route[t]
            if r == 'SELL': sv[k] = n[k]
            elif r == 'FAST': fv[k] = n[k]
            else: cand += [(I.VPU[t]-I.VS[t], 'U', k)]*n[k]
        ss = set(); ps = set()
        for j in range(len(scr)):
            m = I.VPS[scr[j]] - I.VS[scr[j][0]]
            if m > 0: cand.append((m, 'S', j))
            else: ss.add(j)
        cand.sort(key=lambda x: -x[0])
        for m, kind, arg in cand[:I.CAP]:
            if kind == 'U': pu[arg] += 1
            else: ps.add(arg)
        rest = cand[I.CAP:]
        held = 0
        # 보유 한도: 신속검사로 넘긴 대수(fv)도 자리를 차지한다. 그 위에 reserve 만큼
        # 도착 여유를 남긴다 — 넘치면 규칙이 가장 싼 유형부터 강제매각하기 때문이다.
        hmax = min(hold_max, max(int(W - sum(fv) - reserve), 0))
        # 보유 판정: 한계가치가 그림자가격 μ 를 넘고(유체 극한에서 결국 처리되는 후보),
        # 그 초과분이 기대 대기비용 κ·h·r/CAP 을 덮어야 한다. μ 를 빼지 않으면 구조적
        # 과부하(λ_eff ≫ CAP)에서 영원히 처리되지 않을 후보까지 쌓여 창고가 차고
        # 보관비·강제매각이 g 를 갉아먹는다 (컨테이너 시험: M=10 에서 갭 4.4% → 수정).
        # beta 가 주어지면 보유 판정을 바꾼다: 순위 r ≤ beta·CAP 인 **유한 버퍼**만 보유하고
        # (m 은 μ 와 무관하게 전체 한계가치), 그 안에서도 대기비용 κ·h·r/CAP 을 덮어야 한다.
        # 노는 용량(도착이 적은 날)을 채우기 위한 완충이며, 버퍼가 유한하므로 구조적 과부하에서도
        # 창고가 차지 않는다. beta=None 은 종전 규칙 (m − μ 기준).
        for r, (m, kind, arg) in enumerate(rest, start=1):
            if beta is None:
                keep = (kappa is not None) and held < hmax and (m - mu) >= kappa*h*(r/CAP)
            else:
                keep = (kappa is not None) and held < hmax and r <= beta*CAP and m >= kappa*h*(r/CAP)
            if keep:
                held += 1                        # 상태에 그대로 남는다
            else:
                if kind == 'U': sv[arg] += 1
                else: ss.add(arg)
        return (tuple(sv), tuple(fv), tuple(pu), frozenset(ss), frozenset(ps))
    return act


# ───────────────────────── 평가 ─────────────────────────
def evaluate(sim, act_of, T: int, seed: int, burn: int = 0, st=None):
    """단일 긴 궤적의 평균보상과 운영통계. dict 반환."""
    I = sim.I; rng = np.random.default_rng(seed)
    if st is None:
        st = (tuple([0]*len(I.TY)), ())
    for _ in range(burn):
        _, st = sim.step(st, act_of(st), rng)
    tot = 0.0; nprec = 0; nfast = 0; nsell = 0; forced = 0
    occs = np.zeros(T, np.int64)                 # 기간 시작 점유 (도착 포함) — 창고 설계용
    batch = []; bsum = 0.0; B = max(T//20, 1)
    for i in range(T):
        a = act_of(st)
        sv, fv, pu, ss, ps = a
        nprec += sum(pu) + len(ps); nfast += sum(fv); nsell += sum(sv) + len(ss)
        occs[i] = sum(st[0]) + len(st[1])
        r, st = sim.step(st, a, rng)
        forced += getattr(sim, 'last_forced', 0)
        tot += r; bsum += r
        if (i+1) % B == 0:
            batch.append(bsum/B); bsum = 0.0
    batch = np.array(batch)
    return dict(g=tot/T, se=float(batch.std(ddof=1)/np.sqrt(len(batch))) if len(batch) > 1 else 0.0,
                occ=float(occs.mean()), occ_p99=float(np.percentile(occs, 99)),
                occ_max=int(occs.max()),
                prec_per=nprec/T, fast_per=nfast/T, sell_per=nsell/T,
                forced_per=forced/T, T=T, seed=seed)
