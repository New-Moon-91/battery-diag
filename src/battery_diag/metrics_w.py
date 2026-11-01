# -*- coding: utf-8 -*-
"""RCR 편 순환 성과지표 — **W 정식화** 전용. torch 불필요(순수 numpy).

W 정식화의 손실 경로는 NMAX 정식화와 다르다.
  · NMAX 판: 도착이 유형별 상한에 걸리면 조용히 사라진다(거부)
  · **W 판: 창고 초과분은 미검사 상태로 즉시 재활용 매각된다**(강제매각)
    — 수익은 들어오지만 **재사용 가능했을 배터리가 진단 없이 재활용으로 간다.**

따라서 좌초 손실은 세 경로에서 발생한다.
  L_sell    정책이 미검사 상태로 판 것 중 참 PASS 였던 용량
  L_screen  신속검사 후 판 것 중 (사후확률 기준) PASS 였던 용량
  L_forced  **창고 초과로 강제매각된 것 중 참 PASS 였던 용량**  ← W 판의 핵심
정밀검사에서 탈락한 몫은 참 FAIL 이므로 좌초가 아니다.

**L_forced 를 빼먹으면 전수평가(S1)를 과대평가한다.** 전수평가는 용량이 조이면
손실을 '오분류'가 아니라 '창고 넘침'으로 옮기기 때문이다. 그리고 그 넘침은
**규제 미이행**이기도 하다 — 평가 없이 매각되므로.
"""
from __future__ import annotations
import numpy as np


# ------------------------------------------------------------------ 정상분포
def policy_rows(arrays: dict, acts_local: np.ndarray) -> dict:
    ip, ix, pr, aptr = arrays['indptr'], arrays['indices'], arrays['probs'], arrays['aptr']
    nS = len(aptr) - 1
    rows = aptr[:-1] + np.asarray(acts_local, dtype=np.int64)
    starts = ip[rows]; cnt = ip[rows + 1] - starts
    total = int(cnt.sum()); offs = np.cumsum(cnt) - cnt
    pos = np.repeat(starts - offs, cnt) + np.arange(total)
    return dict(cols=ix[pos].astype(np.int64), vals=pr[pos],
                src=np.repeat(np.arange(nS), cnt), nS=nS, r=arrays['rsa'][rows])


def stationary(P: dict, tol: float = 1e-13, itmax: int = 200_000) -> np.ndarray:
    nS = P['nS']; mu = np.full(nS, 1.0/nS)
    for _ in range(itmax):
        nxt = np.zeros(nS)
        np.add.at(nxt, P['cols'], mu[P['src']] * P['vals'])
        s = nxt.sum()
        if s > 0: nxt /= s
        if np.abs(nxt - mu).max() < tol:
            return nxt
        mu = nxt
    return mu


def gain(P: dict, mu: np.ndarray) -> float:
    return float(mu @ P['r'])


# ------------------------------------------------------------------ 강제매각
def forced_sale_stats(I, slack: int, _cache=None):
    """잔여 자리 slack 에서 이번 기간 강제매각의 기대 (대수, 용량kWh, 좌초kWh).

    Instance._arr_slack 과 **동일한 규칙**(매각가치 낮은 차종부터)을 따르되
    유형별 매각 대수를 추적해 q_P 가중 좌초 용량을 낸다.

    캐시는 **인스턴스에 붙인다**. 종전의 모듈 전역 dict 에 key=(id(I), slack) 는 이전
    인스턴스가 해제된 뒤 새 인스턴스가 같은 id 를 받으면 다른 셀(다른 lam·NARR·가격)의
    값을 돌려주는 버그가 있었다 (RCR 용어 ㉑, 서버패치 0002 에 해당).
    """
    cache = I.__dict__.setdefault('_forced_cache', {})
    key = slack
    if key in cache: return cache[key]
    nT = len(I.TY); n_sold = kwh = strand = 0.0
    for a, pr in I._ARRP:
        excess = sum(a) - slack
        if excess <= 0: continue
        aa = list(a); e = excess
        for k in I._sell_order:
            if e <= 0: break
            take = min(aa[k], e); aa[k] -= take; e -= take
            t = I.TY[k]
            kwh += pr * take * I.KWH[t]
            strand += pr * take * I.QP[t] * I.KWH[t]
        n_sold += pr * excess
    out = (n_sold, kwh, strand)
    cache[key] = out
    return out


def _survivor_dist(I, fv):
    """신속검사 fv 건 중 결함 미검출(=선별완료로 남는) 개수의 분포."""
    dist = {0: 1.0}
    for k, t in enumerate(I.TY):
        if fv[k] == 0: continue
        p_surv = 1.0 - (1.0 - I.QP[t]) * I.P_DET
        for _ in range(fv[k]):
            nd = {}
            for m, p in dist.items():
                nd[m] = nd.get(m, 0.) + p * (1 - p_surv)
                nd[m+1] = nd.get(m+1, 0.) + p * p_surv
            dist = nd
    return dist


# ------------------------------------------------------------------ 상태-행동 기여
def sa_contrib(I, st, a) -> dict:
    n, scr = st; sv, fv, pu, ss, ps = a
    TY, KWH, QP, QPOST = I.TY, I.KWH, I.QP, I.QPOST

    sold_kwh = sum(sv[k]*KWH[t] for k, t in enumerate(TY))
    L_sell = sum(sv[k]*QP[t]*KWH[t] for k, t in enumerate(TY))
    L_screen = 0.0
    for j in ss:
        t = scr[j][0]; sold_kwh += KWH[t]; L_screen += QPOST[t]*KWH[t]

    prec_kwh = sum(pu[k]*KWH[t] for k, t in enumerate(TY)) + sum(KWH[scr[j][0]] for j in ps)
    prec_cnt = sum(pu) + len(ps)
    reuse_kwh = (sum(pu[k]*QP[t]*KWH[t] for k, t in enumerate(TY))
                 + sum(QPOST[scr[j][0]]*KWH[scr[j][0]] for j in ps))
    fast_cnt = sum(fv)
    fast_rej_kwh = sum(fv[k]*(1-QP[t])*I.P_DET*KWH[t] for k, t in enumerate(TY))

    # 행동 후 점유 → 강제매각 (신속검사 결과에 따라 점유가 달라지므로 분포로 처리)
    rem = [n[k]-sv[k]-fv[k]-pu[k] for k in range(len(TY))]
    base = sum(rem) + len([j for j in range(len(scr)) if j not in ss and j not in ps])
    f_cnt = f_kwh = f_str = 0.0
    if I.W is not None:
        for m, p in _survivor_dist(I, fv).items():
            c1, c2, c3 = forced_sale_stats(I, max(I.W - (base + m), 0))
            f_cnt += p*c1; f_kwh += p*c2; f_str += p*c3

    return dict(prec_cnt=prec_cnt, prec_kwh=prec_kwh, reuse_kwh=reuse_kwh, fast_cnt=fast_cnt,
                recycle_kwh=sold_kwh + (prec_kwh-reuse_kwh) + fast_rej_kwh + f_kwh,
                L_sell=L_sell, L_screen=L_screen, L_forced=f_str,
                forced_cnt=f_cnt, forced_kwh=f_kwh)


# ------------------------------------------------------------------ 집계
def outcome_metrics(I, arrays, acts_local, mu=None, eps_new=74.0, eps_repurp=7.73,
                    eps_diag=0.319, beta=1.0, rho_required=None) -> dict:
    """기간당 기대 성과. eps_* 단위 kgCO2e/kWh.

    rho_required  처리량 환산에 쓸 '소요' 정밀 사용률.
                  S1(전수평가)은 정의상 1.0 — 실현 rho 는 강제매각 때문에 1보다 작다.
    """
    P = policy_rows(arrays, acts_local)
    if mu is None: mu = stationary(P)
    keys = ['prec_cnt','prec_kwh','reuse_kwh','fast_cnt','recycle_kwh',
            'L_sell','L_screen','L_forced','forced_cnt','forced_kwh']
    acc = {k: 0.0 for k in keys}
    for si, st in enumerate(I.ST):
        w = mu[si]
        if w < 1e-15: continue
        c = sa_contrib(I, st, I.actions(st)[int(acts_local[si])])
        for k in keys: acc[k] += w*c[k]

    lam_eff = I.cfg.lam * I.cfg.NARR
    L_total = acc['L_sell'] + acc['L_screen'] + acc['L_forced']
    dE = acc['reuse_kwh']*(beta*eps_new - eps_repurp) - acc['prec_kwh']*eps_diag
    rho = acc['prec_cnt']/lam_eff if lam_eff > 0 else 0.0
    return dict(
        g=gain(P, mu), rho=rho,
        rho_required=(rho if rho_required is None else rho_required),
        prec_per_period=acc['prec_cnt'], fast_per_period=acc['fast_cnt'],
        U_kwh=acc['reuse_kwh'],
        L_sell_kwh=acc['L_sell'], L_screen_kwh=acc['L_screen'],
        L_forced_kwh=acc['L_forced'], L_total_kwh=L_total,
        D_kwh=acc['prec_kwh'], recycle_kwh=acc['recycle_kwh'],
        forced_cnt=acc['forced_cnt'], forced_rate=acc['forced_cnt']/lam_eff if lam_eff else 0.0,
        forced_kwh=acc['forced_kwh'], dE_kg=dE, lam_eff=lam_eff)


def required_throughput(rho_required, Lambda, n_center=60, D=245):
    """q = Lambda x rho / (n_center x D) — 검사소당 소요 일처리 건수."""
    return dict(national_per_day=Lambda*rho_required/D,
                per_center_per_day=Lambda*rho_required/(n_center*D))
