# -*- coding: utf-8 -*-
"""선별센터 평균보상 MDP 인스턴스 정의.

자원 (2026-08 정정판)
  · 신속검사(EIS 진단기, 10~15분) — 충방전기와 **별도 장비**. 용량 제약 없음, 비용 C_f.
  · 정밀검사(충방전 용량시험, T_p=14h) — 충방전기 M_cyc 대.
  · 자가방전(T_hold=24h) — 항온실 H_slot 자리.
      유효 정밀 용량  CAP = min(M_cyc, floor(H_slot * T_p / T_hold))
      설계규칙        H_slot >= M_cyc * (T_hold / T_p)   (=1.71 M @ 14h/24h)

행동 = 배터리별 배정. 미검사 (sv, fv, pu) 유형별 개수 / 선별완료 ss(판매), ps(정밀).
지배 가지치기: R1 유형 즉시판매 강제, V^PS <= V^S 선별품 즉시판매 강제.
"""
from __future__ import annotations
import itertools, json
import numpy as np
from math import erf, sqrt, factorial
from dataclasses import dataclass, field

_Phi = lambda z: 0.5 * (1 + erf(z / sqrt(2)))


@dataclass
class PriceParams:
    """2층 가격구조: P = M(s) * g(s, cap).  g=예정가격 헤도닉, M=낙찰배수(로그정규)"""
    g_c0: float; g_cap: float; g_s: float
    M_c0: float; M_s: float; M_sd: float
    p_rc: float
    @classmethod
    def from_json(cls, path):
        d = json.load(open(path))
        return cls(d['g_c0'], d['g_cap'], d['g_s'], d['M_c0'], d['M_s'], d['M_sd'], d['p_rc'])
    def g(self, s, cap):
        return np.exp(self.g_c0 + self.g_cap*np.log(cap) + self.g_s*np.log(np.clip(s,1e-3,None)))
    def Erev(self, s, cap):
        return np.exp(self.M_c0 + self.M_s*s + 0.5*self.M_sd**2) * self.g(s, cap)


@dataclass
class Config:
    lam: float = 0.9          # 도착 슬롯당 확률
    NARR: int = 1             # 기간당 도착 슬롯 수 (배치 도착)
    Cf: float = 20_000.       # 신속검사 변동비
    Cp: float = 500_000.      # 정밀검사 변동비 (환경부 2025.5.14 목표치 기준)
    h: float = 3_000.         # 보관비/개·기간
    phi: float = 1.0          # 미공개 결함 중 신속검사로 못 잡는 비율
    Mcyc: int = 1             # 충방전기 대수
    Hslot: int | None = None  # 항온실 자리 (None → ceil(Mcyc*Thold/Tp))
    Tp: float = 14.0
    Thold: float = 24.0
    sig_f: float = 0.05       # EIS SOH 추정 오차
    NMAX: int = 2             # 유형별 미검사 버퍼 상한 (W=None 일 때만 유효)
    SMAX: int = 2             # 선별완료 버퍼 상한 (W=None 일 때만 유효)
    W: int | None = None      # 창고 총 수용 대수. 지정하면 W-정식화
    SB: int = 3               # 신호 구간 수
    prune: bool = True
    F_E: float = 278/354      # 신속검사 확정 판별 결함 비중
    F_U: float = 62/354       # 미공개


class Instance:
    def __init__(self, types: dict, price: PriceParams, cfg: Config):
        self.types, self.pp, self.cfg = types, price, cfg
        c = cfg
        self.TY = list(types)
        self.QP  = {t: types[t][0] for t in self.TY}
        self.KWH = {t: types[t][1] for t in self.TY}
        self.MU  = {t: types[t][2] for t in self.TY}
        self.SD  = {t: types[t][3] for t in self.TY}
        self.MIX = {t: types[t][4] for t in self.TY}
        self.Hslot = c.Hslot if c.Hslot is not None else int(np.ceil(c.Mcyc*c.Thold/c.Tp))
        self.CAP = max(int(min(c.Mcyc, np.floor(self.Hslot*c.Tp/c.Thold))), 0)
        self.P_DET = c.F_E + (1-c.phi)*c.F_U
        # 신호 구간 (3분위) 과 사후평균
        self.BP = {}
        for t in self.TY:
            mu, sd = self.MU[t], self.SD[t]
            sig = np.sqrt(sd**2 + c.sig_f**2); cut = [mu-.5*sig, mu+.5*sig]
            p = np.array([_Phi((cut[0]-mu)/sig),
                          _Phi((cut[1]-mu)/sig)-_Phi((cut[0]-mu)/sig),
                          1-_Phi((cut[1]-mu)/sig)])
            m = [float(np.clip(x,.05,1.)) for x in (mu-1.1*sig, mu, mu+1.1*sig)]
            self.BP[t] = (p, m)
        # 처분 가치
        self.VS, self.VPU, self.VPS, self.QPOST = {}, {}, {}, {}
        for t in self.TY:
            q = self.QP[t]; pb, ms = self.BP[t]; cap = self.KWH[t]
            self.VS[t] = price.p_rc * cap
            self.VPU[t] = q*sum(pb[b]*price.Erev(ms[b],cap) for b in range(c.SB)) \
                          + (1-q)*self.VS[t] - c.Cp
            qq = q/(q+(1-q)*(1-self.P_DET)) if q > 0 else 0.0
            self.QPOST[t] = qq
            for b in range(c.SB):
                self.VPS[(t,b)] = qq*price.Erev(ms[b],cap) + (1-qq)*self.VS[t] - c.Cp
        # 경제영역 R1/R2/R3
        self.REG = {}
        for t in self.TY:
            hi = max(self.VPS[(t,b)] for b in range(c.SB))
            lo = min(self.VPS[(t,b)] for b in range(c.SB))
            self.REG[t] = 'R1' if hi <= self.VS[t] else ('R2' if lo > self.VS[t] else 'R3')
        self.DUMP  = {t for t in self.TY if c.prune and self.REG[t]=='R1'}
        self.SIT   = [(t,b) for t in self.TY for b in range(c.SB)]
        self.SDUMP = {x for x in self.SIT if c.prune and self.VPS[x] <= self.VS[x[0]]}
        self.PUOK  = {t for t in self.TY if self.VPU[t] > self.VS[t] and t not in self.DUMP}
        # 상태공간
        self.W = c.W
        if self.W is None:
            self.SCR = [()] + [tuple(sorted(x)) for k in range(1, c.SMAX+1)
                               for x in itertools.combinations_with_replacement(self.SIT, k)]
            self.NV = list(itertools.product(range(c.NMAX+1), repeat=len(self.TY)))
            self.ST = [(n, sc) for n in self.NV for sc in self.SCR]
        else:
            W = self.W; nT = len(self.TY)
            scr_k = [[()] if k == 0 else
                     [tuple(sorted(x)) for x in
                      itertools.combinations_with_replacement(self.SIT, k)]
                     for k in range(W+1)]
            self.SCR = [sc for k in range(W+1) for sc in scr_k[k]]
            self.NV = [n for s in range(W+1)
                       for n in itertools.product(range(s+1), repeat=nT) if sum(n) == s]
            self.ST = [(n, sc) for n in self.NV
                       for k in range(W - sum(n) + 1) for sc in scr_k[k]]
            # 강제매각 순서: 재활용 매각가치 p_rc*kWh 가 낮은 차종부터
            self._sell_order = sorted(range(nT), key=lambda k: self.VS[self.TY[k]])
            self._ARRP = self._arr_dist()
            self._arrw = {}
        self.SI = {x: i for i, x in enumerate(self.ST)}
        self._acts = {}

    # ---------- 도착 (W-정식화) ----------
    def _arr_dist(self):
        """NARR 슬롯 도착의 차종별 개수 분포 — (개수벡터, 확률) 목록.

        각 슬롯은 확률 lam 로 도착하고 차종은 MIX 를 따른다. 슬롯이 독립이므로
        개수벡터는 다항분포다. 기존 arr() 의 축차 합성곱과 달리 유형별 상한이
        없으므로 닫힌 형태로 한 번에 만든다.
        """
        c = self.cfg; nT = len(self.TY)
        ps = [c.lam*self.MIX[t] for t in self.TY]; p0 = 1 - c.lam
        out = []
        for a in itertools.product(range(c.NARR+1), repeat=nT):
            A = sum(a)
            if A > c.NARR: continue
            coef = factorial(c.NARR)
            for x in a: coef //= factorial(x)
            coef //= factorial(c.NARR - A)
            pr = coef * p0**(c.NARR-A)
            for k in range(nT): pr *= ps[k]**a[k]
            if pr > 0: out.append((a, pr))
        return out

    def _arr_slack(self, slack):
        """잔여 자리 slack 에서의 (확률, 수용 개수벡터, 강제매각 수익).

        도착이 slack 을 넘치면 초과분은 미검사 상태로 즉시 재활용 매각된다.
        어느 것을 파는지는 정책이 아니라 규칙이 정한다 — 매각가치가 낮은 차종부터.
        (회피하려면 정책이 미리 팔아 자리를 비우면 된다. 그 회피 행동이 곧 결과다.)
        """
        if slack in self._arrw: return self._arrw[slack]
        nT = len(self.TY); agg = {}
        for a, pr in self._ARRP:
            excess = sum(a) - slack
            rev = 0.0
            if excess > 0:
                aa = list(a)
                for k in self._sell_order:
                    if excess <= 0: break
                    take = min(aa[k], excess)
                    aa[k] -= take; excess -= take; rev += take*self.VS[self.TY[k]]
                a = tuple(aa)
            key = (a, rev)
            agg[key] = agg.get(key, 0.) + pr
        out = [(p, a, rev) for (a, rev), p in agg.items()]
        self._arrw[slack] = out
        return out

    # ---------- 도착 (기존 NMAX 정식화) ----------
    def arr(self, n):
        o = {tuple(n): 1.0}
        for _ in range(self.cfg.NARR):
            nx = {}
            for nn, p0 in o.items():
                nx[nn] = nx.get(nn, 0.) + p0*(1-self.cfg.lam)
                for k, t in enumerate(self.TY):
                    n2 = list(nn); n2[k] = min(n2[k]+1, self.cfg.NMAX); n2 = tuple(n2)
                    nx[n2] = nx.get(n2, 0.) + p0*self.cfg.lam*self.MIX[t]
            o = nx
        return list(o.items())

    # ---------- 행동 ----------
    def actions(self, st):
        if st in self._acts: return self._acts[st]
        n, scr = st; A = []
        forced = frozenset(j for j in range(len(scr)) if scr[j] in self.SDUMP)
        free_j = [j for j in range(len(scr)) if j not in forced]
        per = []
        for k, t in enumerate(self.TY):
            if t in self.DUMP:
                per.append([(n[k], 0, 0)])
            else:
                o = []
                for sell in range(n[k]+1):
                    for fast in range(n[k]-sell+1):
                        pmax = n[k]-sell-fast if t in self.PUOK else 0
                        for pu in range(pmax+1): o.append((sell, fast, pu))
                per.append(o)
        sopt = []
        for combo in itertools.product(*[['s','p','h']]*len(free_j)):
            ss, ps = set(forced), set()
            for j, ch in zip(free_j, combo):
                if ch == 's':   ss.add(j)
                elif ch == 'p': ps.add(j)
            sopt.append((frozenset(ss), frozenset(ps)))
        for cell in itertools.product(*per):
            sv = tuple(x[0] for x in cell); fv = tuple(x[1] for x in cell); pu = tuple(x[2] for x in cell)
            npu = sum(pu)
            if npu > self.CAP: continue
            for ss, ps in sopt:
                if npu + len(ps) > self.CAP: continue
                A.append((sv, fv, pu, ss, ps))
        self._acts[st] = A
        return A

    def n_actions(self):
        return sum(len(self.actions(st)) for st in self.ST)

    # ---------- 전이 ----------
    def step_dist(self, st, a):
        """(확률, 다음상태 인덱스, 즉시보상) 목록. 보상은 커밋 시점 기대수익 계상."""
        n, scr = st; sv, fv, pu, ss, ps = a; c = self.cfg
        r = (sum(sv[k]*self.VS[t] for k, t in enumerate(self.TY))
             + sum(pu[k]*self.VPU[t] for k, t in enumerate(self.TY))
             + sum(self.VS[scr[j][0]] for j in ss)
             + sum(self.VPS[scr[j]] for j in ps)
             - c.Cf*sum(fv))
        rem = [n[k]-sv[k]-fv[k]-pu[k] for k in range(len(self.TY))]
        sc2 = [scr[j] for j in range(len(scr)) if j not in ss and j not in ps]
        combos = [(1.0, tuple(sc2), 0.0)]
        for k, t in enumerate(self.TY):
            q = self.QP[t]; pdet = (1-q)*self.P_DET; pb, _ = self.BP[t]
            for _ in range(fv[k]):
                nxt = []
                for p0, sc, rr in combos:
                    nxt.append((p0*pdet, sc, rr+self.VS[t]))          # 결함 검출 → 재활용
                    for b in range(c.SB):
                        # W-정식화에서는 신속검사가 미검사→선별완료 이동일 뿐이라
                        # 총 점유가 변하지 않는다. 넘칠 자리가 없으므로 강제매각도 없다.
                        if self.W is not None or len(sc) < c.SMAX:
                            nxt.append((p0*(1-pdet)*pb[b], tuple(sorted(sc+((t,b),))), rr))
                        else:                                          # 선별버퍼 초과 → 즉시 매각
                            nxt.append((p0*(1-pdet)*pb[b], sc, rr+self.VS[t]))
                combos = nxt
        out = []
        nT = len(self.TY)
        for p0, sc, rr in combos:
            occ = sum(rem) + len(sc)
            hc = c.h*occ                       # 보관비는 도착 전 점유에 부과
            if self.W is None:
                for nn, pa in self.arr(tuple(rem)):
                    out.append((p0*pa, self.SI[(nn, sc)], r+rr-hc))
            else:
                for pa, aa, rev in self._arr_slack(self.W - occ):
                    nn = tuple(rem[k]+aa[k] for k in range(nT))
                    out.append((p0*pa, self.SI[(nn, sc)], r+rr-hc+rev))
        return out

    def summary(self):
        cap_tot = self.W if self.W is not None else (self.cfg.NMAX*len(self.TY)+self.cfg.SMAX)
        d = dict(nS=len(self.ST), nA=self.n_actions(), CAP=self.CAP, Hslot=self.Hslot,
                 lam_eff=self.cfg.lam*self.cfg.NARR,
                 selectivity=self.CAP/cap_tot, REG=dict(self.REG))
        if self.W is not None: d['W'] = self.W
        return d
