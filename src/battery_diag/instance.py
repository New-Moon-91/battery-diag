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
    """2층 가격구조: P = M(s) * g(s, cap).  g=예정가격 헤도닉, M=낙찰배수(로그정규)

    w4 재캘리브레이션(2026-08) 이후 **M_s = 0** 이다. 낙찰배수가 SOH 에 의존하지
    않는다는 뜻이다. 근거: 2024-01 이후 실거래에서 낙찰배수를 SOH 로 회귀하면
    R^2 = 0.0009 이고 부호도 기존 추정(-0.9472)과 반대다. 낙찰배수를 실제로 움직이는
    것은 입찰참가자 수인데 이는 모형 외생이므로 M_s=0 으로 두고 로그정규 잡음의
    평균만 맞춘다.

    참가자수 회귀의 창을 정정한다 (w5 에서 원자료로 재현하며 확인).
      전기간 273건 : ln(참가자수) 계수 +0.2841, R^2=0.0736  (2명 1.83배 → 9명 2.81배)
      2024~  94건 : 계수 +0.2713, R^2=0.0521  (2명 1.92배 → 9명 2.89배)
    정합성 메모 §2.1 과 w4 는 앞의 값(+0.284, R^2=0.074)을 «2024-01 이후» 절에
    적었으나 그것은 전기간 값이다. 결론은 어느 창에서도 같다.
    따라서 Erev 의 SOH 의존성은 전부 헤도닉 g 의 g_s 항에서 나온다.
    exp(M_s*s) 항은 항등적으로 1 이 되지만 식은 그대로 둔다 — 구 파라미터
    (data/params_v5.json) 로 기존 결과를 재현할 때 같은 코드 경로를 쓰기 위해서다.
    """
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
    def VS(self, cap):
        """재활용 매각가치. 용량에 선형이라고 본 판이다 — w5 에서 폐기된다."""
        return self.p_rc * cap
    def key(self):
        v = (self.g_c0, self.g_cap, self.g_s, self.M_c0, self.M_s, self.M_sd, self.p_rc)
        return {} if v == _PRICE_V5 else {'price': list(v)}


# 구 파라미터(v5)의 7-튜플. 이 값과 정확히 같을 때만 캐시 해시에서 가격을 빼서
# "가격이 해시에 없던 시절" 의 사전을 복원한다 — 379GB 짜리 기존 캐시를 살리기 위해서다.
_PRICE_V5 = (8.256862475075309, 1.5815474565093999, 2.8545182109885903,
             1.5666485932513265, -0.9472082008955554, 0.545511013478258, 8708.0)


@dataclass
class PriceW5:
    """w5 가격구조: 실거래 낙찰가 **직접회귀**. 2층(예정가×낙찰배수) 구조를 폐기한다.

    폐기 이유. w4 재캘리브레이션에서 낙찰배수의 SOH 계수가 0 이 되면서(R²=0.0008,
    부호도 반대) 낙찰배수가 상수 잡음일 뿐이 됐다. 층을 나눌 의미가 사라졌으므로
    예정가격을 거치지 않고 낙찰가를 바로 회귀한다.

        ln P_reuse   = c0 + a·ln(cap) + b·ln(s) + c·ln(P_Li) + ε,  ε~N(0, σ²)
        ln V_recycle = d0 + e·ln(cap)           + f·ln(P_Li) + η,  η~N(0, τ²)

    **시점 효과를 리튬 시세로 설명한다.** 재활용가가 3~6배 변동한 것은 없앨
    노이즈가 아니라 설명해야 할 신호다. 월 더미 대신 리튬을 직접 넣으면 재활용
    R² 가 0.742(용량만) → 0.827 로, 월FE 상한 0.916 과의 격차를 절반가량 메운다.
    코발트는 넣지 않는다 — 계수가 음수로 나와 경제적 의미가 없고 27개 시점의
    다중공선성이 만든 인공물로 판단된다.

    리튬은 자료기간 시점평균 `Li_ref` 에 고정한다. 즉 **평균적 리튬 시세를 가정한
    정상상태 분석**이다. 리튬 탄력성이 재활용(0.501)이 재사용(0.295)보다 크므로
    리튬 강세기에는 두 채널의 격차가 좁아져 검사 유인이 줄어든다. 그 동학은
    실물옵션 등 후속 확장의 몫이다.

    **로그정규 보정 (`smear`).** 회귀가 ln 스케일이므로 exp(Xβ) 는 중앙값이지
    기댓값이 아니다. MDP 는 기대보상을 최대화하므로 기댓값
    E[P|X] = exp(Xβ + σ²/2) 를 써야 한다 (Duan smearing). 구 코드의 Erev 도
    같은 이유로 `0.5*M_sd**2` 를 달고 있었다. 보정은 재사용 +13.4%,
    재활용 +14.3% 의 수준 이동이고 σ 가 두 층에서 비슷해 **배율은 거의 불변**이다
    (0.8% 차). smear=False 로 두면 정합성 메모의 `*_c0_eff` 와 같은 중앙값 판이 된다.
    """
    reuse_c0: float; reuse_cap: float; reuse_s: float; reuse_li: float; reuse_sd: float
    recyc_c0: float; recyc_cap: float; recyc_li: float; recyc_sd: float
    Li_ref: float
    smear: bool = True

    @classmethod
    def from_json(cls, path):
        d = json.load(open(path, encoding='utf-8'))
        return cls(d['reuse_c0'], d['reuse_cap'], d['reuse_s'], d['reuse_li'], d['reuse_sd'],
                   d['recyc_c0'], d['recyc_cap'], d['recyc_li'], d['recyc_sd'],
                   d['Li_ref'], bool(d.get('smear', True)))

    # 리튬을 기준값에 고정해 상수항에 접어 넣은 유효 절편.
    @property
    def _re0(self):
        return (self.reuse_c0 + self.reuse_li*np.log(self.Li_ref)
                + (0.5*self.reuse_sd**2 if self.smear else 0.0))

    @property
    def _rc0(self):
        return (self.recyc_c0 + self.recyc_li*np.log(self.Li_ref)
                + (0.5*self.recyc_sd**2 if self.smear else 0.0))

    def Erev(self, s, cap):
        """정밀검사 통과 후 재사용 채널 기대 낙찰가 (팩당, 원)."""
        return np.exp(self._re0 + self.reuse_cap*np.log(cap)
                      + self.reuse_s*np.log(np.clip(s, 1e-3, None)))

    def VS(self, cap):
        """재활용 매각가치 (팩당, 원).

        구 판의 `p_rc * cap` 은 용량 지수를 1.0 으로 못박은 것인데, 실측 지수는
        월FE 통제하에 1.187 (t=24.8, R²=0.916) 이다. 즉 pooled p_rc 는 대용량 팩의
        재활용 가치를 과소평가한다. 여기서는 지수를 자유롭게 추정한 값으로 쓴다.

        한계: 원리적으로 회수가치를 정하는 것은 화학조성이지만, 한국 시장에서는
        조성과 용량이 거의 같은 변수라(LFP 용량중앙 16kWh, NCM 59.4kWh) 둘을 함께
        넣으면 NCM 더미가 유의성을 잃고 부호가 뒤집힌다 — 식별 불가. 용량을
        대리변수로 쓴다.
        """
        return np.exp(self._rc0 + self.recyc_cap*np.log(cap))

    def p_rc_at(self, cap):
        """참고용 kWh당 재활용 단가. 용량에 의존한다 (지수가 1이 아니므로)."""
        return self.VS(cap)/cap

    def key(self):
        return {'price_w5': [self.reuse_c0, self.reuse_cap, self.reuse_s, self.reuse_li,
                             self.reuse_sd, self.recyc_c0, self.recyc_cap, self.recyc_li,
                             self.recyc_sd, self.Li_ref, self.smear]}


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
    credit_loss: bool = False # 기존 정식화에서 min(n+1,NMAX) 로 잘린 도착에
                              # 재활용 매각수익 p_rc*kWh 를 계상 (소실보정 MDP)
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
            self.VS[t] = float(price.VS(cap))
            self.VPU[t] = q*sum(pb[b]*price.Erev(ms[b],cap) for b in range(c.SB)) \
                          + (1-q)*self.VS[t] - c.Cp
            qq = q/(q+(1-q)*(1-self.P_DET)) if q > 0 else 0.0
            self.QPOST[t] = qq
            for b in range(c.SB):
                self.VPS[(t,b)] = qq*price.Erev(ms[b],cap) + (1-qq)*self.VS[t] - c.Cp
        # 경제영역 R1/R2/R3
        #
        # 두 갈래로 매긴다.
        #  · REG      — 모형 내부 기준. hi = max_b V^PS, lo = min_b V^PS 를 V^S 와 비교.
        #               **가지치기(DUMP)는 반드시 이쪽을 써야 한다.** R1 은 "모든 신호에서
        #               정밀검사가 지배당함" 이라는 증명 가능한 명제이고, 그래야 즉시매각
        #               강제가 최적성을 해치지 않는다. 실측 배수로 바꿔 끼우면 근거 없이
        #               최적행동을 잘라내게 된다.
        #  · REG_EMP  — w4 [2] 의 실측 기준. 모형을 전혀 쓰지 않고 2024-01 이후 실거래
        #               낙찰가만으로 계산한 손익배수 (E[재사용]-V^S)/(C_p/q_P) 를
        #               임계 0.8/1.5 로 자른다 (battery_diag.data.reg_from_ratio).
        #               논문·보고서에 싣는 차종 분류는 이쪽이 1차 기준이다.
        #
        # 둘이 일치하는지가 그 자체로 검증이다(REG_AGREE). w4 인스턴스
        # (SM3·쏘울·볼트·코나)에서는 넷 다 일치한다 — 구 파라미터에서는 SM3·쏘울이
        # 어긋났다. results/w4/validation.md §3 참조.
        self.REG = {}
        for t in self.TY:
            hi = max(self.VPS[(t,b)] for b in range(c.SB))
            lo = min(self.VPS[(t,b)] for b in range(c.SB))
            self.REG[t] = 'R1' if hi <= self.VS[t] else ('R2' if lo > self.VS[t] else 'R3')
        # 실측 손익배수 — 유형키가 `차종_용량` 이면 w5 표(types_w5.json), 아니면
        # w4 의 차종 단위 표를 쓴다. 둘은 정의가 다르다:
        #   w4  r = (E - V^S) / (C_p / q̄_P)     전체 평균 통과확률로 고정
        #   w5  r = q_P (E - V^S) / C_p          유형별 통과확률 (정합성 메모 §4.2)
        from .data import reg_from_ratio as _r2r
        try:
            from .data import ratio_emp_w5 as _rw5
            _RE = dict(_rw5())
        except Exception:
            _RE = {}
        if not any(t in _RE for t in self.TY):
            from .data import RATIO_EMP as _RE4
            _RE = dict(_RE4)
        self.RATIO_EMP = {t: _RE[t] for t in self.TY if t in _RE}
        self.REG_EMP = {t: _r2r(r) for t, r in self.RATIO_EMP.items()}
        self.REG_AGREE = {t: (self.REG[t] == self.REG_EMP[t]) for t in self.REG_EMP}
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
            self._arrc = {}
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
        """잔여 자리 slack 에서의 (확률, 수용 개수벡터, 강제매각 수익, 강제매각 대수).

        도착이 slack 을 넘치면 초과분은 미검사 상태로 즉시 재활용 매각된다.
        어느 것을 파는지는 정책이 아니라 규칙이 정한다 — 매각가치가 낮은 차종부터.
        (회피하려면 정책이 미리 팔아 자리를 비우면 된다. 그 회피 행동이 곧 결과다.)
        """
        if slack in self._arrw: return self._arrw[slack]
        nT = len(self.TY); agg = {}
        for a, pr in self._ARRP:
            excess = sum(a) - slack
            rev = 0.0; n_sold = max(excess, 0)
            if excess > 0:
                aa = list(a)
                for k in self._sell_order:
                    if excess <= 0: break
                    take = min(aa[k], excess)
                    aa[k] -= take; excess -= take; rev += take*self.VS[self.TY[k]]
                a = tuple(aa)
            key = (a, rev, n_sold)
            agg[key] = agg.get(key, 0.) + pr
        out = [(p, a, rev, ns) for (a, rev, ns), p in agg.items()]
        self._arrw[slack] = out
        return out

    def arr_credit(self, n):
        """소실보정판 도착 — (확률, 다음 미검사벡터, 조건부 기대 매각수익).

        기존 arr() 은 n_t 가 NMAX 면 도착분을 조용히 버린다(무료 소실). 여기서는
        같은 전이를 그대로 두되 버려진 대수마다 p_rc*kWh 를 보상에 계상한다.
        상태공간·전이확률은 arr() 과 완전히 같고 보상만 다르다 — 그래야 "보정 MDP"
        가 원 MDP 와 같은 커널 위에서 정의된다.
        """
        key = tuple(n)
        if key in self._arrc: return self._arrc[key]
        c = self.cfg
        o = {key: (1.0, 0.0)}
        for _ in range(c.NARR):
            nx = {}
            def add(k2, p, dv, cv):
                a0, b0 = nx.get(k2, (0., 0.))
                nx[k2] = (a0+p, b0+p*(cv+dv))
            for nn, (p0, cv) in o.items():
                add(nn, p0*(1-c.lam), 0., cv)
                for k, t in enumerate(self.TY):
                    p = p0*c.lam*self.MIX[t]
                    if nn[k] >= c.NMAX:
                        add(nn, p, self.VS[t], cv)      # 잘림 → 매각수익 계상
                    else:
                        n2 = list(nn); n2[k] += 1
                        add(tuple(n2), p, 0., cv)
            o = {k: (v[0], v[1]/max(v[0], 1e-300)) for k, v in nx.items()}
        out = [(p, nn, rev) for nn, (p, rev) in o.items()]
        self._arrc[key] = out
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
                if c.credit_loss:
                    for pa, nn, rev in self.arr_credit(tuple(rem)):
                        out.append((p0*pa, self.SI[(nn, sc)], r+rr-hc+rev))
                else:
                    for nn, pa in self.arr(tuple(rem)):
                        out.append((p0*pa, self.SI[(nn, sc)], r+rr-hc))
            else:
                for pa, aa, rev, _ns in self._arr_slack(self.W - occ):
                    nn = tuple(rem[k]+aa[k] for k in range(nT))
                    out.append((p0*pa, self.SI[(nn, sc)], r+rr-hc+rev))
        return out

    def summary(self):
        cap_tot = self.W if self.W is not None else (self.cfg.NMAX*len(self.TY)+self.cfg.SMAX)
        d = dict(nS=len(self.ST), nA=self.n_actions(), CAP=self.CAP, Hslot=self.Hslot,
                 lam_eff=self.cfg.lam*self.cfg.NARR,
                 selectivity=self.CAP/cap_tot, REG=dict(self.REG),
                 REG_EMP=dict(self.REG_EMP), RATIO_EMP=dict(self.RATIO_EMP),
                 REG_AGREE=dict(self.REG_AGREE))
        if self.W is not None: d['W'] = self.W
        return d
