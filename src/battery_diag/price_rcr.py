# -*- coding: utf-8 -*-
"""RCR 편 가격모형 — 2층(행정 예정가 x 낙찰배수) + 리튬.

    P_reuse   = M x g(cap, s, Li),   ln M ~ N(M_c0, M_sd^2)  속성·리튬 독립
    V_recycle = exp(rc0 + rcap ln cap + rli ln Li)

EJOR 편의 PriceW5(낙찰가 직접회귀)와 **수준은 일치**하고 구조만 다르다.
분해가 필요한 이유는 예정가 g 가 **정부가 쥔 정책 레버**이기 때문이다 —
"예정가 산정식을 바꾸면 순환 경로 배분이 어떻게 달라지는가"를 물을 수 있다.

주의: 재활용 용량지수는 1.0 이 아니다(1.2675). 선형 근사를 쓰면 소형 팩의
재활용 가치가 3배 가까이 과소평가되어 **유형 등급이 전부 R2 로 무너진다.**
"""
from __future__ import annotations
import json
import numpy as np
from dataclasses import dataclass


@dataclass
class PriceRCR:
    g_c0: float; g_cap: float; g_s: float; g_li: float; g_sd: float
    M_c0: float; M_sd: float
    recyc_c0: float; recyc_cap: float; recyc_li: float; recyc_sd: float
    Li: float

    @classmethod
    def from_json(cls, path, Li=None):
        d = json.load(open(path))
        return cls(d['g_c0'], d['g_cap'], d['g_s'], d['g_li'], d['g_sd'],
                   d['M_c0'], d['M_sd'],
                   d['recyc_c0'], d['recyc_cap'], d['recyc_li'], d['recyc_sd'],
                   float(Li if Li is not None else d['Li_ref']))

    def with_Li(self, Li):
        from dataclasses import replace
        return replace(self, Li=float(Li))

    def g(self, s, cap):
        """행정 예정가 (팩당, 원) — 정책 레버."""
        return np.exp(self.g_c0 + self.g_cap*np.log(cap)
                      + self.g_s*np.log(np.clip(s, 1e-3, None))
                      + self.g_li*np.log(self.Li))

    def Erev(self, s, cap):
        """정밀검사 통과 후 재사용 채널 기대 낙찰가.
        로그정규 smearing 은 낙찰배수와 예정가 잔차 **양쪽**에 적용한다."""
        return np.exp(self.M_c0 + 0.5*(self.M_sd**2 + self.g_sd**2)) * self.g(s, cap)

    def VS(self, cap):
        """재활용 매각가치 — 용량지수 자유추정(1.2675). 선형 근사 금지."""
        return np.exp(self.recyc_c0 + self.recyc_cap*np.log(cap)
                      + self.recyc_li*np.log(self.Li) + 0.5*self.recyc_sd**2)

    def key(self):
        return {'price_rcr': [self.g_c0, self.g_cap, self.g_s, self.g_li, self.g_sd,
                              self.M_c0, self.M_sd, self.recyc_c0, self.recyc_cap,
                              self.recyc_li, self.recyc_sd, self.Li]}


# ---------------------------------------------------------------------------
# 유찰 보정판 (2026-11-01) — Omega 입찰규칙 원고의 E[max(R, W)] 를 2층 구조에 반영
# ---------------------------------------------------------------------------
from math import erf, sqrt as _sqrt
_Phi = lambda z: 0.5*(1 + erf(z/_sqrt(2)))


@dataclass
class PriceRCRCensored(PriceRCR):
    """PriceRCR + 유찰 보정.

    낙찰배수 M 은 **낙찰 표본**(273건)의 평균 2.28 이다. 그러나 재사용급 로트의 35~65% 가
    유찰되어 예정가 ≈ R 에 수의계약되므로, 판매자가 받는 배수의 기대값은 E[max(1, M~)]
    (M~ = W/R 잠재 배수) 이다. ln M~ 의 Tobit(scripts/fit_tobit_mult.py; 공변량 ln cap·ln s·ln Li,
    σ=0.691)로 r = E[max(1,M~)]/E[M~|M~≥1] 을 구해 Erev 에 곱한다 — EJOR 편 PriceCensored 와
    같은 보정이라 두 편의 수준이 함께 움직인다. 재활용 채널은 관측 유찰률·수의계약가로 만든
    r_rc(scripts/fit_recycle_fail.py, 0.924)를 V^S 에 곱한다.

    **리튬 스윕에 새 경로가 생긴다.** r 의 리튬 탄력성 +0.34 — 리튬이 낮을수록 유찰이 늘어
    재사용 채널이 추가로 좁아진다. `with_Li` 가 보정비도 함께 움직인다.
    """
    t_m0: float = 0.0; t_cap: float = 0.0; t_s: float = 0.0; t_li: float = 0.0; t_sig: float = 1.0
    rc_ratio: float = 1.0

    @classmethod
    def from_json(cls, path, Li=None, tobit_path='data/params_tobit_mult.json',
                  recycle_path='data/params_recycle_fail.json'):
        d = json.load(open(path))
        T = json.load(open(tobit_path, encoding='utf-8')); c = T['coef']
        rc = float(json.load(open(recycle_path, encoding='utf-8'))['r_rc']) if recycle_path else 1.0
        return cls(d['g_c0'], d['g_cap'], d['g_s'], d['g_li'], d['g_sd'],
                   d['M_c0'], d['M_sd'],
                   d['recyc_c0'], d['recyc_cap'], d['recyc_li'], d['recyc_sd'],
                   float(Li if Li is not None else d['Li_ref']),
                   c['m0'], c['m_cap'], c['m_s'], c['m_li'], T['sigma'], rc)

    def _mu(self, s, cap):
        return (self.t_m0 + self.t_cap*np.log(cap) + self.t_s*np.log(np.clip(s, 1e-3, None))
                + self.t_li*np.log(self.Li))

    def G(self, s, cap):
        return _Phi(-self._mu(s, cap)/self.t_sig)

    def ratio(self, s, cap):
        m, sg = self._mu(s, cap), self.t_sig
        tail = np.exp(m + 0.5*sg**2)*_Phi((m + sg**2)/sg)
        return (_Phi(-m/sg) + tail)/(tail/max(_Phi(m/sg), 1e-12))

    def Erev(self, s, cap):
        return super().Erev(s, cap)*self.ratio(s, cap)

    def VS(self, cap):
        return super().VS(cap)*self.rc_ratio

    def key(self):
        k = super().key()
        k['censor'] = [self.t_m0, self.t_cap, self.t_s, self.t_li, self.t_sig, self.rc_ratio]
        return k
